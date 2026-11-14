"""Shared HTTP boundary for the local workspace and retained v1 API."""

import asyncio
import base64
import binascii
from collections import deque
from dataclasses import dataclass, field
import ipaddress
import os
import secrets
import threading
import time
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


def is_loopback(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
        mapped = getattr(address, "ipv4_mapped", None)
        return (mapped or address).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class AccessPolicy:
    api_key: str = field(default="", repr=False)
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
    dev_origins: tuple[str, ...] = ()
    max_body_bytes: int = 2_100_000
    body_timeout: float = 15
    writes_per_minute: int = 60
    concurrent_jobs: int = 2
    concurrent_bodies: int = 8

    def __post_init__(self):
        if (
            min(
                self.max_body_bytes,
                self.body_timeout,
                self.writes_per_minute,
                self.concurrent_jobs,
                self.concurrent_bodies,
            )
            <= 0
        ):
            raise ValueError("Request limits must be positive.")
        if self.api_key and (
            len(self.api_key) < 32 or any(not 33 <= ord(c) <= 126 for c in self.api_key)
        ):
            raise ValueError(
                "SODIUM_API_KEY must contain at least 32 printable ASCII characters without spaces."
            )
        if not self.allowed_hosts or any(
            not host or any(c in host for c in "/*@?# ") for host in self.allowed_hosts
        ):
            raise ValueError("Configure explicit allowed hostnames; wildcards are disabled.")

    @classmethod
    def from_env(cls):
        hosts = tuple(
            name.strip().lower()
            for name in os.getenv("SODIUM_ALLOWED_HOSTS", "localhost,127.0.0.1,::1").split(",")
        )
        origins = (
            ("http://localhost:5173", "http://127.0.0.1:5173")
            if os.getenv("SODIUM_DEV_MODE") == "1"
            else ()
        )
        return cls(
            api_key=os.getenv("SODIUM_API_KEY", ""), allowed_hosts=hosts, dev_origins=origins
        )


def require_safe_bind(host: str, policy: AccessPolicy):
    if host != "localhost" and not is_loopback(host) and not policy.api_key:
        raise ValueError("Network binding requires SODIUM_API_KEY. Use loopback for local mode.")


class WorkspaceBoundary:
    """Validate hosts, peer access, browser origins, body bytes, and work admission.

    Limits are per process. Remote operators share one workspace; an API key is
    an access boundary, not per-user authorization. Use TLS outside loopback.
    """

    def __init__(self, app, policy: AccessPolicy):
        self.app, self.policy = app, policy
        self._writes, self._failures = deque(), deque()
        self._active = 0
        self._reading = 0
        self._lock = threading.Lock()

    def _admit(self, queue, limit):
        now = time.monotonic()
        with self._lock:
            while queue and queue[0] <= now - 60:
                queue.popleft()
            if len(queue) >= limit:
                return False
            queue.append(now)
            return True

    def _authenticated(self, headers):
        value = headers.get("authorization", "")
        if len(value) > 2048:
            return False
        scheme, _, credential = value.partition(" ")
        if scheme.lower() == "basic":
            try:
                decoded = base64.b64decode(credential, validate=True).decode("utf-8")
            except (ValueError, UnicodeError, binascii.Error):
                return False
            user, _, credential = decoded.partition(":")
            if user != "workspace":
                return False
        elif scheme.lower() != "bearer":
            return False
        return secrets.compare_digest(credential.encode(), self.policy.api_key.encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        policy = self.policy
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        request_id = secrets.token_hex(12)

        async def secure_send(message):
            if message["type"] == "http.response.start":
                protected = {
                    b"x-request-id": request_id.encode(),
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"same-origin",
                    b"cache-control": b"no-store",
                    b"content-security-policy": b"frame-ancestors 'none'; object-src 'none'; base-uri 'self'",
                }
                message = dict(message)
                message["headers"] = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() not in protected
                ] + list(protected.items())
            await send(message)

        async def reject(status, detail, extra=None):
            await JSONResponse({"detail": detail}, status_code=status, headers=extra)(
                scope, receive, secure_send
            )

        host_values = [v for k, v in scope["headers"] if k.lower() == b"host"]
        try:
            if len(host_values) != 1:
                raise ValueError()
            authority = host_values[0].decode("ascii")
            parsed = urlsplit("http://" + authority)
            if (
                parsed.hostname not in policy.allowed_hosts
                or parsed.username is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            ):
                raise ValueError()
        except (ValueError, UnicodeError):
            return await reject(400, "Untrusted Host header.")

        peer = (scope.get("client") or ("", 0))[0]
        if not policy.api_key and not is_loopback(peer):
            return await reject(403, "Remote access requires a configured workspace API key.")
        origin = headers.get("origin")
        same_origin = f"{scope.get('scheme', 'http')}://{authority}"
        if origin and origin not in (same_origin, *policy.dev_origins):
            return await reject(403, "Cross-origin requests are disabled.")
        if headers.get("sec-fetch-site") == "cross-site" and (
            scope["path"].startswith("/api/") or scope["method"] not in {"GET", "HEAD"}
        ):
            return await reject(403, "Cross-site workspace access is disabled.")
        if policy.api_key and not self._authenticated(headers):
            if not self._admit(self._failures, 30):
                return await reject(429, "Too many authentication attempts.", {"Retry-After": "60"})
            return await reject(
                401,
                "Workspace authentication required.",
                {"WWW-Authenticate": 'Basic realm="Sodium Twin", charset="UTF-8"'},
            )

        lengths = [v for k, v in scope["headers"] if k.lower() == b"content-length"]
        try:
            if len(lengths) > 1 or (lengths and not lengths[0].isdigit()):
                raise ValueError()
            length = int(lengths[0]) if lengths else None
        except ValueError:
            return await reject(400, "Invalid Content-Length header.")
        if length is not None and length > policy.max_body_bytes:
            return await reject(413, "Request exceeds the 2 MB import limit.")
        if lengths and "transfer-encoding" in headers:
            return await reject(400, "Ambiguous request framing.")
        write = scope["method"] not in {"GET", "HEAD", "OPTIONS"}
        if write and not self._admit(self._writes, policy.writes_per_minute):
            return await reject(429, "Workspace write limit reached.", {"Retry-After": "60"})

        async def read_bounded_body():
            parts, total = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return None
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > policy.max_body_bytes:
                    raise OverflowError()
                parts.append(chunk)
                if not message.get("more_body", False):
                    return b"".join(parts)

        with self._lock:
            reading = self._reading < policy.concurrent_bodies
            if reading:
                self._reading += 1
        if not reading:
            return await reject(429, "Too many concurrent request bodies.", {"Retry-After": "5"})
        try:
            body = await asyncio.wait_for(read_bounded_body(), policy.body_timeout)
        except OverflowError:
            return await reject(413, "Request exceeds the 2 MB import limit.")
        except asyncio.TimeoutError:
            return await reject(408, "Request body timed out.")
        finally:
            with self._lock:
                self._reading -= 1
        if body is None:
            return
        if length is not None and len(body) != length:
            return await reject(400, "Content-Length does not match the request body.")
        sent_body = False

        async def replay():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        admitted = False
        if write:
            with self._lock:
                if self._active < policy.concurrent_jobs:
                    self._active += 1
                    admitted = True
            if not admitted:
                return await reject(
                    429,
                    "Workspace is busy; retry after the current jobs finish.",
                    {"Retry-After": "5"},
                )
        try:
            await self.app(scope, replay, secure_send)
        finally:
            if admitted:
                with self._lock:
                    self._active -= 1
