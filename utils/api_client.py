"""前端访问FastAPI的统一客户端。"""

import os
from typing import Any, Dict

import requests


API_BASE = os.getenv("BATTERY_API_BASE", "http://localhost:8000").rstrip("/")


class BackendAPIError(RuntimeError):
    pass


def get_json(path: str, timeout: float = 8.0) -> Dict[str, Any]:
    return _request("GET", path, timeout=timeout)


def post_json(path: str, payload: Dict[str, Any], timeout: float = 180.0) -> Dict[str, Any]:
    return _request("POST", path, payload=payload, timeout=timeout)


def _request(method: str, path: str, payload=None, timeout=30.0):
    try:
        response = requests.request(
            method,
            f"{API_BASE}{path}",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise BackendAPIError(f"无法连接后端服务：{exc}") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendAPIError(f"后端返回 {response.status_code}：{detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise BackendAPIError("后端响应不是有效JSON") from exc
