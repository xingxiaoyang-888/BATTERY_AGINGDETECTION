# Security and deployment boundaries

Sodium Twin is a single-operator research workspace. An API key grants access to the whole workspace; it is not a tenant, role, or per-user authorization system. The [2026-11-14 review](docs/SECURITY_REVIEW.md) records findings, fixes, and remaining actions.

## Supported runtime

Use Python 3.11+ in a dedicated virtual environment. The security release was validated with Python 3.12.16 and PyTorch 2.15.0 CPU. Install with the supplied security constraints:

```powershell
python -m pip install -c constraints-security.txt -e ".[test,ml]" -r requirements.txt
python -m pip install pip-audit
python tools/audit_dependencies.py
```

The audit normalizes a CPU wheel's version before advisory lookup, so PyTorch is not silently skipped. Re-run it after dependency changes. A clean advisory scan is a point-in-time result, not a guarantee against unknown vulnerabilities.

## Local and network access

- Keyless mode accepts loopback peers and explicit local Host headers only. Other local OS users/processes may reach loopback services; configure a key on shared computers.
- Set a random `SODIUM_API_KEY` of at least 32 printable ASCII characters to protect all routes, including documents, exports, and API metadata. Browser authentication uses **username `workspace`** and the key as its password. CLI clients use `Authorization: Bearer <key>`.
- Non-loopback binding requires a key. For a custom hostname, set the comma-separated `SODIUM_ALLOWED_HOSTS` explicitly. Wildcards are rejected. Neither forwarded headers nor the request's own Host value grant trust.
- Use TLS for network access. The application does not terminate TLS or provide identity management, password recovery, MFA, or role-based authorization. An operator should place it behind a correctly configured TLS reverse proxy and firewall; do not publish a bare HTTP port.
- `SODIUM_DEV_MODE=1` opts into the two localhost Vite origins. Leave this off outside development.
- Rotate a workspace key through the process environment and restart the service. Do not place a key in URLs, frontend code, Git, screenshots, or shared logs.

For example, generate a key into the current PowerShell process without hard-coding it:

```powershell
$env:SODIUM_API_KEY = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep the generated value in the operator's password manager. Docker Compose requires this key even when publishing only on host loopback, since the container sees a bridge-network peer. The container runs without root, Linux capabilities, or a writable root filesystem; its database volume remains writable.

## Database and request limits

Both SQLite paths use bound query parameters, scoped transactions, and foreign-key enforcement. Newly created private directories use mode 0700 and database files use mode 0600 on POSIX. On Windows, access is governed by the containing directory's ACL; use a private user profile rather than a shared folder. SQLite data is **not encrypted at rest**. Use OS disk encryption and protect backups separately.

The v2 workspace is bounded to 50 datasets, 1,000 run records, and approximately 256 MiB of SQLite pages. Capacity errors preserve existing records and return HTTP 507. The service never silently deletes research records; archive the workspace before creating a new one. Existing oversized databases are not automatically truncated. Journal files and temporary serialization may need additional disk space.

HTTP bodies are limited by actual received bytes, including chunked requests, with a 15-second deadline and at most eight concurrent body readers. Per process, the gateway allows 60 writes per minute, two concurrent write jobs, and 30 failed authentication attempts per minute. The CLI server also limits connection concurrency to 32. These are local process limits, not a distributed abuse-protection service.

## Legacy accounts

The retained Streamlit interface binds to loopback by default, keeps XSRF protection enabled, and disables self-registration unless `SODIUM_ALLOW_REGISTRATION=1` is explicitly set. Provision a new administrator using `SODIUM_ADMIN_USER` and `SODIUM_ADMIN_PASSWORD`; no default password exists.

Weak passwords and the password-hash fingerprint disclosed in the old public database cannot authenticate. Five failed attempts lock an account for five minutes. Rotate an existing password locally without putting it in shell history:

```powershell
python -m utils.account_admin admin
```

This interactive command preserves account history. Password rotation invalidates the previous Streamlit session revision on its next interaction. Existing databases are not overwritten just because an environment password changes.

## Models and native code

New learned artifacts use **manifest schema 2** and `weights.safetensors`; pickle-based artifacts do not activate in the new inference path. Manifests, dimensions, scaler metadata, provenance, cell splits, metrics, checksums, and finite tensor values are checked. Checksums establish consistency, not publisher identity: a trusted operator still controls artifact installation.

Legacy joblib/pickle loaders are disabled unless `SODIUM_TRUST_LEGACY_ARTIFACTS=1` is deliberately enabled. That flag permits potentially executable serialization and is only for artifacts whose source the operator trusts. Legacy tensor loading additionally rejects known vulnerable PyTorch version families. Prefer the new Safetensors path.

FMUs contain native executable code. Only install trusted binaries. Their time limit and concurrency controls do not sandbox native code; strong isolation requires dedicated worker processes or containers with a separate data boundary.

## Historical disclosure

The old `battery_app.db` was committed to public Git history. Removing it from the current branch does not erase old commits, clones, forks, or caches. Treat the exposed credential as compromised and rotate any reused password. A coordinated history rewrite is a separate, potentially disruptive repository-owner action; the security patch does not force-push or rewrite existing commits.
