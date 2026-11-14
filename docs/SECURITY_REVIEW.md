# Security review — 2026-11-14

Scope: the v2 FastAPI workspace, SQLite persistence, legacy account database and Streamlit entry point, v1 API, model serialization, request/data boundaries, dependencies, deployment defaults, and relevant Git history. This was a source review with local reproductions and regression tests, not a certification or external penetration test.

## Findings and disposition

| ID | Priority | Evidence | Disposition |
|---|---|---|---|
| S1 | High | The database at revision `ddd6e97` contains one account with a bcrypt hash and 18 simulation records. It is recoverable from Git history. | Removed from the current tree earlier; the disclosed hash fingerprint now cannot authenticate or maintain a valid legacy session. Historical erasure and password-reuse rotation remain owner actions. |
| S2 | High when reachable outside the intended local boundary | A test peer at `198.51.100.10`, with an attacker-controlled Host and matching Origin, could read datasets and create a simulation: both returned 200 before the fix. | Both API generations now validate explicit hostnames and peers, require a key for remote access, protect every route when a key is configured, and reject cross-site requests. The same untrusted Host test now returns 400; a non-loopback peer without a key returns 403. |
| S3 | Medium | A chunked request containing over 2.2 MB of padding and valid JSON bypassed the Content-Length-only check and created a run with HTTP 200. | Byte-counted streaming limit returns 413 before persistence. Length ambiguity, body timeout, body-reader count, write rate, and concurrent jobs are bounded. |
| S4 | Medium | Database file creation inherited broad POSIX defaults; database growth and number of imports/runs were unbounded. Legacy foreign keys were declared but not enabled. | Owner-only POSIX database files, foreign keys, bounded SQLite pages and record counts, transactional quota checks, and explicit 507 errors. Windows remains subject to the user's filesystem ACL. No data was silently deleted. |
| S5 | High for a shared legacy deployment | Registration was open, existing weak credentials still worked, login guessing was not throttled, and old Streamlit sessions survived password changes. | Registration disabled by default; weak/disclosed credentials denied; persistent five-attempt throttling; local password-rotation CLI; session revision checks; loopback Streamlit configuration. |
| S6 | High if a malicious artifact is installed | The new path used `torch.load(weights_only=True)` with a runtime old enough to be affected by published loader vulnerabilities. Legacy joblib/scaler loading could deserialize code-bearing artifacts. | New serving/training artifacts use Safetensors and schema 2. Legacy pickle loading requires explicit trust. Supported ML dependencies require PyTorch 2.15+. The unit round-trip forbids `torch.load` and verifies identical outputs without optimization. |
| S7 | Version-dependent supply-chain exposure | The original environment's selected project dependency graph produced 179 advisory entries across 15 packages; duplicate IDs/aliases mean this is not 179 distinct vulnerabilities. Its CPU-tagged old PyTorch could not be audited by an unnormalized PyPI lookup and was checked against upstream advisories separately. | A separate Python 3.12 environment was created, patched dependencies installed, security constraints added, and a strict normalized audit added to CI. The original global Python environment was not overwritten. |

## Checks with no exploitable finding in scope

- Database statements use bound parameters. Injected dataset names and SQL-like identifiers did not change query structure or return another record.
- React renders dataset names as text. The new UI contains no raw HTML insertion path for uploaded content.
- Database files, environment files, and parent-directory traversals are not exposed by static routes. The Git index contains no current database or secret environment file.
- CSV handling validates identifiers, units, finite numbers, row counts, and duplicate cycles. The import API does not accept model/FMU uploads or arbitrary filesystem paths.
- Model checksums are verified and the new registry rejects unsafe formats, malformed metadata, and excessive model dimensions before allocation.

## Verification

- Full Python regression suite: **129 passed, 1 skipped**, with six non-failing warnings, on Windows / Python 3.12.16 / PyTorch 2.15.0 CPU. Command: `python -m pytest tests test -q`.
- Strict Python dependency audit: **106 packages checked, zero skipped, zero reported advisories**. The [machine-readable snapshot](security-dependency-audit.json) includes the exact audited versions.
- Frontend: TypeScript compilation and Vite production build passed; `npm audit --audit-level=low` reported **zero vulnerabilities**.
- English source/documentation check and `git diff --check` passed.
- Real loopback HTTP checks with an isolated temporary database: authenticated health 200, unauthenticated dataset access 401, untrusted Host 400, cross-origin access 403, oversized chunked body 413, and the native Windows 8S2P FMU simulation 200.
- Bandit scan of `sodium_twin`, legacy account storage, and the account administration CLI: one medium-severity B614 finding at the explicitly gated legacy PyTorch loader; no high-severity findings in that scan. This retained risk is described below.

The Windows POSIX-permissions test is intentionally skipped; Linux CI exercises the 0600 assertion. Deprecation warnings from the test client and older research conversion paths are distinct from failures. Docker configuration is reviewed but not executed locally because Docker is unavailable.

Static analysis flags the retained PyTorch deserializer. Its call is behind an explicit trust opt-in and a patched-version check; the risk of trusted legacy executable serialization is documented rather than represented as eliminated. No training, optimizer step, credential guessing against real accounts, or live production attack was performed.

## Remaining owner decisions

1. **Historical database removal:** the exact target is `battery_app.db` in Git history. A safe cleanup requires a private mirror backup, filtering that path from affected history, validating the resulting trees, and coordinated force updates of affected refs. Commit IDs and existing clones would change. No history rewrite or force-push was performed during this review. Existing forks/caches cannot be guaranteed erased.
2. **Credential rotation:** any password reused from the exposed legacy account should be changed. New deployments do not create that default account, and the known disclosed hash is blocked even if restored from a backup.
3. **Shared/public deployment:** use a key on shared computers and TLS on any network deployment. The key grants one operator access to all data; per-user and per-tenant authorization are not implemented.
4. **Storage protection:** SQLite and its backups are plaintext files. OS encryption, private ACLs, backup access, retention, and native-worker isolation remain deployment responsibilities.

## Primary references

- [PyTorch checkpoint RCE advisory, CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).
- [PyTorch weights-only loader memory-corruption advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
- [PyTorch 2.14 weights-only arbitrary-file-read advisory; patched in 2.15](https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83).
- [Safetensors format and loading documentation](https://huggingface.co/docs/safetensors/index).
- [SQLite parameter binding and connection documentation](https://docs.python.org/3/library/sqlite3.html).

These references support the affected mechanisms; local exploitability and mitigation results above come from repository review and isolated tests.
