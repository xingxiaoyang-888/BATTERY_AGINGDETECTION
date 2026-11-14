"""Launch the local, single-user API and compiled web application."""

import argparse
import sys
import uvicorn
from .security import AccessPolicy, require_safe_bind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11+ is required. Activate the updated project environment first.")
    try:
        require_safe_bind(args.host, AccessPolicy.from_env())
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(
        "sodium_twin.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        limit_concurrency=32,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
