"""Launch the local, single-user API and compiled web application."""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("sodium_twin.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
