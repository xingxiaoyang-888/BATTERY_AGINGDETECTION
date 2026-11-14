"""Rotate a legacy local account without placing its new password in shell history."""

import argparse
from getpass import getpass
from .db_manager import init_db, reset_password


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    args = parser.parse_args()
    password = getpass("New password (12–72 UTF-8 bytes): ")
    if password != getpass("Confirm new password: "):
        parser.error("Passwords do not match.")
    init_db()
    try:
        reset_password(args.username, password)
    except ValueError as exc:
        parser.error(str(exc))
    print("Password rotated. Existing account data was preserved.")


if __name__ == "__main__":
    main()
