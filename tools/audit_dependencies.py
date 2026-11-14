"""Audit the active project environment, including CPU-specific PyTorch versions."""

from importlib.metadata import distributions
from pathlib import Path
import subprocess
import sys
import tempfile

from packaging.utils import canonicalize_name
from packaging.version import Version


def main():
    parser_args = sys.argv[1:]
    packages = {}
    for distribution in distributions():
        name = canonicalize_name(distribution.metadata["Name"])
        if name == "sodium-twin":
            continue
        # PyPI advisory lookup uses the public version, without a +cpu wheel suffix.
        packages[name] = Version(distribution.version).public
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit-requirements.txt"
        path.write_text(
            "\n".join(f"{name}=={version}" for name, version in sorted(packages.items())),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(path),
                "--no-deps",
                "--disable-pip",
                "--strict",
                "--progress-spinner",
                "off",
                *parser_args,
            ],
            check=False,
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
