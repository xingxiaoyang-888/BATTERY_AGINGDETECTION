"""Environment configuration without import-time database or model side effects."""

from dataclasses import dataclass, field
from pathlib import Path
import os
from .security import AccessPolicy

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_dir: Path
    web_dir: Path
    access: AccessPolicy = field(default_factory=AccessPolicy)

    @classmethod
    def from_env(cls):
        return cls(
            Path(os.getenv("SODIUM_DATA_DIR", str(ROOT / "data" / "workspace"))),
            Path(os.getenv("SODIUM_MODEL_DIR", str(ROOT / "artifacts"))),
            Path(os.getenv("SODIUM_WEB_DIR", str(ROOT / "web" / "dist"))),
            AccessPolicy.from_env(),
        )
