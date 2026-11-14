"""Explicit trust boundary for retained pickle-based offline research artifacts."""

import os
import re


def require_legacy_trust():
    if os.getenv("SODIUM_TRUST_LEGACY_ARTIFACTS") != "1":
        raise ValueError(
            "Legacy pickle artifacts are disabled. Only an operator who trusts their source may set "
            "SODIUM_TRUST_LEGACY_ARTIFACTS=1. Prefer the new safetensors workflow."
        )


def load_pickle_model(path):
    require_legacy_trust()
    import joblib

    return joblib.load(path)


def load_tensor_checkpoint(path, **kwargs):
    require_legacy_trust()
    import torch

    version = tuple(int(v) for v in re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__).groups())
    if version < (2, 10, 0) or version[:2] == (2, 14):
        raise ValueError(
            "This PyTorch version has known checkpoint-loader vulnerabilities; upgrade first."
        )
    kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def load_numpy_archive(path, **kwargs):
    import numpy as np

    kwargs["allow_pickle"] = os.getenv("SODIUM_TRUST_LEGACY_ARTIFACTS") == "1"
    return np.load(path, **kwargs)
