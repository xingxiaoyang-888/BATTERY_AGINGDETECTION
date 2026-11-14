"""Explicit native FMU smoke command; importing this module performs no simulation."""

import argparse
from pathlib import Path
from utils.fmu_interface import FMUClient


def run_ignition_test(path=None):
    path = path or Path(__file__).resolve().parent / "fmu_models" / "SystemForFMI_8x2.fmu"
    engine = FMUClient(str(path))
    frame, _, _, _ = engine.run_simulation(stop_time=10, inputs={"I_load_external": 10})
    if frame is None or frame.empty:
        raise RuntimeError("Native solver returned no data. Check the OpenModelica runtime.")
    print(frame[["time", "pack.V_pack", "pack.T_max"]].tail())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmu", type=Path)
    run_ignition_test(parser.parse_args().fmu)
