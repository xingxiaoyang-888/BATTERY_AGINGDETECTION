import os
import re
import logging
import fmpy
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - FMU Engine - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def discover_available_fmus(fmu_dir: str = "fmu_models") -> List[Dict]:
    """Discover exported FMUs from their series/parallel filenames."""
    configs = []
    if not os.path.isdir(fmu_dir):
        return configs

    pattern = re.compile(r"SystemForFMI_(\d+)x(\d+)\.fmu$")
    for fname in sorted(os.listdir(fmu_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        ns, np = int(m.group(1)), int(m.group(2))
        fpath = os.path.join(fmu_dir, fname)
        configs.append(
            {
                "ns": ns,
                "np": np,
                "filename": fname,
                "path": os.path.abspath(fpath),
                "total_cells": ns * np,
                "nominal_voltage": round(ns * 3.1, 1),
            }
        )
    return configs


STANDARD_PACK_CONFIGS: List[Tuple[int, int]] = [
    (8, 2),
    (12, 2),
    (16, 1),
    (16, 2),
    (16, 3),
    (16, 4),
    (20, 2),
    (24, 2),
    (32, 2),
    (48, 1),
    (48, 2),
    (96, 1),
    (96, 2),
    (96, 3),
    (100, 1),
    (100, 2),
    (108, 2),
    (128, 1),
    (128, 2),
    (192, 1),
    (192, 2),
    (200, 1),
    (216, 1),
    (256, 1),
    (288, 1),
    (480, 1),
]


def discover_available_models(models_dir: str = "mo_system_models") -> List[Dict]:
    """Discover Modelica topologies, including configurations without an exported FMU."""
    configs = []
    if not os.path.isdir(models_dir):
        return configs

    file_list = sorted(os.listdir(models_dir))

    has_template = any(
        f.startswith("SystemForFMI.") and f.endswith(".mo") and not re.search(r"\d+x\d+", f)
        for f in file_list
    )

    if has_template:
        logger.info(f" 检测到参数化 FMU 模板，共 {len(STANDARD_PACK_CONFIGS)} 种标准规格可用")
        for ns, np_val in STANDARD_PACK_CONFIGS:
            fname = f"SystemForFMI_{ns}x{np_val}.mo"
            configs.append(
                {
                    "ns": ns,
                    "np": np_val,
                    "filename": fname,
                    "total_cells": ns * np_val,
                    "nominal_voltage": round(ns * 3.1, 1),
                }
            )
        return configs

    pattern = re.compile(r"SystemForFMI_(\d+)x(\d+)\.mo$")
    for fname in file_list:
        m = pattern.match(fname)
        if not m:
            continue
        ns, np_val = int(m.group(1)), int(m.group(2))
        configs.append(
            {
                "ns": ns,
                "np": np_val,
                "filename": fname,
                "total_cells": ns * np_val,
                "nominal_voltage": round(ns * 3.1, 1),
            }
        )
    return configs


def get_ns_np_options(models: List[Dict] = None) -> Dict[int, List[int]]:
    """Group available parallel counts by series count."""
    if models is None:
        models = discover_available_models()
    options = {}
    for m in models:
        ns = m["ns"]
        np = m["np"]
        if ns not in options:
            options[ns] = []
        options[ns].append(np)

    for ns in options:
        options[ns] = sorted(options[ns])
    return options


class FMUClient:
    """FMI co-simulation adapter with model metadata and cell matrix extraction."""

    def __init__(self, fmu_path: str):
        self.fmu_path = fmu_path
        self._dll_directory_handles = []
        self._setup_windows_env()

        try:
            self.model_description = fmpy.read_model_description(self.fmu_path)
            logger.info(f" FMU 模型加载成功: {self.model_description.modelName}")
        except Exception as e:
            logger.error(f" 解析 FMU 失败，请检查路径: {e}")
            raise

        self.Ns, self.Np = self._detect_dimensions()

    def _setup_windows_env(self):
        """Use an explicit runtime location or an installation discovered on PATH."""
        if os.name != "nt":
            return
        import shutil
        from pathlib import Path

        configured = os.getenv("OPENMODELICA_BIN")
        if not configured and os.getenv("OPENMODELICAHOME"):
            configured = str(Path(os.environ["OPENMODELICAHOME"]) / "bin")
        executable = shutil.which("omc")
        candidate = configured or (str(Path(executable).parent) if executable else None)
        if candidate:
            directory = Path(candidate).expanduser().resolve()
            if not directory.is_dir():
                raise ValueError("OPENMODELICA_BIN must point to an existing directory.")
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                self._dll_directory_handles.append(os.add_dll_directory(str(directory)))

    def _detect_dimensions(self) -> Tuple[int, int]:
        """Read cell-array dimensions from modelDescription.xml."""
        import zipfile

        try:
            with zipfile.ZipFile(self.fmu_path, "r") as z:
                xml = z.read("modelDescription.xml").decode("utf-8")

            matches = re.findall(r"pack\.TCell\[(\d+),\s*(\d+)\]", xml)
            if not matches:
                logger.warning("⚠️ 未在 FMU 中找到 pack.TCell 变量，使用默认 8×2")
                return 8, 2
            max_s = max(int(m[0]) for m in matches)
            max_p = max(int(m[1]) for m in matches)
            logger.info(f" 自动检测 FMU 维度: Ns={max_s}, Np={max_p} ({len(matches)} 个电芯)")
            return max_s, max_p
        except Exception as e:
            logger.warning(f"⚠️ 维度检测失败: {e}，使用默认 8×2")
            return 8, 2

    def run_simulation(
        self,
        stop_time: float,
        inputs: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, float]] = None,
        current_profile: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[
        Optional[pd.DataFrame],
        List[List[List[float]]],
        List[List[List[float]]],
        List[List[List[float]]],
    ]:
        """Run native co-simulation and reconstruct temperature, SOC, and SOH matrices."""
        Ns, Np = self.Ns, self.Np
        inputs = dict(inputs or {})
        parameters = dict(parameters or {})
        logger.info(
            " 启动 FMU 解算 [%ss%sp] 时长: %ss, 标量输入: %s, 参数覆盖: %s, 工况点: %s",
            Ns,
            Np,
            stop_time,
            inputs,
            sorted(parameters),
            len(current_profile) if current_profile else 0,
        )

        output_vars = [
            "time",
            "pack.V_pack",
            "pack.I_pack",
            "pack.T_max",
            "pack.T_min",
            "pack.SOC_min",
            "pack.SOH_min",
        ]

        for s in range(1, Ns + 1):
            for p in range(1, Np + 1):
                output_vars.append(f"pack.TCell[{s},{p}]")
                output_vars.append(f"pack.SOCCell[{s},{p}]")
                output_vars.append(f"pack.SOHCell[{s},{p}]")

        try:
            start_values = dict(parameters)
            input_table = None
            if current_profile:
                input_table = self._build_current_profile(current_profile, stop_time)
                inputs.pop("I_load_external", None)
            start_values.update(inputs)

            result = fmpy.simulate_fmu(
                filename=self.fmu_path,
                start_time=0.0,
                stop_time=stop_time,
                start_values=start_values,
                input=input_table,
                output_interval=max(1.0, stop_time / 300.0),
                timeout=120.0,
                output=output_vars,
            )

            df = pd.DataFrame(result)

            temp_matrix_frames = self._extract_spatial_matrix(df, Ns, Np, prefix="pack.TCell")
            soc_matrix_frames = self._extract_spatial_matrix(df, Ns, Np, prefix="pack.SOCCell")
            soh_matrix_frames = self._extract_spatial_matrix(df, Ns, Np, prefix="pack.SOHCell")

            logger.info(" FMU 联合解算与矩阵重组完成！(T+OC+SOH)")
            return df, temp_matrix_frames, soc_matrix_frames, soh_matrix_frames

        except Exception as e:
            logger.error(f" FMU 仿真执行崩溃: {e}")
            return None, [], [], []

    @staticmethod
    def _build_current_profile(profile: List[Tuple[float, float]], stop_time: float) -> np.ndarray:
        """Convert ordered current points into the structured FMI input table."""
        if len(profile) < 2:
            raise ValueError("电流工况至少需要两个时间点")
        points = [(float(t), float(i)) for t, i in profile]
        times = np.asarray([point[0] for point in points], dtype=float)
        currents = np.asarray([point[1] for point in points], dtype=float)
        if not np.isfinite(times).all() or not np.isfinite(currents).all():
            raise ValueError("电流工况包含非有限值")
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("电流工况时间必须非负且严格递增")
        if times[-1] > stop_time + 1e-9:
            raise ValueError("电流工况末端时间不能超过仿真时长")
        if times[0] > 0:
            points.insert(0, (0.0, currents[0]))
        if points[-1][0] < stop_time:
            points.append((float(stop_time), points[-1][1]))
        table = np.empty(
            len(points),
            dtype=[("time", np.float64), ("I_load_external", np.float64)],
        )
        table["time"] = [point[0] for point in points]
        table["I_load_external"] = [point[1] for point in points]
        return table

    def _extract_spatial_matrix(
        self, df: pd.DataFrame, Ns: int, Np: int, prefix: str
    ) -> List[List[List[float]]]:
        """Reassemble flattened output columns as [time,series,parallel] matrices."""
        col_names = np.empty((Ns, Np), dtype=object)
        for s in range(1, Ns + 1):
            for p in range(1, Np + 1):
                name_with_space = f"{prefix}[{s}, {p}]"
                name_no_space = f"{prefix}[{s},{p}]"
                col_names[s - 1, p - 1] = (
                    name_with_space if name_with_space in df.columns else name_no_space
                )

        is_temperature = "TCell" in prefix
        matrix_frames = []
        for i in range(len(df)):
            frame = np.zeros((Ns, Np))
            for s in range(Ns):
                for p in range(Np):
                    raw = df.iloc[i][col_names[s, p]]
                    if is_temperature:
                        frame[s, p] = round(raw - 273.15, 2)
                    else:
                        frame[s, p] = round(float(raw), 6)
            matrix_frames.append(frame.tolist())

        return matrix_frames
