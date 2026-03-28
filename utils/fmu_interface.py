# utils/fmu_interface.py
import os
import logging
import fmpy
from fmpy import read_model_description, simulate_fmu
import pandas as pd
from typing import Optional, Dict, Any

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - FMU Interface - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FMUClient:
    """
    轻量级 FMI/FMU 工业标准接口封装类
    用于替代原有的 OpenModelica ZMQ 通信，实现云端 SaaS 化的高并发计算
    """
    def __init__(self, fmu_path: str):
        self.fmu_path = fmu_path
        self.model_description = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """解析 FMU 描述文件，获取接口变量 (无环境依赖)"""
        if not os.path.exists(self.fmu_path):
            logger.error(f"❌ 找不到 FMU 文件: {self.fmu_path}")
            return False
            
        try:
            # 瞬间读取 xml，不需要启动任何外部求解器
            self.model_description = read_model_description(self.fmu_path)
            logger.info(f"📥 成功加载 FMU 资产: {self.model_description.modelName}")
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"❌ 解析 FMU 失败: {e}")
            return False

    def execute_simulation(self, stop_time: float, inputs: Dict[str, float], output_interval: float = 1.0) -> Optional[pd.DataFrame]:
        """
        执行联合仿真 (Co-Simulation)
        :param stop_time: 仿真总时长 (秒)
        :param inputs: 传入 FMU 的初始参数/输入变量，例如 {"I_load": 50.0, "T_env": 25.0}
        :param output_interval: 采样间隔
        """
        if not self.is_loaded:
            logger.error("⚠️ 模型未加载，请先执行 load_model()")
            return None
            
        logger.info(f"⚙️ 正在启动 FMU 极速解算... 输入条件: {inputs}")
        
        try:
            # simulate_fmu 底层直接调用 C/C++ 动态链接库，速度极快
            result = simulate_fmu(
                filename=self.fmu_path,
                start_time=0.0,
                stop_time=stop_time,
                start_values=inputs,     # 将前端传来的网页配置直接注入
                output_interval=output_interval,
                validate=False           # 生产环境关闭验证以提高吞吐量
            )
            
            # 将 numpy structured array 转为标准的 Pandas DataFrame 供外部使用
            df = pd.DataFrame(result)
            logger.info("📊 FMU 仿真计算完成")
            return df
            
        except Exception as e:
            logger.error(f"❌ 仿真执行崩溃: {e}")
            return None