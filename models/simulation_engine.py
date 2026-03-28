# models/simulation_engine.py
import numpy as np
import pandas as pd
import os
import time
import joblib 

# 导入独立的工业级老化算法模块
from models.aging_algorithm import AgingModel

class BatteryDigitalTwin:
    """
     动力电池数字孪生引擎 V1.0 (OMC + Aging + AI Hybrid Model)
    
    架构特性:
    1. 物理层: OpenModelica (OMC) 处理电-热耦合。
    2. 数据层: 引入机器学习 (RandomForest) 预测非线性产热特征。
    3. 寿命层: Python (AgingModel) 处理日历与循环老化闭环。
    4. 策略层: 动态 SOP 与安全状态机。
    """
    def __init__(self, cell_capacity_ah=100.0, cell_resistance_ohm=0.003, series_num=96, parallel_num=1):
        # === 基础 BOM 参数 ===
        self.cell_capacity = cell_capacity_ah
        self.base_resistance = cell_resistance_ohm
        self.series_num = series_num
        self.parallel_num = parallel_num
        
        # Pack 级衍生参数
        self.pack_capacity = self.cell_capacity * self.parallel_num
        self.pack_energy_kwh = (self.pack_capacity * 3.7 * self.series_num) / 1000.0
        
        # === AI 模型初始化 ===
        self.ai_heat_model = None
        self._load_ai_model()

        # === 引擎初始化状态 ===
        self.use_omc = False
        self.omc = None
        self._init_openmodelica()

    def _load_ai_model(self):
        """[新增] 尝试加载训练好的 AI 产热模型"""
        try:
            model_path = os.path.join(os.getcwd(), 'models', 'weights', 'heat_ai_model.pkl')
            if os.path.exists(model_path):
                self.ai_heat_model = joblib.load(model_path)
                print("🤖 AI 产热模型 (Data-Driven) 加载成功！")
            else:
                print("ℹ️ 未找到 AI 产热模型，将使用机理物理公式计算产热。")
        except Exception as e:
            print(f"⚠️ AI 模型加载异常: {e}")

    def _init_openmodelica(self):
        """尝试建立与 OpenModelica 的 ZMQ 通信"""
        try:
            from OMPython import OMCSessionZMQ
            self.omc = OMCSessionZMQ()
            
            # 加载 package.mo
            current_dir = os.getcwd()
            pkg_path = os.path.join(current_dir, "models", "BatterySystem", "package.mo")
            pkg_path = pkg_path.replace("\\", "/")
            
            if not os.path.exists(pkg_path):
                print(f"⚠️ 关键文件缺失: {pkg_path}")
                self.use_omc = False
                return
                
            load_status = self.omc.sendExpression(f'loadFile("{pkg_path}")')
            classes = self.omc.sendExpression("getClassNames()")
            
            if not load_status or "BatterySystem" not in str(classes):
                base_path = os.path.dirname(pkg_path).replace("\\", "/")
                self.omc.sendExpression(f'loadFile("{base_path}/CoolingSystem.mo")')
                self.omc.sendExpression(f'loadFile("{base_path}/BatteryCell.mo")')
                
            self.use_omc = True
            print("✅ OpenModelica 工业级内核已就绪 (BatterySystem Loaded)")
                
        except Exception as e:
            print(f"⚠️ OMC 连接异常: {e}")
            self.use_omc = False

    # ==========================================
    # 核心入口: 运行工况 (带熔断机制)
    # ==========================================
    def run_profile(self, time_total, pack_current_a, env_temp_c, init_soc, init_soh, cooling_type):
        self.pack_capacity = self.cell_capacity * self.parallel_num
        
        if self.use_omc:
            try:
                return self._run_with_openmodelica(time_total, pack_current_a, env_temp_c, init_soc, init_soh, cooling_type)
            except Exception as e:
                print(f"🔴 仿真中断: {e}")
                self.use_omc = False # 熔断降级
                return self._run_with_python_surrogate(time_total, pack_current_a, env_temp_c, init_soc, init_soh, cooling_type)
        else:
            return self._run_with_python_surrogate(time_total, pack_current_a, env_temp_c, init_soc, init_soh, cooling_type)

    # ==========================================
    # 核心 A: OpenModelica 驱动逻辑
    # ==========================================
    def _run_with_openmodelica(self, duration, current, temp, soc, init_soh, cooling):
        mode_map = {"Natural": 1, "Air Cooling": 2, "Liquid Cooling": 3, "Liquid Heating": 4, "Immersion": 3}
        c_mode = mode_map.get(cooling, 1)
        
        i_cell = current / self.parallel_num
        
        sim_cmd = (
            f"simulate(BatterySystem.BatteryCell, stopTime={duration}, numberOfIntervals=200, "
            f"simflags=\"-override I_load={i_cell},T_env_input={temp+273.15},SOC_init={soc/100},cooling_mode={c_mode}\")"
        )
        
        self.omc.sendExpression(sim_cmd)
        
        raw_time = self.omc.sendExpression("val(time)")
        raw_v = self.omc.sendExpression("val(V_terminal)")
        
        if raw_time is None or raw_v is None:
            raise RuntimeError(f"OMC 无数据返回: {self.omc.sendExpression('getErrorString()')}")
            
        times = np.array(raw_time)
        v_cell = np.array(raw_v)
        t_cell_k = np.array(self.omc.sendExpression("val(T_cell)"))
        soc_res = np.array(self.omc.sendExpression("val(SOC)"))
        
        v_pack = v_cell * self.series_num
        t_cell_c = t_cell_k - 273.15
        
        grad_factor = 0.05
        if c_mode == 3: grad_factor = 0.2
        if c_mode == 2: grad_factor = 0.1
        
        delta_t_arr = t_cell_c * grad_factor * (abs(current)/100)
        t_max = t_cell_c + delta_t_arr/2
        t_min = t_cell_c - delta_t_arr/2
        
        sop_chg, sop_dch = [], []
        for v, t, s in zip(v_pack, t_max, soc_res):
             p_d, _, p_c = self._calculate_sop_logic(v, self.base_resistance*self.series_num/self.parallel_num, t, s)
             sop_chg.append(abs(p_c))
             sop_dch.append(abs(p_d))
             
        # 集成 AgingModel 计算 SOH
        current_soh = init_soh
        soh_arr = []
        total_loss = 0.0
        dt_list = np.diff(times, prepend=0)
        
        for i in range(len(times)):
            step_loss = AgingModel.calculate_step_loss(
                temp_c=t_max[i], 
                current_a=i_cell, 
                soc_pct=soc_res[i]*100, 
                dt_seconds=dt_list[i], 
                cell_capacity_ah=self.cell_capacity
            )
            total_loss += step_loss
            current_soh -= step_loss
            soh_arr.append(current_soh)
        
        df = pd.DataFrame({
            "Time": times, "Pack_Voltage": v_pack, "Pack_Current": [current] * len(times),
            "SOC": soc_res * 100, "SOH": soh_arr,
            "Max_Temp": t_max, "Min_Temp": t_min, "Delta_T": delta_t_arr,
            "SOP_Charge_kW": np.array(sop_chg)/1000, "SOP_Discharge_kW": np.array(sop_dch)/1000
        })
        
        kpis = self._generate_kpis(df, soc, init_soh, total_loss)
        return df, kpis

    # ==========================================
    # 核心 B: Python 高保真代理模型 (AI Enhanced)
    # ==========================================
    def _run_with_python_surrogate(self, time_total, pack_current_a, env_temp_c, init_soc_pct, init_soh_pct, cooling_type):
        steps = 200
        dt = time_total / steps
        t_arr = np.linspace(0, time_total, steps)
        
        current_soc = init_soc_pct / 100.0
        current_soh = init_soh_pct / 100.0
        temp_core = env_temp_c
        temp_surface = env_temp_c
        
        h_cooling_map = {"Natural": 0.5, "Air Cooling": 5.0, "Liquid Cooling": 30.0, "Liquid Heating": 15.0, "Immersion": 100.0}
        h_coef = h_cooling_map.get(cooling_type, 1.0)
        grad_factor = 2.0 if "Liquid" in cooling_type else 1.2

        results = {
            "Time": t_arr, "Pack_Voltage": [], "Pack_Current": [],
            "SOC": [], "SOH": [], "Max_Temp": [], "Min_Temp": [], "Delta_T": [],
            "SOP_Charge_kW": [], "SOP_Discharge_kW": [], "Warning_Msg": [], "Warning_Level": []
        }
        
        total_loss = 0.0
        
        #  提取 AI 需要的特征特征: 倍率和充放电状态
        is_discharge = 1 if pack_current_a > 0 else 0
        c_rate = abs(pack_current_a) / self.pack_capacity
        
        for i in range(steps):
            i_cell = pack_current_a / self.parallel_num
            
            # 1. 动态内阻
            r_dynamic = self._get_dynamic_resistance(current_soc, temp_core, i_cell)
            r_aging = r_dynamic * (1 + (1-current_soh)*1.5)
            
            # 2. 电气状态
            ocv = self._get_complex_ocv(current_soc)
            v_polar = i_cell * 0.001 * (1 - np.exp(-t_arr[i]/100))
            v_cell = ocv - (i_cell * r_aging) - v_polar
            v_pack = v_cell * self.series_num
            
            # 3. 热力学计算 ( 物理机理) 高倍率接管 残差预测
            if self.ai_heat_model is not None:
                #  AI 产热模型接管：利用 RandomForest 预测极化尖峰
                current_time_minutes = t_arr[i] / 60.0 
                features = pd.DataFrame([[current_time_minutes, c_rate, is_discharge]], 
                                        columns=['Time', 'C_Rate', 'Is_Discharge'])
                #极化尖峰
                heat_gen = self.ai_heat_model.predict(features)[0]
            else:
                # 降级：传统的焦耳热模型
                heat_gen = (i_cell ** 2) * r_aging
            #热熔  
            thermal_mass = 900 
            dT_core = (heat_gen - h_coef*0.2*(temp_core - temp_surface)) / thermal_mass * dt
            #散热系数
            #差分方程
            dT_surf = (h_coef*0.2*(temp_core - temp_surface) - h_coef*(temp_surface - env_temp_c)) / (thermal_mass*0.5) * dt
            temp_core += dT_core
            temp_surface += dT_surf
            
            dynamic_spread = (abs(pack_current_a) / 100) * grad_factor * (1 - np.exp(-t_arr[i]/200))
            t_max = temp_core + dynamic_spread
            t_min = temp_surface - (dynamic_spread * 0.2)
            
            # 4. SOP 计算
            r_pack_total = (r_aging * self.series_num) / self.parallel_num
            p_peak, _, p_chg = self._calculate_sop_logic(v_pack, r_pack_total, t_max, current_soc)
            
            # 5. SOC 与 老化更新
            soc_change = -(i_cell * dt / 3600) / self.cell_capacity
            current_soc = np.clip(current_soc + soc_change, 0, 1)
            
            step_loss = AgingModel.calculate_step_loss(
                temp_c=t_max, current_a=i_cell, soc_pct=current_soc * 100,
                dt_seconds=dt, cell_capacity_ah=self.cell_capacity
            )
            
            current_soh -= step_loss
            total_loss += step_loss
            
            # 6. 数据记录
            results["Pack_Voltage"].append(v_pack)
            results["Pack_Current"].append(pack_current_a)
            results["SOC"].append(current_soc * 100)
            results["SOH"].append(current_soh * 100)
            results["Max_Temp"].append(t_max)
            results["Min_Temp"].append(t_min)
            results["Delta_T"].append(t_max - t_min)
            results["SOP_Charge_kW"].append(abs(p_chg) / 1000)
            results["SOP_Discharge_kW"].append(abs(p_peak) / 1000)
            results["Warning_Msg"].append("Normal") 
            results["Warning_Level"].append(0)

        df = pd.DataFrame(results)
        kpis = self._generate_kpis(df, init_soc_pct, init_soh_pct, total_loss)
        return df, kpis

    # ==========================================
    # 辅助物理算法 (Shared Logic)
    # ==========================================
    def _get_complex_ocv(self, soc):
        return 3.0 + (soc * 0.9) + 0.3 * (soc**0.5) - 0.1 * (soc**3) + 0.1 * np.exp(-15 * (1-soc))

    def _get_dynamic_resistance(self, soc, temp_c, current_a):
        temp_k = temp_c + 273.15
        temp_factor = np.exp(1000 * (1/temp_k - 1/298.15))
        soc_factor = 1.0 + 2.0 * np.exp(-10 * soc) + 0.2 * np.exp(-10 * (1-soc))
        rate_factor = 1.0 + 0.05 * (abs(current_a) / self.cell_capacity)
        return self.base_resistance * temp_factor * soc_factor * rate_factor

    def _calculate_sop_logic(self, v_pack, r_total, temp_c, soc):
        v_max = 4.2 * self.series_num
        v_min = 2.8 * self.series_num
        i_max_dch = (v_pack - v_min) / r_total
        i_max_chg = (v_max - v_pack) / r_total
        
        derating = 1.0
        if temp_c < 10: derating *= (temp_c + 10) / 20.0
        if temp_c < 0: derating = 0.1
        if soc < 0.1: derating *= 0.2
        
        p_dch_peak = v_pack * i_max_dch * derating
        p_dch_cont = p_dch_peak * 0.6
        p_chg_peak = v_pack * i_max_chg * derating
        return p_dch_peak, p_dch_cont, p_chg_peak

    def _generate_kpis(self, df, init_soc, init_soh, total_loss):
        last = df.iloc[-1]
        w_msg = "Normal"
        if last["Max_Temp"] > 55: w_msg = "🔴 Risk: Overheat"
        elif last["Delta_T"] > 8: w_msg = "🟡 Warning: High Delta-T"
        elif last["SOC"] < 10: w_msg = "🟡 Warning: Low SOC"
            
        return {
            "soc": last["SOC"],
            "soh": init_soh if self.use_omc else last["SOH"],
            "soh_loss": total_loss,
            "max_temp": last["Max_Temp"],
            "avg_delta_t": df["Delta_T"].mean(),
            "sop_dch": last["SOP_Discharge_kW"],
            "warning": w_msg
        }

# 单例实例化
engine = BatteryDigitalTwin()