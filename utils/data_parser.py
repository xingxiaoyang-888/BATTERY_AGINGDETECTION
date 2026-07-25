# utils/data_parser.py
import pandas as pd
import numpy as np

def parse_profile_csv(uploaded_file):
    """
    解析用户上传的工况 CSV 文件
    要求包含列: 'Time' (秒), 'Current' (安培, +放电/-充电)
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # 1. 基础校验：检查列名
        required_cols = ['Time', 'Current']
        if not all(col in df.columns for col in required_cols):
            return None, "❌ 格式错误：CSV 必须包含 'Time' 和 'Current' 列"
            
        # 2. 数据清洗与严格校验
        df = df[required_cols].copy()
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Current'] = pd.to_numeric(df['Current'], errors='coerce')
        if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
            return None, "❌ 格式错误：Time/Current 必须是有限数值"
        df = df.sort_values(by='Time').reset_index(drop=True)
        if len(df) < 2:
            return None, "❌ 工况至少需要两个时间点"
        if df['Time'].duplicated().any():
            return None, "❌ Time 不允许重复"
        df['Time'] = df['Time'] - df['Time'].iloc[0]
        if (df['Time'].diff().dropna() <= 0).any():
            return None, "❌ Time 必须严格递增"
        
        # 3. 提取关键信息
        duration = df['Time'].max() - df['Time'].min()
        avg_current = df['Current'].mean()
        
        # 4. 返回标准化数据
        profile_data = {
            'duration': float(duration),
            'avg_current': float(avg_current),
            'points': [
                {'time_s': float(row.Time), 'current_a': float(row.Current)}
                for row in df.itertuples(index=False)
            ],
            'dataframe': df
        }
        return profile_data, "✅ 文件解析成功"
        
    except Exception as e:
        return None, f"❌ 解析异常: {str(e)}"
