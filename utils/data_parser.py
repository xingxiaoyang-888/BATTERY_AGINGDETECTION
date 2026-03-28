# utils/data_parser.py
import pandas as pd
import streamlit as st

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
            
        # 2. 数据清洗
        df = df.sort_values(by='Time').reset_index(drop=True)
        
        # 3. 提取关键信息
        duration = df['Time'].max() - df['Time'].min()
        avg_current = df['Current'].mean()
        
        # 4. 返回标准化数据
        profile_data = {
            'duration': float(duration),
            'avg_current': float(avg_current),
            'dataframe': df
        }
        return profile_data, "✅ 文件解析成功"
        
    except Exception as e:
        return None, f"❌ 解析异常: {str(e)}"