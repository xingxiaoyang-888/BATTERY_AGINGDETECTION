# utils/db_manager.py
import sqlite3
import bcrypt
import json
from datetime import datetime
import os

# 数据库文件路径
DB_PATH = "battery_app.db"

def init_db():
    """初始化数据库表结构"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. 用户表
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL
        )
    ''')
    
    # 2. 仿真历史表 (表名统一为 history)
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp DATETIME,
            config TEXT,  -- 存 JSON 字符串
            kpis TEXT,    -- 存 JSON 字符串
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    
    # 创建默认管理员 (密码: 1234)
    try:
        pwd_hash = bcrypt.hashpw("1234".encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                  ("admin", pwd_hash))
    except sqlite3.IntegrityError:
        pass # admin 已存在
        
    conn.commit()
    conn.close()

def verify_user(username, password):
    """验证登录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        if bcrypt.checkpw(password.encode('utf-8'), result[0]):
            return True
    return False

def register_user(username, password):
    """注册新用户"""
    conn = sqlite3.connect(DB_PATH)
    try:
        pwd_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    finally:
        conn.close()

def save_simulation(username, config_dict, kpi_dict):
    """保存仿真记录"""
    conn = sqlite3.connect(DB_PATH)
    
    # 简单的 numpy 类型清洗，防止 JSON 序列化报错
    clean_kpis = {}
    for k, v in kpi_dict.items():
        if hasattr(v, 'item'): 
            clean_kpis[k] = v.item()
        else:
            clean_kpis[k] = v
            
    conn.execute("INSERT INTO history (username, timestamp, config, kpis) VALUES (?, ?, ?, ?)",
                 (username, datetime.now().strftime("%Y-%m-%d %H:%M"), 
                  json.dumps(config_dict), json.dumps(clean_kpis)))
    conn.commit()
    conn.close()

# [关键修复] 函数名必须是 get_history，以匹配 dashboard_view.py 的导入
def get_history(username):
    """获取用户历史记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 倒序查询最近 10 条
    c.execute("SELECT timestamp, config, kpis FROM history WHERE username=? ORDER BY id DESC LIMIT 10", (username,))
    rows = c.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "time": r[0],
            "config": json.loads(r[1]),
            "kpis": json.loads(r[2])
        })
    return history