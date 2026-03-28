import numpy as np
import pandas as pd
import os
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# --- 引入深度学习框架 (用于高级特征提取与非线性映射) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# 1. 深度学习网络架构定义 (PyTorch)
class BatteryThermalDNN(nn.Module):
    """
    动力电池热力学深度神经网络 (Deep Neural Network)
    用于捕捉高倍率下极化生热的非线性瞬态特征
    """
    def __init__(self, input_dim=3, hidden_dim=64):
        super(BatteryThermalDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),#矩阵乘法
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 1) # 输出层：产热功率 (Heat_Power)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
    def forward(self, x):
        return self.network(x)

def load_and_augment_data():
    """加载真实数据并进行工业级数据增强 (Data Augmentation)"""
    print("📂 正在读取并融合真实 CSV 数据...")
    files = [
        ('1c_discharge.csv', 1.0, 1),
        ('1c_charge.csv', 1.0, 0),
        ('0.5c_discharge.csv', 0.5, 1),
        ('0.5c_charge.csv', 0.5, 0)
    ]
    
    df_list = []
    for f, c_rate, is_dch in files:
        if os.path.exists(f):
            temp_df = pd.read_csv(f, header=None, names=['Time', 'Heat_Power'])
            temp_df['C_Rate'] = c_rate
            temp_df['Is_Discharge'] = is_dch
            df_list.append(temp_df)
    
    if not df_list:
        raise FileNotFoundError("未找到任何训练 CSV 文件！")
        
    df = pd.concat(df_list, ignore_index=True)
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Heat_Power'] = pd.to_numeric(df['Heat_Power'], errors='coerce')
    df = df.dropna()
    
    #  数据增强：引入传感器白噪声模拟真实 BMS 环境
    print("🧬 正在执行特征工程与数据增强...")
    augmented_dfs = [df]
    for _ in range(2): # 扩增两倍数据
        noise_df = df.copy()
        noise_df['Heat_Power'] += np.random.normal(0, 0.05, len(df)) # 添加 5% 高斯噪声
        noise_df['Time'] += np.random.uniform(-0.5, 0.5, len(df)) # 时间轴抖动
        augmented_dfs.append(noise_df)
        
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    return final_df

def train_dual_models():
    """训练双轨 AI 模型 (Random Forest + PyTorch DNN)"""
    df = load_and_augment_data()
    print(f"📊 成功加载数据！扩增后总计 {len(df)} 个有效数据点。")
    
    X = df[['Time', 'C_Rate', 'Is_Discharge']].values
    y = df['Heat_Power'].values.reshape(-1, 1)
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # ==========================================
    # 阶段 1: 训练 Random Forest (鲁棒性基线)
    # ==========================================
    print("\n🌲 [Phase 1] 训练随机森林回归器...")
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_split=4, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train.ravel())
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    print(f"✅ RF 训练完成 | 耗时: {time.time()-start_time:.2f}s | MSE: {rf_mse:.4f}")
    
    # ==========================================
    # 阶段 2: 训练 PyTorch DNN (高精尖预测)
    # ==========================================
    print("\n🧠 [Phase 2] 训练深度神经网络 (PyTorch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用计算加速设备: {device}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    dnn_model = BatteryThermalDNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(dnn_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    epochs = 50
    for epoch in range(epochs):
        dnn_model.train()#开启训练模式
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)#瞎猜
            optimizer.zero_grad()#梯度清零
            outputs = dnn_model(batch_X)#前向传播
            loss = criterion(outputs, batch_y)#计算损失
            loss.backward()#复盘
            optimizer.step()#更新权重
            epoch_loss += loss.item()
            
        # 验证评估
        dnn_model.eval()
        with torch.no_grad():
            val_X = torch.FloatTensor(X_test).to(device)
            val_y = torch.FloatTensor(y_test).to(device)
            val_pred = dnn_model(val_X)
            val_loss = criterion(val_pred, val_y)
            scheduler.step(val_loss)
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss.item():.4f}")
            
    print("✅ PyTorch DNN 训练收敛完成。")
    
    # ==========================================
    # 阶段 3: 权重导出与持久化
    # ==========================================
    os.makedirs('models/weights', exist_ok=True)
    
    # 统一打包保存 (包含模型和标准化器)
    export_package = {
        'rf_model': rf_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metadata': {'version': '2.0', 'features': ['Time', 'C_Rate', 'Is_Discharge']}
    }
    
    joblib.dump(export_package, 'models/weights/heat_ai_model.pkl')
    torch.save(dnn_model.state_dict(), 'models/weights/dnn_weights.pth')
    print("\n💾 模型联合权重与预处理器已安全持久化至 models/weights/ 目录！")

if __name__ == "__main__":
    train_dual_models()
