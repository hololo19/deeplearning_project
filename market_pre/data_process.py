import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('/home/holo/market_pre/portfolio_data.csv')

# 数据预处理
print("原始数据信息:")
print(df.info())
print("\n数据前5行:")
print(df.head())
print("\n数据统计信息:")
print(df.describe())

# 检查缺失值
print("\n缺失值检查:")
print(df.isnull().sum())

# 将日期列转换为datetime类型
df['Date'] = pd.to_datetime(df['Date'])

# 按日期排序
df = df.sort_values('Date').reset_index(drop=True)

print("\n数据预处理后信息:")
print(f"数据集形状: {df.shape}")
print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")

# 创建技术指标特征
def create_features(data):
    """创建技术指标特征"""
    df_features = data.copy()
    
    # 计算每日收益率
    for col in ['AMZN', 'DPZ', 'BTC', 'NFLX']:
        df_features[f'{col}_returns'] = df_features[col].pct_change()
        
        # 计算移动平均线
        df_features[f'{col}_ma_5'] = df_features[col].rolling(window=5).mean()
        df_features[f'{col}_ma_20'] = df_features[col].rolling(window=20).mean()
        
        # 计算波动率(标准差)
        df_features[f'{col}_vol_5'] = df_features[col].rolling(window=5).std()
        df_features[f'{col}_vol_20'] = df_features[col].rolling(window=20).std()
        
    return df_features

# 创建特征
df_processed = create_features(df)

# 删除包含NaN的行(由于计算移动平均线和波动率产生)
df_processed = df_processed.dropna().reset_index(drop=True)

print(f"\n特征工程后数据形状: {df_processed.shape}")

# 准备用于训练的数据
# 选择价格和收益率作为主要特征
feature_columns = ['AMZN', 'DPZ', 'BTC', 'NFLX', 
                   'AMZN_returns', 'DPZ_returns', 'BTC_returns', 'NFLX_returns',
                   'AMZN_ma_5', 'DPZ_ma_5', 'BTC_ma_5', 'NFLX_ma_5',
                   'AMZN_ma_20', 'DPZ_ma_20', 'BTC_ma_20', 'NFLX_ma_20']

X = df_processed[feature_columns]
y = df_processed[['AMZN', 'DPZ', 'BTC', 'NFLX']]  # 预测目标为四个资产的价格

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标矩阵形状: {y.shape}")

# 数据标准化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
)

# 为了时间序列预测，我们也可以创建序列数据
def create_sequences(data, seq_length=10):
    """创建时间序列数据"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 创建序列数据(用于RNN/LSTM等模型)
seq_length = 10
X_seq, y_seq = create_sequences(y_scaled, seq_length)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

print(f"\n序列数据形状:")
print(f"X序列训练集: {X_seq_train.shape}")
print(f"X序列测试集: {X_seq_test.shape}")
print(f"y序列训练集: {y_seq_train.shape}")
print(f"y序列测试集: {y_seq_test.shape}")

print(f"\n传统数据划分:")
print(f"X训练集: {X_train.shape}")
print(f"X测试集: {X_test.shape}")
print(f"y训练集: {y_train.shape}")
print(f"y测试集: {y_test.shape}")

# 保存处理后的数据
processed_data = {
    'df_original': df,
    'df_processed': df_processed,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_seq_train': X_seq_train,
    'X_seq_test': X_seq_test,
    'y_seq_train': y_seq_train,
    'y_seq_test': y_seq_test,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_columns': feature_columns
}

print("\n数据预处理完成！")
print("数据已准备好用于股票预测模型训练。")

# 显示一些处理后的数据示例
print("\n处理后数据示例 (前5行):")
print(df_processed[['Date'] + feature_columns].head())