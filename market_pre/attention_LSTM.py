# attention_LSTM.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

# 加载预处理数据
print("加载预处理数据...")
exec(open('/home/holo/market_pre/data_process.py').read())

# 选择单个资产进行训练
asset_name = 'BTC'  # 可以更改为 'DPZ', 'BTC', 'NFLX'

# 使用序列数据，但仅选择一个资产
X_train_seq_full = processed_data['X_seq_train']
X_test_seq_full = processed_data['X_seq_test']
y_train_seq_full = processed_data['y_seq_train']
y_test_seq_full = processed_data['y_seq_test']

# 准备单个资产数据
asset_index = {'AMZN': 0, 'DPZ': 1, 'BTC': 2, 'NFLX': 3}[asset_name]
y_train_seq_single = y_train_seq_full[:, asset_index:asset_index+1]
y_test_seq_single = y_test_seq_full[:, asset_index:asset_index+1]

# 为单个资产创建新的标准化器
scaler_y_single = MinMaxScaler()

# 缩放单个资产数据
y_train_single_prices = processed_data['y_train'][:, asset_index:asset_index+1]
y_test_single_prices = processed_data['y_test'][:, asset_index:asset_index+1]
y_train_single_scaled = scaler_y_single.fit_transform(y_train_single_prices)
y_test_single_scaled = scaler_y_single.transform(y_test_single_prices)

# 为单个资产创建序列数据
def create_single_sequences(data, seq_length=10):
    """为单个资产创建时间序列数据"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 重新创建序列数据，仅使用单个资产的历史价格
seq_length = 10
single_asset_prices = processed_data['df_processed'][asset_name].values.reshape(-1, 1)
single_asset_scaled = scaler_y_single.fit_transform(single_asset_prices)

X_seq_single, y_seq_single = create_single_sequences(single_asset_scaled, seq_length)

# 分割为训练/测试集
split_idx = int(len(X_seq_single) * 0.8)
X_train_seq_single = X_seq_single[:split_idx]
X_test_seq_single = X_seq_single[split_idx:]
y_train_seq_single = y_seq_single[:split_idx]
y_test_seq_single = y_seq_single[split_idx:]

print(f"单个资产 ({asset_name}) 训练数据形状: {X_train_seq_single.shape}")
print(f"单个资产 ({asset_name}) 测试数据形状: {X_test_seq_single.shape}")

# 构建带注意力机制的LSTM模型
def create_attention_lstm_model(input_shape):
    """
    创建带注意力机制的LSTM模型
    """
    # 输入层
    inputs = Input(shape=input_shape)
    
    # LSTM层 - 返回序列以用于注意力机制
    lstm_out = LSTM(units=100, return_sequences=True, name='lstm_encoder')(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    
    lstm_out = LSTM(units=100, return_sequences=True, name='lstm_decoder')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # 注意力机制
    # 创建查询、键和值
    query = lstm_out
    key = lstm_out
    value = lstm_out
    
    # 注意力层
    attention_output = Attention(name='attention_layer')([query, key, value])
    
    # 使用注意力输出的最后一个时间步
    # 通过全局平均池化处理注意力输出
    attention_last = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
    
    # 全连接层
    dense_out = Dense(units=50, activation='relu', name='dense_1')(attention_last)
    dense_out = Dropout(0.2)(dense_out)
    
    # 输出层 - 预测单个资产价格
    outputs = Dense(units=1, name='output')(dense_out)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs, name='attention_lstm_model')
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    
    return model

# 创建模型
input_shape = (X_train_seq_single.shape[1], X_train_seq_single.shape[2])
model = create_attention_lstm_model(input_shape)

print("模型结构:")
model.summary()

# 训练模型
print("开始训练模型...")
history = model.fit(
    X_train_seq_single, y_train_seq_single,
    epochs=500,
    batch_size=32,
    validation_data=(X_test_seq_single, y_test_seq_single),
    verbose=1,
    shuffle=False
)

# 绘制训练历史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 模型预测
print("进行预测...")
train_predictions = model.predict(X_train_seq_single)
test_predictions = model.predict(X_test_seq_single)

# 反标准化预测结果
train_predictions_rescaled = scaler_y_single.inverse_transform(train_predictions)
test_predictions_rescaled = scaler_y_single.inverse_transform(test_predictions)
y_train_actual = scaler_y_single.inverse_transform(y_train_seq_single)
y_test_actual = scaler_y_single.inverse_transform(y_test_seq_single)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_rescaled))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_rescaled))
test_mae = mean_absolute_error(y_test_actual, test_predictions_rescaled)

print(f"\n模型评估结果 ({asset_name}):")
print(f"训练集 RMSE: {train_rmse:.2f}")
print(f"测试集 RMSE: {test_rmse:.2f}")
print(f"测试集 MAE: {test_mae:.2f}")

# 绘制预测结果
plt.figure(figsize=(12, 6))

# 绘制实际值
plt.plot(y_test_actual, label=f'实际 {asset_name} 价格', color='blue')

# 绘制预测值
plt.plot(test_predictions_rescaled, label=f'预测 {asset_name} 价格', color='red', linestyle='--')

plt.title(f'{asset_name} 价格预测')
plt.xlabel('时间')
plt.ylabel('价格')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 预测未来价格 - 参考 SINGLE_train.py
def predict_future_single(model, last_sequence, scaler, days=5):
    """
    预测未来价格
    
    参数:
    model: 训练好的注意力LSTM模型
    last_sequence: 最后序列数据
    scaler: 用于反标准化的缩放器
    days: 预测天数
    
    返回:
    未来几天的预测价格
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # 预测下一天
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0])
        
        # 更新序列: 移除最旧的一天，添加预测的一天
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred[0]
    
    # 反标准化预测结果
    future_predictions = np.array(future_predictions)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)
    
    return future_predictions_rescaled

# 使用最后10天数据预测接下来5天
last_sequence = X_test_seq_single[-1]
future_prices = predict_future_single(model, last_sequence, scaler_y_single, days=5)

print(f"\n未来5天 {asset_name} 价格预测:")
print(future_prices.flatten())

# 保存模型
model.save(f'/home/holo/market_pre/attention_lstm_{asset_name.lower()}_model.keras')
print(f"\n模型已保存到: /home/holo/market_pre/attention_lstm_{asset_name.lower()}_model.keras")

print(f"\n{asset_name} 注意力LSTM模型训练和预测完成!")