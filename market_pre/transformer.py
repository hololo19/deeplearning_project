# TRANSFORMER_TRAIN.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以获得可重现的结果
np.random.seed(42)
tf.random.set_seed(42)

# 加载预处理数据
print("加载预处理数据...")
exec(open('/home/holo/market_pre/data_process.py').read())

# 选择单个资产进行训练（使用BTC作为示例）
asset_name = 'NFLX'  # 可更改为'DPZ', 'BTC', 'NFLX'

# 使用序列数据进行Transformer训练，但仅选择一个资产
y_train_seq_full = processed_data['y_seq_train']
y_test_seq_full = processed_data['y_seq_test']

# 确定资产在数据中的索引位置
assets = ['AMZN', 'DPZ', 'BTC', 'NFLX']
asset_index = assets.index(asset_name)

# 选择单个资产数据
y_train_seq_single = y_train_seq_full[:, asset_index:asset_index+1]  # 保持2D数组
y_test_seq_single = y_test_seq_full[:, asset_index:asset_index+1]
scaler_y_single = MinMaxScaler()

# 重新缩放单个资产数据
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

# 使用单个资产的历史价格重新创建序列数据
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

# Transformer位置编码函数
def positional_encoding(position, d_model):
    """计算位置编码"""
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    # 将 sin 应用于偶数索引；将 cos 应用于奇数索引
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# 创建Transformer编码器层
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")
    
    # 多头注意力
    attention = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model, dropout=dropout)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # 前馈神经网络
    outputs = Dense(units=units, activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

# 创建编码器
def encoder(time_steps, d_model, num_heads, dropout, 
            num_layers, dff, name="encoder"):
    inputs = Input(shape=(time_steps, 1), name="inputs")
    
    # 嵌入层 - 将1维输入扩展到d_model维
    embedding = Dense(d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    # 添加位置编码
    pos_encoding = positional_encoding(time_steps, d_model)
    embedding += pos_encoding[:, :time_steps, :]
    
    outputs = Dropout(dropout)(embedding)
    
    # 多层编码器
    for i in range(num_layers):
        outputs = encoder_layer(
            units=dff,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )(outputs)
        
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

# 创建Transformer模型
def create_transformer_model(time_steps, d_model, num_heads, 
                            num_layers, dff, dropout):
    inputs = Input(shape=(time_steps, 1))
    
    # 编码器
    enc_outputs = encoder(
        time_steps=time_steps,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        dff=dff)(inputs)
    
    # 全局平均池化
    outputs = tf.keras.layers.GlobalAveragePooling1D()(enc_outputs)
    
    # 输出层
    outputs = Dense(25, activation='relu')(outputs)
    outputs = Dropout(dropout)(outputs)
    outputs = Dense(1)(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# 创建模型
time_steps = X_train_seq_single.shape[1]  # 序列长度
d_model = 64  # 模型维度
num_heads = 4  # 注意力头数
num_layers = 2  # 编码器层数
dff = 128  # 前馈网络维度
dropout_rate = 0.2  # Dropout率

model = create_transformer_model(
    time_steps=time_steps,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dff=dff,
    dropout=dropout_rate
)

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
    metrics=['mae']
)

print("模型结构:")
model.summary()

# 训练模型
print("开始模型训练...")
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
print("正在进行预测...")
train_predictions = model.predict(X_train_seq_single)
test_predictions = model.predict(X_test_seq_single)

# 反向变换预测结果
train_predictions_rescaled = scaler_y_single.inverse_transform(train_predictions)
test_predictions_rescaled = scaler_y_single.inverse_transform(test_predictions)
y_train_actual = scaler_y_single.inverse_transform(y_train_seq_single)
y_test_actual = scaler_y_single.inverse_transform(y_test_seq_single)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_rescaled))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_rescaled))
test_mae = mean_absolute_error(y_test_actual, test_predictions_rescaled)

print(f"\n模型评估结果 ({asset_name}):")
print(f"训练RMSE: {train_rmse:.2f}")
print(f"测试RMSE: {test_rmse:.2f}")
print(f"测试MAE: {test_mae:.2f}")

# 绘制预测结果 - 修正版本
plt.figure(figsize=(12, 6))

# 确保数据是一维的用于绘图
y_test_actual_flat = y_test_actual.flatten()
test_predictions_rescaled_flat = test_predictions_rescaled.flatten()

# 绘制实际值
plt.plot(y_test_actual_flat, label=f'实际 {asset_name} 价格', color='blue')

# 绘制预测值
plt.plot(test_predictions_rescaled_flat, label=f'预测 {asset_name} 价格', color='red', linestyle='--')

plt.title(f'{asset_name} 价格预测')
plt.xlabel('时间')
plt.ylabel('价格')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 预测未来价格
def predict_future_transformer(model, last_sequence, scaler, days=5):
    """
    预测未来价格
    
    参数:
    model: 训练好的Transformer模型
    last_sequence: 最后序列数据
    scaler: 用于反向变换的缩放器
    days: 预测天数
    
    返回:
    未来几天的预测价格
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # 预测下一天
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
        future_predictions.append(next_pred[0])
        
        # 更新序列：移除最旧的一天，添加预测的一天
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred[0]
    
    # 反向变换预测结果
    future_predictions = np.array(future_predictions)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)
    
    return future_predictions_rescaled

# 使用最后10天数据预测接下来5天
last_sequence = X_test_seq_single[-1]
future_prices = predict_future_transformer(model, last_sequence, scaler_y_single, days=5)

print(f"\n未来5天 {asset_name} 价格预测:")
print(future_prices.flatten())

# 保存模型
model.save(f'/home/holo/market_pre/transformer_{asset_name.lower()}_model.keras')
print(f"\n模型已保存至: /home/holo/market_pre/transformer_{asset_name.lower()}_model.keras")

print(f"\n{asset_name} Transformer模型训练和预测完成!")