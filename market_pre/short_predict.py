import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname('/home/holo/market_pre/'))

def load_processed_data():
    """
    加载预处理后的数据
    """
    print("正在加载预处理数据...")
    # 执行数据处理脚本
    global processed_data
    exec(open('/home/holo/market_pre/data_process.py').read())
    print("数据加载完成!")
    return locals()['processed_data']

def predict_future_with_actual(model, processed_data, start_index, days=5):
    """
    从指定位置开始预测未来几天的价格，并获取实际值进行对比
    
    Parameters:
    model: 训练好的模型
    processed_data: 预处理后的数据
    start_index: 开始预测的位置（在测试集中的索引）
    days: 预测天数
    
    Returns:
    future_predictions: 未来价格预测
    actual_future: 对应的实际价格
    """
    print(f"\n从测试集中第 {start_index} 个样本开始预测未来 {days} 天的价格...")
    
    X_test_seq = processed_data['X_seq_test']
    y_test_seq = processed_data['y_seq_test']
    scaler_y = processed_data['scaler_y']
    
    # 确保索引有效
    if start_index >= len(X_test_seq) - days:
        raise ValueError("起始索引过大，无法获取足够的未来真实数据")
    
    # 使用指定位置的数据作为起点
    current_sequence = X_test_seq[start_index].copy()
    
    future_predictions = []
    temp_sequence = current_sequence.copy()
    
    # 预测未来指定天数
    for _ in range(days):
        # 预测下一天
        next_pred = model.predict(temp_sequence.reshape(1, temp_sequence.shape[0], temp_sequence.shape[1]), verbose=0)
        future_predictions.append(next_pred[0])
        
        # 更新序列（模拟真实的预测场景）
        temp_sequence = np.roll(temp_sequence, -1, axis=0)
        temp_sequence[-1] = next_pred[0]
    
    # 反标准化预测结果
    future_predictions = np.array(future_predictions)
    future_predictions_rescaled = scaler_y.inverse_transform(future_predictions)
    
    # 获取对应的实际未来数据
    actual_future = y_test_seq[start_index+1:start_index+1+days]  # 从下一天开始的真实数据
    actual_future_rescaled = scaler_y.inverse_transform(actual_future)
    
    assets = ['AMZN', 'DPZ', 'BTC', 'NFLX']
    print("\n=== 未来价格预测 vs 实际值 ===")
    for i, asset in enumerate(assets):
        print(f"\n{asset}:")
        print(f"  预测值: {future_predictions_rescaled[:, i]}")
        print(f"  实际值: {actual_future_rescaled[:, i]}")
    
    return future_predictions_rescaled, actual_future_rescaled

def visualize_future_vs_actual(future_predictions, actual_future, title_suffix=""):
    """
    可视化未来预测价格与实际价格的对比
    
    Parameters:
    future_predictions: 预测的未来价格
    actual_future: 实际的未来价格
    title_suffix: 图表标题后缀
    """
    print("正在生成未来价格预测可视化图表...")
    
    assets = ['AMZN', 'DPZ', 'BTC', 'NFLX']
    asset_names = {'AMZN': 'Amazon', 'DPZ': 'Domino\'s Pizza', 'BTC': 'Bitcoin', 'NFLX': 'Netflix'}
    days = len(future_predictions)
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    for i, asset in enumerate(assets):
        plt.subplot(2, 2, i+1)
        
        # 绘制实际值
        x_actual = range(1, days+1)
        plt.plot(x_actual, actual_future[:, i], label=f'{asset_names[asset]} Real', 
                color='blue', marker='o', linewidth=2)
        
        # 绘制预测值
        x_pred = range(1, days+1)
        plt.plot(x_pred, future_predictions[:, i], label=f'{asset_names[asset]} predict', 
                color='red', marker='s', linestyle='--', linewidth=2)
        
        plt.title(f'{asset_names[asset]} future{days}predict {title_suffix}')
        plt.xlabel('days')
        plt.ylabel('price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/home/holo/market_pre/future_prediction_vs_actual{title_suffix.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"未来价格预测图表已保存到: /home/holo/market_pre/future_prediction_vs_actual{title_suffix.replace(' ', '_')}.png")

def main():
    """
    主函数
    """
    # 加载模型
    print("正在加载模型...")
    model = load_model('/home/holo/market_pre/lstm_stock_model.keras')
    print("模型加载成功!")
    
    # 加载数据
    processed_data = load_processed_data()
    
    # 选择测试集中的一个位置进行预测（例如中间位置）
    start_index = len(processed_data['X_seq_test']) // 3
    
    # 预测未来5天价格并与实际值对比
    future_predictions, actual_future = predict_future_with_actual(
        model, processed_data, start_index, days=5)
    
    # 可视化预测结果与实际值对比
    visualize_future_vs_actual(future_predictions, actual_future, f"(起始索引: {start_index})")
    
    print("\n预测完成!")

if __name__ == "__main__":
    main()