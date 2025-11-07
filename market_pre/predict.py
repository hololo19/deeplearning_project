# predict.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')



def load_trained_model(model_path='/home/holo/market_pre/lstm_stock_model.keras'):
    """
    加载训练好的模型
    
    Parameters:
    model_path: 模型保存路径
    
    Returns:
    model: 加载的模型
    """
    print("正在加载模型...")
    model = load_model(model_path)
    print("模型加载成功!")
    return model

def load_processed_data():
    """
    加载预处理后的数据
    
    Returns:
    processed_data: 包含预处理数据的字典
    """
    print("正在加载预处理数据...")
    # 导入数据处理模块
    import sys
    import os
    sys.path.append(os.path.dirname('/home/holo/market_pre/'))
    
    # 直接导入数据处理过程
    global processed_data
    exec(open('/home/holo/market_pre/data_process.py').read())
    print("数据加载完成!")
    return locals()['processed_data']

def make_predictions(model, processed_data):
    """
    使用模型进行预测
    
    Parameters:
    model: 训练好的模型
    processed_data: 预处理后的数据
    
    Returns:
    predictions: 预测结果字典
    """
    print("正在进行预测...")
    
    # 获取测试数据
    X_test_seq = processed_data['X_seq_test']
    y_test_seq = processed_data['y_seq_test']
    scaler_y = processed_data['scaler_y']
    
    # 模型预测
    test_predictions = model.predict(X_test_seq)
    
    # 反标准化预测结果
    test_predictions_rescaled = scaler_y.inverse_transform(test_predictions)
    y_test_actual = scaler_y.inverse_transform(y_test_seq)
    
    predictions = {
        'predicted': test_predictions_rescaled,
        'actual': y_test_actual
    }
    
    return predictions

def visualize_predictions(predictions):
    """
    可视化预测结果并与真值比较
    
    Parameters:
    predictions: 预测结果字典
    """
    print("正在生成可视化图表...")
    
    predicted = predictions['predicted']
    actual = predictions['actual']
    
    assets = ['AMZN', 'DPZ', 'BTC', 'NFLX']
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    for i, asset in enumerate(assets):
        plt.subplot(2, 2, i+1)
        
        # 绘制实际值
        plt.plot(actual[:, i], label=f'{asset} Real price', color='blue')
        
        # 绘制预测值
        plt.plot(predicted[:, i], label=f'{asset} Predict preice', color='red', linestyle='--')
        
        plt.title(f'{asset} Predict vs Real')
        plt.xlabel('Time')
        plt.ylabel('PRice')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/holo/market_pre/prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存到: /home/holo/market_pre/prediction_comparison.png")

def calculate_metrics(predictions):
    """
    计算预测指标
    
    Parameters:
    predictions: 预测结果字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    predicted = predictions['predicted']
    actual = predictions['actual']
    
    print("\n=== 模型预测评估结果 ===")
    
    assets = ['AMZN', 'DPZ', 'BTC', 'NFLX']
    for i, asset in enumerate(assets):
        rmse = np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))
        mae = mean_absolute_error(actual[:, i], predicted[:, i])
        print(f"{asset} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")


def main():
    """
    主函数
    """
    # 加载模型
    model = load_trained_model()
    
    # 加载数据
    processed_data = load_processed_data()
    
    # 进行预测
    predictions = make_predictions(model, processed_data)
    
    # 计算评估指标
    calculate_metrics(predictions)
    
    # 可视化预测结果
    visualize_predictions(predictions)
    


if __name__ == "__main__":
    main()