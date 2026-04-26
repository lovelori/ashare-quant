"""
A股量化推荐系统 - 配置文件 (幻方量化增强版)
"""
import os
from datetime import datetime, timedelta

CONFIG = {
    # 股票池: 'hs300' = 沪深300, 'all' = 全市场, 或自定义代码列表
    'stock_pool': 'hs300',

    # 技术分析参数
    'kline_days': 120,          # 每支股获取多少天K线数据
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'boll_period': 20,
    'boll_std': 2,
    'ma_periods': [5, 10, 20, 60],

    # 评分权重 (传统多因子 60% + ML策略 40%)
    'weights': {
        'trend': 0.12,          # 趋势因子 (均线排列)
        'momentum': 0.10,       # 动量因子 (RSI, MACD)
        'volume': 0.10,         # 量价因子 (成交量变化)
        'volatility': 0.08,     # 波动率因子 (ATR, 布林带宽)
        'sentiment': 0.10,      # 舆情情绪因子
        'fundamental': 0.05,    # 基本面因子
        'pattern': 0.05,        # K线形态因子
    },

    # 幻方AI策略权重 (与weights一起构成综合评分)
    'hf_weights': {
        'ml_score': 0.25,       # ML策略评分
        'alpha_score': 0.10,    # Alpha因子评分
        'deep_score': 0.05,     # 深度因子评分
    },

    # 舆情参数
    'news_days': 3,             # 取几天内的新闻
    'max_news_per_stock': 20,   # 每支股最多取多少条新闻

    # 卖出信号参数
    'sell_signals': {
        'ma_break_pct': -0.03,      # 跌破MA60超过3%
        'rsi_overbought_threshold': 75,
        'rsi_overbought_drop': -0.05,  # 从超买区回落5%
        'volume_divergence': 0.3,   # 量价背离阈值
        'stop_loss_pct': -0.08,     # 止损 -8%
        'take_profit_pct': 0.20,    # 止盈 +20%
    },

    # AI策略参数
    'hf_config': {
        'enable_ml': True,          # 启用ML策略
        'enable_alpha_factors': True,  # 启用Alpha因子
        'enable_deep_features': True,  # 启用深度因子
        'adaptive_weights': True,   # 自适应权重
        'market_regime_detection': True,  # 市场状态检测
    },

    # 每日推荐的股票数量
    'top_n': 6,

    # 运行模式
    'mode': 'daily',            # 'daily' 每日推荐, 'backtest' 历史回测

    # 数据路径 (相对于项目根目录)
    'data_dir': 'data',
    'reports_dir': 'reports',

    # 行业黑名单 (排除的行业)
    'exclude_industries': [],
}

# 数据路径 (基于项目根目录自动解析)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG['data_dir'] = os.path.join(_PROJECT_ROOT, CONFIG['data_dir'])
CONFIG['reports_dir'] = os.path.join(_PROJECT_ROOT, CONFIG['reports_dir'])

# 自动计算日期
CONFIG['end_date'] = datetime.now().strftime('%Y%m%d')
CONFIG['start_date'] = (datetime.now() - timedelta(days=CONFIG['kline_days'] + 30)).strftime('%Y%m%d')
CONFIG['news_since'] = (datetime.now() - timedelta(days=CONFIG['news_days'])).strftime('%Y-%m-%d')
