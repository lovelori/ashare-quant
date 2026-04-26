"""
A股量化推荐系统 - 配置文件
"""
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

    # 评分权重
    'weights': {
        'trend': 0.20,          # 趋势因子 (均线排列)
        'momentum': 0.15,       # 动量因子 (RSI, MACD)
        'volume': 0.15,         # 量价因子 (成交量变化)
        'volatility': 0.10,     # 波动率因子 (ATR, 布林带宽)
        'sentiment': 0.15,      # 舆情情绪因子
        'fundamental': 0.15,    # 基本面因子 (换手率等)
        'pattern': 0.10,        # K线形态因子
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

    # 每日推荐的股票数量
    'top_n': 6,

    # 运行模式
    'mode': 'daily',            # 'daily' 每日推荐, 'backtest' 历史回测

    # 数据路径
    'data_dir': '/home/cdh/ashare_quant/data',
    'reports_dir': '/home/cdh/ashare_quant/reports',

    # 行业黑名单 (排除的行业)
    'exclude_industries': [],
}

# 自动计算日期
CONFIG['end_date'] = datetime.now().strftime('%Y%m%d')
CONFIG['start_date'] = (datetime.now() - timedelta(days=CONFIG['kline_days'] + 30)).strftime('%Y%m%d')
CONFIG['news_since'] = (datetime.now() - timedelta(days=CONFIG['news_days'])).strftime('%Y-%m-%d')
