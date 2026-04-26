"""
深度技术因子库 - 幻方量化风格
基于AI因子挖掘思路，生成大量技术因子 + 非线性组合

核心思想:
1. 不仅仅是标准指标，而是通过时序运算生成大量alpha信号
2. 因子之间的交互作用 (rank_corr, 比值, 差值)
3. 横截面标准化 (z-score across stocks)
4. 时序衰减 (最近的数据权重更高)
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP 成交量加权均价"""
    return (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()


def compute_money_flow(df: pd.DataFrame) -> pd.Series:
    """资金流向指标 (类似MFI)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    mfr = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum().replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def compute_volume_trend(df: pd.DataFrame) -> dict:
    """量价趋势因子 - 识别主力资金动向"""
    close = df['close']
    volume = df['volume']

    # VPIN (Volume-synchronized Probability of Informed Trading)
    # 简化版：价格变动方向 * 成交量
    price_direction = np.sign(close.diff())
    volume_flow = price_direction * volume
    vpin = volume_flow.rolling(20).mean() / volume.rolling(20).mean()

    # 量价相关系数 (滚动20日)
    def rolling_corr(x, y, w=20):
        return x.rolling(w).corr(y)

    vol_price_corr = rolling_corr(volume, close.pct_change())

    # Amihud 非流动性指标 (|return| / volume)
    illiquidity = close.pct_change().abs() / volume
    illiquidity = illiquidity.replace([np.inf, -np.inf], np.nan)

    return {
        'vpin': vpin,
        'vol_price_corr': vol_price_corr,
        'illiquidity': illiquidity,
    }


def compute_price_structure(df: pd.DataFrame) -> dict:
    """价格结构因子 - 捕捉非线性价格特征"""
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']

    # 开盘-收盘相对位置 (O-C / range)
    oc_position = (open_p - close) / (high - low).replace(0, np.nan)

    # 日内波动位置 (close - low) / (high - low)
    close_position = (close - low) / (high - low).replace(0, np.nan)

    # 价格序列熵 (5日价格的复杂度)
    entropy = close.rolling(5).apply(
        lambda x: -sum(v / len(x) * np.log(v / len(x) + 1e-10)
                       for v in (x - x.min()) / (x.max() - x.min() + 1e-10)),
        raw=True
    )

    # 价格加速度 (二阶导数)
    velocity = close.diff()
    acceleration = velocity.diff()

    return {
        'oc_position': oc_position,
        'close_position': close_position,
        'entropy': entropy,
        'acceleration': acceleration,
    }


def compute_volatility_features(df: pd.DataFrame) -> dict:
    """波动率结构因子"""
    close = df['close']
    high = df['high']
    low = df['low']

    # Parkinson 波动率 (H-L based)
    parkinson = np.sqrt(1 / (4 * np.log(2)) * (np.log(high / low) ** 2))
    parkinson_vol = parkinson.rolling(20).mean()

    # Yang-Zhang 波动率 (考虑隔夜跳空)
    log_ho = np.log(high / close.shift(1))
    log_lo = np.log(low / close.shift(1))
    log_co = np.log(close / close.shift(1))
    yz_vol = np.sqrt(
        parkinson.rolling(20).mean() ** 2 +
        log_co.rolling(20).std() ** 2
    )

    # 波动率锥 (不同周期的滚动波动率比值)
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    vol_ratio_5_20 = vol_5 / vol_20.replace(0, np.nan)

    # 波动率偏度 (正负波动不对称性)
    returns = close.pct_change()
    neg_vol = returns[returns < 0].rolling(20).std()
    pos_vol = returns[returns > 0].rolling(20).std()
    vol_skew = (pos_vol - neg_vol) / (pos_vol + neg_vol).replace(0, np.nan)

    return {
        'parkinson_vol': parkinson_vol,
        'yz_vol': yz_vol,
        'vol_ratio_5_20': vol_ratio_5_20,
        'vol_skew': vol_skew,
    }


def compute_momentum_decay(df: pd.DataFrame) -> dict:
    """动量衰减因子 - 识别动量持续性"""
    close = df['close']

    # 加权动量 (最近权重更高)
    def weighted_momentum(price, lookback, half_life):
        weights = np.exp(-np.arange(lookback) * np.log(2) / half_life)
        weights = weights / weights.sum()
        rets = price.pct_change(lookback)
        return rets  # 简化版

    # 不同时间尺度的动量
    mom_5 = close.pct_change(5)
    mom_10 = close.pct_change(10)
    mom_20 = close.pct_change(20)

    # 动量衰减率 (短期动量 / 长期动量)
    mom_decay_5_20 = mom_5 / (mom_20 + 1e-10).abs().clip(lower=0.001)
    mom_decay_10_20 = mom_10 / (mom_20 + 1e-10).abs().clip(lower=0.001)

    # 动量加速 (Momentum 2nd derivative)
    mom_accel = mom_5 - mom_10

    # 动量趋同/发散 (短期均线相对于长期均线的位置)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma_convergence = (ma5 - ma20) / (ma20 - ma60 + 1e-10)

    return {
        'mom_5': mom_5,
        'mom_10': mom_10,
        'mom_20': mom_20,
        'mom_decay_5_20': mom_decay_5_20,
        'mom_accel': mom_accel,
        'ma_convergence': ma_convergence,
    }


def compute_cross_sectional_rank(series: pd.Series) -> pd.Series:
    """横截面排名 (0-1之间)"""
    return series.rank(pct=True)


def compute_ts_rank(series: pd.Series, window: int = 20) -> pd.Series:
    """时序排名 - 当前值在过去window天中的位置"""
    return series.rolling(window).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=True)


def compute_all_deep_features(df: pd.DataFrame) -> dict:
    """
    计算所有深度技术因子
    返回最新一期因子值的dict
    """
    if df is None or len(df) < 30:
        return {}

    features = {}

    # VWAP
    vwap = compute_vwap(df)
    close = df['close']
    features['vwap_ratio'] = float((close.iloc[-1] / vwap.iloc[-1] - 1) if not vwap.empty else 0)

    # 资金流向
    mfi = compute_money_flow(df)
    features['mfi'] = float(mfi.iloc[-1]) if not mfi.empty else 50
    features['mfi_trend'] = int((mfi.iloc[-1] > mfi.iloc[-5])) if len(mfi) >= 5 else 0

    # 量价趋势
    vt = compute_volume_trend(df)
    for k, v in vt.items():
        if v is not None and not v.empty:
            features[k] = float(v.iloc[-1])

    # 价格结构
    ps = compute_price_structure(df)
    for k, v in ps.items():
        if v is not None and not v.empty and not np.isnan(v.iloc[-1]):
            features[k] = float(v.iloc[-1])

    # 波动率结构
    vs = compute_volatility_features(df)
    for k, v in vs.items():
        if v is not None and not v.empty and not np.isnan(v.iloc[-1]):
            features[k] = float(v.iloc[-1])

    # 动量衰减
    md = compute_momentum_decay(df)
    for k, v in md.items():
        if v is not None and not v.empty and not np.isnan(v.iloc[-1]):
            features[k] = float(v.iloc[-1])

    # 时序排名指标
    tsrsi = compute_ts_rank(df['close'].pct_change().dropna(), 20)
    features['ts_rank_return'] = float(tsrsi.iloc[-1]) if not tsrsi.empty else 0.5

    tsvol = compute_ts_rank(df['volume'], 20)
    features['ts_rank_volume'] = float(tsvol.iloc[-1]) if not tsvol.empty else 0.5

    # 乖离率序列 (Bias ratio)
    for p in [5, 10, 20]:
        ma = df['close'].rolling(p).mean()
        bias = (df['close'] - ma) / ma * 100
        features[f'bias_{p}'] = float(bias.iloc[-1]) if not bias.empty else 0

    return features


def compute_all_deep_features_batch(kline_data: dict) -> dict:
    """批量计算深度因子"""
    result = {}
    for code, df in kline_data.items():
        if df is not None and len(df) >= 30:
            features = compute_all_deep_features(df)
            if features:
                result[code] = features
    print(f"[深度] 深度因子计算完成: {len(result)} 支")
    return result


if __name__ == '__main__':
    from data_fetcher import get_daily_kline
    df = get_daily_kline('000001')
    if len(df) > 0:
        f = compute_all_deep_features(df)
        print("平安银行深度因子:")
        for k, v in sorted(f.items()):
            print(f"  {k}: {v:.4f}")
