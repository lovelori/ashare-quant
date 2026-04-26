"""
技术指标计算引擎
计算各类技术分析指标用于评分
"""
import numpy as np
import pandas as pd
from config import CONFIG


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 指标"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD 指标"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    histogram = 2 * (dif - dea)
    return dif, dea, histogram


def compute_bollinger(series: pd.Series, period=20, std_dev=2):
    """布林带指标"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / middle * 100  # 布林带宽 %
    b_position = (series - lower) / (upper - lower)  # %B
    return upper, middle, lower, bandwidth, b_position


def compute_ma(series: pd.Series, periods: list) -> dict:
    """移动均线"""
    return {f'ma{p}': series.rolling(window=p).mean() for p in periods}


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR 平均真实波幅"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_volume_indicators(df: pd.DataFrame) -> dict:
    """量价指标"""
    volume = df['volume']
    close = df['close']

    # 成交量均线
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()

    # 量比 (当前量 / 5日均量)
    vol_ratio = volume / vol_ma5.replace(0, np.nan)

    # 量价配合 (上涨放量, 下跌缩量)
    vol_trend = (close.pct_change() > 0) & (vol_ratio > 1.2)

    # OBV (简易版)
    obv = (volume * np.sign(close.diff())).fillna(0).cumsum()

    return {
        'vol_ma5': vol_ma5,
        'vol_ma20': vol_ma20,
        'vol_ratio': vol_ratio,
        'vol_trend': vol_trend.astype(int),
        'obv': obv,
    }


def compute_momentum(series: pd.Series, periods=[5, 10, 20]) -> dict:
    """动量指标"""
    result = {}
    for p in periods:
        result[f'mom_{p}'] = series.pct_change(p)
    return result


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """
    K线形态识别 (最近3根K线)
    """
    if len(df) < 5:
        return {}

    recent = df.tail(5).copy()
    open_p = recent['open'].values
    close = recent['close'].values
    high = recent['high'].values
    low = recent['low'].values

    patterns = {}

    # 锤子线 / 吊人线 (下影线长, 实体小)
    body = abs(close - open_p)
    lower_shadow = np.minimum(open_p, close) - low
    upper_shadow = high - np.maximum(open_p, close)
    hammer = (lower_shadow > body * 2) & (upper_shadow < body * 0.3)
    patterns['hammer_last'] = int(hammer[-1])
    patterns['hammer_last3'] = int(hammer[-3:].any())

    # 吞没形态
    if len(recent) >= 2:
        prev_body = abs(close[-2] - open_p[-2])
        curr_body = abs(close[-1] - open_p[-1])
        prev_bull = close[-2] > open_p[-2]  # 前一根阳线
        curr_bear = close[-1] < open_p[-1]  # 当前阴线 (看跌吞没)
        # 看涨吞没: 前阴后阳, 阳包阴
        engulfing_bull = (close[-2] < open_p[-2] and close[-1] > open_p[-1]
                          and open_p[-1] <= close[-2] and close[-1] >= open_p[-2])
        patterns['engulfing_bull'] = int(engulfing_bull)

    # 十字星 (Doji)
    doji = body < (high - low) * 0.1
    patterns['doji_last'] = int(doji[-1])

    # 三连阳 / 三连阴
    trends = np.sign(close - open_p)
    patterns['three_up'] = int(len(recent) >= 3 and all(trends[-3:] > 0))
    patterns['three_down'] = int(len(recent) >= 3 and all(trends[-3:] < 0))

    return patterns


def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    计算所有技术指标，返回最新的指标值 (dict)
    """
    if df is None or len(df) < 30:
        return {}

    close = df['close']
    volume = df['volume']

    # RSI
    rsi = compute_rsi(close, CONFIG['rsi_period'])
    rsi_current = rsi.iloc[-1] if not rsi.empty else 50

    # MACD
    dif, dea, hist = compute_macd(close, CONFIG['macd_fast'], CONFIG['macd_slow'], CONFIG['macd_signal'])

    # 布林带
    upper, middle, lower, bandwidth, b_position = compute_bollinger(
        close, CONFIG['boll_period'], CONFIG['boll_std']
    )

    # 均线
    mas = compute_ma(close, CONFIG['ma_periods'])

    # ATR
    atr = compute_atr(df)
    atr_pct = (atr / close * 100).iloc[-1] if not atr.empty else 0

    # 量价
    vol_indicators = compute_volume_indicators(df)

    # 动量
    momentum = compute_momentum(close)

    # 均线排列状态
    latest_mas = {k: v.iloc[-1] for k, v in mas.items() if v is not None and len(v) > 0}
    ma_keys = sorted(latest_mas.keys())
    ma_alignment = 0
    if len(ma_keys) >= 3:
        # 多头排列: MA5 > MA10 > MA20 > MA60
        values = [latest_mas[k] for k in ma_keys]
        ma_alignment = sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])

    # 最新价格与均线的关系
    price_vs_ma = {}
    current_price = close.iloc[-1]
    for k, v in mas.items():
        if v is not None and len(v) > 0 and not np.isnan(v.iloc[-1]):
            price_vs_ma[k] = (current_price - v.iloc[-1]) / v.iloc[-1] * 100

    result = {
        # 最新价格
        'current_price': close.iloc[-1],

        # RSI
        'rsi': rsi_current,
        'rsi_trend': int(rsi.iloc[-1] > rsi.iloc[-5]) if len(rsi) >= 5 else 0,
        'rsi_overbought': int(rsi_current > CONFIG['sell_signals']['rsi_overbought_threshold']),
        'rsi_oversold': int(rsi_current < 30),

        # MACD
        'macd_dif': dif.iloc[-1] if not dif.empty else 0,
        'macd_dea': dea.iloc[-1] if not dea.empty else 0,
        'macd_hist': hist.iloc[-1] if not hist.empty else 0,
        'macd_golden_cross': int(
            len(hist) >= 2 and hist.iloc[-2] <= 0 and hist.iloc[-1] > 0
        ) if not hist.empty else 0,
        'macd_death_cross': int(
            len(hist) >= 2 and hist.iloc[-2] >= 0 and hist.iloc[-1] < 0
        ) if not hist.empty else 0,
        'macd_bullish': int(dif.iloc[-1] > dea.iloc[-1]) if not dif.empty else 0,

        # 布林带
        'boll_b_position': b_position.iloc[-1] if not b_position.empty else 0.5,
        'boll_bandwidth': bandwidth.iloc[-1] if not bandwidth.empty else 0,
        'boll_upper_touch': int(close.iloc[-1] >= (upper.iloc[-1] * 0.99)) if not upper.empty else 0,
        'boll_lower_touch': int(close.iloc[-1] <= (lower.iloc[-1] * 1.01)) if not lower.empty else 0,

        # 均线
        'ma_alignment': ma_alignment,
        'price_above_ma5': int(current_price > latest_mas.get('ma5', current_price)),
        'price_above_ma20': int(current_price > latest_mas.get('ma20', current_price)),
        'price_above_ma60': int(current_price > latest_mas.get('ma60', current_price)),
        **{f'price_vs_{k}': v for k, v in price_vs_ma.items()},

        # 量价
        'vol_ratio': vol_indicators['vol_ratio'].iloc[-1] if not vol_indicators['vol_ratio'].empty else 1,
        'vol_trend': int(vol_indicators['vol_trend'].iloc[-1]) if not vol_indicators['vol_trend'].empty else 0,
        'vol_above_ma5': int(volume.iloc[-1] > vol_indicators['vol_ma5'].iloc[-1]) if not vol_indicators['vol_ma5'].empty else 0,
        'vol_above_ma20': int(volume.iloc[-1] > vol_indicators['vol_ma20'].iloc[-1]) if not vol_indicators['vol_ma20'].empty else 0,

        # 动量
        **{f'mom_{p}': v.iloc[-1] if v is not None and len(v) > 0 else 0 for p, v in momentum.items()},

        # ATR
        'atr_pct': atr_pct,

        # K线形态
        **detect_candlestick_patterns(df),

        # 近期涨跌
        'pct_1d': df['pct_chg'].iloc[-1] if 'pct_chg' in df.columns else 0,
        'pct_5d': df['pct_chg'].tail(5).sum() if len(df) >= 5 else 0,
        'pct_20d': df['pct_chg'].tail(20).sum() if len(df) >= 20 else 0,
    }

    return result


def compute_all_indicators_batch(kline_data: dict) -> dict:
    """
    批量计算所有股票的技术指标
    返回: {code: {indicator_dict}}
    """
    result = {}
    for code, df in kline_data.items():
        if df is not None and len(df) >= 30:
            result[code] = compute_all_indicators(df)
    print(f"[技术] 指标计算完成: {len(result)} 支")
    return result


if __name__ == '__main__':
    # 测试
    from data_fetcher import get_daily_kline
    df = get_daily_kline('000001')
    if df is not None and len(df) > 0:
        ind = compute_all_indicators(df)
        for k, v in sorted(ind.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
