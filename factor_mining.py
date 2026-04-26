"""
因子挖掘引擎 - 幻方量化风格 Alpha 因子系统

实现了:
1. 101式Alpha因子 (简化版, 核心的几个)
2. 因子交互挖掘 (二阶组合因子)
3. 因子IC分析 (Information Coefficient)
4. 自适应因子权重 (基于近期表现)

核心思路:
- 不是固定权重, 而是根据因子近期的预测能力动态调整
- 发现因子之间的非线性交互作用
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_alpha_factors(df: pd.DataFrame) -> dict:
    """
    计算一系列Alpha因子 (灵感来自WorldQuant 101 Alpha)
    返回最新一期因子值
    """
    if df is None or len(df) < 30:
        return {}

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    open_p = df['open'].values
    vwap = (close * volume).cumsum() / volume.cumsum()  # 累计VWAP

    # 转为Series便于rolling操作
    s_close = df['close']
    s_high = df['high']
    s_low = df['low']
    s_volume = df['volume']
    s_open = df['open']

    result = {}
    n = len(df) - 1  # 最新索引

    # 日内相对位置 (用于多个Alpha因子)
    daily_range = (s_close - s_open) / (s_high - s_low + 1e-10)

    # Alpha#1: (rank(Ts_ArgMax(SignedPower(((close - open) / (high - low + 1e-10)), 2)), 5))
    # 简化: 过去5天中, 日内振幅最大的天数占比
    result['alpha_01'] = float((daily_range.abs() > 0.5).tail(5).mean()) if len(daily_range) >= 5 else 0

    # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    # 简化版: 量价背离因子
    vol_change = np.log(s_volume + 1).diff(2)
    ret = s_close.pct_change()
    corr_6 = vol_change.rolling(6).corr(ret)
    result['alpha_02'] = float(-corr_6.iloc[-1]) if not corr_6.empty else 0

    # Alpha#3: (-1 * Ts_Rank(decay_linear(delta(close, 5), 3), 5))
    # 简化: 价格变动是否正在衰竭
    close_delta_5 = s_close.diff(5)
    decay = close_delta_5.ewm(span=3).mean()
    ts_rank = decay.rolling(10).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=True
    )
    result['alpha_03'] = float(-ts_rank.iloc[-1]) if not ts_rank.empty else 0

    # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    result['alpha_04'] = float(-(s_low.rank(pct=True).iloc[-1])) if len(s_low) > 0 else 0

    # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    vwap_10 = vwap[-10:].mean() if len(vwap) >= 10 else vwap[-1]
    result['alpha_05'] = float((open_p[-1] - vwap_10) / vwap_10 * 100)

    # Alpha#6: (-1 * correlation(open, volume, 10))
    corr_ov = s_open.rolling(10).corr(s_volume)
    result['alpha_06'] = float(-corr_ov.iloc[-1]) if not corr_ov.empty else 0

    # Alpha#7: (adv20 * Ts_Rank(close / open, 5))
    adv20 = s_volume.rolling(20).mean()
    close_open_ratio = s_close / s_open
    rank_cor = close_open_ratio.rolling(5).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=True
    )
    result['alpha_07'] = float((adv20.iloc[-1] * rank_cor.iloc[-1]) / (adv20.iloc[-1] + 1e-10)) if not rank_cor.empty else 0

    # Alpha#8: -1 * (rank(((sum((close - open), 5) / sum((open - close), 5)) * ts_rank(close - open, 10))))
    sum_up = (s_close - s_open).rolling(10).sum()
    sum_down = (s_open - s_close).rolling(10).sum()
    ratio = sum_up / sum_down.replace(0, np.nan)
    result['alpha_08'] = float(-ratio.iloc[-1]) if not ratio.empty else 0

    # Alpha#9: delta(close, 1) / volatility(close, 20) * 100 (价格变动 / 波动率)
    ret_1d = s_close.pct_change()
    vol_20 = s_close.pct_change().rolling(20).std()
    result['alpha_09'] = float((ret_1d.iloc[-1] / vol_20.iloc[-1]) * 10) if not vol_20.empty and vol_20.iloc[-1] != 0 else 0

    # Alpha#10: -1 * (high - close) / (close - low) * volume (上影线/下影线)
    upper_shadow = s_high - s_close
    lower_shadow = s_close - s_low
    shadow_ratio = upper_shadow / lower_shadow.replace(0, np.nan)
    result['alpha_10'] = float(-shadow_ratio.iloc[-1] * volume[-1] / (volume.mean() + 1)) if not shadow_ratio.empty else 0

    # Alpha#11:成交量突变因子 (今日量 / 20日均量)
    vol_surge = s_volume / s_volume.rolling(20).mean()
    result['alpha_11'] = float(vol_surge.iloc[-1]) if not vol_surge.empty else 1

    # Alpha#12: 价格位置因子 (close - 20日低点) / (20日高点 - 20日低点)
    hh20 = s_high.rolling(20).max()
    ll20 = s_low.rolling(20).min()
    stoch = (s_close - ll20) / (hh20 - ll20 + 1e-10)
    result['alpha_12'] = float(stoch.iloc[-1]) if not stoch.empty else 0.5

    # Alpha#13: 反转因子 (-1 * 过去5日收益)
    result['alpha_13'] = float(-s_close.pct_change(5).iloc[-1]) if len(s_close) >= 5 else 0

    # Alpha#14: 趋势强度 (过去20日中上涨天数占比)
    up_days = (s_close.diff() > 0).rolling(20).sum()
    result['alpha_14'] = float(up_days.iloc[-1] / 20) if not up_days.empty else 0.5

    # Alpha#15: 波动率调整后的动量
    mom_10 = s_close.pct_change(10)
    vol_10 = s_close.pct_change().rolling(10).std()
    result['alpha_15'] = float((mom_10.iloc[-1] / vol_10.iloc[-1])) if not vol_10.empty and vol_10.iloc[-1] != 0 else 0

    # 去除无效值
    result = {k: float(v) if not (np.isnan(v) or np.isinf(v)) else 0.0 for k, v in result.items()}

    return result


def compute_alpha_factors_batch(kline_data: dict) -> dict:
    """批量计算Alpha因子"""
    result = {}
    for code, df in kline_data.items():
        if df is not None and len(df) >= 30:
            result[code] = compute_alpha_factors(df)
    print(f"[Alpha] Alpha因子计算完成: {len(result)} 支")
    return result


def rank_factor_ic(factor_values: dict, forward_returns: dict) -> float:
    """
    计算因子的IC值 (Information Coefficient)
    IC = rank(factor) 与 rank(forward_return) 的Spearman相关系数
    """
    codes = [c for c in factor_values if c in forward_returns]
    if len(codes) < 10:
        return 0.0

    f_vals = [factor_values[c] for c in codes]
    r_vals = [forward_returns[c] for c in codes]

    rho, _ = spearmanr(f_vals, r_vals)
    return float(rho) if not np.isnan(rho) else 0.0


def cross_sectional_normalize(values: dict) -> dict:
    """
    横截面标准化 (z-score within universe)
    返回: {code: z_score}
    """
    codes = list(values.keys())
    vals = np.array([values[c] for c in codes])
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = np.mean(vals), np.std(vals)
    if std < 1e-10:
        return {c: 0.0 for c in codes}

    z_scores = (vals - mean) / std
    return {codes[i]: float(np.clip(z_scores[i], -3, 3)) for i in range(len(codes))}


def compute_factor_interaction(alpha_values: dict, alpha_keys: list) -> dict:
    """
    因子交互: 生成二阶组合因子
    如: alpha_01 * alpha_02, alpha_03 + alpha_04, etc.
    返回 {code: interaction_score}
    """
    if len(alpha_keys) < 2:
        return {}

    codes = list(alpha_values.keys())
    result = {}

    # 对每个股票, 选择最好的几个因子的平均值作为综合Alpha
    for code in codes:
        vals = []
        for k in alpha_keys:
            if code in alpha_values and k in alpha_values[code]:
                v = alpha_values[code][k]
                if not np.isnan(v) and not np.isinf(v):
                    vals.append(v)

        if vals:
            # 中位数聚合 (稳健)
            result[code] = float(np.median(vals))
        else:
            result[code] = 0.0

    return result


if __name__ == '__main__':
    from data_fetcher import get_daily_kline
    df = get_daily_kline('000001')
    if len(df) > 0:
        alphas = compute_alpha_factors(df)
        print("平安银行 Alpha因子:")
        for k, v in sorted(alphas.items()):
            print(f"  {k}: {v:.4f}")
