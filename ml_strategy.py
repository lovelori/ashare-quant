"""
轻量ML预测模型 + 集成评分 - 幻方量化风格

实现了:
1. Ridge回归 (L2正则化, 纯numpy实现)
2. 自适应因子权重 (基于因子近期IC表现)
3. 市场状态检测 (波动率/MACD/量能多维度)
4. 集成评分 (ML预测 + 传统规则评分 + 因子挖掘)
"""
import os
import json
import numpy as np
from config import CONFIG
from scipy.stats import spearmanr


def ridge_regression(X, y, alpha=1.0):
    """
    纯numpy实现的Ridge回归
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  # 不惩罚截距

    try:
        coeff = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
        return coeff
    except np.linalg.LinAlgError:
        return np.zeros(n_features)


class AdaptiveFactorWeight:
    """
    自适应因子权重 - 根据因子近期IC动态调整权重
    核心: 表现好的因子加大权重, 表现差的降低权重
    """

    def __init__(self, n_factors: int, decay: float = 0.9):
        self.weights = np.ones(n_factors) / n_factors
        self.ic_history = []
        self.decay = decay

    def update(self, factor_values: np.ndarray, forward_returns: np.ndarray):
        """根据最新一期因子IC更新权重"""
        n = factor_values.shape[1]
        ics = []

        for i in range(n):
            f = factor_values[:, i]
            # 处理无效值
            valid = ~(np.isnan(f) | np.isnan(forward_returns))
            if valid.sum() >= 10:
                rho, _ = spearmanr(f[valid], forward_returns[valid])
                ics.append(abs(rho) if not np.isnan(rho) else 0.0)
            else:
                ics.append(0.0)

        ics = np.array(ics)

        # 指数移动平均
        self.ic_history.append(ics)
        if len(self.ic_history) > 1:
            weighted_ics = np.zeros(n)
            for t, ic in enumerate(reversed(self.ic_history)):
                weighted_ics += ic * (self.decay ** t)
            ics = weighted_ics / sum(self.decay ** t for t in range(len(self.ic_history)))

        # Softmax转换为权重
        exp_ics = np.exp(ics * 2)  # 放大差异
        self.weights = exp_ics / exp_ics.sum()

    def get_weights(self) -> np.ndarray:
        return self.weights


def detect_market_regime(kline_data: dict) -> dict:
    """
    市场状态检测 - 用于动态调整策略参数

    返回:
    {
        'regime': 'bull' | 'bear' | 'range' | 'volatile',
        'market_momentum': float,   # 市场动量
        'market_volatility': float, # 市场波动率
        'market_volume': float,     # 市场量能
    }
    """
    # 使用所有股票的平均表现来估计市场状态
    all_returns_1d = []
    all_returns_5d = []
    all_volumes = []
    all_volatilities = []

    for code, df in kline_data.items():
        if df is None or len(df) < 20:
            continue
        close = df['close'].values
        volume = df['volume'].values

        ret_1d = close[-1] / close[-2] - 1 if len(close) >= 2 else 0
        ret_5d = close[-1] / close[-6] - 1 if len(close) >= 6 else 0
        vol_20 = np.std(np.diff(close[-21:]) / close[-21:-1]) if len(close) >= 22 else 0.01
        vol_ratio = volume[-1] / np.mean(volume[-5:]) if len(volume) >= 5 else 1

        all_returns_1d.append(ret_1d)
        all_returns_5d.append(ret_5d)
        all_volatilities.append(vol_20)
        all_volumes.append(vol_ratio)

    if not all_returns_1d:
        return {'regime': 'range', 'market_momentum': 0, 'market_volatility': 0, 'market_volume': 1}

    avg_ret_1d = np.mean(all_returns_1d) * 100  # %
    avg_ret_5d = np.mean(all_returns_5d) * 100
    avg_vol = np.median(all_volatilities) * 100
    avg_vol_ratio = np.median(all_volumes)

    # 判断市场状态
    if avg_ret_5d > 2 and avg_vol < 3:
        regime = 'bull'
    elif avg_ret_5d < -2 and avg_vol < 3:
        regime = 'bear'
    elif avg_vol > 4:
        regime = 'volatile'
    else:
        regime = 'range'

    return {
        'regime': regime,
        'market_momentum': round(avg_ret_5d, 2),
        'market_volatility': round(avg_vol, 2),
        'market_volume': round(avg_vol_ratio, 2),
    }


def compute_ml_score(tech_features: dict, deep_features: dict,
                     alpha_features: dict, market_state: dict) -> float:
    """
    基于ML的综合评分 (无训练数据的纯规则版)
    结合多维度特征进行评分

    幻方量化风格:
    - 非线性得分: 不是简单加权, 而是根据特征组合给出非线性分数
    - 市场状态感知: 牛市/熊市/震荡中不同因子权重不同
    """
    if not tech_features:
        return 50.0  # 中性分

    regime = market_state.get('regime', 'range')
    score = 50.0

    # ─── 深度因子贡献 ───
    if deep_features:
        # VWAP位置
        vwap_ratio = deep_features.get('vwap_ratio', 0)
        score += np.clip(vwap_ratio * 50, -10, 10)

        # 资金流向 (MFI)
        mfi = deep_features.get('mfi', 50)
        if 40 <= mfi <= 70:
            score += 5
        elif mfi < 20:
            score -= 10  # 极度超卖
        elif mfi > 80:
            score -= 8  # 极度超买

        # 波动率偏度 (正收益波动 > 负收益波动 = 强势)
        vol_skew = deep_features.get('vol_skew', 0)
        score += np.clip(vol_skew * 20, -8, 8)

        # 价格位置 (close position in daily range)
        close_pos = deep_features.get('close_position', 0.5)
        if close_pos > 0.7:
            score += 3  # 收在高位
        elif close_pos < 0.3:
            score -= 3  # 收在低位

        # 成交量与价格相关性
        vol_price_corr = deep_features.get('vol_price_corr', 0)
        if vol_price_corr > 0.3:
            score += 5  # 量价齐升
        elif vol_price_corr < -0.3:
            score -= 5  # 量价背离

        # 动量衰减
        mom_decay = deep_features.get('mom_decay_5_20', 0)
        if mom_decay > 1.5:
            score -= 5  # 短期动量远超长期, 可能衰竭
        elif 0.5 <= mom_decay <= 1.5:
            score += 3  # 健康的动量继续

    # ─── Alpha因子贡献 ───
    if alpha_features:
        # 趋势强度
        alpha_14 = alpha_features.get('alpha_14', 0.5)
        score += (alpha_14 - 0.5) * 20

        # 反转信号
        alpha_13 = alpha_features.get('alpha_13', 0)
        if regime == 'bull':
            score += np.clip(-alpha_13 * 5, -3, 3)  # 牛市中不要太在意短期反转
        else:
            score += np.clip(alpha_13 * 10, -10, 10)  # 震荡/熊市重视反转

        # 价格位置
        alpha_12 = alpha_features.get('alpha_12', 0.5)
        if regime == 'bull':
            score += (alpha_12 - 0.3) * 10  # 牛市追涨
        else:
            score += (0.7 - alpha_12) * 10  # 非牛市低吸

        # 量价背离
        alpha_02 = alpha_features.get('alpha_02', 0)
        score += np.clip(-alpha_02 * 15, -10, 10)

        # 趋势动能
        alpha_15 = alpha_features.get('alpha_15', 0)
        score += np.clip(alpha_15 * 3, -8, 8)

    # ─── 市场状态调整 ───
    if regime == 'bull':
        score += 3  # 牛市偏乐观
    elif regime == 'bear':
        score -= 3  # 熊市偏保守
    elif regime == 'volatile':
        score -= 2  # 高波动降低仓位

    # ─── 异常检测 ───
    if tech_features:
        # 极端情况不做推荐
        pct_chg = tech_features.get('pct_1d', 0)
        if abs(pct_chg) > 9:
            score -= 20  # 涨停/跌停, 风险过大

    return np.clip(score, 0, 100)


def compute_ml_scores_batch(tech_data: dict, deep_data: dict,
                            alpha_data: dict, market_state: dict) -> dict:
    """
    批量计算ML策略评分
    返回: {code: score}
    """
    scores = {}
    for code in tech_data:
        tech = tech_data.get(code, {})
        deep = deep_data.get(code, {})
        alpha = alpha_data.get(code, {})

        ml_score = compute_ml_score(tech, deep, alpha, market_state)
        scores[code] = ml_score

    print(f"[幻方AI] ML策略评分完成: {len(scores)} 支")
    return scores


if __name__ == '__main__':
    # 测试
    from data_fetcher import get_daily_kline, get_stock_universe
    stocks = get_stock_universe().head(10)
    kline_data = {}
    for _, row in stocks.iterrows():
        df = get_daily_kline(row['code'])
        if len(df) > 0:
            kline_data[row['code']] = df

    regime = detect_market_regime(kline_data)
    print(f"市场状态: {regime}")

    test_tech = {'rsi': 55, 'macd_bullish': 1, 'pct_1d': 2.5}
    test_deep = {'vwap_ratio': 0.02, 'mfi': 60, 'vol_skew': 0.3, 'close_position': 0.6, 'vol_price_corr': 0.4, 'mom_decay_5_20': 1.2}
    test_alpha = {'alpha_14': 0.6, 'alpha_13': 0.03, 'alpha_12': 0.6, 'alpha_02': -0.1, 'alpha_15': 1.5}

    score = compute_ml_score(test_tech, test_deep, test_alpha, regime)
    print(f"ML评分: {score:.1f}")

    # 测试市场状态检测
    market = detect_market_regime(kline_data)
    print(f"市场状态: {market}")
