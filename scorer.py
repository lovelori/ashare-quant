"""
多因子综合评分 & 排名系统 (幻方量化增强版)
融合传统多因子 + ML策略 + Alpha因子 + 深度因子
"""
import numpy as np
from config import CONFIG
from ml_strategy import compute_ml_score, detect_market_regime


def compute_score(tech: dict, sentiment: dict, deep: dict = None,
                  alpha: dict = None, ml_score: float = None, market_state: dict = None) -> dict:
    """
    计算个股的综合评分 (幻方量化风格综合评分)

    三层评分体系:
    1. 传统多因子评分 (60%)
    2. ML策略评分 (25% + 自适应)
    3. Alpha因子 + 深度因子评分 (15%)
    """
    if tech is None or len(tech) == 0:
        return {'total_score': 0, 'details': {}}

    weights = CONFIG['weights']
    hf_w = CONFIG['hf_weights']
    hf_config = CONFIG['hf_config']

    # ====================================
    # 第一层: 传统多因子评分
    # ====================================

    # ── 趋势因子 [0, 100] ──
    trend_score = 50
    ma_align = tech.get('ma_alignment', 0)
    trend_score += ma_align * 10
    trend_score += 8 if tech.get('price_above_ma5', 0) else -5
    trend_score += 8 if tech.get('price_above_ma20', 0) else -5
    trend_score += 5 if tech.get('price_above_ma60', 0) else -3
    pct_20d = tech.get('pct_20d', 0)
    trend_score += min(pct_20d * 2, 15)
    trend_score = np.clip(trend_score, 0, 100)

    # ── 动量因子 [0, 100] ──
    momentum_score = 50
    rsi = tech.get('rsi', 50)
    if 40 <= rsi <= 65:
        momentum_score += 10
    elif rsi > 70:
        momentum_score -= 15
    elif rsi > 65:
        momentum_score -= 5
    elif rsi < 30:
        momentum_score -= 10
    if tech.get('macd_bullish', 0):
        momentum_score += 10
    if tech.get('macd_golden_cross', 0):
        momentum_score += 15
    if tech.get('macd_death_cross', 0):
        momentum_score -= 20
    pct_1d = tech.get('pct_1d', 0)
    momentum_score += min(pct_1d * 5, 10)
    momentum_score = np.clip(momentum_score, 0, 100)

    # ── 量价因子 [0, 100] ──
    volume_score = 50
    vol_ratio = tech.get('vol_ratio', 1)
    if 1.0 <= vol_ratio <= 2.5:
        volume_score += 15
    elif vol_ratio > 3.0:
        volume_score -= 5
    elif vol_ratio < 0.5:
        volume_score -= 10
    if tech.get('vol_trend', 0):
        volume_score += 15
    if tech.get('vol_above_ma5', 0):
        volume_score += 5
    if tech.get('vol_above_ma20', 0):
        volume_score += 5
    volume_score = np.clip(volume_score, 0, 100)

    # ── 波动率因子 [0, 100] ──
    vol_score = 50
    atr = tech.get('atr_pct', 0)
    if 1.0 <= atr <= 4.0:
        vol_score += 10
    elif atr > 6.0:
        vol_score -= 10
    bandwidth = tech.get('boll_bandwidth', 0)
    if bandwidth > 5 and bandwidth < 30:
        vol_score += 5
    b_pos = tech.get('boll_b_position', 0.5)
    if 0.2 <= b_pos <= 0.8:
        vol_score += 10
    elif b_pos > 1.0:
        vol_score -= 5
    vol_score = np.clip(vol_score, 0, 100)

    # ── 舆情因子 [0, 100] ──
    if sentiment:
        sentiment_score = 50 + sentiment.get('sentiment_score', 0) * 30
        sentiment_score += min(sentiment.get('positive_count', 0) * 3, 10)
        sentiment_score -= min(sentiment.get('negative_count', 0) * 5, 15)
        sentiment_score = np.clip(sentiment_score, 0, 100)
    else:
        sentiment_score = 50

    # ── 基本面因子 [0, 100] ──
    fundamental_score = 50
    turnover = tech.get('turnover_latest', 0)
    if 1 <= turnover <= 5:
        fundamental_score += 10
    elif turnover > 10:
        fundamental_score -= 10
    fundamental_score = np.clip(fundamental_score, 0, 100)

    # ── K线形态因子 [0, 100] ──
    pattern_score = 50
    if tech.get('engulfing_bull', 0):
        pattern_score += 20
    if tech.get('hammer_last', 0):
        pattern_score += 10
    if tech.get('three_up', 0):
        pattern_score += 15
    if tech.get('three_down', 0):
        pattern_score -= 15
    pattern_score = np.clip(pattern_score, 0, 100)

    # ── 传统综合评分 ──
    traditional_score = (
        trend_score * weights['trend'] +
        momentum_score * weights['momentum'] +
        volume_score * weights['volume'] +
        vol_score * weights['volatility'] +
        sentiment_score * weights['sentiment'] +
        fundamental_score * weights['fundamental'] +
        pattern_score * weights['pattern']
    )

    # ====================================
    # 第二层: ML策略评分 (幻方风格)
    # ====================================
    ml_final_score = 50.0
    if hf_config['enable_ml'] and ml_score is not None:
        ml_final_score = ml_score
    else:
        # 如果没传入ML分数, 用规则计算
        ml_final_score = compute_ml_score(tech, deep or {}, alpha or {},
                                          market_state or {'regime': 'range'})

    # ====================================
    # 第三层: Alpha因子 + 深度因子
    # ====================================
    alpha_contribution = 0.0
    if hf_config['enable_alpha_factors'] and alpha:
        # 取Alpha因子的中位数作为贡献
        alpha_vals = [v for v in alpha.values() if not np.isnan(v) and not np.isinf(v)]
        if alpha_vals:
            alpha_contribution = float(np.clip(np.median(alpha_vals) * 5, -15, 15))

    deep_contribution = 0.0
    if hf_config['enable_deep_features'] and deep:
        # VWAP偏离度
        vwap_r = deep.get('vwap_ratio', 0)
        deep_contribution += np.clip(vwap_r * 30, -5, 5)
        # 波动率偏度
        vol_sk = deep.get('vol_skew', 0)
        deep_contribution += np.clip(vol_sk * 15, -5, 5)

    alpha_score = 50 + alpha_contribution
    deep_score = 50 + deep_contribution

    # ====================================
    # 综合评分: 三层融合 (归一化)
    # ====================================
    # 传统评分归一化到0-100
    traditional_weights_sum = sum(weights.values())
    if traditional_weights_sum > 0:
        traditional_score = traditional_score / traditional_weights_sum

    # AI部分归一化到0-100
    hf_weights_sum = sum(hf_w.values())
    if hf_weights_sum > 0:
        hf_part_raw = (
            ml_final_score * hf_w['ml_score'] +
            alpha_score * hf_w['alpha_score'] +
            deep_score * hf_w['deep_score']
        )
        hf_part_normalized = hf_part_raw / hf_weights_sum
    else:
        hf_part_normalized = 50

    # 融合权重
    blend_ratio = hf_weights_sum  # 0.40
    total = traditional_score * (1 - blend_ratio) + hf_part_normalized * blend_ratio

    details = {
        'trend': round(trend_score, 1),
        'momentum': round(momentum_score, 1),
        'volume': round(volume_score, 1),
        'volatility': round(vol_score, 1),
        'sentiment': round(sentiment_score, 1),
        'fundamental': round(fundamental_score, 1),
        'pattern': round(pattern_score, 1),
    }

    # 幻方AI评分详情 (仅在可用时显示)
    hf_details = {}
    if hf_config['enable_ml']:
        hf_details['ml_score'] = round(ml_final_score, 1)
    if hf_config['enable_alpha_factors']:
        hf_details['alpha_score'] = round(alpha_score, 1)
    if hf_config['enable_deep_features']:
        hf_details['deep_score'] = round(deep_score, 1)

    return {
        'total_score': round(total, 2),
        'details': details,
        'hf_details': hf_details,
        'current_price': tech.get('current_price', None),
        'pct_1d': tech.get('pct_1d', 0),
        'pct_5d': tech.get('pct_5d', 0),
        'rsi': tech.get('rsi', 50),
        'vol_ratio': tech.get('vol_ratio', 1),
    }


def rank_stocks(scores: dict) -> list:
    """对所有股票评分进行排名"""
    ranked = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    return ranked


def get_top_picks(scored_stocks: dict, stock_names: dict, top_n: int = None) -> list:
    """获取评分最高的N支股票"""
    if top_n is None:
        top_n = CONFIG['top_n']

    ranked = rank_stocks(scored_stocks)
    picks = []

    for code, score in ranked[:top_n]:
        hf_details = score.get('hf_details', {})
        picks.append({
            'code': code,
            'name': stock_names.get(code, code),
            'total_score': score['total_score'],
            'details': score['details'],
            'hf_details': hf_details,
            'pct_1d': score['pct_1d'],
            'pct_5d': score['pct_5d'],
            'rsi': score['rsi'],
            'vol_ratio': score['vol_ratio'],
            'sentiment_score': score['details'].get('sentiment', 50),
            'current_price': score.get('current_price', None),
            'buy_price': score.get('current_price', None),
        })

    return picks


def compute_all_scores(tech_data: dict, sentiment_data: dict,
                       stock_names: dict, deep_data: dict = None,
                       alpha_data: dict = None, ml_scores: dict = None,
                       market_state: dict = None) -> dict:
    """
    批量计算所有股票的综合评分 (幻方增强版)
    """
    scores = {}
    for code in tech_data:
        tech = tech_data[code]
        sentiment = sentiment_data.get(code, None)
        deep = deep_data.get(code, {}) if deep_data else None
        alpha = alpha_data.get(code, {}) if alpha_data else None
        ml_score = ml_scores.get(code, None) if ml_scores else None

        score = compute_score(tech, sentiment, deep, alpha, ml_score, market_state)
        scores[code] = score

    print(f"[评分] 幻方综合评分完成: {len(scores)} 支")
    return scores


if __name__ == '__main__':
    # 测试
    test_tech = {
        'rsi': 55, 'macd_bullish': 1, 'macd_golden_cross': 0, 'macd_death_cross': 0,
        'ma_alignment': 2, 'price_above_ma5': 1, 'price_above_ma20': 1, 'price_above_ma60': 1,
        'pct_1d': 1.5, 'pct_5d': 3.0, 'pct_20d': 5.0,
        'vol_ratio': 1.5, 'vol_trend': 1, 'vol_above_ma5': 1, 'vol_above_ma20': 1,
        'atr_pct': 2.5, 'boll_bandwidth': 10, 'boll_b_position': 0.6,
        'boll_upper_touch': 0, 'boll_lower_touch': 0,
        'engulfing_bull': 1, 'hammer_last': 0, 'three_up': 1, 'three_down': 0,
    }
    test_sentiment = {'sentiment_score': 0.6, 'news_count': 5, 'positive_count': 3, 'negative_count': 1, 'neutral_count': 1, 'sentiment_label': 'positive'}
    test_deep = {'vwap_ratio': 0.02, 'mfi': 60, 'vol_skew': 0.3, 'close_position': 0.6, 'vol_price_corr': 0.4, 'mom_decay_5_20': 1.2}
    test_alpha = {'alpha_14': 0.6, 'alpha_13': 0.03, 'alpha_12': 0.6, 'alpha_02': -0.1, 'alpha_15': 1.5}
    test_market = {'regime': 'range', 'market_momentum': 0.5, 'market_volatility': 2.0, 'market_volume': 1.2}

    score = compute_score(test_tech, test_sentiment, test_deep, test_alpha, None, test_market)
    print(f"幻方综合评分: {score['total_score']:.1f}")
    print(f"传统因子: {score['details']}")
    print(f"AI因子: {score['hf_details']}")
