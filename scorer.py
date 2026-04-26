"""
多因子综合评分 & 排名系统
"""
import numpy as np
from config import CONFIG


def compute_score(tech: dict, sentiment: dict, info: dict = None) -> dict:
    """
    计算个股的综合评分

    因子说明:
    - trend: 趋势因子 (均线多头排列, 价格在均线上方)
    - momentum: 动量因子 (RSI, MACD金叉, 近期涨幅)
    - volume: 量价因子 (成交量放大, 量价配合)
    - volatility: 波动率因子 (ATR适中, 布林带宽扩张)
    - sentiment: 舆情情绪因子
    - fundamental: 基本面因子 (换手率健康)
    - pattern: K线形态因子
    """
    if tech is None or len(tech) == 0:
        return {'total_score': 0, 'details': {}}

    weights = CONFIG['weights']

    # ── 趋势因子 [0, 100] ──
    trend_score = 50
    # 均线多头排列 (ma_alignment: 0~3, 每项约10分)
    ma_align = tech.get('ma_alignment', 0)
    trend_score += ma_align * 10
    # 价格在均线上方加分
    trend_score += 8 if tech.get('price_above_ma5', 0) else -5
    trend_score += 8 if tech.get('price_above_ma20', 0) else -5
    trend_score += 5 if tech.get('price_above_ma60', 0) else -3
    # 20日涨幅为正加分
    pct_20d = tech.get('pct_20d', 0)
    trend_score += min(pct_20d * 2, 15)
    trend_score = np.clip(trend_score, 0, 100)

    # ── 动量因子 [0, 100] ──
    momentum_score = 50
    rsi = tech.get('rsi', 50)
    # RSI在40-65之间最健康
    if 40 <= rsi <= 65:
        momentum_score += 10
    elif rsi > 70:
        momentum_score -= 15  # 超买风险，加大惩罚
    elif rsi > 65:
        momentum_score -= 5  # 偏高预警
    elif rsi < 30:
        momentum_score -= 10  # 弱势
    # MACD多头加分
    if tech.get('macd_bullish', 0):
        momentum_score += 10
    if tech.get('macd_golden_cross', 0):
        momentum_score += 15
    if tech.get('macd_death_cross', 0):
        momentum_score -= 20
    # 1日涨幅
    pct_1d = tech.get('pct_1d', 0)
    momentum_score += min(pct_1d * 5, 10)
    momentum_score = np.clip(momentum_score, 0, 100)

    # ── 量价因子 [0, 100] ──
    volume_score = 50
    vol_ratio = tech.get('vol_ratio', 1)
    # 量比在1.0-2.0之间为健康放量
    if 1.0 <= vol_ratio <= 2.5:
        volume_score += 15
    elif vol_ratio > 3.0:
        volume_score -= 5  # 放量过大可能是出货
    elif vol_ratio < 0.5:
        volume_score -= 10  # 缩量
    # 量价配合 (上涨放量)
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
    # ATR适中 (1-4%) 为健康
    if 1.0 <= atr <= 4.0:
        vol_score += 10
    elif atr > 6.0:
        vol_score -= 10  # 波动过大
    # 布林带宽扩张表示趋势启动
    bandwidth = tech.get('boll_bandwidth', 0)
    if bandwidth > 5 and bandwidth < 30:
        vol_score += 5
    # %B位置 (0.2-0.8为健康区间)
    b_pos = tech.get('boll_b_position', 0.5)
    if 0.2 <= b_pos <= 0.8:
        vol_score += 10
    elif b_pos > 1.0:
        vol_score -= 5  # 突破上轨, 可能超买
    vol_score = np.clip(vol_score, 0, 100)

    # ── 舆情因子 [0, 100] ──
    if sentiment:
        sentiment_score = 50 + sentiment.get('sentiment_score', 0) * 30
        # 积极新闻数量加分
        sentiment_score += min(sentiment.get('positive_count', 0) * 3, 10)
        sentiment_score -= min(sentiment.get('negative_count', 0) * 5, 15)
        sentiment_score = np.clip(sentiment_score, 0, 100)
    else:
        sentiment_score = 50  # 无舆情数据给中性分

    # ── 基本面因子 [0, 100] ──
    fundamental_score = 50
    # 换手率 (1-5%为健康)
    turnover = tech.get('turnover_latest', 0)
    if 1 <= turnover <= 5:
        fundamental_score += 10
    elif turnover > 10:
        fundamental_score -= 10
    # 市净率等 (从info获取)
    if info:
        pass  # 未来可以扩展
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

    # ── 综合评分 ──
    total = (
        trend_score * weights['trend'] +
        momentum_score * weights['momentum'] +
        volume_score * weights['volume'] +
        vol_score * weights['volatility'] +
        sentiment_score * weights['sentiment'] +
        fundamental_score * weights['fundamental'] +
        pattern_score * weights['pattern']
    )

    details = {
        'trend': round(trend_score, 1),
        'momentum': round(momentum_score, 1),
        'volume': round(volume_score, 1),
        'volatility': round(vol_score, 1),
        'sentiment': round(sentiment_score, 1),
        'fundamental': round(fundamental_score, 1),
        'pattern': round(pattern_score, 1),
    }

    return {
        'total_score': round(total, 2),
        'details': details,
        'current_price': tech.get('current_price', None),
        'pct_1d': tech.get('pct_1d', 0),
        'pct_5d': tech.get('pct_5d', 0),
        'rsi': tech.get('rsi', 50),
        'vol_ratio': tech.get('vol_ratio', 1),
    }


def rank_stocks(scores: dict) -> list:
    """
    对所有股票评分进行排名
    返回: [(code, score_dict), ...] 按总分降序
    """
    ranked = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    return ranked


def get_top_picks(scored_stocks: dict, stock_names: dict, top_n: int = None) -> list:
    """
    获取评分最高的N支股票
    返回: [{'code': code, 'name': name, 'score': ..., ...}, ...]
    """
    if top_n is None:
        top_n = CONFIG['top_n']

    ranked = rank_stocks(scored_stocks)
    picks = []

    for code, score in ranked[:top_n]:
        picks.append({
            'code': code,
            'name': stock_names.get(code, code),
            'total_score': score['total_score'],
            'details': score['details'],
            'pct_1d': score['pct_1d'],
            'pct_5d': score['pct_5d'],
            'rsi': score['rsi'],
            'vol_ratio': score['vol_ratio'],
            'sentiment_score': score['details'].get('sentiment', 50),
            'current_price': score.get('current_price', None),
            'buy_price': score.get('current_price', None),
        })

    return picks


def compute_all_scores(tech_data: dict, sentiment_data: dict, stock_names: dict) -> dict:
    """
    批量计算所有股票的综合评分
    返回: {code: score_dict}
    """
    scores = {}
    for code in tech_data:
        tech = tech_data[code]
        sentiment = sentiment_data.get(code, None)
        info = None

        score = compute_score(tech, sentiment, info)
        scores[code] = score

    print(f"[评分] 已完成 {len(scores)} 支股票评分")
    return scores


if __name__ == '__main__':
    # 测试
    test_tech = {
        'rsi': 55,
        'macd_bullish': 1,
        'macd_golden_cross': 0,
        'macd_death_cross': 0,
        'ma_alignment': 2,
        'price_above_ma5': 1,
        'price_above_ma20': 1,
        'price_above_ma60': 1,
        'pct_1d': 1.5,
        'pct_5d': 3.0,
        'pct_20d': 5.0,
        'vol_ratio': 1.5,
        'vol_trend': 1,
        'vol_above_ma5': 1,
        'vol_above_ma20': 1,
        'atr_pct': 2.5,
        'boll_bandwidth': 10,
        'boll_b_position': 0.6,
        'boll_upper_touch': 0,
        'boll_lower_touch': 0,
        'engulfing_bull': 1,
        'hammer_last': 0,
        'three_up': 1,
        'three_down': 0,
    }
    test_sentiment = {
        'sentiment_score': 0.6,
        'news_count': 5,
        'positive_count': 3,
        'negative_count': 1,
        'neutral_count': 1,
        'sentiment_label': 'positive',
    }

    score = compute_score(test_tech, test_sentiment)
    print(f"总评分: {score['total_score']}")
    for k, v in score['details'].items():
        print(f"  {k}: {v}")
