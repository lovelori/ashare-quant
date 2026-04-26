"""
舆情情绪分析模块
基于新闻标题和内容进行情感分析
"""
import re
import numpy as np
import pandas as pd
from collections import Counter
from config import CONFIG

# 中文情感词典 (金融领域)
BULLISH_WORDS = set([
    # 强烈积极
    '涨停', '大涨', '暴涨', '飙升', '狂飙', '爆发', '井喷', '引爆',
    '强势', '领涨', '上攻', '突破', '创新高', '新高', '放量突破',
    # 业绩利好
    '净利润增长', '营收增长', '业绩大增', '扭亏为盈', '超预期', '预增',
    '分红', '送转', '回购', '增持', '业绩预增', '大幅增长', '同比增长',
    # 政策/行业利好
    '利好', '政策支持', '重点扶持', '战略合作', '中标', '重大合同',
    '获批', '核准', '放行', '准入', '降价', '减税', '补贴',
    # 机构观点
    '买入评级', '推荐', '增持评级', '看好', '目标价', '上调评级',
    '主力流入', '北向资金', '外资买入', '资金流入',
    # 技术面
    '金叉', '底背离', '双底', 'W底', '头肩底', '突破颈线',
    '缩量调整', '企稳回升', '触底反弹',
])

BEARISH_WORDS = set([
    # 强烈消极
    '跌停', '大跌', '暴跌', '跳水', '闪崩', '崩盘', '重挫', '下挫',
    '弱势', '领跌', '下探', '破位', '创新低', '新低', '跌破',
    # 业绩利空
    '净利润下降', '营收下降', '亏损', '预亏', '业绩下滑', '不及预期',
    '商誉减值', '计提', '债务违约', 'st', '退市', '暂停上市',
    # 政策/行业利空
    '利空', '监管', '处罚', '罚款', '立案调查', '问询', '警示',
    '减持', '套现', '解禁', '配股', '增发',
    # 风险提示
    '风险提示', '资金链断裂', '流动性危机', '信用评级下调',
    '下调评级', '卖出评级', '减持评级',
    # 技术面
    '死叉', '顶背离', 'M头', '双头', '头肩顶', '破颈线',
    '放量下跌', '缩量上涨', '天量见天价',
])

# 强度修饰词
INTENSIFIERS = {
    '大幅': 1.5, '明显': 1.3, '持续': 1.2, '加速': 1.4,
    '严重': 1.5, '强烈': 1.4, '显著': 1.3,
}

# 否定词 (翻转情绪)
NEGATORS = {'不', '未', '无', '没有', '并非', '并非', '尚未', '并非'}


def analyze_news_sentiment(news_df: pd.DataFrame) -> dict:
    """
    对个股的新闻进行情感分析
    返回: {sentiment_score, news_count, positive_count, negative_count, ...}
    """
    if news_df is None or len(news_df) == 0:
        return {
            'sentiment_score': 0,
            'news_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'sentiment_label': 'neutral',
            'top_positive_news': [],
            'top_negative_news': [],
        }

    scores = []
    news_items = []

    for _, row in news_df.iterrows():
        title = str(row.get('title', ''))
        content = str(row.get('content', ''))
        combined = title + ' ' + content[:500]  # 只取内容前500字

        score = _score_text(combined)
        scores.append(score)

        news_items.append({
            'title': title[:80],
            'score': score,
            'source': row.get('source', ''),
            'time': str(row.get('pub_time', ''))[:19],
        })

    # 汇总
    sentiment_score = np.mean(scores) if scores else 0
    positive = sum(1 for s in scores if s > 0.15)
    negative = sum(1 for s in scores if s < -0.15)
    neutral = len(scores) - positive - negative

    # 取最积极和最消极的新闻
    sorted_items = sorted(news_items, key=lambda x: x['score'], reverse=True)

    return {
        'sentiment_score': round(sentiment_score, 4),
        'news_count': len(scores),
        'positive_count': positive,
        'negative_count': negative,
        'neutral_count': neutral,
        'sentiment_label': 'positive' if sentiment_score > 0.15 else ('negative' if sentiment_score < -0.15 else 'neutral'),
        'top_positive_news': [n['title'] for n in sorted_items[:3] if n['score'] > 0],
        'top_negative_news': [n['title'] for n in sorted_items[-3:] if n['score'] < 0],
    }


def _score_text(text: str) -> float:
    """
    对文本进行情感评分
    返回 [-1, 1] 的分数
    """
    if not text:
        return 0

    # 分词 (简单按词切割中文)
    words = _tokenize(text)
    word_set = set(words)

    bullish_hits = word_set & BULLISH_WORDS
    bearish_hits = word_set & BEARISH_WORDS

    # 计算基础分数
    bullish_score = sum(1.0 for w in bullish_hits)
    bearish_score = sum(1.0 for w in bearish_hits)

    # 应用强度修饰
    for i, w in enumerate(words):
        if w in INTENSIFIERS:
            factor = INTENSIFIERS[w]
            if i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in BULLISH_WORDS:
                    bullish_score += (factor - 1)
                elif next_word in BEARISH_WORDS:
                    bearish_score += (factor - 1)

    # 应用否定翻转
    for i, w in enumerate(words):
        if w in NEGATORS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in BULLISH_WORDS:
                bearish_score += 1.0
                bullish_score -= 1.0
            elif next_word in BEARISH_WORDS:
                bullish_score += 1.0
                bearish_score -= 1.0

    # 归一化到 [-1, 1]
    total = bullish_score + bearish_score
    if total == 0:
        return 0

    score = (bullish_score - bearish_score) / max(total, 1)

    # 非线性映射，增强区分度
    score = np.tanh(score * 2)

    return score


def _tokenize(text: str) -> list:
    """
    简单中文分词 (基于正则)
    """
    # 去除标点和空格
    text = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    # 返回2-6个字的连续字符作为候选词
    tokens = []
    for match in re.finditer(r'[\u4e00-\u9fff]{2,6}', text):
        tokens.append(match.group())

    # 也加入单个有意义的字
    for word in text.split():
        if word in BULLISH_WORDS or word in BEARISH_WORDS:
            tokens.append(word)

    return tokens


def compute_sentiment_batch(news_data: dict) -> dict:
    """
    批量计算所有股票的舆情分数
    返回: {code: sentiment_dict}
    """
    result = {}
    for code, news_df in news_data.items():
        result[code] = analyze_news_sentiment(news_df)
    print(f"[舆情] 情感分析完成: {sum(1 for v in result.values() if v['news_count'] > 0)} 支有舆情数据")
    return result


if __name__ == '__main__':
    # 测试
    test_news = [
        {'title': '业绩超预期，净利润大幅增长，机构看好后市'},
        {'title': '公司股价涨停，主力资金大幅流入，强势突破压力位'},
        {'title': '公司发布减持公告，股价大跌，监管问询'},
        {'title': '公司召开股东大会，讨论年度报告'},
    ]
    from data_fetcher import get_stock_news
    news_df = get_stock_news('000001')
    if news_df is not None and len(news_df) > 0:
        result = analyze_news_sentiment(news_df)
        print(f"平安银行舆情: {result['sentiment_label']} (score={result['sentiment_score']:.3f})")
        print(f"  新闻数: {result['news_count']}, 积极: {result['positive_count']}, 消极: {result['negative_count']}")
        if result['top_positive_news']:
            print(f"  积极头条: {result['top_positive_news'][0]}")
        if result['top_negative_news']:
            print(f"  消极头条: {result['top_negative_news'][0]}")
