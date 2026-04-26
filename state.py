"""
状态管理模块 - 存储和读取历史推荐记录
"""
import os
import json
from datetime import datetime
from config import CONFIG


STATE_FILE = os.path.join(CONFIG['data_dir'], 'recommendations.json')


def _get_default_state() -> dict:
    return {
        'last_update': '',
        'history': [],       # 所有历史推荐记录
        'daily_records': {},  # {date: [{code, name, score, price}, ...]}
    }


def load_state() -> dict:
    """读取历史推荐状态"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[状态] 读取失败: {e}, 使用默认状态")
            return _get_default_state()
    return _get_default_state()


def save_state(state: dict):
    """保存推荐状态"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"[状态] 已保存到 {STATE_FILE}")


def record_recommendation(picks: list):
    """
    记录今日推荐
    picks: [{'code': ..., 'name': ..., 'total_score': ..., ...}, ...]
    """
    state = load_state()
    today = datetime.now().strftime('%Y-%m-%d')

    # 创建今日记录
    today_picks = []
    for pick in picks:
        today_picks.append({
            'code': pick['code'],
            'name': pick['name'],
            'score': pick['total_score'],
            'price': None,  # 会在update_prices里更新
            'date': today,
        })

    # 尝试获取推荐时的价格 (从提供的picks中取)
    for pick, raw in zip(today_picks, picks):
        if 'current_price' in raw and raw['current_price']:
            pick['price'] = float(raw['current_price'])
        else:
            pick['price'] = None
    # 兼容旧字段: price_at_recommend = price
    for p in today_picks:
        p['price_at_recommend'] = p['price']

    # 更新状态
    state['daily_records'][today] = today_picks
    state['history'].extend(today_picks)
    state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 只保留最近90天的历史
    if len(state['history']) > 90 * 6:
        state['history'] = state['history'][-90 * 6:]

    save_state(state)
    return state


def get_recent_recommendations(days: int = 30) -> list:
    """
    获取最近N天推荐的股票列表 (用于卖出信号检查)
    """
    state = load_state()
    history = state.get('history', [])

    # 去重取最新
    seen = set()
    recent = []
    for item in reversed(history):
        code = item.get('code', '')
        if code not in seen:
            seen.add(code)
            recent.append(item)

    return recent


def clear_history():
    """清空历史记录"""
    save_state(_get_default_state())


if __name__ == '__main__':
    # 测试
    state = load_state()
    print(f"最后更新: {state.get('last_update', 'N/A')}")
    print(f"历史记录: {len(state.get('history', []))} 条")
    print(f"历史天数: {len(state.get('daily_records', {}))} 天")
