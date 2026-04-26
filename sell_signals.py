"""
卖出信号监测模块
监控历史推荐的股票是否出现卖出信号
"""
import numpy as np
import pandas as pd
from config import CONFIG


def check_sell_signals(code: str, name: str, kline_df: pd.DataFrame,
                       tech: dict, buy_price: float = None) -> dict:
    """
    检查个股是否出现卖出信号

    信号类型:
    1. 均线破位 (跌破MA60)
    2. RSI超买后回落
    3. MACD死叉
    4. 量价背离 (价格上涨但成交量萎缩)
    5. 止损/止盈触发
    6. 布林带上轨突破后回落
    7. K线见顶形态 (吊人线、看跌吞没)
    """
    if kline_df is None or len(kline_df) < 20 or tech is None:
        return {
            'code': code,
            'name': name,
            'has_sell_signal': False,
            'signals': [],
            'signal_count': 0,
            'signal_strength': 0,
        }

    close = kline_df['close'].values
    signals = []
    strength = 0

    s = CONFIG['sell_signals']

    # ─── 信号1: 均线破位 ───
    price_vs_ma60 = tech.get('price_vs_ma60', 0)
    if price_vs_ma60 is not None and price_vs_ma60 < s['ma_break_pct'] * 100:
        signals.append({
            'type': 'ma_break',
            'label': '🔴 均线破位',
            'detail': f"价格跌破MA60 {price_vs_ma60:.1f}%",
            'severity': 'high',
        })
        strength += 3

    # ─── 信号2: RSI超买后回落 ───
    rsi = tech.get('rsi', 50)
    if rsi > s['rsi_overbought_threshold']:
        # 检查RSI是否在回落
        if len(kline_df) >= 5:
            rsi_series = _calc_rsi_series(kline_df['close'].values, 14)
            if len(rsi_series) >= 5:
                rsi_now = rsi_series[-1]
                rsi_prev = rsi_series[-5]
                if rsi_now < rsi_prev and (rsi_prev - rsi_now) > 5:
                    signals.append({
                        'type': 'rsi_overbought_drop',
                        'label': '🟠 RSI超买回落',
                        'detail': f"RSI从{rsi_prev:.0f}回落到{rsi_now:.0f}",
                        'severity': 'medium',
                    })
                    strength += 2

    # ─── 信号3: MACD死叉 ───
    if tech.get('macd_death_cross', 0):
        signals.append({
            'type': 'macd_death_cross',
            'label': '🔴 MACD死叉',
            'detail': 'MACD出现死叉信号',
            'severity': 'high',
        })
        strength += 3

    # ─── 信号4: 量价背离 ───
    if len(kline_df) >= 10 and tech.get('vol_trend') == 0:
        # 价格创新高但量能缩小
        high_10d = kline_df['high'].rolling(10).max()
        if len(high_10d) > 0:
            vol_10d = kline_df['volume'].rolling(5).mean()
            recent_high = kline_df['close'].iloc[-1] >= high_10d.iloc[-2]
            recent_vol_low = vol_10d.iloc[-1] < vol_10d.iloc[-5]
            if recent_high and recent_vol_low:
                signals.append({
                    'type': 'volume_divergence',
                    'label': '🟡 量价背离',
                    'detail': '价格高位但成交量萎缩',
                    'severity': 'medium',
                })
                strength += 2

    # ─── 信号5: 止损/止盈 ───
    if buy_price is not None and buy_price > 0:
        current_price = close[-1]
        pnl_pct = (current_price - buy_price) / buy_price * 100

        if pnl_pct <= s['stop_loss_pct'] * 100:
            signals.append({
                'type': 'stop_loss',
                'label': '🔴 触发止损',
                'detail': f"亏损 {pnl_pct:.1f}% (止损线 {s['stop_loss_pct']*100:.0f}%)",
                'severity': 'high',
            })
            strength += 4

        if pnl_pct >= s['take_profit_pct'] * 100:
            signals.append({
                'type': 'take_profit',
                'label': '🟢 触发止盈',
                'detail': f"盈利 {pnl_pct:.1f}% (止盈线 {s['take_profit_pct']*100:.0f}%)",
                'severity': 'low',
            })
            strength += 1

    # ─── 信号6: K线见顶形态 ───
    if tech.get('three_down', 0):
        signals.append({
            'type': 'three_black_crows',
            'label': '🔴 三连阴',
            'detail': '连续三天下跌',
            'severity': 'high',
        })
        strength += 2

    has_sell = len(signals) > 0

    return {
        'code': code,
        'name': name,
        'has_sell_signal': has_sell,
        'signals': signals,
        'signal_count': len(signals),
        'signal_strength': strength,
        'current_price': close[-1],
        'buy_price': buy_price,
    }


def _calc_rsi_series(prices, period=14):
    """计算RSI序列"""
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def batch_check_sell_signals(stock_names: dict, kline_data: dict,
                             tech_data: dict, history: list) -> list:
    """
    批量检查所有历史推荐股票的卖出信号
    返回: [sell_signal_dict, ...]
    """
    if not history:
        return []

    result = []

    for item in history:
        code = item.get('code', '')
        name = item.get('name', stock_names.get(code, code))
        buy_price = item.get('price_at_recommend', None)
        buy_date = item.get('date', '')

        kline_df = kline_data.get(code, None)
        tech = tech_data.get(code, None)

        check = check_sell_signals(code, name, kline_df, tech, buy_price)
        if check['has_sell_signal']:
            check['buy_date'] = buy_date
            check['buy_price'] = buy_price
            result.append(check)

    # 按信号强度排序
    result.sort(key=lambda x: x['signal_strength'], reverse=True)

    if result:
        print(f"[卖出] 检测到 {len(result)} 支历史推荐股有卖出信号")
    return result


if __name__ == '__main__':
    import akshare as ak
    from data_fetcher import get_daily_kline
    from indicators import compute_all_indicators

    kline = get_daily_kline('000001')
    if kline is not None and len(kline) > 0:
        tech = compute_all_indicators(kline)
        result = check_sell_signals('000001', '平安银行', kline, tech, buy_price=12.0)
        print(f"卖出信号: {result['has_sell_signal']}")
        for sig in result['signals']:
            print(f"  {sig['label']}: {sig['detail']}")
