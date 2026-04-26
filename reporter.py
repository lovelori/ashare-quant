"""
日报生成器
生成每日A股推荐报告，支持终端打印和文件保存
"""
import os
from datetime import datetime
from config import CONFIG


def generate_report(picks: list, sell_signals: list, stock_count: int,
                    date_str: str = None) -> str:
    """
    生成完整的日报文本
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    lines = []
    separator = '═' * 72

    lines.append(separator)
    lines.append(f"  📊 A股量化推荐系统 · 每日报告")
    lines.append(f"  {date_str}  |  沪深300成分股分析  |  共{stock_count}支")
    lines.append(separator)
    lines.append('')

    # ─── 卖出信号 (放前面，紧急!) ───
    if sell_signals:
        lines.append('  🚨 【卖出信号 · 历史推荐监测】')
        lines.append('─' * 68)
        for sig in sell_signals:
            code = sig['code']
            name = sig['name']
            severity = sig['signal_strength']
            signals_str = ' | '.join([s['label'] for s in sig['signals']])
            lines.append(f'  ⚠ {code} {name} (强度{severity})')
            for s in sig['signals']:
                lines.append(f'     {s["label"]} → {s["detail"]}')
            lines.append('')
        lines.append('─' * 68)
        lines.append('')

    # ─── 今日推荐 (TOP 6) ───
    lines.append(f'  🏆 【今日推荐 · TOP {CONFIG["top_n"]}】')
    lines.append('─' * 68)

    header = f'  {"排名":>3} {"代码":>7} {"名称":>10} {"总分":>6} {"趋势":>6} {"动量":>6} {"量价":>6} {"情绪":>6} {"形态":>6}'
    lines.append(header)
    lines.append('  ' + '─' * 66)

    for i, pick in enumerate(picks):
        rank = i + 1
        medal = '🥇' if rank == 1 else ('🥈' if rank == 2 else ('🥉' if rank == 3 else f'  {rank} '))
        code = pick['code']
        name = pick['name'][:8]
        score = pick['total_score']
        d = pick['details']
        trend = f"{d['trend']:.0f}"
        mom = f"{d['momentum']:.0f}"
        vol = f"{d['volume']:.0f}"
        sent = f"{d['sentiment']:.0f}"
        pat = f"{d['pattern']:.0f}"

        emoji = '🟢' if score >= 75 else ('🟡' if score >= 60 else '🔴')
        lines.append(f'  {medal} {code:>7} {name:>8} {emoji}{score:>5.1f} {trend:>5} {mom:>5} {vol:>5} {sent:>5} {pat:>5}')

    lines.append('')

    # 推荐理由
    lines.append('  📋 【推荐理由】')
    lines.append('─' * 68)
    for i, pick in enumerate(picks[:3]):
        reasons = _generate_reasons(pick)
        lines.append(f'  {i+1}. {pick["name"]}({pick["code"]}) 总分 {pick["total_score"]:.1f}')
        for r in reasons:
            lines.append(f'     → {r}')
        lines.append('')

    lines.append('─' * 68)

    # ─── 评分分布 ───
    lines.append('')
    lines.append('  📈 【评分统计】')
    lines.append('─' * 68)
    scores = [p['total_score'] for p in picks]
    if scores:
        lines.append(f'  最高分: {max(scores):.1f}  |  最低分: {min(scores):.1f}  |  平均: {sum(scores)/len(scores):.1f}')
    lines.append('')

    # ─── 短线建议 ───
    lines.append('  💡 【短线参考】')
    lines.append('─' * 68)
    for pick in picks:
        rsi = pick.get('rsi', 50)
        vol_r = pick.get('vol_ratio', 1.0)
        advice = []
        if rsi > 70:
            advice.append('RSI偏高 >70, 不宜追高')
        elif rsi < 30:
            advice.append('RSI偏弱 <30, 等待企稳')
        if vol_r > 2.5:
            advice.append(f'量比{vol_r:.1f}, 放量明显')
        if advice:
            lines.append(f'  {pick["name"]}: {" | ".join(advice)}')
    if not any(p.get('rsi', 50) > 70 or p.get('rsi', 50) < 30 for p in picks):
        pass  # 所有RSI正常
    lines.append('')

    lines.append(separator)
    lines.append('  🤖 A股量化推荐系统 v1.0')
    lines.append('  ⚠ 以上仅供参考，股市有风险，投资需谨慎')
    lines.append(separator)

    return '\n'.join(lines)


def _generate_reasons(pick: dict) -> list:
    """为推荐生成简洁的理由"""
    reasons = []
    d = pick['details']

    if d['trend'] >= 70:
        reasons.append('均线多头排列，趋势向上')
    if d['momentum'] >= 70:
        reasons.append('动量指标强势 (MACD多头/RSI健康)')
    if d['volume'] >= 70:
        reasons.append('成交量配合良好，放量上攻')
    if d['sentiment'] >= 65:
        reasons.append('舆情偏积极，近期有正面消息')
    if d['pattern'] >= 65:
        reasons.append('出现看涨K线形态')
    if d['volatility'] >= 65:
        reasons.append('波动率适中，布林带开口扩张')

    if not reasons:
        reasons.append('综合评分排名靠前')

    return reasons[:3]


def save_report(report: str) -> str:
    """保存报告到文件"""
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(CONFIG['reports_dir'], f'report_{date_str}.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[报告] 已保存: {report_path}")
    return report_path


def print_report(report: str):
    """打印报告到终端"""
    print('\n' + report + '\n')


if __name__ == '__main__':
    # 测试报告生成
    test_picks = [
        {'code': '000001', 'name': '平安银行', 'total_score': 85.5,
         'details': {'trend': 80, 'momentum': 75, 'volume': 70,
                     'volatility': 65, 'sentiment': 72, 'fundamental': 60, 'pattern': 68},
         'pct_1d': 2.3, 'pct_5d': 5.1, 'rsi': 58, 'vol_ratio': 1.5, 'sentiment_score': 72},
        {'code': '000002', 'name': '万科A', 'total_score': 78.2,
         'details': {'trend': 75, 'momentum': 72, 'volume': 68,
                     'volatility': 60, 'sentiment': 65, 'fundamental': 65, 'pattern': 60},
         'pct_1d': 1.5, 'pct_5d': 3.2, 'rsi': 55, 'vol_ratio': 1.2, 'sentiment_score': 65},
    ]
    test_sell = [
        {'code': '600519', 'name': '贵州茅台', 'signal_strength': 5,
         'signals': [
             {'label': '🔴 MACD死叉', 'detail': 'MACD出现死叉信号'},
             {'label': '🟠 RSI超买回落', 'detail': 'RSI从78回落到65'},
         ]},
    ]
    report = generate_report(test_picks, test_sell, 300)
    print_report(report)
    save_report(report)
