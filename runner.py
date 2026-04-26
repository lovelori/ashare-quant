#!/usr/bin/env python3
"""
A股量化推荐 CI 入口 (幻方量化增强版)
- 运行完整分析流程
- 推送结果到飞书机器人
- 只在 CI 环境中使用
"""
import os
import sys
import json
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from data_fetcher import get_stock_universe, batch_fetch_klines, batch_fetch_news
from indicators import compute_all_indicators_batch
from sentiment import compute_sentiment_batch
from deep_features import compute_all_deep_features_batch
from factor_mining import compute_alpha_factors_batch
from ml_strategy import compute_ml_scores_batch, detect_market_regime
from scorer import compute_all_scores, get_top_picks
from sell_signals import batch_check_sell_signals
from state import load_state, record_recommendation, get_recent_recommendations
from reporter import generate_report, save_report

FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')


def feishu_notify(picks, sell_signals, stock_count, report_path, market_state=None):
    """推送报告到飞书 (含AI策略信息)"""
    if not FEISHU_WEBHOOK:
        print("[飞书] 未设置 FEISHU_WEBHOOK，跳过推送")
        return False

    today = datetime.now().strftime('%Y-%m-%d')

    # 市场状态
    regime_text = ""
    if market_state:
        r = market_state.get('regime', '?')
        regime_text = f" 市场: {r} | 动量{market_state.get('market_momentum',0):.1f}% | 波动{market_state.get('market_volatility',0):.1f}%\n\n"

    # 构建TOP 6表格
    picks_md = ""
    for i, p in enumerate(picks[:6]):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        picks_md += f"**{emoji} {p['name']}({p['code']})**  总分: {p['total_score']:.1f}\n"
        d = p['details']
        picks_md += f"    趋势{d['trend']:.0f}·动量{d['momentum']:.0f}·量价{d['volume']:.0f}·情绪{d['sentiment']:.0f}·形态{d['pattern']:.0f}\n"
        # 幻方AI评分
        if 'hf_details' in p:
            hf = p['hf_details']
            if hf:
                parts = [f"AI:{v:.0f}" for k, v in sorted(hf.items())]
                picks_md += f"    ({' '.join(parts)})\n"
        picks_md += "\n"

    # 卖出信号
    sell_md = ""
    if sell_signals:
        sell_md = "**🚨 卖出信号**\n"
        for sig in sell_signals:
            sell_md += f"⚠ {sig['name']}({sig['code']}): "
            sell_md += " | ".join([s['label'] for s in sig['signals'][:2]])
            sell_md += "\n"
        sell_md += "\n"

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📊 A股量化推荐 · {today}"},
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**覆盖:** {stock_count}支沪深300成分股\n**策略:** 幻方量化AI增强 (传统多因子+ML+Alpha因子)\n{regime_text}**推荐:** TOP 6 最值得关注标的"
                    }
                },
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": picks_md.strip()}},
            ]
        }
    }

    if sell_md:
        card["card"]["elements"].append({"tag": "hr"})
        card["card"]["elements"].append({
            "tag": "div", "text": {"tag": "lark_md", "content": sell_md.strip()}
        })

    card["card"]["elements"].append({"tag": "hr"})
    card["card"]["elements"].append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": "🤖 幻方量化AI策略 · 仅供参考 · 股市有风险投资需谨慎"}]
    })

    try:
        resp = requests.post(FEISHU_WEBHOOK, json=card, timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            if result.get('code') == 0:
                print(f"[飞书] 推送成功")
                return True
            else:
                print(f"[飞书] 推送失败: {result}")
        else:
            print(f"[飞书] HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[飞书] 异常: {e}")

    # 备选：纯文本
    try:
        text_report = f"📊 A股量化推荐 {today}\n{'─'*30}\n"
        for i, p in enumerate(picks[:6]):
            text_report += f"{i+1}. {p['name']}({p['code']}) {p['total_score']:.1f}分\n"
        if sell_signals:
            text_report += f"\n🚨 卖出信号 {len(sell_signals)}个"
        text_report += "\n\n🤖 幻方量化AI策略 · 仅供参考"
        requests.post(FEISHU_WEBHOOK, json={"msg_type": "text", "content": {"text": text_report}}, timeout=10)
    except Exception:
        pass

    return False


def run():
    """执行完整 CI 流程 (幻方量化增强版)"""
    print(f"\n{'='*60}")
    print(f"  A股量化推荐 · CI 运行 (幻方量化增强版)")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ─── 步骤1: 股票池 ───
    print("[1/7] 获取股票池...")
    stocks = get_stock_universe()
    stock_names = dict(zip(stocks['code'], stocks['name']))
    print(f"  → {len(stocks)} 支股票\n")

    # ─── 步骤2: K线数据 ───
    print("[2/7] 获取K线数据...")
    kline_data = batch_fetch_klines(stocks)
    if len(kline_data) < 10:
        print("[错误] K线数据不足")
        return 1
    print()

    # ─── 步骤3: 技术指标 ───
    print("[3/7] 计算技术指标...")
    tech_data = compute_all_indicators_batch(kline_data)

    # AI因子
    hf_config = CONFIG['hf_config']
    deep_data = compute_all_deep_features_batch(kline_data) if hf_config['enable_deep_features'] else {}
    alpha_data = compute_alpha_factors_batch(kline_data) if hf_config['enable_alpha_factors'] else {}
    market_state = detect_market_regime(kline_data) if hf_config['market_regime_detection'] else {}
    if market_state:
        print(f"  → 市场状态: {market_state['regime']} (动量{market_state['market_momentum']:.1f}% 波动{market_state['market_volatility']:.1f}%)")
    print()

    # ─── 步骤4: 舆情 ───
    print("[4/7] 获取舆情数据...")
    news_data = batch_fetch_news(stocks)
    sentiment_data = compute_sentiment_batch(news_data)
    print()

    # ─── 步骤5: 评分 ───
    print("[5/7] 幻方综合评分 & 排名...")
    ml_scores = compute_ml_scores_batch(tech_data, deep_data, alpha_data, market_state) if hf_config['enable_ml'] else None
    scores = compute_all_scores(tech_data, sentiment_data, stock_names, deep_data, alpha_data, ml_scores, market_state)
    picks = get_top_picks(scores, stock_names, 6)
    print(f"  → TOP 6 已选出\n")

    # ─── 步骤6: 卖出信号 ───
    print("[6/7] 检测卖出信号...")
    history = get_recent_recommendations(30)
    sell_signals = batch_check_sell_signals(stock_names, kline_data, tech_data, history)
    print()

    # ─── 步骤7: 报告 ───
    print("[7/7] 生成报告...")
    report = generate_report(picks, sell_signals, len(kline_data), market_state=market_state)
    report_path = save_report(report)
    print(report)

    record_recommendation(picks)

    # 推送飞书
    print(f"\n{'─'*60}")
    print("[推送] 发送到飞书...")
    feishu_notify(picks, sell_signals, len(kline_data), report_path, market_state)

    print(f"\n{'='*60}")
    print(f"  ✅ CI 完成")
    print(f"{'='*60}")
    return 0


if __name__ == '__main__':
    sys.exit(run())
