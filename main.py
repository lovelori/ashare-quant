#!/usr/bin/env python3
"""
A股量化推荐系统 v1.0
主入口脚本 - 每日分析 + 推荐 + 卖出信号检测

用法:
    python3 main.py                  # 执行每日分析推荐
    python3 main.py --mode test      # 快速测试 (只分析10支)
    python3 main.py --mode backtest  # 模拟回测 (TODO)
"""
import os, sys, time, json, argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from data_fetcher import (get_stock_universe, batch_fetch_klines,
                          batch_fetch_news, get_stock_info)
from indicators import compute_all_indicators_batch
from sentiment import compute_sentiment_batch
from scorer import compute_all_scores, get_top_picks
from sell_signals import batch_check_sell_signals
from state import load_state, record_recommendation, get_recent_recommendations
from reporter import generate_report, save_report, print_report


def run_daily_analysis(test_mode: bool = False):
    """
    执行完整的每日分析流程
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"  A股量化推荐系统 · 每日分析")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ─── 步骤1: 获取股票池 ───
    print("[1/6] 获取股票池...")
    stocks = get_stock_universe()
    if test_mode and len(stocks) > 20:
        stocks = stocks.head(20)
    stock_names = dict(zip(stocks['code'], stocks['name']))
    print(f"  → {len(stocks)} 支股票待分析\n")

    # ─── 步骤2: 获取K线数据 ───
    print("[2/6] 获取K线数据...")
    kline_data = batch_fetch_klines(stocks)
    if len(kline_data) < 5:
        print("[错误] 获取到的K线数据太少，无法分析")
        return
    print()

    # ─── 步骤3: 计算技术指标 ───
    print("[3/6] 计算技术指标...")
    tech_data = compute_all_indicators_batch(kline_data)
    print()

    # ─── 步骤4: 获取舆情数据 ───
    print("[4/6] 获取舆情数据...")
    news_data = batch_fetch_news(stocks)
    sentiment_data = compute_sentiment_batch(news_data)
    print()

    # ─── 步骤5: 评分 & 排名 ───
    print("[5/6] 多因子评分 & 排名...")
    scores = compute_all_scores(tech_data, sentiment_data, stock_names)
    picks = get_top_picks(scores, stock_names, CONFIG['top_n'])
    print(f"  → TOP {CONFIG['top_n']} 已选出，最高分: {picks[0]['total_score']:.1f}")
    print()

    # ─── 步骤6: 卖出信号检测 ───
    print("[6/6] 检测历史推荐卖出信号...")
    history = get_recent_recommendations(30)
    sell_signals = batch_check_sell_signals(stock_names, kline_data, tech_data, history)
    print()

    # ─── 生成报告 ───
    report = generate_report(
        picks=picks,
        sell_signals=sell_signals,
        stock_count=len(kline_data),
        date_str=datetime.now().strftime('%Y-%m-%d'),
    )
    print_report(report)
    report_path = save_report(report)

    # ─── 保存推荐记录 ───
    record_recommendation(picks)

    elapsed = time.time() - start_time
    print(f"\n[完成] 总耗时: {elapsed:.1f}秒  |  报告: {report_path}")
    print()

    return picks, sell_signals, report_path


def main():
    parser = argparse.ArgumentParser(description='A股量化推荐系统')
    parser.add_argument('--mode', choices=['daily', 'test', 'backtest'],
                       default='daily', help='运行模式')
    parser.add_argument('--top-n', type=int, default=None,
                       help=f'推荐股票数量 (默认: {CONFIG["top_n"]})')
    args = parser.parse_args()

    if args.top_n:
        CONFIG['top_n'] = args.top_n

    if args.mode == 'test':
        print("[模式] 快速测试模式 (仅分析前20支)\n")
        run_daily_analysis(test_mode=True)
    elif args.mode == 'daily':
        run_daily_analysis(test_mode=False)
    elif args.mode == 'backtest':
        print("[模式] 回测模式 (TODO)")
        pass


if __name__ == '__main__':
    main()
