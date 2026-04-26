#!/usr/bin/env python3
"""
A股量化推荐系统 v2.0 (幻方量化增强版)
主入口脚本

用法:
    python3 main.py                 # 完整分析 (传统+AI)
    python3 main.py --mode test     # 快速测试 (10支)
    python3 main.py --mode classic  # 仅传统多因子
"""
import os, sys, time, json, argparse
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
from reporter import generate_report, save_report, print_report


def run_daily_analysis(test_mode: bool = False, classic_mode: bool = False):
    """
    执行完整的每日分析流程 (幻方量化增强版)
    """
    start_time = time.time()
    mode_label = "幻方量化增强版" if not classic_mode else "传统多因子版"
    print(f"\n{'='*60}")
    print(f"  A股量化推荐系统 · {mode_label}")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ─── 步骤1: 获取股票池 ───
    print("[1/7] 获取股票池...")
    stocks = get_stock_universe()
    if test_mode and len(stocks) > 20:
        stocks = stocks.head(20)
    stock_names = dict(zip(stocks['code'], stocks['name']))
    print(f"  → {len(stocks)} 支股票待分析\n")

    # ─── 步骤2: 获取K线数据 ───
    print("[2/7] 获取K线数据...")
    kline_data = batch_fetch_klines(stocks)
    if len(kline_data) < 5:
        print("[错误] 获取到的K线数据太少")
        return
    print()

    # ─── 步骤3: 计算技术指标 ───
    print("[3/7] 计算技术指标...")
    tech_data = compute_all_indicators_batch(kline_data)
    print()

    # ─── 步骤3b: 幻方AI因子计算 (可选) ───
    deep_data = {}
    alpha_data = {}
    ml_scores = {}
    market_state = {}
    hf_config = CONFIG['hf_config']

    if not classic_mode and hf_config['enable_deep_features']:
        print("[3b/7] 计算深度技术因子 (幻方风格)...")
        deep_data = compute_all_deep_features_batch(kline_data)
        print()

    if not classic_mode and hf_config['enable_alpha_factors']:
        print("[3c/7] 计算Alpha因子...")
        alpha_data = compute_alpha_factors_batch(kline_data)
        print()

    if not classic_mode and hf_config['market_regime_detection']:
        print("[3d/7] 检测市场状态...")
        market_state = detect_market_regime(kline_data)
        print(f"  → 市场: {market_state['regime']} (动量:{market_state['market_momentum']:.1f}% "
              f"波动:{market_state['market_volatility']:.1f}% "
              f"量比:{market_state['market_volume']:.1f})\n")

    # ─── 步骤4: 获取舆情数据 ───
    print("[4/7] 获取舆情数据...")
    news_data = batch_fetch_news(stocks)
    sentiment_data = compute_sentiment_batch(news_data)
    print()

    # ─── 步骤5: 评分 & 排名 ───
    print("[5/7] 幻方综合评分 & 排名...")
    if not classic_mode and hf_config['enable_ml']:
        ml_scores = compute_ml_scores_batch(tech_data, deep_data, alpha_data, market_state)
    else:
        ml_scores = None

    scores = compute_all_scores(tech_data, sentiment_data, stock_names,
                                deep_data, alpha_data, ml_scores, market_state)
    picks = get_top_picks(scores, stock_names, CONFIG['top_n'])
    print(f"  → TOP 6 已选出，最高分: {picks[0]['total_score']:.1f}\n")

    # ─── 步骤6: 卖出信号检测 ───
    print("[6/7] 检测历史推荐卖出信号...")
    history = get_recent_recommendations(30)
    sell_signals = batch_check_sell_signals(stock_names, kline_data, tech_data, history)
    print()

    # ─── 步骤7: 生成报告 ───
    print("[7/7] 生成格式化报告...")
    report = generate_report(
        picks=picks,
        sell_signals=sell_signals,
        stock_count=len(kline_data),
        date_str=datetime.now().strftime('%Y-%m-%d'),
        market_state=market_state,
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
    parser = argparse.ArgumentParser(description='A股量化推荐系统 (幻方量化增强版)')
    parser.add_argument('--mode', choices=['daily', 'test', 'classic'],
                        default='daily', help='运行模式: daily完整分析, test快速测试, classic传统模式')
    parser.add_argument('--top-n', type=int, default=None,
                        help=f'推荐股票数量 (默认: {CONFIG["top_n"]})')
    args = parser.parse_args()

    if args.top_n:
        CONFIG['top_n'] = args.top_n

    if args.mode == 'test':
        print("[模式] 快速测试模式 (20支)\n")
        run_daily_analysis(test_mode=True)
    elif args.mode == 'classic':
        print("[模式] 传统多因子模式 (禁用AI策略)\n")
        for k in ['enable_ml', 'enable_alpha_factors', 'enable_deep_features']:
            CONFIG['hf_config'][k] = False
        run_daily_analysis(test_mode=False, classic_mode=True)
    else:
        run_daily_analysis(test_mode=False)


if __name__ == '__main__':
    main()
