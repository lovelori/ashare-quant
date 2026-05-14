"""
数据获取模块 - A股行情、K线、新闻数据
优先使用 Sina API (requests, fast & reliable from China)
"""
import re
import json
import time
import requests
import pandas as pd
import numpy as np
import akshare as ak
from config import CONFIG


# ─── K线数据 (Sina API - 最快最稳定) ───

def get_daily_kline(symbol: str, retries: int = 2) -> pd.DataFrame:
    """
    获取个股日K线 (Sina Finance API)
    返回 DataFrame [date, open, close, high, low, volume, amount]
    """
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    days = CONFIG.get('kline_days', 120)
    url = (
        f'https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_daily/'
        f'CN_MarketData.getKLineData?symbol={prefix}{symbol}&scale=240&ma=no&datalen={days}'
    )

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            match = re.search(r'\((\[.*\])\)', r.text)
            if not match:
                raise ValueError('No JSONP match')

            raw = json.loads(match.group(1))
            if not raw or len(raw) < 20:
                raise ValueError(f'Too few bars: {len(raw) if raw else 0}')

            # 转为 DataFrame
            rows = []
            for d in raw:
                rows.append({
                    'date': d['day'][:10],
                    'open': float(d['open']),
                    'high': float(d['high']),
                    'low': float(d['low']),
                    'close': float(d['close']),
                    'volume': int(float(d.get('volume', 0))),
                    'amount': float(d.get('amount', 0)),
                })

            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 计算涨跌幅和换手率
            df['pct_chg'] = df['close'].pct_change() * 100
            df['change'] = df['close'].diff()
            df['pct_chg'] = df['pct_chg'].fillna(0)
            df['change'] = df['change'].fillna(0)
            df['turnover'] = 0.0  # 换手率从 Sina API 拿不到，忽略
            df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
            df['amplitude'] = df['amplitude'].fillna(0)

            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return pd.DataFrame()

    return pd.DataFrame()


# ─── 市值排名 (Sina行情中心API) ───

def fetch_market_caps(max_stocks: int = 6000) -> dict:
    """
    从新浪行情中心获取全A股实时市值数据
    返回: {code: {'name': str, 'mktcap': float(元), 'trade': float}}
    """
    nodes = [
        ('sh_a', '沪市主板'),
        ('sz_a', '深市主板'),
        ('kcb', '科创板'),
    ]

    all_stocks = {}
    for node, desc in nodes:
        for page in range(1, 60):
            url = (
                f'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
                f'Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1'
                f'&node={node}&symbol=&_s_r_a=init'
            )
            try:
                r = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                data = r.json()
            except Exception:
                break
            if not data or len(data) == 0:
                break
            for s in data:
                code = s['code']
                if code not in all_stocks:
                    mktcap = float(s.get('mktcap', 0)) * 10000  # 新浪mktcap单位是万元 -> 元
                    all_stocks[code] = {
                        'name': s['name'],
                        'mktcap': mktcap,
                        'trade': float(s.get('trade', 0)),
                    }
            if len(data) < 100:
                break
            time.sleep(0.12)

    return all_stocks


def get_top_n_by_market_cap(n: int = 1000) -> pd.DataFrame:
    """
    获取A股市值前N的股票
    返回: DataFrame [code, name, mktcap]
    """
    print(f"[市值] 获取全A股市值排名 TOP {n}...")
    t_start = time.time()

    caps = fetch_market_caps()

    # 按市值降序排序
    sorted_stocks = sorted(caps.items(), key=lambda x: x[1]['mktcap'], reverse=True)

    rows = []
    for code, info in sorted_stocks[:n]:
        rows.append({
            'code': code,
            'name': info['name'],
            'mktcap': info['mktcap'],
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - t_start
    print(f"[市值] TOP {len(df)} 股票已选出 (总耗时 {elapsed:.0f}s)")
    if len(df) > 0:
        threshold = df['mktcap'].iloc[-1] / 1e8
        print(f"[市值] 门槛: {threshold:.1f}亿 ({df.iloc[-1]['name']})")

        # 打印TOP 10预览
        print(f"[市值] TOP 10预览:")
        for i, (_, row) in enumerate(df.head(10).iterrows()):
            print(f"  {i+1:>2}. {row['code']} {row['name']} 市值:{row['mktcap']/1e8:.1f}亿")

    return df


# ─── 股票池 ───

def get_stock_universe() -> pd.DataFrame:
    """
    获取股票池列表
    返回: DataFrame [code, name]
    """
    pool_type = CONFIG['stock_pool']

    if pool_type == 'hs300':
        try:
            df = ak.index_stock_cons_csindex(symbol='000300')
            result = df[['成分券代码', '成分券名称']].copy()
            result.columns = ['code', 'name']
            result['code'] = result['code'].astype(str).str.zfill(6)
            print(f"[池] 沪深300成分股: {len(result)} 支")
            return result
        except Exception as e:
            print(f"[池] 获取沪深300失败: {e}, 使用全市场列表")
            return _fallback_stock_list()

    elif pool_type == 'top_1000':
        n = CONFIG.get('market_cap_top_n', 1000)
        df = get_top_n_by_market_cap(n)
        print(f"  → A股市值TOP {n} 股票池: {len(df)} 支")
        return df

    elif pool_type == 'hs300_zz500':
        try:
            df300 = ak.index_stock_cons_csindex(symbol='000300')
            df500 = ak.index_stock_cons_csindex(symbol='000905')
            df300 = df300[['成分券代码', '成分券名称']].copy()
            df500 = df500[['成分券代码', '成分券名称']].copy()
            df300.columns = ['code', 'name']
            df500.columns = ['code', 'name']
            df300['code'] = df300['code'].astype(str).str.zfill(6)
            df500['code'] = df500['code'].astype(str).str.zfill(6)
            result = pd.concat([df300, df500]).drop_duplicates(subset='code').reset_index(drop=True)
            print(f"[池] 沪深300+中证500合并: {len(result)} 支 (沪深300:{len(df300)}, 中证500:{len(df500)})")
            return result
        except Exception as e:
            print(f"[池] 获取沪深300+中证500失败: {e}, 使用全市场列表")
            return _fallback_stock_list()

    elif pool_type == 'all':
        return _fallback_stock_list()

    elif isinstance(pool_type, list):
        return pd.DataFrame(pool_type, columns=['code', 'name'])

    else:
        try:
            df = ak.index_stock_cons_csindex(symbol='000300')
            result = df[['成分券代码', '成分券名称']].copy()
            result.columns = ['code', 'name']
            result['code'] = result['code'].astype(str).str.zfill(6)
            return result
        except Exception:
            return _fallback_stock_list()


def _fallback_stock_list() -> pd.DataFrame:
    """备用: 获取全市场股票列表"""
    try:
        df = ak.stock_info_a_code_name()
        print(f"[池] A股全市场: {len(df)} 支")
        return df
    except Exception as e:
        print(f"[池] 全市场获取失败: {e}")
        return pd.DataFrame(columns=['code', 'name'])


# ─── 新闻舆情 ───

def get_stock_news(symbol: str, retries: int = 2) -> pd.DataFrame:
    """
    获取个股新闻舆情数据 (akshare)
    """
    for attempt in range(retries):
        try:
            df = ak.stock_news_em(symbol=symbol)
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    '关键词': 'symbol',
                    '新闻标题': 'title',
                    '新闻内容': 'content',
                    '发布时间': 'pub_time',
                    '文章来源': 'source',
                    '新闻链接': 'url',
                })
                df['pub_time'] = pd.to_datetime(df['pub_time'], errors='coerce')
                since = pd.Timestamp(CONFIG['news_since'])
                df = df[df['pub_time'] >= since]
                return df.head(CONFIG['max_news_per_stock'])
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


# ─── 批量获取 ───

def batch_fetch_klines(stocks: pd.DataFrame) -> dict:
    """
    批量获取所有股票K线数据
    返回: {code: kline_df}
    """
    result = {}
    codes = stocks['code'].tolist()
    names = dict(zip(stocks['code'], stocks['name']))
    total = len(codes)

    print(f"\n[K线] 获取 {total} 支股票日K线...")
    success, failed = 0, 0
    t_start = time.time()

    for i, code in enumerate(codes):
        df = get_daily_kline(code)
        if df is not None and len(df) > 20:
            result[code] = df
            success += 1
        else:
            failed += 1

        if (i + 1) % 20 == 0 or i == total - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / total * 100
            print(f"  [{i+1}/{total} {pct:.0f}%] ✓{success} ✗{failed} | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        time.sleep(0.1)

    print(f"[K线] 完成: ✓{success} ✗{failed} | 耗时 {time.time()-t_start:.0f}s")
    return result


def batch_fetch_news(stocks: pd.DataFrame) -> dict:
    """
    批量获取新闻舆情
    返回: {code: news_df}
    """
    result = {}
    codes = stocks['code'].tolist()
    total = len(codes)

    print(f"\n[新闻] 获取 {total} 支股票新闻...")
    success = 0

    for i, code in enumerate(codes):
        df = get_stock_news(code)
        if df is not None and len(df) > 0:
            result[code] = df
            success += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] 有新闻: {success}")

    print(f"[新闻] 完成: {success} 支有近期新闻")
    return result


def get_stock_info(symbol: str) -> dict:
    """获取个股基本信息"""
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        result = {}
        for _, row in df.iterrows():
            result[row['item']] = row['value']
        return result
    except Exception:
        return {}


if __name__ == '__main__':
    # 测试
    stocks = get_stock_universe()
    print(stocks.head(3))

    k = get_daily_kline('000001')
    print(f"\n平安银行K线: {len(k)} 行")
    if len(k) > 0:
        print(k[['date', 'close', 'volume', 'pct_chg']].tail(5))

    n = get_stock_news('000001')
    print(f"\n新闻: {len(n) if n is not None else 0} 条")
    if n is not None and len(n) > 0:
        print(n[['title', 'pub_time']].head(2))
