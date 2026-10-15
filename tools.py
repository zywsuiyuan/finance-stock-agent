# tools.py —— Agent 的全部工具
# 职责单一：只定义工具。检索器从 kb 拿，财务工具从 finance_tools 拿
import re
import time
from datetime import datetime, timedelta
from functools import lru_cache
from finance_tools import _resolve_stock_code
from langchain.tools import tool
import akshare as ak

from kb import retriever                                 # 知识库检索器（来自 kb.py）
from finance_tools import compare_financial_metrics, get_financial_metrics  # 财务工具（已装饰，直接转出口）

# ===== 辅助函数：交易日历 =====
@lru_cache(maxsize=1)
def _get_trade_date_set() -> set[str]:
    """拉取A股官方交易日历（含过去和未来），结果缓存，整个程序只请求一次。"""
    df = ak.tool_trade_date_hist_sina()
    return set(df["trade_date"].astype(str).str[:10])

def _last_trade_date() -> str:
    """（普通函数）从今天往前找最近一个交易日。自己的代码里调用这个。"""
    current_date = datetime.now().date()
    trade_dates = _get_trade_date_set()
    while current_date.strftime("%Y-%m-%d") not in trade_dates:
        current_date -= timedelta(days=1)
    return current_date.strftime("%Y-%m-%d")

# ===== 工具1：RAG 知识库检索 =====
@tool
def search_finance_kb(query: str) -> str:
    """搜索财务知识库，获取财务指标定义、估值逻辑、选股规则、交易规则等知识。
    Args:
        query: 要查询的财务问题或关键词
    """
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中没有找到相关内容。"
    return "\n".join(f"- {doc.page_content}" for doc in docs)

# ===== 工具2：获取最近A股交易日 =====
@tool
def get_last_trade_date() -> str:
    """获取最近一个A股交易日的日期字符串，格式：YYYY-MM-DD"""
    return _last_trade_date()

# ===== 工具3：A股实时行情查询 =====
@tool
def get_stock_realtime(stock_code: str) -> str:
    """查询A股单只股票的实时最新行情。
    Args:
        stock_code: 6位数字股票代码，例如 600519，也支持 sh600519、600519.SH
    """
    try:
        code, symbol = _resolve_stock_code(stock_code)
    except ValueError as e:
        return f"{str(e)}"

    # ===== 主源：东财实时报价，失败自动重试2次 =====
    quote = None
    for attempt in range(3):
        try:
            quote = ak.stock_bid_ask_em(symbol=code)
            break
        except Exception:
            if attempt < 2:
                time.sleep(1)
            quote = None

    # ===== 公司信息（尽力而为：失败不致命，用兜底值）=====
    try:
        info = ak.stock_individual_info_em(symbol=code)
        i = dict(zip(info["item"], info["value"]))
        name = i.get("股票简称", code)
        market_cap = float(i.get("总市值", 0)) / 100_000_000
        industry = i.get("行业", "暂无")
    except Exception:
        name = code
        market_cap = 0.0
        industry = "暂无"

    if quote is not None:
        # ===== 主源成功：用实时五档报价 =====
        q = dict(zip(quote["item"], quote["value"]))
        price = float(q.get("最新"))
        pct = float(q.get("涨幅"))
        high = float(q.get("最高"))
        low = float(q.get("最低"))
        open_p = float(q.get("今开"))
        prev_close = float(q.get("昨收"))
        volume = float(q.get("总手"))
        amount = float(q.get("金额"))
        turnover = float(q.get("换手"))
        source = "东方财富（实时）"
    else:
        # ===== 主源失败：切换腾讯日线备用源 =====
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
        hist = None
        for attempt in range(3):
            try:
                prefix = symbol.split(".")[1].lower()  # "SH" → "sh"，深市 → "sz"
                hist = ak.stock_zh_a_hist_tx(
                    symbol=f"{prefix}{code}", start_date=start, end_date=today, adjust=""
                )
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1)
        if hist is None or len(hist) == 0:
            return "查询行情失败：主源与备用源均暂时不可用，请稍后重试。"
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else last
        price = float(last["close"])
        prev_close = float(prev["close"])
        pct = (price - prev_close) / prev_close * 100
        high = float(last["high"])
        low = float(last["low"])
        open_p = float(last["open"])
        volume = float(last["volume"])
        amount = float(last["amount"])
        turnover = 0.0
        source = "腾讯日线（备用源，最近交易日收盘数据）"

    trade_date = _last_trade_date()
    return f"""股票：{name}（{code}）
最新价：{price:.2f} 元
涨跌幅：{pct:.2f}%
今开：{open_p:.2f} 元 / 昨收：{prev_close:.2f} 元
最高：{high:.2f} 元 / 最低：{low:.2f} 元
成交量：{volume:.0f} 手 / 成交额：{amount / 100_000_000:.2f} 亿元
换手率：{turnover:.2f}%
总市值：{market_cap:.2f} 亿元
行业：{industry}
数据源：{source}
更新时间：{trade_date}"""

