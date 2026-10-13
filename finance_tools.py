from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import akshare as ak
import pandas as pd
from langchain.tools import tool

_CODE_PATTERN = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _exchange_suffix(code: str) -> str:
    """根据6位代码判断交易所。"""
    if code.startswith("6"):
        return "SH"
    if code.startswith(("0", "3")):
        return "SZ"
    if code.startswith(("4", "8", "92")):
        return "BJ"
    raise ValueError(f"无法识别股票代码所属交易所：{code}")


@lru_cache(maxsize=1)
def _get_name_code_map() -> dict[str, str]:
    """获取A股简称和代码的映射，只在首次使用公司简称时查询。"""
    df = ak.stock_info_a_code_name()

    result: dict[str, str] = {}
    for row in df.itertuples(index=False):
        name = str(getattr(row, "name", "")).strip()
        code = str(getattr(row, "code", "")).strip()
        if name and code:
            result[name] = code.zfill(6)
    return result


def _resolve_stock_code(stock_code: str) -> tuple[str, str]:
    """支持 600519、600519.SH、SH600519、贵州茅台。"""
    text = stock_code.strip()

    match = _CODE_PATTERN.search(text)
    if match:
        code = match.group(1)
        return code, f"{code}.{_exchange_suffix(code)}"

    name_map = _get_name_code_map()

    if text in name_map:
        code = name_map[text]
        return code, f"{code}.{_exchange_suffix(code)}"

    matches = [
        (name, code)
        for name, code in name_map.items()
        if text in name
    ]

    if len(matches) == 1:
        name, code = matches[0]
        return code, f"{code}.{_exchange_suffix(code)}"

    if len(matches) > 1:
        examples = "、".join(f"{name}({code})" for name, code in matches[:5])
        raise ValueError(f"公司简称不唯一，请使用股票代码。匹配结果：{examples}")

    raise ValueError(f"没有找到股票：{stock_code}")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _format_amount(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "暂无"
    return f"{number / 100_000_000:.2f}亿元"


def _format_percent(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "暂无"
    return f"{number:.2f}%"


def _format_number(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "暂无"
    return f"{number:.2f}"


def _load_metrics(stock_code: str) -> tuple[str, str, pd.DataFrame]:
    """返回股票代码、公司名称、按报告期倒序排列的财务指标。"""
    code, symbol = _resolve_stock_code(stock_code)

    df = ak.stock_financial_analysis_indicator_em(
        symbol=symbol,
        indicator="按报告期",
    )

    if df is None or df.empty:
        raise ValueError(f"未查询到 {code} 的财务指标")

    df = df.copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
    df = df.dropna(subset=["REPORT_DATE"])
    df = df.sort_values("REPORT_DATE", ascending=False).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"未查询到 {code} 的有效报告期数据")

    company_name = str(
        df.iloc[0].get("SECURITY_NAME_ABBR") or code
    ).strip()

    return code, company_name, df


def _format_metric_row(row: pd.Series) -> str:
    report_date = row.get("REPORT_DATE")
    date_text = (
        report_date.strftime("%Y-%m-%d")
        if hasattr(report_date, "strftime")
        else str(report_date)[:10]
    )

    season = row.get("SEASON_LABEL")
    season_text = "" if pd.isna(season) else f"（{season}）"

    return (
        f"{date_text}{season_text}："
        f"营业收入 {_format_amount(row.get('TOTALOPERATEREVE'))}"
        f"（同比 {_format_percent(row.get('TOTALOPERATEREVETZ'))}）；"
        f"归母净利润 {_format_amount(row.get('PARENTNETPROFIT'))}"
        f"（同比 {_format_percent(row.get('PARENTNETPROFITTZ'))}）；"
        f"扣非归母净利润 {_format_amount(row.get('DEDU_PARENT_PROFIT'))}；"
        f"ROE（摊薄）{_format_percent(row.get('ROE_DILUTED'))}；"
        f"毛利率 {_format_percent(row.get('GROSS_PROFIT_RATIO'))}；"
        f"净利率 {_format_percent(row.get('NET_PROFIT_RATIO'))}；"
        f"每股收益 {_format_number(row.get('EPSJB'))}元；"
        f"每股净资产 {_format_number(row.get('BPS'))}元"
    )


@tool
def get_financial_metrics(stock_code: str, periods: int = 8) -> str:
    """查询A股公司最近若干报告期的核心财务指标。

    可以查询：营业收入、归母净利润、扣非归母净利润、ROE、
    毛利率、净利率、每股收益、每股净资产，以及相关同比增速。

    Args:
        stock_code: 6位A股代码或公司简称，例如 600519、600519.SH、贵州茅台。
        periods: 返回最近多少个报告期，默认8，范围1到40。
    """
    try:
        periods = max(1, min(int(periods), 40))
    except (TypeError, ValueError):
        return "periods 必须是1到40之间的整数。"

    try:
        code, company_name, df = _load_metrics(stock_code)
    except Exception as exc:
        return (
            f"财务指标查询失败：{type(exc).__name__}: {exc}。"
            "请检查股票代码、网络连接和AkShare版本，不要根据自己的记忆编造数据。"
        )

    lines = [
        f"{company_name}（{code}）核心财务指标",
        "数据源：东方财富F10，经AkShare获取；金额单位为亿元。",
        "",
    ]

    for _, row in df.head(periods).iterrows():
        lines.append(_format_metric_row(row))

    lines.extend(
        [
            "",
            "说明：ROE和营业收入均为报告期累计值；"
            "一季报、半年报、三季报ROE不能直接与全年ROE比较。"
            "第三方数据仅供学习分析，最终请以公司公告为准。",
        ]
    )

    return "\n".join(lines)


@tool
def     compare_financial_metrics(stock_codes: str, periods: int = 1) -> str:
    """对比多家A股公司的核心财务指标。

    Args:
        stock_codes: 多个股票代码或公司简称，用逗号分隔，最多5个，
                     例如 "600519,000858,000568"。
        periods: 每家公司返回最近多少个报告期，默认1，范围1到8。
    """
    try:
        periods = max(1, min(int(periods), 8))
    except (TypeError, ValueError):
        return "periods 必须是1到8之间的整数。"

    items = [
        item
        for item in re.split(r"[,，;；\s]+", stock_codes.strip())
        if item
    ]

    if not items:
        return "请至少提供一个股票代码或公司简称。"

    if len(items) > 5:
        return "一次最多对比5家公司。"

    lines = [
        "多公司核心财务指标对比",
        "数据源：东方财富F10，经AkShare获取；金额单位为亿元。",
        "",
    ]

    for item in items:
        try:
            code, company_name, df = _load_metrics(item)
            lines.append(f"【{company_name}（{code}）】")

            for _, row in df.head(periods).iterrows():
                lines.append(_format_metric_row(row))

            lines.append("")

        except Exception as exc:
            lines.append(
                f"【{item}】查询失败："
                f"{type(exc).__name__}: {exc}\n"
            )

    return "\n".join(lines).strip()