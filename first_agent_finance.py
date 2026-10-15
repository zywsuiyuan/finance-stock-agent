# ========== 1. 环境初始化 ==========
import os
import re
import time
import sqlite3
from dotenv import load_dotenv
load_dotenv()

from finance_tools import (
    compare_financial_metrics,
    get_financial_metrics,
)
from datetime import datetime, timedelta
from functools import lru_cache

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.checkpoint.sqlite import SqliteSaver
import akshare as ak

# ========== 2. 构建 RAG 知识库 ==========
# 2.1 加载知识库

KB_FILES = ["stock_knowledge.txt", "myself_knowledge.txt"]

documents = []
for _file in KB_FILES:
    if not os.path.exists(_file) or os.path.getsize(_file) == 0:
        print(f"[提示] 知识文件 {_file} 不存在或为空，已跳过")
        continue
    documents.extend(TextLoader(_file, encoding="utf-8").load())

if not documents:
    raise RuntimeError("没有任何可用的知识文件，请先填写 stock_knowledge.txt 或 myself_knowledge.txt")


# 2.2 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", " "]
)
chunks = text_splitter.split_documents(documents)

# 2.3 初始化百炼向量模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=10
)

# 2.4 加载/创建 Chroma 向量库（持久化，第一次运行创建，后续直接加载）
# 2.4 加载/创建 Chroma 向量库
# 已存在就只“连接”不插入；首次运行才真正写入
CHROMA_DIR = "./finance_chroma_db"
if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    # 已有向量库：直接连接
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="finance_knowledge",
    )
else:
    # 首次运行：把切分好的 chunks 写入向量库
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="finance_knowledge",
    )
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ========== 3. 定义 Agent 工具 ==========
# 工具1：RAG 知识库检索
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


# 工具3：获取最近A股交易日（用官方交易日历，而不是“工作日”）
@lru_cache(maxsize=1)
def _get_trade_date_set() -> set[str]:
    """拉取A股官方交易日历（含过去和未来），结果缓存，整个程序只请求一次。"""
    df = ak.tool_trade_date_hist_sina()
    return set(df["trade_date"].astype(str).str[:10])

def _last_trade_date() -> str:
    """（普通函数）从今天往前找最近一个交易日。自己的代码里调用这个。"""
    current_date = datetime.now().date()
    trade_dates = _get_trade_date_set()          # ← 保持原样！是"取交易日集合"
    while current_date.strftime("%Y-%m-%d") not in trade_dates:
        current_date -= timedelta(days=1)
    return current_date.strftime("%Y-%m-%d")

@tool
def get_last_trade_date() -> str:
    """获取最近一个A股交易日的日期字符串，格式：YYYY-MM-DD"""
    return _last_trade_date()



# 工具2：A股实时行情查询（东财单只接口，秒回，不再拉全市场快照）
# 工具2：A股实时行情查询（主源东财，失败自动重试并切换腾讯备用源）
@tool
def get_stock_realtime(stock_code: str) -> str:
    """查询A股单只股票的实时最新行情。
    Args:
        stock_code: 6位数字股票代码，例如 600519，也支持 sh600519、600519.SH
    """
    try:
        match = re.search(r"(\d{6})", stock_code.strip())
        if not match:
            return f"无法识别股票代码：{stock_code}，请输入6位数字代码。"
        code = match.group(1)

        # ===== 主源：东财实时报价，失败自动重试2次 =====
        quote = None
        for attempt in range(3):          # 第1次 + 2次重试，最多试3次
            try:
                quote = ak.stock_bid_ask_em(symbol=code)
                break                      # 成功就跳出循环
            except Exception:
                if attempt < 2:
                    time.sleep(1)          # 失败先等1秒，再试下一次
                quote = None

        # ===== 公司信息（尽力而为：失败不致命，用兜底值）=====
        try:
            info = ak.stock_individual_info_em(symbol=code)
            i = dict(zip(info["item"], info["value"]))
            name = i.get("股票简称", code)
            market_cap = float(i.get("总市值", 0)) / 100_000_000
            industry = i.get("行业", "暂无")
        except Exception:
            name = code        # 名称拿不到 → 用代码代替
            market_cap = 0.0   # 市值拿不到 → 显示 0
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
                    hist = ak.stock_zh_a_hist_tx(
                        symbol=f"sh{code}", start_date=start, end_date=today, adjust=""
                    )
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(1)
            if hist is None or len(hist) == 0:
                return "查询行情失败：主源与备用源均暂时不可用，请稍后重试。"
            last = hist.iloc[-1]                     # 最近一天
            prev = hist.iloc[-2] if len(hist) >= 2 else last   # 前一天，用来算涨跌幅
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

    except Exception as e:
        return f"查询行情失败：{str(e)}，请稍后重试。"


# ========== 4. 初始化对话记忆 ==========
conn = sqlite3.connect("finance_chat.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# ========== 5. 初始化大模型 ==========
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.2)

# ========== 6. 创建财务分析 Agent ==========
agent = create_agent(
    model=model,
    tools=[
    search_finance_kb,
    get_stock_realtime,
    get_last_trade_date,
    get_financial_metrics,
    compare_financial_metrics,
],
    checkpointer=checkpointer,
    system_prompt="""你是我的专属财务分析助手，专注于A股市场分析。
## 你的职责
1.  财务指标解释、估值逻辑、选股规则、交易规则等知识类问题，使用 search_finance_kb 工具查询知识库
2.  个股实时股价、涨跌幅等行情问题，使用 get_stock_realtime 工具查询
3. 查询某家公司具体报告期的ROE、净利润、营业收入、毛利率等数据时，
   必须使用 get_financial_metrics 工具，不能使用知识库中的定义代替真实数据。
4. 对比多家公司的财务指标时，必须使用 compare_financial_metrics 工具。
5. 财务数据必须注明公司、报告期和数据来源，不要根据记忆编造数字。
6. 工具查询失败时，应如实说明查询失败及原因，建议用户去券商APP自行查询，
   也不要猜测数据。
7. 帮我查询贵州茅台最近四个报告期的ROE、归母净利润和毛利率。
对比贵州茅台、五粮液、泸州老窖最新报告期的ROE和净利率。 财务知识使用 search_finance_kb；
   实时行情使用 get_stock_realtime；
   公司财务指标使用 get_financial_metrics；
   多公司比较使用 compare_financial_metrics。
## 行为准则
- 不知道的内容就查询知识库，查不到就诚实说明
- 不做投资建议，只做财务知识和行情数据的分析解读
- 保持严谨客观的语气"""
)

# ========== 7. 对话接口 ==========
def chat(thread_id: str, message: str) -> str:
    """和财务助手对话
    Args:
        thread_id: 会话ID，同一个用户用同一个ID可以保留对话记忆
        message: 用户的问题
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    return result["messages"][-1].content

# ========== 8. 测试运行 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("       财务分析智能助手")
    print("输入问题即可对话，输入 exit 退出")
    print("=" * 40 + "\n")

    # 固定会话ID，全程保留对话记忆
    thread_id = "user_001"

    # 无限循环，直到输入exit
    while True:
        # 接收用户输入
        user_input = input("你：")

        # 去除首尾空格，转小写后判断是否退出
        if user_input.strip().lower() == "exit":
            print("\n👋 对话已结束，祝你投资顺利")
            break

        # 空输入跳过
        if not user_input.strip():
            continue

        # 调用智能体并打印回答
        try:
            answer = chat(thread_id, user_input)
            print(f"助手：{answer}\n")
        except Exception as e:
            print(f"助手：抱歉，处理出错了——{str(e)}\n")

    # 退出前关闭数据库连接
    conn.close()