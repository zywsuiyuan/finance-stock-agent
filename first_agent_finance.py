# ========== 1. 环境初始化 ==========
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

from finance_tools import (
    compare_financial_metrics,
    get_financial_metrics,
)
from datetime import datetime, timedelta
from chinese_calendar import is_workday

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
loader = TextLoader("stock_knowledge.txt", encoding="utf-8")
documents = loader.load()

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
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./finance_chroma_db",
    collection_name="finance_knowledge"
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

#工具3 具体返回最近的交易日时间
@tool
def get_last_trade_date() -> str:
    """获取最近一个A股交易日的日期字符串，格式：YYYY-MM-DD"""
    current_date = datetime.now().date()
    # 从今天往前找，直到找到工作日（A股交易日 = 工作日）
    while not is_workday(current_date):
        current_date -= timedelta(days=1)
    return current_date.strftime("%Y-%m-%d")


# 工具2：A股实时行情查询（新浪源，网络兼容性更好）
@tool
def get_stock_realtime(stock_code: str) -> str:
    """查询A股单只股票的实时最新行情。
    Args:
        stock_code: 6位数字股票代码，例如 600519
    """
    try:
        # 自动补全市场前缀
        code = stock_code.strip()
        if code.startswith("6"):
            full_code = f"sh{code}"
        elif code.startswith(("0", "3")):
            full_code = f"sz{code}"
        elif code.startswith(("8", "4")):
            full_code = f"bj{code}"
        else:
            full_code = code

        # 获取实时行情
        df = ak.stock_zh_a_spot()
        target = df[df["代码"] == full_code]

        if target.empty:
            return f"未找到代码为 {stock_code} 的股票，请检查代码是否正确。"

        stock = target.iloc[0]
        full_time = f"{get_last_trade_date()} {stock['时间戳']}"
        return f"""股票：{stock['名称']}（{stock_code}）
最新价：{stock['最新价']} 元
涨跌幅：{stock['涨跌幅']}%
最低：{stock['最低']} 元
成交量：{stock['成交量']} 手
成交额：{stock['成交额']} 万元
更新时间：{full_time}"""

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
6. 工具查询失败时，应如实说明查询失败及原因，不要建议用户去券商APP自行查询，
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