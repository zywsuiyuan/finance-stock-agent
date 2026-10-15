# agent.py —— 创建 Agent + 对话接口
# 职责单一：组装 Agent（模型+工具+记忆+人设），对外只暴露 chat()
import sqlite3

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config import MODEL_NAME, MODEL_TEMPERATURE, DB_FILE
from tools import (
    search_finance_kb,
    get_stock_realtime,
    get_last_trade_date,
    get_financial_metrics,
    compare_financial_metrics,
)

# ===== 初始化对话记忆 =====
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
checkpointer = SqliteSaver(conn)

# ===== 初始化大模型 =====
model = init_chat_model(MODEL_NAME, temperature=MODEL_TEMPERATURE)

# ===== 创建财务分析 Agent =====
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

# ===== 对话接口 =====
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
