# 基于 LangChain 的 A股财务分析智能 Agent

一个基于大语言模型的 A 股数据分析助手，能够通过自然语言对话完成实时行情查询、财务指标分析、多公司对比、投资知识问答等任务。

## 技术栈

- **大模型**：DeepSeek Chat（通过 LangChain 调用）
- **Agent 框架**：LangChain / LangGraph（create_agent + 工具调用）
- **RAG 检索**：Chroma 向量数据库 + 阿里云百炼 text-embedding-v4
- **多轮记忆**：SqliteSaver（按 thread_id 持久化对话历史）
- **数据源**：AkShare（A股实时行情、财务数据）
- **开发语言**：Python

## 核心功能

1. **实时行情查询**：自然语言查询个股最新价格、涨跌幅
2. **财务指标分析**：查询市盈率、市净率、ROE、营收等核心指标
3. **多公司对比**：同时对比多家公司的关键财务数据
4. **投资知识问答（RAG）**：基于本地投教知识库检索增强，减少大模型幻觉
5. **交易日历计算**：自动判断交易日、计算距今天数
6. **多轮对话记忆**：按用户会话持久化上下文，支持连续追问

## 项目结构

├── first_agent_finance.py   # Agent 主入口
├── finance_tools.py        # 5 个自定义工具（@tool）
├── rag_build.py             # RAG 向量库构建
├── retriver_chorm.py        # RAG 检索器
├── main.py                  # 程序入口
├── stock_knowledge.txt      # 投教知识库原始文本
└── knowledge.txt            # 切分后的知识片段


## 运行方式
```bash
# 安装依赖
pip install langchain langchain-openai chromadb akshare

# 配置环境变量（在 .env 文件中）
# DEEPSEEK_API_KEY=你的key
# DASHSCOPE_API_KEY=你的key

# 运行
python main.py

```
## 作者

**张育玮**


