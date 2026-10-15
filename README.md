# A股财务分析智能助手

基于 **LangChain + akshare** 的 A 股财务分析智能体（Agent）。通过自然语言对话，即可查询股票实时行情、财务指标、多公司对比与投资知识问答，支持命令行与 Web 双端交互。

## 功能特性

- 自然语言对话：输入"贵州茅台今天的股价""对比茅台和五粮液的ROE"，自动理解意图并执行
- Agent 工具调用：5 个工具自动路由（行情 / 财务指标 / 多公司对比 / 知识检索 / 交易日历）
- 多源容错：东方财富主源 + 腾讯备用源，接口失败自动重试切换
- 股票容错：支持 6 位代码 / 简称 / 模糊匹配，打错字自动推荐候选（difflib 相似度）
- RAG 知识库：Chroma 向量检索，回答基于你的专属投教知识，减少幻觉
- 多轮对话记忆：SqliteSaver 持久化对话历史，支持连续追问
- 双端交互：命令行（main.py）+ Web（Streamlit app.py）

## 技术栈

| 层 | 技术 |
|----|------|
| 大模型 | DeepSeek（通过 langchain-deepseek 调用） |
| Agent 框架 | LangChain 1.4 create_agent + LangGraph（状态管理与记忆） |
| RAG | Chroma 向量库 + 阿里云百炼 text-embedding-v4 |
| 数据源 | akshare（A股实时行情、财务指标、交易日历） |
| 记忆 | SQLite（langgraph-checkpoint-sqlite） |
| Web | Streamlit |

## 项目结构

├── config.py           # 配置中心：模型名、路径、知识库文件等常量
├── kb.py               # 知识库构建：加载→切分→向量化→retriever
├── finance_tools.py    # 财务工具实现：行情解析、财务指标、多公司对比
├── tools.py            # 工具出口：5 个 Agent 工具统一注册
├── agent.py            # Agent 组装：模型 + 工具 + 记忆 + 人设，对外提供 chat ()
├── main.py             # 命令行入口：对话主循环
├── app.py              # Web 入口：Streamlit 网页界面
├── stock_knowledge.txt # 投教知识库原始文本
├── myself_knowledge.txt# 个人知识库（可选，空文件自动跳过）
├── .env                # 环境变量（已 gitignore，不提交）
└── requirements.txt    # 依赖清单


## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/zywsuiyuan/finance-stock-agent.git
cd finance-stock-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（复制 .env.example 并重命名为 .env，填入真实 key）
#    DEEPSEEK_API_KEY=你的key
#    DASHSCOPE_API_KEY=你的key

# 4. 命令行运行
python main.py

# 5. Web 界面运行
streamlit run app.py
# 浏览器访问 http://localhost:8501
```
## 配置说明（.env）

| 变量 | 说明 | 必填 |
|------|------|------|
| DEEPSEEK_API_KEY | DeepSeek 大模型 API Key | 是 |
| DASHSCOPE_API_KEY | 阿里云百炼 Key（知识库向量化 text-embedding-v4） | 是 |
| LANGSMITH_TRACING | LangSmith 链路追踪开关（true/false） | 否 |
| LANGSMITH_API_KEY | LangSmith API Key | 否 |
| LANGSMITH_PROJECT | LangSmith 项目名 | 否 |

## 架构设计
config（配置）→ kb（知识库）→ tools（工具出口）→ agent（大脑）→ main /app（入口）
↑
finance_tools（财务工具实现）
- 单一职责：每个模块只做一件事，互不干扰
- 统一出口：Agent 只认识 tools.py 一个工具集合
- 数据流：用户问题 → agent 判断 → 调用工具 → akshare/知识库返回 → 模型组织回答

## 常见问题

| 问题 | 解决 |
|------|------|
| 行情接口报错/超时 | 已内置重试 3 次 + 自动切换腾讯备用源 |
| 打错股票名查不到 | 支持 difflib 相似度推荐，自动提示"你是不是想查 XXX" |
| 报 400 tool_calls 错误 | 对话记忆被污染，删除 finance_chat.db 后重启 |
| Web 界面改了代码不生效 | Streamlit 需重启（Ctrl+C 后重新 streamlit run app.py） |

## 更新日志

- v1.2（2026-10）：新增 Streamlit Web 界面、侧边栏与清空对话；收紧 system_prompt
- v1.1（2026-10）：五模块工程化重构（config/kb/tools/agent/main）
- v1.0（2026-09）：单文件 Agent，支持行情/财务/知识问答


##作者
张育玮
