# app.py —— Web 界面（Streamlit）
# 职责单一：负责"好看地展示"，业务逻辑全在 agent.py 的 chat()
import streamlit as st

from agent import chat

st.set_page_config(page_title="A股财务分析助手", page_icon="📈")
st.title("📈 A股财务分析助手")
st.caption("基于 LangChain + akshare 的 A 股财务分析智能体")

# ===== 会话历史（存网页内存，刷新不丢）=====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== 显示历史消息 =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== 输入框：用户打字的地方 =====
if prompt := st.chat_input("问我，比如：贵州茅台今天的股价"):
    # 1. 用户的话上屏
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调 agent（记忆用固定会话ID，全程连续）
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer = chat("web_user", prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
