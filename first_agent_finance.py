# ========== 1. 环境初始化 ==========
from dotenv import load_dotenv
load_dotenv()

from agent import chat

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
