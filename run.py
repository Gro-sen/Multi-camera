#!/usr/bin/env python3
"""
启动脚本 - 使用新的app结构
"""
import sys
import os
import signal
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("DEBUG", "False")


def force_exit(signum, frame):
    """强制退出处理"""
    print("\n[INFO] 收到终止信号，强制退出...")
    os._exit(0)


def choose_provider() -> str:
    print("\n请选择推理提供方：")
    print("1) 阿里云 API")
    print("2) Ollama 本地模型")
    choice = input("输入 1 或 2（默认 1）: ").strip()

    if choice == "2":
        return "ollama"
    return "aliyun"


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)
    
    provider = choose_provider()
    os.environ["MODEL_PROVIDER"] = provider
    print(f"[启动] 当前推理提供方: {provider}")

    # 延迟导入，确保上面的环境变量先设置
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False) 