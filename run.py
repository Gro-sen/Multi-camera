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


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)
    
    import uvicorn
    from app.main import app
    from app.core import get_logger
    
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("智能安防视频分析系统 - 启动")
    logger.info("="*60)
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"访问地址: http://localhost:8000")
    logger.info(f"API文档: http://localhost:8000/docs")
    logger.info(f"WebSocket: ws://localhost:8000/ws")
    logger.info("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )