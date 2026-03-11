"""
统一配置管理模块
"""
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

class Config:
    """应用配置类"""
    
    # ===== 路径配置 =====
    BASE_DIR = Path(__file__).parent.parent.parent
    ALARM_DIR = BASE_DIR / "alarms"
    SOUND_DIR = BASE_DIR / "sounds"
    LOGS_DIR = BASE_DIR / "logs"
    KB_DIR = BASE_DIR / "kb"
    
    # 创建必要的目录
    for directory in [ALARM_DIR, SOUND_DIR, LOGS_DIR, KB_DIR]:
        directory.mkdir(exist_ok=True)
    
    # ===== FastAPI 配置 =====
    APP_TITLE = "智能安防视频分析系统"
    APP_VERSION = "1.0.0"
    
    # ===== RTSP 流配置 =====
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://<user>:<pass>@<ip>:<port>/stream1")
    RTSP_URL_2 = os.getenv("RTSP_URL_2", "")
    CAMERA_SOURCES = [
        {
            "id": "cam1",
            "name": "摄像头1",
            "rtsp_url": RTSP_URL,
        }
    ]
    if RTSP_URL_2:
        CAMERA_SOURCES.append({
            "id": "cam2",
            "name": "摄像头2",
            "rtsp_url": RTSP_URL_2,
        })
    
    # ===== 推理参数 =====
    INFER_INTERVAL = float(os.getenv("INFER_INTERVAL", "2.0"))  # 秒
    MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
    
    # ===== 模型配置 =====
    ALIBABA_VISION_MODEL = os.getenv("ALIBABA_VISION_MODEL", "qwen3-vl-8b-thinking")
    ALIBABA_REASONING_MODEL = os.getenv("ALIBABA_REASONING_MODEL", "qwen2.5-7b-instruct")
    OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")
    OLLAMA_REASONING_MODEL = os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:7b")

    
    # ===== 知识库配置 =====
    KB_SIMILARITY_THRESHOLD = float(os.getenv("KB_SIMILARITY_THRESHOLD", "0.3"))
    KB_RETRIEVAL_TOP_K = int(os.getenv("KB_RETRIEVAL_TOP_K", "3"))
    KB_CHUNK_SIZE = 500  # 知识库分块大小
    KB_INDEX_UPDATE_THRESHOLD = int(os.getenv("KB_INDEX_UPDATE_THRESHOLD", "20"))
    
    # ===== 报警配置 =====
    ALARM_CONFIDENCE_THRESHOLD = float(os.getenv("ALARM_CONFIDENCE_THRESHOLD", "0.6"))
    
    # ===== 报警声音配置 =====
    ALARM_SOUNDS: Dict[str, str] = {
        "一般": str(SOUND_DIR / "normal.mp3"),
        "严重": str(SOUND_DIR / "severe.mp3"),
        "紧急": str(SOUND_DIR / "critical.mp3"),
    }
    
    # ===== 摄像头配置 =====
    RTSP_BUFFER_SIZE = 1024000
    RTSP_MAX_DELAY = 500000
    RTSP_TIMEOUT = 5000000
    
    # ===== 日志配置 =====
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ===== 调试配置 =====
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"


# 创建全局配置实例
config = Config()