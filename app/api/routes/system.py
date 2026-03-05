"""
系统状态相关路由
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import psutil
import time
import os
import signal  
from app.core import get_logger, config, state
from app.services.camera import CameraService

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

# 全局摄像头服务实例
_camera_service: CameraService = None


def get_camera_service() -> CameraService:
    """获取摄像头服务（延迟初始化）"""
    global _camera_service
    if _camera_service is None:
        _camera_service = CameraService()
    return _camera_service


@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取系统信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # 获取摄像头状态
        camera_service = get_camera_service()
        camera_stats_map = camera_service.get_all_stats()
        camera_stats = {camera_id: stats.dict() for camera_id, stats in camera_stats_map.items()}
        
        # 获取知识库状态
        kb_stats = {}
        try:
            from kb import kb
            kb_stats = kb.get_statistics()
        except Exception as e:
            logger.warning(f"获取知识库状态失败: {e}")
            kb_stats = {"status": "unavailable"}
        
        # 获取报警统计
        results = state.get_recognition_results(limit=10000)
        alarms_count = len([r for r in results if r.get("is_alarm") == "是"])
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        alarms_today = len([
            r for r in results 
            if r.get("is_alarm") == "是" and 
            datetime.fromisoformat(r.get("timestamp", "")) >= today_start
        ])
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
            },
            "camera": camera_stats,
            "knowledge_base": kb_stats,
            "alarms": {
                "total": len(results),
                "alarms_count": alarms_count,
                "alarms_today": alarms_today,
            }
        }
        
        return status
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统状态失败")


@router.get("/config")
async def get_config():
    """获取系统配置"""
    try:
        provider = os.getenv("MODEL_PROVIDER", "aliyun").lower()
        if provider == "ollama":
            vision_model = config.OLLAMA_VISION_MODEL
            reasoning_model = config.OLLAMA_REASONING_MODEL
        else:
            vision_model = config.ALIBABA_VISION_MODEL
            reasoning_model = config.ALIBABA_REASONING_MODEL

        return {
            "rtsp_url": config.RTSP_URL[:30] + "***" if len(config.RTSP_URL) > 30 else config.RTSP_URL,
            "camera_sources": [
                {
                    "id": camera.get("id"),
                    "name": camera.get("name"),
                    "rtsp_url": (camera.get("rtsp_url", "")[:30] + "***")
                    if len(camera.get("rtsp_url", "")) > 30
                    else camera.get("rtsp_url", "")
                }
                for camera in config.CAMERA_SOURCES
            ],
            "model_provider": provider,
            "infer_interval": config.INFER_INTERVAL,
            "vision_model": vision_model,
            "reasoning_model": reasoning_model,
            "kb_similarity_threshold": config.KB_SIMILARITY_THRESHOLD,
            "kb_retrieval_top_k": config.KB_RETRIEVAL_TOP_K,
            "alarm_confidence_threshold": config.ALARM_CONFIDENCE_THRESHOLD,
        }
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail="获取配置失败")


@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查关键组件
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "camera": "unknown",
                "models": "unknown",
                "knowledge_base": "unknown",
            }
        }
        
        # 检查摄像头
        try:
            camera_service = get_camera_service()
            stats_map = camera_service.get_all_stats()
            has_frames = any(stats.frames_received > 0 for stats in stats_map.values())
            health["components"]["camera"] = "healthy" if has_frames else "unhealthy"
        except:
            health["components"]["camera"] = "error"
        
        # 检查知识库
        try:
            from kb import kb
            kb_stats = kb.get_statistics()
            health["components"]["knowledge_base"] = "healthy" if kb_stats.get("status") == "ready" else "degraded"
        except:
            health["components"]["knowledge_base"] = "error"
        
        # 检查模型
        try:
            from app.models import VisionModelFactory, ReasoningModelFactory
            vision_model = VisionModelFactory.get_default_model()
            reasoning_model = ReasoningModelFactory.get_default_model()
            health["components"]["models"] = "healthy"
        except:
            health["components"]["models"] = "error"
        
        # 总体状态
        if any(v == "error" for v in health["components"].values()):
            health["status"] = "degraded"
        elif any(v == "unhealthy" for v in health["components"].values()):
            health["status"] = "degraded"
        
        return health
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.post("/restart-camera")
async def restart_camera():
    """重启摄像头"""
    try:
        camera_service = get_camera_service()
        camera_service.stop()
        camera_service.start()
        return {"status": "success", "message": "摄像头已重启"}
    except Exception as e:
        logger.error(f"重启摄像头失败: {e}")
        raise HTTPException(status_code=500, detail="重启摄像头失败")


@router.get("/cameras")
async def get_cameras():
    """获取摄像头列表"""
    try:
        return {
            "count": len(config.CAMERA_SOURCES),
            "data": [
                {
                    "id": camera.get("id"),
                    "name": camera.get("name"),
                    "rtsp_url": (camera.get("rtsp_url", "")[:30] + "***")
                    if len(camera.get("rtsp_url", "")) > 30
                    else camera.get("rtsp_url", "")
                }
                for camera in config.CAMERA_SOURCES
            ]
        }
    except Exception as e:
        logger.error(f"获取摄像头列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取摄像头列表失败")

@router.post("/shutdown")
async def shutdown_system():
    """关闭系统（需要管理员权限）"""
    try:
        logger.info("收到系统关闭请求")
        
        # 设置停止标志
        state.is_running = False
        
        # 获取当前进程ID
        pid = os.getpid()
        
        # 发送SIGTERM信号（优雅终止）
        os.kill(pid, signal.SIGTERM)
        
        return {
            "status": "success",
            "message": "系统正在关闭...",
            "pid": pid
        }
    except Exception as e:
        logger.error(f"关闭系统失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))