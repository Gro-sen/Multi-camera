"""
系统状态相关路由
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import csv
import subprocess
import psutil
import time
import os
import signal  
import threading
from typing import Optional
from app.core import get_logger, config, state
from app.services.camera import CameraService

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

# 全局摄像头服务实例
_camera_service: CameraService = None
_MODEL_CLOSE_DELAY_SECONDS = 45


def get_camera_service() -> CameraService:
    """获取摄像头服务（延迟初始化）"""
    global _camera_service
    if _camera_service is not None:
        return _camera_service

    try:
        from app.main import lifecycle
        if getattr(lifecycle, "camera_service", None) is not None:
            _camera_service = lifecycle.camera_service
            return _camera_service
    except Exception as e:
        logger.debug(f"从 lifecycle 获取摄像头服务失败: {e}")

    _camera_service = CameraService()
    return _camera_service


def _safe_close_model(model) -> None:
    """尝试关闭模型客户端连接。"""
    try:
        if model is not None and hasattr(model, "close"):
            model.close()
    except Exception as e:
        logger.debug(f"关闭模型连接时异常: {e}")


def _schedule_close_model(model, delay_seconds: int = _MODEL_CLOSE_DELAY_SECONDS) -> None:
    """延迟关闭旧模型，避免切换瞬间并发请求使用已关闭客户端。"""
    if model is None:
        return

    def _close_later():
        try:
            _safe_close_model(model)
        except Exception:
            pass

    timer = threading.Timer(delay_seconds, _close_later)
    timer.daemon = True
    timer.start()


def _get_gpu_vram_status() -> dict:
    """获取 NVIDIA GPU 专用显存占用情况。"""
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        output = (completed.stdout or "").strip()
        if not output:
            return {
                "available": False,
                "status": "unavailable",
                "gpu_name": None,
                "vram_total_gb": None,
                "vram_used_gb": None,
                "vram_free_gb": None,
                "vram_percent": None,
            }

        row = next(csv.reader([output]))
        if len(row) < 4:
            return {
                "available": False,
                "status": "parse_error",
                "gpu_name": None,
                "vram_total_gb": None,
                "vram_used_gb": None,
                "vram_free_gb": None,
                "vram_percent": None,
            }

        gpu_name = row[0].strip()
        total_mb = float(row[1])
        used_mb = float(row[2])
        free_mb = float(row[3])
        total_gb = total_mb / 1024.0
        used_gb = used_mb / 1024.0
        free_gb = free_mb / 1024.0
        percent = (used_mb / total_mb * 100.0) if total_mb > 0 else None

        return {
            "available": True,
            "status": "ok",
            "gpu_name": gpu_name,
            "vram_total_gb": total_gb,
            "vram_used_gb": used_gb,
            "vram_free_gb": free_gb,
            "vram_percent": percent,
        }
    except FileNotFoundError:
        return {
            "available": False,
            "status": "nvidia-smi_not_found",
            "gpu_name": None,
            "vram_total_gb": None,
            "vram_used_gb": None,
            "vram_free_gb": None,
            "vram_percent": None,
        }
    except Exception as e:
        logger.debug(f"获取 GPU 显存状态失败: {e}")
        return {
            "available": False,
            "status": "error",
            "gpu_name": None,
            "vram_total_gb": None,
            "vram_used_gb": None,
            "vram_free_gb": None,
            "vram_percent": None,
        }


def _apply_model_env(provider: str, vision_model: Optional[str], reasoning_model: Optional[str]) -> None:
    provider = (provider or "").lower().strip()
    if provider not in {"aliyun", "ollama"}:
        raise ValueError("provider 必须是 aliyun 或 ollama")

    os.environ["MODEL_PROVIDER"] = provider

    if provider == "ollama":
        if vision_model:
            os.environ["OLLAMA_VISION_MODEL"] = vision_model.strip()
        if reasoning_model:
            os.environ["OLLAMA_REASONING_MODEL"] = reasoning_model.strip()
    else:
        if vision_model:
            os.environ["ALIBABA_VISION_MODEL"] = vision_model.strip()
        if reasoning_model:
            os.environ["ALIBABA_REASONING_MODEL"] = reasoning_model.strip()


def _reload_runtime_models() -> None:
    """重载全局模型并让推理服务使用新模型。"""
    from app.models.factory import create_models

    old_vision = state.vision_model
    old_reasoning = state.reasoning_model

    new_vision, new_reasoning = create_models()

    state.vision_model = new_vision
    state.reasoning_model = new_reasoning

    # 让已创建的摄像头服务实例切换到新模型
    try:
        from app.main import lifecycle
        worker = getattr(lifecycle, "inference_worker", None)
        services = getattr(worker, "services", None)
        if isinstance(services, dict):
            for svc in services.values():
                try:
                    svc.vision_model = state.vision_model
                    svc.reasoning_model = state.reasoning_model
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"刷新运行中服务模型引用失败: {e}")

    # 延迟关闭旧模型，给进行中的请求留出完成窗口，避免 "client has been closed"。
    _schedule_close_model(old_vision)
    _schedule_close_model(old_reasoning)


@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取系统信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        gpu_vram = _get_gpu_vram_status()
        
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
                "gpu_vram": gpu_vram,
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
            vision_model = os.getenv("OLLAMA_VISION_MODEL", config.OLLAMA_VISION_MODEL)
            reasoning_model = os.getenv("OLLAMA_REASONING_MODEL", config.OLLAMA_REASONING_MODEL)
        else:
            vision_model = os.getenv("ALIBABA_VISION_MODEL", config.ALIBABA_VISION_MODEL)
            reasoning_model = os.getenv("ALIBABA_REASONING_MODEL", config.ALIBABA_REASONING_MODEL)

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


@router.post("/model/switch")
async def switch_model(payload: dict):
    """运行时切换大模型提供方与模型名。"""
    try:
        provider = (payload.get("provider") or "").strip().lower()
        vision_model = (payload.get("vision_model") or "").strip() or None
        reasoning_model = (payload.get("reasoning_model") or "").strip() or None

        if provider not in {"aliyun", "ollama"}:
            raise HTTPException(status_code=400, detail="provider 必须是 aliyun 或 ollama")

        _apply_model_env(provider, vision_model, reasoning_model)
        _reload_runtime_models()

        if provider == "ollama":
            current_vision = os.getenv("OLLAMA_VISION_MODEL", config.OLLAMA_VISION_MODEL)
            current_reasoning = os.getenv("OLLAMA_REASONING_MODEL", config.OLLAMA_REASONING_MODEL)
        else:
            current_vision = os.getenv("ALIBABA_VISION_MODEL", config.ALIBABA_VISION_MODEL)
            current_reasoning = os.getenv("ALIBABA_REASONING_MODEL", config.ALIBABA_REASONING_MODEL)

        logger.info(
            "模型已切换: provider=%s vision=%s reasoning=%s",
            provider,
            current_vision,
            current_reasoning,
        )

        return {
            "status": "success",
            "message": "模型切换成功",
            "data": {
                "provider": provider,
                "vision_model": current_vision,
                "reasoning_model": current_reasoning,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型切换失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型切换失败: {e}")


@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查关键组件
        camera_service = get_camera_service()
        camera_stats_map = camera_service.get_all_stats()
        active_count = state.get_active_inferences_count()
        queue_backlog = state.broadcast_queue.qsize()
        global_latency = state.get_inference_latency_stats()

        cameras = []
        for camera_id, stats in camera_stats_map.items():
            worker = camera_service.workers.get(camera_id)
            latency_stats = state.get_inference_latency_stats(camera_id)
            cameras.append({
                "camera_id": camera_id,
                "is_running": worker.is_running if worker is not None else False,
                "analysis_enabled": state.is_camera_analysis_enabled(camera_id),
                "fps": stats.fps,
                "frame_delay_seconds": stats.frame_delay_seconds,
                "is_connected": stats.is_connected,
                "connection_status": stats.connection_status,
                "frames_received": stats.frames_received,
                "connection_errors": stats.connection_errors,
                "latency": latency_stats,
            })

        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "camera": "unknown",
                "models": "unknown",
                "knowledge_base": "unknown",
            }
            ,"metrics": {
                "active_tasks": active_count,
                "queue_backlog": queue_backlog,
                "inference_latency": global_latency,
                "cameras": cameras,
            }
        }
        
        # 检查摄像头
        try:
            has_frames = any(stats.frames_received > 0 for stats in camera_stats_map.values())
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
            # 使用启动阶段已加载的全局模型实例，避免健康检查再次触发模型工厂
            # （否则会打印“使用阿里云模型”的误导日志）
            if state.vision_model is not None and state.reasoning_model is not None:
                health["components"]["models"] = "healthy"
            else:
                health["components"]["models"] = "unhealthy"
        except Exception:
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
        camera_service = get_camera_service()
        status_map = camera_service.get_all_camera_status()
        return {
            "status": "success",
            "count": len(config.CAMERA_SOURCES),
            "data": [
                {
                    "id": camera.get("id"),
                    "name": camera.get("name"),
                    "rtsp_url": (camera.get("rtsp_url", "")[:30] + "***")
                    if len(camera.get("rtsp_url", "")) > 30
                    else camera.get("rtsp_url", ""),
                    "is_running": status_map.get(camera.get("id"), {}).get("is_running", False),
                    "analysis_enabled": status_map.get(camera.get("id"), {}).get("analysis_enabled", True),
                    "stats": status_map.get(camera.get("id"), {}).get("stats", {}),
                }
                for camera in config.CAMERA_SOURCES
            ]
        }
    except Exception as e:
        logger.error(f"获取摄像头列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取摄像头列表失败")


@router.post("/camera/{camera_id}/start")
async def start_camera(camera_id: str):
    """启动单路摄像头采集并恢复分析"""
    try:
        camera_service = get_camera_service()
        if not camera_service.start_camera(camera_id):
            raise HTTPException(status_code=404, detail="摄像头不存在")
        state.set_camera_analysis_enabled(camera_id, True)
        return {
            "status": "success",
            "message": f"摄像头 {camera_id} 已启动",
            "camera": camera_service.get_camera_status(camera_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动摄像头失败: {e}")
        raise HTTPException(status_code=500, detail="启动摄像头失败")


@router.post("/camera/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """停止单路摄像头采集并暂停分析"""
    try:
        camera_service = get_camera_service()
        if not camera_service.stop_camera(camera_id):
            raise HTTPException(status_code=404, detail="摄像头不存在")
        state.set_camera_analysis_enabled(camera_id, False)
        return {
            "status": "success",
            "message": f"摄像头 {camera_id} 已停止",
            "camera": camera_service.get_camera_status(camera_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止摄像头失败: {e}")
        raise HTTPException(status_code=500, detail="停止摄像头失败")

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