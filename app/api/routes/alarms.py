"""
报警相关路由
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import os

from app.core import config
from app.core import get_logger, state
from app.models.types import RecognitionRecord

logger = get_logger(__name__)

router = APIRouter(prefix="/alarms", tags=["alarms"])


def _build_alarm_image_url(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    filename = Path(str(image_path)).name
    if not filename:
        return None
    return f"/api/alarms/image/{filename}"


@router.get("/history")
async def get_alarm_history(
    limit: int = Query(50, ge=1, le=500),
    camera_id: Optional[str] = Query(None, description="筛选摄像头ID"),
    alarm_level: Optional[str] = Query(None, description="筛选报警级别"),
    is_alarm_only: bool = Query(False, description="仅显示报警记录")
):
    """获取历史报警记录"""
    try:
        results = state.get_recognition_results(limit)
        
        # 筛选
        if is_alarm_only:
            results = [r for r in results if r.get("is_alarm") == "是"]

        if camera_id:
            results = [r for r in results if r.get("camera_id") == camera_id]
        
        if alarm_level:
            results = [r for r in results if r.get("alarm_level") == alarm_level]

        enriched = []
        for record in results:
            item = dict(record)
            item["image_url"] = _build_alarm_image_url(item.get("image_path"))
            enriched.append(item)
        
        return {
            "total": len(enriched),
            "data": enriched
        }
    except Exception as e:
        logger.error(f"获取报警历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取报警历史失败")


@router.get("/image/{filename}")
async def get_alarm_image(filename: str):
    """获取报警图片文件"""
    try:
        safe_name = Path(filename).name
        if not safe_name:
            raise HTTPException(status_code=404, detail="图片不存在")

        image_path = (config.ALARM_DIR / safe_name).resolve()
        alarm_dir = config.ALARM_DIR.resolve()
        if alarm_dir not in image_path.parents and image_path != alarm_dir:
            raise HTTPException(status_code=400, detail="非法图片路径")
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")

        return FileResponse(str(image_path), media_type="image/jpeg", filename=safe_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取报警图片失败: {e}")
        raise HTTPException(status_code=500, detail="获取报警图片失败")


@router.get("/statistics")
async def get_alarm_statistics(
    hours: int = Query(24, ge=1, le=720, description="过去N小时的统计")
):
    """获取报警统计信息"""
    try:
        results = state.get_recognition_results(limit=10000)
        
        # 计算时间范围
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 统计
        stats = {
            "total": 0,
            "alarmed": 0,
            "normal": 0,
            "by_level": {
                "无": 0,
                "一般": 0,
                "严重": 0,
                "紧急": 0,
            },
            "average_confidence": 0.0,
        }
        
        valid_records = []
        confidence_sum = 0
        
        for record in results:
            try:
                timestamp = datetime.fromisoformat(record.get("timestamp", ""))
                if timestamp < cutoff_time:
                    continue
                
                valid_records.append(record)
                stats["total"] += 1
                
                is_alarm = record.get("is_alarm", "否")
                alarm_level = record.get("alarm_level", "无")
                confidence = record.get("confidence", 0.0)
                
                if is_alarm == "是":
                    stats["alarmed"] += 1
                else:
                    stats["normal"] += 1
                
                if alarm_level in stats["by_level"]:
                    stats["by_level"][alarm_level] += 1
                
                confidence_sum += confidence
            except:
                continue
        
        if stats["total"] > 0:
            stats["average_confidence"] = confidence_sum / stats["total"]
        
        return stats
    except Exception as e:
        logger.error(f"获取报警统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取报警统计失败")


@router.delete("/clear")
async def clear_alarm_history():
    """清空报警历史"""
    try:
        state.clear_recognition_results()
        return {"status": "success", "message": "报警历史已清空"}
    except Exception as e:
        logger.error(f"清空报警历史失败: {e}")
        raise HTTPException(status_code=500, detail="清空报警历史失败")