"""
报警相关路由
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from app.core import get_logger, state
from app.models.types import RecognitionRecord

logger = get_logger(__name__)

router = APIRouter(prefix="/alarms", tags=["alarms"])


@router.get("/history")
async def get_alarm_history(
    limit: int = Query(50, ge=1, le=500),
    alarm_level: Optional[str] = Query(None, description="筛选报警级别"),
    is_alarm_only: bool = Query(False, description="仅显示报警记录")
):
    """获取历史报警记录"""
    try:
        results = state.get_recognition_results(limit)
        
        # 筛选
        if is_alarm_only:
            results = [r for r in results if r.get("is_alarm") == "是"]
        
        if alarm_level:
            results = [r for r in results if r.get("alarm_level") == alarm_level]
        
        return {
            "total": len(results),
            "data": results
        }
    except Exception as e:
        logger.error(f"获取报警历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取报警历史失败")


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