"""
报警服务层
"""
import threading
import os
from typing import Optional
from datetime import datetime
from app.core import get_logger, config, state
from app.models.types import RecognitionRecord, AlarmDecision

logger = get_logger(__name__)


class AlarmService:
    """报警服务"""
    
    ALARM_LEVEL_PRIORITY = {
        "无": 0,
        "一般": 1,
        "严重": 2,
        "紧急": 3,
    }
    
    def __init__(self):
        self.sound_lock = threading.Lock()
        self._sound_available = self._check_sound_module()
    
    def _check_sound_module(self) -> bool:
        """检查声音模块是否可用"""
        try:
            import playsound
            return True
        except ImportError:
            logger.warning("playsound模块不可用，告警声音功能将被禁用")
            return False
    
    def save_alarm_image(self, frame, alarm_level: str, case_id: Optional[str] = None, camera_id: Optional[str] = None) -> Optional[str]:
        """保存报警图片"""
        if frame is None or frame.size == 0:
            return None
        
        try:
            import cv2
            
            level_map = {"一般": "normal", "严重": "severe", "紧急": "critical"}
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            camera_tag = f"_{camera_id}" if camera_id else ""
            if case_id:
                filename = f"{case_id}{camera_tag}_{level_map.get(alarm_level, 'unknown')}.jpg"
            else:
                filename = f"{ts}{camera_tag}_{level_map.get(alarm_level, 'unknown')}.jpg"
            
            path = config.ALARM_DIR / filename
            cv2.imwrite(str(path), frame)
            logger.info(f"报警图片已保存: {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"保存报警图片失败: {e}", exc_info=True)
            return None
    
    def play_alarm_sound(self, alarm_level: str) -> None:
        """播放报警声音"""
        if not self._sound_available:
            return
        
        sound_path = config.ALARM_SOUNDS.get(alarm_level)
        if not sound_path or not os.path.exists(sound_path):
            logger.warning(f"报警声音文件不存在: {sound_path}")
            return
        
        def _play():
            with self.sound_lock:
                try:
                    from playsound import playsound
                    playsound(sound_path)
                    logger.debug(f"报警声音已播放: {alarm_level}")
                except Exception as e:
                    logger.warning(f"报警声音播放失败: {e}")
        
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
    
    def record_alarm(self, record: RecognitionRecord) -> None:
        """记录报警信息"""
        state.add_recognition_result(record.dict())
        logger.info(f"报警记录已添加: {record.alarm_level}")
        
        # 如果有警报，播放声音
        if record.is_alarm == "是" and record.alarm_level != "无":
            self.play_alarm_sound(record.alarm_level)
    
    def broadcast_alarm(self, record: RecognitionRecord) -> None:
        """广播报警信息到WebSocket"""
        message = {
            "type": "alarm" if record.is_alarm == "是" else "normal",
            "timestamp": record.timestamp,
            "is_alarm": record.is_alarm,
            "alarm_level": record.alarm_level,
            "alarm_reason": record.alarm_reason,
            "confidence": record.confidence,
            "camera_id": record.camera_id,
        }
        state.queue_broadcast_message(message)

    def record_inference_time(self, final_decision, elapsed: float) -> None:
        """记录推理耗时（占位实现，便于后续统计）"""
        try:
            # 这里可以接入日志/指标系统
            logger.info(f"推理耗时: {elapsed:.3f}s, 报警: {final_decision.get('is_alarm')}, 等级: {final_decision.get('alarm_level')}")
        except Exception:
            pass