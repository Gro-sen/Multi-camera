"""
视频流服务
"""
import time
import cv2
from typing import Generator
from app.core import get_logger, state

logger = get_logger(__name__)


class StreamService:
    """视频流服务"""
    
    @staticmethod
    def generate_frames(camera_id: str) -> Generator[bytes, None, None]:
        """生成视频流帧"""
        while True:
            try:
                # 使用双缓冲读取
                frame = state.get_buffered_frame(camera_id)
                if frame is None:
                    time.sleep(0.03)
                    continue
                
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                
                time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"视频流生成异常: {e}")
                time.sleep(0.1)
                continue