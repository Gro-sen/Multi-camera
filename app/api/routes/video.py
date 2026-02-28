"""
视频相关路由
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core import get_logger, state
from app.services.stream import StreamService

logger = get_logger(__name__)

router = APIRouter(prefix="/video", tags=["video"])


@router.get("/feed/{camera_id}")
async def get_video_feed(camera_id: str):
    """获取视频流"""
    try:
        return StreamingResponse(
            StreamService.generate_frames(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        logger.error(f"视频流生成失败: {e}")
        raise HTTPException(status_code=500, detail="视频流生成失败")


@router.get("/frame/{camera_id}")
async def get_current_frame(camera_id: str):
    """获取当前帧（JPEG格式）"""
    try:
        # 使用双缓冲读取
        frame = state.get_buffered_frame(camera_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="当前没有可用的帧")
        
        import cv2
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            raise HTTPException(status_code=500, detail="帧编码失败")
        
        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取当前帧失败: {e}")
        raise HTTPException(status_code=500, detail="获取帧失败")