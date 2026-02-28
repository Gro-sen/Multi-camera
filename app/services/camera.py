"""
摄像头服务层
"""
import cv2
import threading
import time
from typing import Optional, Dict
from app.core import get_logger, config, state
from app.models.types import CameraStats
from app.core.exceptions import CameraException

logger = get_logger(__name__)


class RTSPMonitor:
    """RTSP连接监控器"""
    
    def __init__(self):
        self.connection_start_time: Optional[float] = None
        self.frames_received: int = 0
        self.connection_errors: int = 0
        self.last_error_time: Optional[float] = None
        self.last_error_message: Optional[str] = None
        self.last_frame_time: Optional[float] = None
    
    def on_connection_start(self) -> None:
        """连接开始"""
        self.connection_start_time = time.time()
        self.frames_received = 0
        logger.info(f"RTSP连接开始")
    
    def on_frame_received(self) -> None:
        """帧接收"""
        self.frames_received += 1
        self.last_frame_time = time.time()
        
        # 每30帧打印一次统计
        if self.frames_received % 30 == 0:
            uptime = time.time() - self.connection_start_time if self.connection_start_time else 0
            fps = self.frames_received / uptime if uptime > 0 else 0
            logger.debug(f"RTSP状态: {self.frames_received}帧, {fps:.2f}fps, 运行时间{uptime:.0f}s")
    
    def on_error(self, error_msg: str) -> None:
        """错误处理"""
        self.connection_errors += 1
        self.last_error_time = time.time()
        self.last_error_message = error_msg
        logger.error(f"RTSP错误 #{self.connection_errors}: {error_msg}")
    
    def get_stats(self) -> CameraStats:
        """获取统计信息"""
        uptime = time.time() - self.connection_start_time if self.connection_start_time else 0
        fps = self.frames_received / uptime if uptime > 0 else 0
        
        return CameraStats(
            is_connected=self.frames_received > 0,
            frames_received=self.frames_received,
            connection_errors=self.connection_errors,
            fps=fps,
            uptime_seconds=uptime,
            last_frame_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_frame_time)) if self.last_frame_time else None,
            last_error_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_error_time)) if self.last_error_time else None,
            last_error_message=self.last_error_message,
        )


class CameraWorker:
    """单路摄像头采集"""

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.monitor = RTSPMonitor()
        self.is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _create_capture(self) -> cv2.VideoCapture:
        """创建RTSP捕获对象"""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        return cap

    def start(self) -> None:
        """启动摄像头采集"""
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=False,
            name=f"CameraWorker-{self.camera_id}"
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """捕获循环"""
        try:
            self.cap = self._create_capture()
            self.monitor.on_connection_start()
            state.register_camera(self.camera_id)

            while self.is_running and not self._stop_event.is_set():
                ret, frame = self.cap.read()

                if not ret:
                    self.monitor.on_error("无法读取帧")
                    time.sleep(1)
                    continue

                self.monitor.on_frame_received()
                state.update_frame(self.camera_id, frame)
                state.swap_buffers(self.camera_id)
                time.sleep(0.01)

        except Exception as e:
            self.monitor.on_error(str(e))
            logger.error(f"摄像头捕获异常: {e}", exc_info=True)
        finally:
            if self.cap:
                self.cap.release()
            logger.info(f"摄像头捕获线程已退出: {self.camera_id}")

    def stop(self) -> None:
        """停止摄像头采集"""
        self.is_running = False
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=5)

    def get_stats(self) -> CameraStats:
        """获取统计信息"""
        return self.monitor.get_stats()


class CameraService:
    """摄像头服务"""

    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}
        self.is_running = False

        for camera in config.CAMERA_SOURCES:
            camera_id = camera.get("id")
            rtsp_url = camera.get("rtsp_url")
            if not camera_id or not rtsp_url:
                continue
            self.workers[camera_id] = CameraWorker(camera_id, rtsp_url)

    def start(self) -> None:
        """启动摄像头服务"""
        if self.is_running:
            logger.warning("摄像头服务已在运行")
            return

        self.is_running = True
        for camera_id, worker in self.workers.items():
            worker.start()
            logger.info(f"摄像头已启动: {camera_id}")

    def stop(self) -> None:
        """停止摄像头服务"""
        self.is_running = False
        for worker in self.workers.values():
            worker.stop()
        logger.info("摄像头服务已停止")

    def get_stats(self, camera_id: str) -> Optional[CameraStats]:
        """获取指定摄像头统计信息"""
        worker = self.workers.get(camera_id)
        if worker is None:
            return None
        return worker.get_stats()

    def get_all_stats(self) -> Dict[str, CameraStats]:
        """获取全部摄像头统计信息"""
        return {camera_id: worker.get_stats() for camera_id, worker in self.workers.items()}

    def get_current_frame(self, camera_id: str) -> Optional[bytes]:
        """获取当前帧（用于视频流）"""
        frame = state.get_frame(camera_id)
        if frame is None:
            return None

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            return buffer.tobytes()
        return None