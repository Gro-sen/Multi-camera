"""
全局应用状态管理 (单例模式)
"""
import threading
import queue
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from app.core import config as app_config

class FrameBuffer:
    """双缓冲类，减少锁竞争"""
    
    def __init__(self):
        self.front_buffer: Optional[np.ndarray] = None
        self.back_buffer: Optional[np.ndarray] = None
        self.lock = threading.Lock()
    
    def write(self, frame: np.ndarray) -> None:
        """写入新帧到后端缓冲"""
        with self.lock:
            self.back_buffer = frame
    
    def read(self) -> Optional[np.ndarray]:
        """读取前端缓冲的帧"""
        with self.lock:
            if self.front_buffer is not None:
                return self.front_buffer.copy()
            return None
    
    def swap(self) -> None:
        """交换前后缓冲区"""
        with self.lock:
            self.front_buffer = self.back_buffer


class AppState:
    """全局应用状态管理（单例）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # ===== 帧管理 =====
        self.frame_buffers: Dict[str, FrameBuffer] = {}
        self.latest_frame_lock = threading.Lock()
        self.latest_frames: Dict[str, np.ndarray] = {}
        self.camera_ids: List[str] = []
        
        # ===== 推理管理 =====
        self.inference_lock = threading.Lock()
        self.inference_semaphore = threading.Semaphore(app_config.MAX_CONCURRENT_INFERENCES)
        self.last_infer_times: Dict[str, float] = {}
        
        # ===== 通信 =====
        self.broadcast_queue: queue.Queue = queue.Queue()
        self.sound_lock = threading.Lock()
        
        # ===== 识别结果历史 =====
        self.recognition_results: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()
        
        # ===== WebSocket连接 =====
        self.ws_connections: List = []
        self.ws_lock = threading.Lock()
        
        # ===== 系统状态 =====
        self.is_running = True
        self.start_time = None
        
        self._initialized = True
    
    # ===== 帧操作 =====
    def register_camera(self, camera_id: str) -> None:
        """注册摄像头"""
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = FrameBuffer()
        if camera_id not in self.camera_ids:
            self.camera_ids.append(camera_id)
        if camera_id not in self.last_infer_times:
            self.last_infer_times[camera_id] = 0.0

    def update_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """更新指定摄像头帧"""
        self.register_camera(camera_id)
        with self.latest_frame_lock:
            self.latest_frames[camera_id] = frame
        self.frame_buffers[camera_id].write(frame)
    
    def get_buffered_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """从双缓冲区获取帧（供视频流使用）"""
        buffer = self.frame_buffers.get(camera_id)
        if buffer is None:
            return None
        return buffer.read()
    
    def swap_buffers(self, camera_id: str) -> None:
        """交换缓冲区"""
        buffer = self.frame_buffers.get(camera_id)
        if buffer is None:
            return
        buffer.swap()
    
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """获取当前帧（供推理使用）"""
        with self.latest_frame_lock:
            frame = self.latest_frames.get(camera_id)
            if frame is not None:
                return frame.copy()
            return None
    
    # ===== 推理管理 =====
    def acquire_inference_lock(self, timeout: float = 0.1) -> bool:
        """尝试获取推理锁"""
        return self.inference_semaphore.acquire(timeout=timeout)
    
    def release_inference_lock(self) -> None:
        """释放推理锁"""
        self.inference_semaphore.release()
    
    def update_infer_time(self, camera_id: str, timestamp: float) -> None:
        """更新最后推理时间"""
        with self.inference_lock:
            self.last_infer_times[camera_id] = timestamp
    
    def get_last_infer_time(self, camera_id: str) -> float:
        """获取最后推理时间"""
        with self.inference_lock:
            return self.last_infer_times.get(camera_id, 0.0)

    def get_camera_ids(self) -> List[str]:
        """获取已注册摄像头ID列表"""
        return list(self.camera_ids)
    
    # ===== 结果管理 =====
    def add_recognition_result(self, result: Dict[str, Any]) -> None:
        """添加识别结果"""
        with self.results_lock:
            self.recognition_results.append(result)
    
    def get_recognition_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取识别结果历史"""
        with self.results_lock:
            return self.recognition_results[-limit:]
    
    def clear_recognition_results(self) -> None:
        """清空识别结果"""
        with self.results_lock:
            self.recognition_results.clear()
    
    # ===== 通信管理 =====
    def queue_broadcast_message(self, message: Dict[str, Any]) -> None:
        """队列广播消息"""
        self.broadcast_queue.put(message)
    
    def get_broadcast_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取待广播消息"""
        try:
            return self.broadcast_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def broadcast_queue_task_done(self) -> None:
        """标记广播任务完成"""
        self.broadcast_queue.task_done()
    
    # ===== WebSocket管理 =====
    def register_ws_connection(self, ws) -> None:
        """注册WebSocket连接"""
        with self.ws_lock:
            self.ws_connections.append(ws)
    
    def unregister_ws_connection(self, ws) -> None:
        """取消注册WebSocket连接"""
        with self.ws_lock:
            if ws in self.ws_connections:
                self.ws_connections.remove(ws)
    
    def get_ws_connections(self) -> List:
        """获取所有WebSocket连接"""
        with self.ws_lock:
            return list(self.ws_connections)
    
    def get_ws_connection_count(self) -> int:
        """获取WebSocket连接数"""
        with self.ws_lock:
            return len(self.ws_connections)


# 创建全局单例
state = AppState()