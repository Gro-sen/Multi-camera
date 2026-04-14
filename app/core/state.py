"""
全局应用状态管理 (单例模式)
"""
import threading
import queue
from datetime import datetime
from collections import deque
from statistics import median
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
        self.latest_frame_timestamps: Dict[str, str] = {}
        self.camera_ids: List[str] = []
        self.camera_analysis_enabled: Dict[str, bool] = {}
        
        # ===== 推理管理 =====
        self.inference_lock = threading.Lock()
        self.vision_model = None
        self.reasoning_model = None
        self.face_service = None
        self.face_service_lock = threading.Lock()
        self.camera_inference_locks: Dict[str, threading.Semaphore] = {}
        self.last_infer_times: Dict[str, float] = {}
        self.active_inferences: Dict[str, float] = {}  # camera_id -> 开始推理的时间戳
        self.active_lock = threading.Lock()
        self.inference_latencies: Dict[str, deque] = {}
        self.latency_lock = threading.Lock()
        self.global_inference_latencies: deque = deque(maxlen=500)
        
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
        if camera_id not in self.camera_analysis_enabled:
            self.camera_analysis_enabled[camera_id] = True
        if camera_id not in self.last_infer_times:
            self.last_infer_times[camera_id] = 0.0
        if camera_id not in self.inference_latencies:
            self.inference_latencies[camera_id] = deque(maxlen=200)

    def update_frame(self, camera_id: str, frame: np.ndarray, frame_timestamp: Optional[str] = None) -> None:
        """更新指定摄像头帧"""
        self.register_camera(camera_id)
        with self.latest_frame_lock:
            self.latest_frames[camera_id] = frame
            self.latest_frame_timestamps[camera_id] = frame_timestamp or datetime.now().isoformat()
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

    def get_frame_with_timestamp(self, camera_id: str) -> tuple[Optional[np.ndarray], Optional[str]]:
        """原子获取当前帧及其采集时间（供推理使用）"""
        with self.latest_frame_lock:
            frame = self.latest_frames.get(camera_id)
            ts = self.latest_frame_timestamps.get(camera_id)
            if frame is not None:
                return frame.copy(), ts
            return None, ts
    
    # ===== 推理管理 =====
    def init_models(self) -> None:
        """系统启动时预加载模型（全局共享）"""
        if self.vision_model is not None and self.reasoning_model is not None:
            return  # 已初始化
        
        try:
            from app.models.factory import create_models
            self.vision_model, self.reasoning_model = create_models()
            from app.core import get_logger
            logger = get_logger(__name__)
            logger.info("✓ 全局模型已预加载并缓存")
        except Exception as e:
            from app.core import get_logger
            logger = get_logger(__name__)
            logger.error(f"模型预加载失败: {e}", exc_info=True)
            raise

    def init_face_service(self):
        """系统启动时预加载人脸识别服务（全局单例共享）"""
        if self.face_service is not None:
            return self.face_service

        with self.face_service_lock:
            if self.face_service is not None:
                return self.face_service

            from app.services.face_recognition import FaceRecognitionService

            self.face_service = FaceRecognitionService()
            from app.core import get_logger
            logger = get_logger(__name__)
            logger.info("✓ 全局人脸识别服务已初始化并缓存")
            return self.face_service
    
    def acquire_camera_lock(self, camera_id: str, timeout: float = 2.0) -> bool:
        """尝试获取摄像头推理锁（每摄像头独立）"""
        if camera_id not in self.camera_inference_locks:
            with self.inference_lock:
                if camera_id not in self.camera_inference_locks:
                    self.camera_inference_locks[camera_id] = threading.Semaphore(1)
        return self.camera_inference_locks[camera_id].acquire(timeout=timeout)
    
    def release_camera_lock(self, camera_id: str) -> None:
        """释放摄像头推理锁"""
        if camera_id in self.camera_inference_locks:
            try:
                self.camera_inference_locks[camera_id].release()
            except ValueError:
                pass  # 无需释放
    
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

    def set_camera_analysis_enabled(self, camera_id: str, enabled: bool) -> None:
        """设置单路摄像头是否参与推理"""
        with self.inference_lock:
            self.camera_analysis_enabled[camera_id] = enabled

    def is_camera_analysis_enabled(self, camera_id: str) -> bool:
        """检查单路摄像头是否参与推理"""
        with self.inference_lock:
            return self.camera_analysis_enabled.get(camera_id, True)

    def mark_inference_start(self, camera_id: str) -> None:
        """标记推理开始"""
        import time
        with self.active_lock:
            self.active_inferences[camera_id] = time.time()

    def mark_inference_end(self, camera_id: str) -> None:
        """标记推理结束"""
        with self.active_lock:
            self.active_inferences.pop(camera_id, None)

    def get_active_inferences_count(self) -> int:
        """获取当前活跃推理任务数"""
        with self.active_lock:
            return len(self.active_inferences)

    def get_active_inferences_info(self) -> str:
        """获取活跃推理任务信息"""
        import time
        with self.active_lock:
            if not self.active_inferences:
                return "无活跃推理"
            info_parts = []
            now = time.time()
            for cid, start_time in self.active_inferences.items():
                elapsed = now - start_time
                info_parts.append(f"{cid}({elapsed:.1f}s)")
            return f"活跃推理[count={len(self.active_inferences)}]: {', '.join(info_parts)}"

    def record_inference_latency(self, camera_id: Optional[str], elapsed: float) -> None:
        """记录推理耗时样本"""
        if elapsed is None:
            return
        elapsed = float(elapsed)
        if elapsed < 0:
            return

        with self.latency_lock:
            if camera_id:
                if camera_id not in self.inference_latencies:
                    self.inference_latencies[camera_id] = deque(maxlen=200)
                self.inference_latencies[camera_id].append(elapsed)
            self.global_inference_latencies.append(elapsed)

    def _percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])

        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        if lower_index == upper_index:
            return float(ordered[lower_index])

        lower_value = ordered[lower_index]
        upper_value = ordered[upper_index]
        return float(lower_value + (upper_value - lower_value) * (position - lower_index))

    def get_inference_latency_stats(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        """获取推理耗时统计"""
        with self.latency_lock:
            if camera_id:
                samples = list(self.inference_latencies.get(camera_id, []))
            else:
                samples = list(self.global_inference_latencies)

        if not samples:
            return {
                "count": 0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        total = sum(samples)
        return {
            "count": len(samples),
            "avg": total / len(samples),
            "p50": self._percentile(samples, 0.50),
            "p95": self._percentile(samples, 0.95),
            "min": min(samples),
            "max": max(samples),
        }
    
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