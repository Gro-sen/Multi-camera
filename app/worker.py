"""
推理工作循环 - 后台持续执行推理任务
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any
from app.core import get_logger, config, state
from app.services.inference import InferenceService

logger = get_logger(__name__)


class InferenceWorker:
    """推理工作线程管理"""
    
    def __init__(self):
        self.inference_service = InferenceService()
        self.is_running = False  # 初始为 False
        self._stop_event = threading.Event()  # 停止事件
        self._worker_thread = None
        self._executor: ThreadPoolExecutor = None
    
    def start(self) -> None:
        """启动推理工作线程"""
        if self.is_running:
            logger.warning("推理工作线程已在运行")
            return

        self.is_running = True
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_INFERENCES)
        self._worker_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="InferenceWorker-Batch"
        )
        self._worker_thread.start()
        logger.info("推理工作线程已启动")
    
    def stop(self) -> None:
        """停止推理工作线程"""
        logger.info("正在停止推理工作线程...")
        self.is_running = False
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
            if self._worker_thread.is_alive():
                logger.warning("推理线程未能在3秒内停止")

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

        logger.info("推理工作线程已完全停止")

    def _collect_due_frames(self) -> List[Tuple[str, Any]]:
        """收集需要推理的帧"""
        camera_ids = state.get_camera_ids()
        if not camera_ids:
            camera_ids = [
                camera.get("id")
                for camera in config.CAMERA_SOURCES
                if camera.get("id")
            ]

        if not camera_ids:
            return []

        now = time.time()
        tasks: List[Tuple[str, any]] = []
        for camera_id in camera_ids:
            last_infer_time = state.get_last_infer_time(camera_id)
            if now - last_infer_time < config.INFER_INTERVAL:
                continue

            frame = state.get_frame(camera_id)
            if frame is None:
                continue

            state.update_infer_time(camera_id, now)
            tasks.append((camera_id, frame))

        return tasks

    def _broadcast_batch(self, records: List[object], batch_timestamp: float) -> None:
        """批量广播推理结果"""
        message = {
            "type": "batch",
            "timestamp": batch_timestamp,
            "data": [record.dict() for record in records]
        }
        state.queue_broadcast_message(message)

    def _inference_loop(self) -> None:
        """推理循环"""
        logger.info("推理循环开始运行")
        
        try:
            while self.is_running and not self._stop_event.is_set():
                try:
                    tasks = self._collect_due_frames()
                    if not tasks:
                        time.sleep(0.05)
                        continue

                    if not self._executor:
                        time.sleep(0.05)
                        continue

                    futures = [
                        self._executor.submit(
                            self.inference_service.infer,
                            frame,
                            camera_id=camera_id,
                            broadcast=False
                        )
                        for camera_id, frame in tasks
                    ]

                    records = []
                    for future in as_completed(futures):
                        record = future.result()
                        if record:
                            records.append(record)
                            logger.info(f"推理完成: {record.camera_id} {record.alarm_level}级警报")

                    if records:
                        self._broadcast_batch(records, time.time())

                except Exception as e:
                    logger.error(f"推理循环异常: {e}", exc_info=True)
                    if not self._stop_event.is_set():
                        time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("推理线程收到中断信号")
        finally:
            logger.info("推理循环已退出")