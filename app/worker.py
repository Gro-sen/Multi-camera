"""
推理工作循环 - 后台持续执行推理任务
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any, Dict

from app.core import get_logger, config, state
from app.services.inference import InferenceService
from app.models.factory import create_models

logger = get_logger(__name__)


class InferenceWorker:
    """推理工作线程管理"""
    
    def __init__(self):
        # 不再使用单一共享实例，按摄像头按需创建实例
        self.is_running = False
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread = None
        self._executor: ThreadPoolExecutor = None
        # camera_id -> InferenceService
        self.services: Dict[str, InferenceService] = {}

    def start(self) -> None:
        """启动推理工作线程并预热每个摄像头的模型实例（减少首帧卡顿）"""
        if self.is_running:
            logger.warning("推理工作线程已在运行")
            return

        self.is_running = True
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_INFERENCES)

        # 预热：按已注册的摄像头创建模型实例，避免首帧加载延迟
        try:
            camera_ids = [c.get("id") for c in config.CAMERA_SOURCES if c.get("id")]
            for cid in camera_ids:
                if cid:
                    try:
                        self._get_service_for_camera(cid)
                    except Exception as e:
                        logger.warning(f"为摄像头预热实例 {cid} 失败: {e}")
            logger.info("已为配置中的摄像头预热推理实例")
        except Exception as e:
            logger.debug(f"预热过程异常: {e}", exc_info=True)

        self._worker_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="InferenceWorker-Batch"
        )
        self._worker_thread.start()
        logger.info("推理工作线程已启动")
        logger.info(f"[diagnostic] ThreadPoolExecutor max_workers={config.MAX_CONCURRENT_INFERENCES}")

    def stop(self) -> None:
        """停止推理工作线程并清理所有按摄像头创建的实例"""
        logger.info("正在停止推理工作线程...")
        self.is_running = False
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
            if self._worker_thread.is_alive():
                logger.warning("推理线程未能在3秒内停止")

        if self._executor:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                logger.debug("关闭 ThreadPoolExecutor 时异常", exc_info=True)
            self._executor = None

        # 清理按摄像头创建的服务实例（尝试调用 close/shutdown/stop）
        try:
            for cid, svc in list(self.services.items()):
                try:
                    if hasattr(svc, "close"):
                        svc.close()  # 如果 InferenceService 提供 close，优先调用
                    elif hasattr(svc, "shutdown"):
                        svc.shutdown()
                    elif hasattr(svc, "stop"):
                        svc.stop()
                except Exception:
                    logger.debug(f"清理服务 {cid} 时发生异常", exc_info=True)
            self.services.clear()
            logger.info("已清理所有按摄像头创建的推理实例")
        except Exception:
            logger.debug("停止时清理服务发生异常", exc_info=True)

        logger.info("推理工作线程已完全停止")

    def _collect_due_frames(self) -> List[Tuple[str, Any]]:
        """
        收集需要推理的帧（基于每个摄像头的上次推理时间）。

        行为：
        1) 首先收集满足时间间隔 `INFER_INTERVAL` 的摄像头；
        2) 若数量少于可并发槽位（`MAX_CONCURRENT_INFERENCES`），则从剩余摄像头补充（忽略间隔），直到达到槽位上限或无更多帧。
        3) 在将摄像头加入任务队列时立即更新其 last_infer_time（标记为已采样），确保每个摄像头按自己的间隔调度，和推理耗时解耦。
        """
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
        tasks: List[Tuple[str, Any]] = []

        # 先收集已到达间隔的摄像头
        for camera_id in camera_ids:
            last_infer_time = state.get_last_infer_time(camera_id)
            if now - last_infer_time < config.INFER_INTERVAL:
                continue

            frame = state.get_frame(camera_id)
            if frame is None:
                continue

            # 立即标记该摄像头的采样时间（使其下次认为已采样）
            state.update_infer_time(camera_id, now)
            tasks.append((camera_id, frame))
            # 如果已达并发上限则返回
            if len(tasks) >= config.MAX_CONCURRENT_INFERENCES:
                return tasks

        # 若不足并发上限，则从剩余摄像头补采（忽略间隔），以充分利用并发槽位
        if len(tasks) < config.MAX_CONCURRENT_INFERENCES:
            for camera_id in camera_ids:
                if any(t[0] == camera_id for t in tasks):
                    continue
                frame = state.get_frame(camera_id)
                if frame is None:
                    continue
                state.update_infer_time(camera_id, now)
                tasks.append((camera_id, frame))
                if len(tasks) >= config.MAX_CONCURRENT_INFERENCES:
                    break

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
        """主推理循环：收集任务，使用线程池并发执行各摄像头对应的 InferenceService"""
        logger.info("[diagnostic] 推理循环开始运行")

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

                    logger.info(f"[diagnostic] 收集到 {len(tasks)} 个任务，摄像头列表: {[t[0] for t in tasks]}，已创建服务数: {len(self.services)}")
                    futures = []
                    for camera_id, frame in tasks:
                        svc = self._get_service_for_camera(camera_id)
                        futures.append(self._executor.submit(svc.infer, frame, camera_id=camera_id, broadcast=False))

                    records = []
                    for future in as_completed(futures):
                        try:
                            record = future.result()
                        except Exception as e:
                            logger.error(f"推理任务异常: {e}", exc_info=True)
                            record = None
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

    def _get_service_for_camera(self, camera_id: str) -> InferenceService:
        """按摄像头按需创建独立的 InferenceService 实例并缓存"""
        if camera_id in self.services:
            return self.services[camera_id]

        try:
            vision_model, reasoning_model = create_models()
            svc = InferenceService(
                vision_model=vision_model,
                reasoning_model=reasoning_model,
                kb=state.kb
            )
            self.services[camera_id] = svc
            logger.info(f"为摄像头 {camera_id} 创建独立推理实例")
            logger.info(f"[diagnostic] 已创建服务：camera={camera_id} current_services={list(self.services.keys())}")
            return svc
        except Exception as e:
            logger.error(f"为摄像头 {camera_id} 创建模型实例失败: {e}", exc_info=True)
            # 回退：尝试使用共享实例（懒创建）
            if "shared" not in self.services:
                try:
                    self.services["shared"] = InferenceService(kb=state.kb)
                except Exception as ex:
                    logger.error(f"创建共享推理实例失败: {ex}", exc_info=True)
                    raise
            return self.services["shared"]