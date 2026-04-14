"""
人脸识别服务（白名单匹配）
"""
import os
import ctypes
import re
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from app.core import get_logger, config
from app.models.types import FaceRecognitionResult

logger = get_logger(__name__)


class FaceRecognitionService:
    """基于 insightface 的白名单人脸匹配服务"""

    def __init__(self) -> None:
        self.enabled = config.FACE_RECOGNITION_ENABLED
        self.threshold = float(config.FACE_MATCH_THRESHOLD)
        self._app = None
        self._whitelist: List[Tuple[str, np.ndarray]] = []
        self._init_error: Optional[str] = None

        if not self.enabled:
            logger.info("人脸识别已禁用（FACE_RECOGNITION_ENABLED=false）")
            return

        self._initialize_model()
        if self._app is not None:
            self.refresh_whitelist()

    def _initialize_model(self) -> None:
        try:
            self._bootstrap_cuda_runtime_paths()
            from insightface.app import FaceAnalysis
        except Exception as e:
            self._init_error = f"insightface导入失败: {e}"
            logger.warning(self._init_error)
            return

        try:
            det_size = (config.FACE_DETECTION_SIZE, config.FACE_DETECTION_SIZE)
            if config.DEBUG:
                self._app = FaceAnalysis(
                    name=config.FACE_MODEL_NAME,
                    providers=config.FACE_PROVIDERS,
                )
                self._app.prepare(ctx_id=config.FACE_CTX_ID, det_size=det_size)
            else:
                # 降噪：屏蔽 insightface/onnxruntime 初始化时的标准输出和错误输出
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self._app = FaceAnalysis(
                        name=config.FACE_MODEL_NAME,
                        providers=config.FACE_PROVIDERS,
                    )
                    self._app.prepare(ctx_id=config.FACE_CTX_ID, det_size=det_size)
            logger.info(
                "人脸识别模型已加载: model=%s providers=%s",
                config.FACE_MODEL_NAME,
                config.FACE_PROVIDERS,
            )
        except Exception as e:
            self._init_error = f"人脸识别模型初始化失败: {e}"
            logger.warning(self._init_error)
            self._app = None

    def _bootstrap_cuda_runtime_paths(self) -> None:
        """CUDA DLL 路径启动已在 app/__init__.py 中提前处理。此处仅做健康检查。"""
        if os.name != "nt":
            return

        # 健康检查：关键 CUDA 依赖
        required = ["cublasLt64_12.dll", "cufft64_11.dll"]
        missing = []
        for dll in required:
            try:
                ctypes.WinDLL(dll)
            except Exception:
                missing.append(dll)
        if missing:
            logger.warning(
                "检测到 CUDA 关键依赖缺失: %s。请补齐对应 nvidia-*-cu12 包。",
                missing,
            )

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(emb))
        if norm == 0.0:
            return emb
        return emb / norm

    def _extract_main_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._app is None:
            return None

        faces = self._app.get(image)
        if not faces:
            return None

        # 选取面积最大的脸，避免远处小脸干扰
        def face_area(face) -> float:
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) != 4:
                return 0.0
            return max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

        main_face = max(faces, key=face_area)
        embedding = getattr(main_face, "embedding", None)
        if embedding is None:
            return None

        return self._normalize(np.asarray(embedding, dtype=np.float32))

    def refresh_whitelist(self) -> None:
        """重新加载白名单人脸"""
        self._whitelist = []

        if self._app is None:
            return

        whitelist_dir = Path(config.FACE_WHITELIST_DIR)
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(whitelist_dir.glob(ext))

        grouped_embeddings = {}

        for path in sorted(image_files):
            image = cv2.imread(str(path))
            if image is None:
                logger.warning("白名单图片读取失败: %s", path)
                continue

            emb = self._extract_main_embedding(image)
            if emb is None:
                logger.warning("白名单图片未检测到有效人脸: %s", path)
                continue

            identity = self._identity_from_stem(path.stem)
            grouped_embeddings.setdefault(identity, []).append(emb)

        for identity, embeddings in grouped_embeddings.items():
            # 多模板归并：对同一身份样本向量取均值并再归一化
            merged = np.mean(np.stack(embeddings, axis=0), axis=0)
            self._whitelist.append((identity, self._normalize(merged)))

        logger.info(
            "白名单加载完成: %d 人（模板图 %d 张）",
            len(self._whitelist),
            sum(len(v) for v in grouped_embeddings.values()),
        )

    def _identity_from_stem(self, stem: str) -> str:
        """将文件名 stem 归并为身份名：name_1/name-1/name 1 -> name。"""
        cleaned = stem.strip()
        if not cleaned:
            return stem

        merged = re.sub(r"(?:[_\-\s]+\d+)$", "", cleaned)
        return merged if merged else cleaned

    def recognize(self, frame: np.ndarray) -> FaceRecognitionResult:
        """对单帧进行人脸识别"""
        if not self.enabled:
            return FaceRecognitionResult(enabled=False)

        if self._app is None:
            return FaceRecognitionResult(
                enabled=True,
                threshold=self.threshold,
                error=self._init_error or "人脸识别模型不可用",
            )

        try:
            faces = self._app.get(frame)
            if not faces:
                if config.FACE_OUTPUT_LOG_ENABLED:
                    logger.info("人脸识别输出: detected_faces=0 matched=False")
                return FaceRecognitionResult(
                    enabled=True,
                    detected_faces=0,
                    matched=False,
                    threshold=self.threshold,
                )

            probe = getattr(faces[0], "embedding", None)
            if probe is None:
                if config.FACE_OUTPUT_LOG_ENABLED:
                    logger.info("人脸识别输出: detected_faces=%s matched=False error=检测到人脸但未提取到特征", len(faces))
                return FaceRecognitionResult(
                    enabled=True,
                    detected_faces=len(faces),
                    matched=False,
                    threshold=self.threshold,
                    error="检测到人脸但未提取到特征",
                )

            probe = self._normalize(np.asarray(probe, dtype=np.float32))

            if not self._whitelist:
                if config.FACE_OUTPUT_LOG_ENABLED:
                    logger.info("人脸识别输出: detected_faces=%s matched=False error=白名单为空", len(faces))
                return FaceRecognitionResult(
                    enabled=True,
                    detected_faces=len(faces),
                    matched=False,
                    threshold=self.threshold,
                    error="白名单为空",
                )

            best_name = None
            best_score = -1.0
            for name, emb in self._whitelist:
                score = float(np.dot(probe, emb))
                if score > best_score:
                    best_score = score
                    best_name = name

            is_match = best_score >= self.threshold
            if config.FACE_OUTPUT_LOG_ENABLED:
                logger.info(
                    "人脸识别输出: detected_faces=%s matched=%s name=%s similarity=%.3f threshold=%.3f",
                    len(faces),
                    is_match,
                    best_name if is_match else None,
                    max(best_score, 0.0),
                    self.threshold,
                )
            return FaceRecognitionResult(
                enabled=True,
                detected_faces=len(faces),
                matched=is_match,
                best_match_name=best_name if is_match else None,
                best_similarity=max(best_score, 0.0),
                threshold=self.threshold,
            )
        except Exception as e:
            logger.warning("人脸识别失败: %s", e)
            if config.FACE_OUTPUT_LOG_ENABLED:
                logger.info("人脸识别输出: matched=False error=%s", str(e))
            return FaceRecognitionResult(
                enabled=True,
                threshold=self.threshold,
                error=str(e),
            )
