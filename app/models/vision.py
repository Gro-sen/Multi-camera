"""
视觉模型抽象层
"""
import base64
import json
import os
from typing import Optional, Dict, Any
import cv2

from app.core import get_logger, config
from app.models.types import VisionFacts
from app.core.exceptions import ModelException
from app.utils import JSONFixer
from app.models.common_prompt import build_vision_prompt
logger = get_logger(__name__)


def _clip_text(text: str, limit: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return f"{s[:limit]} ...[truncated {len(s) - limit} chars]"


class VisionModelBase:
    """视觉模型基类"""
    
    def analyze(self, frame) -> Optional[VisionFacts]:
        """分析帧"""
        raise NotImplementedError


class AlibabaVisionModel(VisionModelBase):
    """阿里云视觉模型"""
    
    def __init__(self):
        try:
            from app.utils.alibaba_client import AlibabaOpenAIClient
            self.client = AlibabaOpenAIClient()
            self.available = True
        except ImportError:
            logger.warning("阿里云客户端未安装，视觉模型不可用")
            self.available = False

    def _get_model_name(self) -> str:
        return os.getenv("ALIBABA_VISION_MODEL", config.ALIBABA_VISION_MODEL)
    
    def analyze(self, image_base64: str, prompt: str) -> str:
        if not self.available:
            raise ModelException("视觉模型不可用")

        try:
            self.model = self._get_model_name()
            strict_prompt = build_vision_prompt(prompt)
            raw_output = self.client.generate(
                model=self.model,
                prompt=strict_prompt,
                images=[image_base64] if image_base64 else None,
                options={"temperature": 0.1, "top_p": 0.2},
            )
            if config.LLM_OUTPUT_LOG_ENABLED:
                logger.info(
                    "视觉模型输出[%s]: %s",
                    self.model,
                    _clip_text(raw_output, config.LLM_OUTPUT_LOG_MAX_CHARS),
                )
            return raw_output
        except Exception as e:
            logger.error(f"视觉分析失败: {e}", exc_info=True)
            raise ModelException(f"视觉模型分析失败: {e}") from e

    def close(self):
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass

class VisionModelFactory:
    """视觉模型工厂"""
    
    _models = {}
    
    @classmethod
    def get_default_model(cls) -> VisionModelBase:
        """获取默认模型 - 仅使用阿里云"""
        if "alibaba" not in cls._models:
            cls._models["alibaba"] = AlibabaVisionModel()
        
        model = cls._models["alibaba"]
        if model.available:
            logger.info("✓ 使用阿里云视觉模型")
            return model
        else:
            logger.error("❌ 阿里云视觉模型不可用")
            raise ModelException("阿里云视觉模型不可用")
