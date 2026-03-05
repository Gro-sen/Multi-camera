import os
from app.core import get_logger,config
from app.core.exceptions import ModelException
from app.models.common_prompt import build_vision_prompt
from app.utils.ollama_client import OllamaClient

logger = get_logger(__name__)

class OllamaVisionModel:
    def __init__(self):
        try:
            self.client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
            self.available = True
        except Exception:
            self.available = False
        self.model = config.OLLAMA_VISION_MODEL

    def analyze(self, image_base64: str, prompt: str) -> str:
        if not self.available:
            raise ModelException("视觉模型不可用")

        strict_prompt = build_vision_prompt(prompt)
        images = [image_base64] if image_base64 else None

        try:
            try:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=strict_prompt,
                    images=images,
                )
            except TypeError:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=strict_prompt,
                    images=images,
                )
            logger.info(f"【DEBUG】Ollama视觉原始响应: {raw_output}")
            return raw_output
        except Exception as e:
            logger.error(f"Ollama视觉分析失败: {e}", exc_info=True)
            raise ModelException(f"Ollama视觉模型分析失败: {e}") from e