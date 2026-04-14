import os
import re
from app.core import get_logger, config
from app.core.exceptions import ModelException
from app.models.common_prompt import build_vision_prompt
from app.utils.ollama_client import OllamaClient

logger = get_logger(__name__)


def _clip_text(text: str, limit: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return f"{s[:limit]} ...[truncated {len(s) - limit} chars]"


def _clean_ollama_response(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip().strip("`").strip()


class OllamaVisionModel:
    def __init__(self):
        try:
            self.client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
            self.available = True
        except Exception:
            self.available = False

    def _get_model_name(self) -> str:
        return os.getenv("OLLAMA_VISION_MODEL", config.OLLAMA_VISION_MODEL)

    def analyze(self, image_base64: str, prompt: str) -> str:
        if not self.available:
            raise ModelException("视觉模型不可用")

        self.model = self._get_model_name()
        strict_prompt = build_vision_prompt(prompt)
        images = [image_base64] if image_base64 else None

        try:
            try:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=strict_prompt,
                    images=images,
                    timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
                    max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
                )
            except TypeError:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=strict_prompt,
                    images=images,
                )
            if config.DEBUG:
                logger.debug("Ollama视觉原始响应: %s", raw_output)
            cleaned = _clean_ollama_response(raw_output)
            if config.LLM_OUTPUT_LOG_ENABLED:
                logger.info(
                    "Ollama视觉模型输出[%s]: %s",
                    self.model,
                    _clip_text(cleaned, config.LLM_OUTPUT_LOG_MAX_CHARS),
                )
            return cleaned
        except Exception as e:
            logger.error(f"Ollama视觉分析失败: {e}", exc_info=True)
            raise ModelException(f"Ollama视觉模型分析失败: {e}") from e

    def close(self):
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass