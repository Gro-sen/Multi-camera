import os
import re
from app.core import get_logger, config
from app.core.exceptions import ModelException
from app.models.common_prompt import build_reasoning_prompt
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
    # 优先提取 ```json ... ``` 或 ``` ... ``` 中的内容
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 否则去掉前后多余空白和单行反引号
    return text.strip().strip("`").strip()


class OllamaReasoningModel:
    def __init__(self):
        try:
            self.client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
            self.available = True
        except Exception:
            self.available = False

    def _get_model_name(self) -> str:
        return os.getenv("OLLAMA_REASONING_MODEL", config.OLLAMA_REASONING_MODEL)

    def infer(self, facts: dict, cases: list, prompt: str) -> str:
        if not self.available:
            raise ModelException("推理模型不可用")

        try:
            self.model = self._get_model_name()
            final_prompt = build_reasoning_prompt(prompt, facts, cases)
            try:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=final_prompt,
                    timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
                    max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
                )
            except TypeError:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=final_prompt,
                )
            if config.DEBUG:
                logger.debug("Ollama推理原始响应: %s", raw_output)
                logger.debug("打印final_prompt: %s", final_prompt)
            cleaned = _clean_ollama_response(raw_output)
            if config.LLM_OUTPUT_LOG_ENABLED:
                logger.info(
                    "Ollama推理模型输出[%s]: %s",
                    self.model,
                    _clip_text(cleaned, config.LLM_OUTPUT_LOG_MAX_CHARS),
                )
            return cleaned
        except Exception as e:
            logger.error(f"Ollama推理失败: {e}", exc_info=True)
            raise ModelException(f"Ollama推理模型失败: {e}") from e

    def close(self):
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass