import os
from app.core import get_logger,config
from app.core.exceptions import ModelException
from app.models.common_prompt import build_reasoning_prompt
from app.utils.ollama_client import OllamaClient

logger = get_logger(__name__)

class OllamaReasoningModel:
    def __init__(self):
        try:
            self.client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
            self.available = True
        except Exception:
            self.available = False
        self.model = config.OLLAMA_REASONING_MODEL

    def infer(self, facts: dict, cases: list, prompt: str) -> str:
        if not self.available:
            raise ModelException("推理模型不可用")

        try:
            final_prompt = build_reasoning_prompt(prompt, facts, cases)
            try:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=final_prompt,
                )
            except TypeError:
                raw_output = self.client.generate(
                    model=self.model,
                    prompt=final_prompt,
                )
            logger.info(f"【DEBUG】Ollama推理原始响应: {raw_output}")
            logger.info(f"【DEBUG】打印final_prompt: {final_prompt}")
            return raw_output
        except Exception as e:
            logger.error(f"Ollama推理失败: {e}", exc_info=True)
            raise ModelException(f"Ollama推理模型失败: {e}") from e