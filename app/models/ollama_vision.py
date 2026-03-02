import os
from app.utils.ollama_client import OllamaClient


class OllamaVisionModel:
    def __init__(self):
        self.client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
        self.model = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")

    def analyze(self, *args, **kwargs) -> str:
        # 兼容不同调用方式：analyze(image_b64, prompt) / analyze(prompt=..., image_base64=...)
        image_base64 = kwargs.get("image_base64")
        prompt = kwargs.get("prompt", "请描述图像内容，并输出结构化要点。")

        if len(args) >= 1 and isinstance(args[0], str):
            # 可能是 image_base64 或 prompt
            if len(args[0]) > 200:  # 粗略判断base64
                image_base64 = args[0]
            else:
                prompt = args[0]
        if len(args) >= 2 and isinstance(args[1], str):
            prompt = args[1]

        images = [image_base64] if image_base64 else None
        return self.client.generate(model=self.model, prompt=prompt, images=images)