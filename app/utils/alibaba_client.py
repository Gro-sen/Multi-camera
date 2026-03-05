from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class AlibabaOpenAIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        if not HAS_OPENAI:
            raise ImportError("openai库未安装。请运行: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url

    def generate(
        self,
        model: str,
        prompt: str,
        images=None,
        options: Optional[dict] = None,
        format: Optional[str] = None, 
        timeout: int = 30,
    ) -> str:
        options = options or {}
        temperature = options.get("temperature", 0.1)
        top_p = options.get("top_p", 1.0)

        if images:
            image_b64 = images[0]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout
        )
        return completion.choices[0].message.content or ""
