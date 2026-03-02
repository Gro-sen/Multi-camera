import os
from app.utils.ollama_client import OllamaClient


class OllamaReasoningModel:
    def __init__(self):
        self.client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
        self.model = os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:7b")

    def infer(self, *args, **kwargs) -> str:
        """
        支持 infer(facts: dict, cases: list, prompt: str) 或 infer(prompt: str)
        """
        # 兼容推理服务的三参数调用
        if len(args) == 3 and isinstance(args[0], dict) and isinstance(args[2], str):
            facts, cases, prompt = args
            vision_summary = f"## 当前画面分析结果：\n{facts}\n"
            kb_context = ""
            if cases and len(cases) > 0:
                kb_context = "\n## 知识库规则：\n"
                for case in cases[:3]:
                    source = case.get('source', '未知')
                    text = case.get('text', '')
                    kb_context += f"### 【{source}】\n{text}\n\n"
            final_prompt = f"{prompt}\n{vision_summary}\n{kb_context}"
        else:
            # 兼容单 prompt 字符串
            final_prompt = kwargs.get("prompt") or (args[0] if args else "")

        return self.client.generate(model=self.model, prompt=final_prompt)