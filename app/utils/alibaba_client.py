"""
阿里云OpenAI兼容客户端
支持可选安装 - 如果未安装openai库，会优雅地降级
"""
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class AlibabaOpenAIClient:
    """
    阿里云OpenAI兼容客户端
    支持多模态和文本API调用
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        """
        初始化客户端
        
        Args:
            api_key: API密钥，如果为None则从环境变量OPENAI_API_KEY读取
            base_url: API基础URL
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai库未安装。请运行: pip install openai\n"
                "或在不使用阿里云API的情况下，系统会自动降级到本地模型"
            )
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
    
    def call_multimodal_api(
        self,
        prompt: str,
        image_b64: str,
        model: str = "qwen3-vl-8b-thinking"
    ) -> str:
        """
        调用多模态模型API
        
        Args:
            prompt: 提示词
            image_b64: Base64编码的图像
            model: 模型名称
        
        Returns:
            API响应文本
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                temperature=0.1,
                timeout=30
            )
            
            result = completion.choices[0].message.content
            return result
            
        except Exception as e:
            raise RuntimeError(f"阿里云多模态API调用失败: {e}") from e
    
    def call_text_api(
        self,
        prompt: str,
        model: str = "qwen2.5-7b-instruct"
    ) -> str:
        """
        调用纯文本模型API
        
        Args:
            prompt: 提示词
            model: 模型名称
        
        Returns:
            API响应文本
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1,
                timeout=30
            )
            
            result = completion.choices[0].message.content
            return result
            
        except Exception as e:
            raise RuntimeError(f"阿里云文本API调用失败: {e}") from e
    
    def call_api(
        self,
        prompt: str,
        model: str = "2.5-7b-instruct"
    ) -> str:
        """
        通用API调用（别名）
        
        Args:
            prompt: 提示词
            model: 模型名称
        
        Returns:
            API响应文本
        """
        return self.call_text_api(prompt, model)