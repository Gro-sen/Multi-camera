"""
推理模型抽象层
"""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

from app.core import get_logger, config
from app.models.types import VisionFacts, ReasoningResult, AlarmDecision, Analysis
from app.core.exceptions import ModelException
from app.utils import JSONFixer
from app.models.common_prompt import build_reasoning_prompt
logger = get_logger(__name__)


def extract_json_from_response(response_text: str) -> dict:
    """从模型响应中提取 JSON，处理 markdown 代码块和额外文字"""
    
    # 方法1：提取 markdown 代码块中的 JSON
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 方法2：查找最外层的完整 JSON 对象（处理嵌套大括号）
    start_idx = response_text.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(response_text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        json_str = response_text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}, 原文: {json_str[:200]}...")
    
    return {}


class ReasoningModelBase:
    """推理模型基类"""

    def infer(self, facts: Dict[str, Any], cases: List[Dict[str, Any]], prompt: str) -> str:
        raise NotImplementedError


class AlibabaReasoningModel(ReasoningModelBase):
    """阿里云推理模型"""

    def __init__(self, model_name: str = "qwen2.5-7b-instruct"):
        try:
            from app.utils.alibaba_client import AlibabaOpenAIClient
            self.client = AlibabaOpenAIClient()
            self.available = True
        except ImportError:
            logger.warning("阿里云客户端未安装，推理模型不可用")
            self.available = False
        self.model = config.ALIBABA_REASONING_MODEL

    def infer(self, facts: Dict[str, Any], cases: List[Dict[str, Any]], prompt: str) -> str:
        if not self.available:
            raise ModelException("推理模型不可用")

        try:
            final_prompt = build_reasoning_prompt(prompt, facts, cases)
            raw_output = self.client.generate(
                model=self.model,
                prompt=final_prompt,
                options={"temperature": 0.1, "top_p": 0.2},
            )
            return raw_output
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            raise ModelException(f"推理模型失败: {e}") from e

    def close(self):
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass

class ReasoningModelFactory:
    """推理模型工厂"""
    
    _models = {}
    
    @classmethod
    def get_default_model(cls) -> ReasoningModelBase:
        """获取默认推理模型 - 仅使用阿里云"""
        if "alibaba" not in cls._models:
            cls._models["alibaba"] = AlibabaReasoningModel(config.ALIBABA_REASONING_MODEL)
        
        model = cls._models["alibaba"]
        if model.available:
            logger.info("✓ 使用阿里云推理模型")
            return model
        else:
            logger.error("❌ 阿里云推理模型不可用")
            raise ModelException("阿里云推理模型不可用")