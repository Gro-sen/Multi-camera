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
    
    def infer(self, vision_facts: VisionFacts, similar_cases: List[Dict] = None) -> Optional[ReasoningResult]:
        """执行推理"""
        raise NotImplementedError


class AlibabaReasoningModel(ReasoningModelBase):
    """阿里云推理模型"""
    
    def __init__(self, model_name: str = "qwen2.5-7b-instruct"):
        try:
            from app.utils.alibaba_client import AlibabaOpenAIClient
            self.client = AlibabaOpenAIClient()
            self.model_name = model_name
            self.available = True
        except ImportError:
            logger.warning("阿里云客户端未安装，推理模型不可用")
            self.available = False
    
    def infer(self, vision_facts: VisionFacts, similar_cases: List[Dict] = None) -> Optional[ReasoningResult]:
        """执行推理"""
        if not self.available:
            raise ModelException("阿里云推理模型不可用")
        
        try:
            vision_summary = f"""## 当前画面分析结果：
- 有人员：{vision_facts.has_person}
- 工牌状态：{vision_facts.badge_status}
- 进入禁区：{vision_facts.enter_restricted_area}
- 火灾/烟雾：{vision_facts.has_fire_or_smoke}
- 电气风险：{vision_facts.has_electric_risk}
- 场景描述：{vision_facts.scene_summary}
"""
            
            kb_context = ""
            if similar_cases and len(similar_cases) > 0:
                kb_context = "\n## 知识库规则：\n"
                for case in similar_cases[:3]:
                    source = case.get('source', '未知')
                    text = case.get('text', '')
                    kb_context += f"### 【{source}】\n{text}\n\n"
            
            prompt = f"""你是安防系统的决策模块。

**核心原则：严格按照知识库规则执行判断，不得自行放宽或修改条件。**

{vision_summary}
{kb_context}
根据知识库规则和当前画面分析结果，输出JSON格式的决策：
{{
  "final_decision": {{
    "is_alarm": "是/否",
    "alarm_level": "无/一般/严重/紧急",
    "alarm_reason": "原因",
    "confidence": 0.0-1.0
  }},
  "analysis": {{
    "risk_assessment": "风险评估",
    "recommendation": "处置建议",
    "rules_applied": ["应用的规则"]
  }}
}}

只输出JSON，不要其他文字。"""
            
            response = self.client.call_api(
                prompt=prompt,
                model=self.model_name
            )
            
            logger.info(f"【DEBUG】推理模型原始响应: {response}")

            # 使用统一的 JSON 提取函数
            result_dict = extract_json_from_response(response)
            
            if not result_dict:
                try:
                    result_dict = JSONFixer.safe_parse(response)
                except:
                    result_dict = {}
            
            final_decision_dict = result_dict.get('final_decision', {})
            analysis_dict = result_dict.get('analysis', {})
            
            # 确保有默认值
            if not final_decision_dict:
                logger.warning("未能提取 final_decision，使用默认值")
                final_decision_dict = {
                    "is_alarm": "否",
                    "alarm_level": "无",
                    "alarm_reason": "推理模型响应解析失败",
                    "confidence": 0.0
                }
            
            result = ReasoningResult(
                final_decision=AlarmDecision(**final_decision_dict),
                analysis=Analysis(
                    risk_assessment=analysis_dict.get('risk_assessment', ''),
                    recommendation=analysis_dict.get('recommendation', ''),
                    rules_applied=analysis_dict.get('rules_applied', [])
                ),
                metadata={"model": self.model_name, "timestamp": datetime.now().isoformat()}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"阿里云推理失败: {e}", exc_info=True)
            raise ModelException(f"阿里云推理模型失败: {e}") from e


class ReasoningModelFactory:
    """推理模型工厂"""
    
    _models = {}
    
    @classmethod
    def get_default_model(cls) -> ReasoningModelBase:
        """获取默认推理模型 - 仅使用阿里云"""
        if "alibaba" not in cls._models:
            cls._models["alibaba"] = AlibabaReasoningModel(config.REASONING_MODEL)
        
        model = cls._models["alibaba"]
        if model.available:
            logger.info("✓ 使用阿里云推理模型")
            return model
        else:
            logger.error("❌ 阿里云推理模型不可用")
            raise ModelException("阿里云推理模型不可用")