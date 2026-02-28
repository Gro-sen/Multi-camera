"""
视觉模型抽象层
"""
import base64
import json
from typing import Optional, Dict, Any
import cv2

from app.core import get_logger, config
from app.models.types import VisionFacts
from app.core.exceptions import ModelException
from app.utils import JSONFixer
logger = get_logger(__name__)


class VisionModelBase:
    """视觉模型基类"""
    
    def analyze(self, frame) -> Optional[VisionFacts]:
        """分析帧"""
        raise NotImplementedError


class AlibabaVisionModel(VisionModelBase):
    """阿里云视觉模型"""
    
    def __init__(self):
        try:
            from app.utils.alibaba_client import AlibabaOpenAIClient
            self.client = AlibabaOpenAIClient()
            self.available = True
        except ImportError:
            logger.warning("阿里云客户端未安装，视觉模型不可用")
            self.available = False
    
    def frame_to_base64(self, frame) -> str:
        """将帧转换为Base64编码"""
        frame = cv2.resize(frame, (640, 360))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()
    
    def analyze(self, frame) -> Optional[VisionFacts]:
        """使用阿里云API分析帧"""
        if not self.available:
            raise ModelException("视觉模型不可用")
        
        try:
            image_b64 = self.frame_to_base64(frame)
            
            vision_prompt = """
你是公司内部安防系统的【视觉感知模块】。
只输出 JSON，不要解释，不要多余文字。
格式如下：
{
  "has_person": true/false,
  "badge_status": "佩戴" / "未佩戴" / "无法确认" / "不适用",
  "enter_restricted_area": true/false,
  "has_fire_or_smoke": true/false,
  "has_electric_risk": true/false,
  "scene_summary": "一句话描述画面",
  "object_details": {
    "person_count": 数量,
    "person_positions": ["位置描述"],
    "environment_status": "环境状态描述"
  }
}
"""
            
            raw_output = self.client.call_multimodal_api(
                prompt=vision_prompt,
                image_b64=image_b64,
                model=config.VISION_MODEL
            )
            
            logger.info(f"【DEBUG】视觉模型原始响应: {raw_output}")
            
            # 尝试修复JSON
            try:
                vision_dict = JSONFixer.safe_parse(raw_output)
            except:
                vision_dict = json.loads(raw_output)
            
            vision_facts = VisionFacts(**vision_dict)
            logger.debug(f"视觉分析完成: {vision_facts.scene_summary}")
            return vision_facts
            
        except Exception as e:
            logger.error(f"视觉分析失败: {e}", exc_info=True)
            raise ModelException(f"视觉模型分析失败: {e}") from e


class VisionModelFactory:
    """视觉模型工厂"""
    
    _models = {}
    
    @classmethod
    def get_default_model(cls) -> VisionModelBase:
        """获取默认模型 - 仅使用阿里云"""
        if "alibaba" not in cls._models:
            cls._models["alibaba"] = AlibabaVisionModel()
        
        model = cls._models["alibaba"]
        if model.available:
            logger.info("✓ 使用阿里云视觉模型")
            return model
        else:
            logger.error("❌ 阿里云视觉模型不可用")
            raise ModelException("阿里云视觉模型不可用")
