"""
推理服务层 - 整合视觉模型和推理模型
"""
import base64
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
import cv2

from app.core import get_logger, config, state
from app.models.types import VisionFacts, ReasoningResult, RecognitionRecord, AlarmDecision, Analysis
from app.core.exceptions import InferenceException
from app.services.alarm import AlarmService
from app.utils import JSONFixer
logger = get_logger(__name__)


class InferenceService:
    """推理服务"""
    
    def __init__(self):
        self.alarm_service = AlarmService()
        self.vision_client = None
        self.reasoning_model = None
        self.kb = None
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """初始化模型"""
        try:
            from app.utils.alibaba_client import AlibabaOpenAIClient
            self.vision_client = AlibabaOpenAIClient()
            logger.info("视觉模型客户端已初始化")
        except ImportError as e:
            logger.warning(f"视觉模型初始化失败: {e}")
        
        try:
            from app.models import ReasoningModelFactory
            self.reasoning_model = ReasoningModelFactory.get_default_model()
            logger.info("推理模型已初始化")
        except ImportError as e:
            logger.warning(f"推理模型初始化失败: {e}")
        
        try:
            from kb import kb
            self.kb = kb
            logger.info("知识库已初始化")
        except ImportError as e:
            logger.warning(f"知识库初始化失败: {e}")
    
    def frame_to_base64(self, frame) -> str:
        """将帧转换为Base64编码"""
        frame = cv2.resize(frame, (640, 360))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()
    
    def analyze_vision(self, frame) -> Optional[VisionFacts]:
        """视觉模型分析"""
        if self.vision_client is None:
            logger.error("视觉模型客户端未初始化")
            return None
        
        try:
            image_b64 = self.frame_to_base64(frame)
            
            vision_prompt = """
你是公司内部安防系统的【视觉感知模块】。
只输出 JSON，不要解释，不要多余文字。

重要规则：
- 仔细观察画面，即使人员较小或在边缘也要识别
- 看到人但没看到工牌 → badge_status 填 "未佩戴"
- 只有背对镜头或严重遮挡才填 "无法确认"

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
            
            raw_output = self.vision_client.call_multimodal_api(
                prompt=vision_prompt,
                image_b64=image_b64,
                model=config.VISION_MODEL
            )
            
            # ← 添加这行调试日志
            logger.info(f"【DEBUG】视觉模型原始响应: {raw_output}")
            
            try:
                vision_dict = JSONFixer.safe_parse(raw_output)
            except:
                vision_dict = json.loads(raw_output)
            
            # 如果视觉模型返回空结果，使用默认值
            if not vision_dict:
                logger.warning("视觉模型返回空结果，使用默认值")
                vision_dict = {
                    "has_person": False,
                    "badge_status": "不适用",
                    "enter_restricted_area": False,
                    "has_fire_or_smoke": False,
                    "has_electric_risk": False,
                    "scene_summary": "视觉分析失败",
                }

            vision_facts = VisionFacts(**vision_dict)
            logger.debug(f"视觉分析完成: {vision_facts.dict()}")
            return vision_facts
            
        except Exception as e:
            logger.error(f"视觉分析失败: {e}", exc_info=True)
            return None
    
    def get_similar_cases(self, vision_facts: VisionFacts) -> list:
        """从知识库获取相似案例"""
        if self.kb is None:
            return []
        
        try:
            query_parts = []
            
            # 始终包含核心检测项
            if vision_facts.has_person:
                query_parts.append("人员检测 工牌佩戴")
            else:
                query_parts.append("无人员场景")
            
            # 工牌状态相关查询
            if vision_facts.badge_status in ["未佩戴", "无法确认"]:
                query_parts.append("工牌 未佩戴 告警")
            
            if vision_facts.enter_restricted_area:
                query_parts.append("禁区 入侵")
            if vision_facts.has_fire_or_smoke:
                query_parts.append("火灾 烟雾")
            if vision_facts.has_electric_risk:
                query_parts.append("电气风险 触电")
            
            query_text = " ".join(query_parts)
            similar_cases = self.kb.get_similar_cases(
                query_text,
                top_k=config.KB_RETRIEVAL_TOP_K,
                similarity_threshold=config.KB_SIMILARITY_THRESHOLD
            )
            logger.debug(f"检索到 {len(similar_cases)} 个相似案例")
            return similar_cases
            
        except Exception as e:
            logger.error(f"知识库查询失败: {e}")
            return []
    
    def reasoning_inference(self, vision_facts: VisionFacts, similar_cases: list) -> Optional[ReasoningResult]:
        """推理模型推理"""
        if self.reasoning_model is None:
            logger.error("推理模型未初始化")
            return None
        
        try:
            reasoning_result = self.reasoning_model.infer(vision_facts, similar_cases)
            logger.debug(f"推理完成: {reasoning_result.dict()}")
            return reasoning_result
            
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            # 返回安全的默认决策
            return ReasoningResult(
                final_decision=AlarmDecision(
                    is_alarm="否",
                    alarm_level="无",
                    alarm_reason=f"推理系统异常",
                    confidence=0.0
                ),
                analysis=Analysis(
                    risk_assessment="推理系统故障",
                    recommendation="请检查推理模型",
                    rules_applied=["错误处理"]
                ),
                metadata={
                    "model": "fallback",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )
    
    def infer(self, frame, camera_id: Optional[str] = None, broadcast: bool = True) -> Optional[RecognitionRecord]:
        """完整推理流程"""
        # 获取推理锁（防止并发推理）
        if not state.acquire_inference_lock(timeout=0.5):
            logger.debug("推理锁获取失败，跳过本次推理")
            return None
        
        try:
            start_time = time.time()
            
            # 第一阶段：视觉分析
            logger.info("开始视觉分析...")
            vision_facts = self.analyze_vision(frame)
            if vision_facts is None:
                return None
            
            # 第二阶段：知识库查询
            logger.info("查询知识库...")
            similar_cases = self.get_similar_cases(vision_facts)
            
            # 第三阶段：推理分析
            logger.info("执行推理...")
            reasoning_result = self.reasoning_inference(vision_facts, similar_cases)
            if reasoning_result is None:
                return None
            
            # 第四阶段：保存结果
            final_decision = reasoning_result.final_decision
            
            # 保存报警图片
            image_path = None
            if final_decision.is_alarm == "是":
                image_path = self.alarm_service.save_alarm_image(
                    frame, 
                    final_decision.alarm_level,
                    camera_id=camera_id
                )
            
            # 创建识别记录
            record = RecognitionRecord(
                is_alarm=final_decision.is_alarm,
                alarm_level=final_decision.alarm_level,
                alarm_reason=final_decision.alarm_reason,
                confidence=final_decision.confidence,
                image_path=image_path,
                camera_id=camera_id,
                vision_facts=vision_facts,
                analysis=reasoning_result.analysis,
                model_version=config.REASONING_MODEL,
            )
            
            # 记录和广播
            self.alarm_service.record_alarm(record)
            if broadcast:
                self.alarm_service.broadcast_alarm(record)
            
            # 🔥 新增：如果是报警，写入知识库
            if final_decision.is_alarm == "是" and final_decision.alarm_level != "无":
                try:
                    self._save_to_knowledge_base(record, vision_facts, reasoning_result, similar_cases)
                except Exception as e:
                    logger.warning(f"写入知识库失败: {e}")
            
            # 记录推理耗时
            self.alarm_service.record_inference_time(final_decision, time.time() - start_time)
            
            elapsed = time.time() - start_time
            logger.info(f"推理完成 ({elapsed:.2f}s): {final_decision.alarm_level}级警报 (置信度: {final_decision.confidence:.2f})")
            
            return record
            
        except Exception as e:
            logger.error(f"推理流程异常: {e}", exc_info=True)
            return None
        finally:
            state.release_inference_lock()
            if camera_id:
                state.update_infer_time(camera_id, time.time())

    def _save_to_knowledge_base(self, record: RecognitionRecord, vision_facts: VisionFacts, 
                                 reasoning_result: ReasoningResult, similar_cases: list) -> None:
        """将报警案例保存到知识库"""
        if self.kb is None:
            return
        
        case_data = {
            "case_id": record.case_id,
            "timestamp": record.timestamp,
            "camera_id": record.camera_id,
            "alarm_level": record.alarm_level,
            "alarm_reason": record.alarm_reason,
            "scene_summary": vision_facts.scene_summary,
            "is_alarm": record.is_alarm,
            "confidence": record.confidence,
            "image_path": record.image_path,
            "final_decision": reasoning_result.final_decision.dict(),
            "analysis": reasoning_result.analysis.dict(),
            "vision_facts": vision_facts.dict(),
            "metadata": {
                "model": config.REASONING_MODEL,
                "kb_cases_used": len(similar_cases) if similar_cases else 0,
                "kb_total_references": len(similar_cases) if similar_cases else 0,
            }
        }
        
        self.kb.add_case(case_data)
        logger.info(f"报警案例已写入知识库: {record.alarm_level}")