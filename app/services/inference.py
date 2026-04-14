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
from app.models.types import VisionFacts, ReasoningResult, RecognitionRecord, AlarmDecision, Analysis, FaceRecognitionResult
from app.core.exceptions import InferenceException
from app.services.alarm import AlarmService
from app.utils import JSONFixer
from multiprocessing import Process, Queue
from sentence_transformers import SentenceTransformer
from kb.rule_source import build_query_text, get_badge_face_rules, get_fast_rules, get_prompt, select_first_rule
logger = get_logger(__name__)

def analyze_worker(camera_id, frame_queue, result_queue):
    inference_service = InferenceService()
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        result = inference_service.infer(frame, camera_id)
        result_queue.put((camera_id, result))

class InferenceService:
    """推理服务"""
    
    def __init__(self, vision_model=None, reasoning_model=None, kb=None, *args, **kwargs):
        self.kb = kb

        # 优先使用外部传入；未传入则按 MODEL_PROVIDER 工厂创建
        if vision_model is None or reasoning_model is None:
            from app.models.factory import create_models
            vision_model, reasoning_model = create_models()

        self.vision_model = vision_model
        self.reasoning_model = reasoning_model
        self.alarm_service = AlarmService()
        self.face_service = state.init_face_service()

        # 只做依赖初始化，不覆盖模型
        self._initialize_models()

    def _initialize_models(self) -> None:
        """初始化依赖（不要在这里覆盖 self.vision_model / self.reasoning_model）"""
        try:
            if self.kb is None:
                from kb import kb
                self.kb = kb
            logger.info("知识库已初始化")
        except ImportError as e:
            logger.warning(f"知识库初始化失败: {e}")
               
    def frame_to_base64(self, frame) -> str:
        """将帧转换为Base64编码"""
        frame = cv2.resize(frame, (config.INFER_FRAME_WIDTH, config.INFER_FRAME_HEIGHT))
        _, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.INFER_JPEG_QUALITY],
        )
        return base64.b64encode(buf).decode()
    
    def analyze_vision(self, frame) -> Optional[VisionFacts]:
        """视觉模型分析"""
        if self.vision_model is None:
            logger.error("视觉模型未初始化")
            return None

        try:
            image_b64 = self.frame_to_base64(frame)
            vision_prompt = get_prompt("vision_prompt")
            # 统一接口：只传两个参数
            raw_output = self.vision_model.analyze(image_b64, vision_prompt)
            if config.DEBUG:
                logger.debug("视觉模型原始响应: %s", raw_output)

            if isinstance(raw_output, VisionFacts):
                return raw_output
            elif isinstance(raw_output, dict):
                vision_dict = raw_output
            else:
                try:
                    vision_dict = JSONFixer.safe_parse(raw_output)
                except Exception:
                    vision_dict = json.loads(raw_output)

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
    
    def get_similar_cases(self, vision_facts: VisionFacts, face_result: Optional[FaceRecognitionResult] = None) -> list:
        """从知识库获取相似案例"""
        if config.SKIP_KB_RETRIEVAL:
            return []

        if self.kb is None:
            return []
        
        try:
            query_text = build_query_text(vision_facts, face_result)
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
    
    def reasoning_inference(
        self,
        vision_facts: VisionFacts,
        similar_cases: list,
        face_result: Optional[FaceRecognitionResult] = None,
    ) -> Optional[ReasoningResult]:
        """推理模型推理"""
        if self.reasoning_model is None:
            logger.error("推理模型未初始化")
            return None

        try:
            reasoning_prompt = get_prompt("reasoning_prompt")
            # 统一接口：只传三个参数
            vision_payload = vision_facts.dict()
            if face_result and face_result.enabled:
                vision_payload["face_identity"] = {
                    "enabled": True,
                    "detected_faces": face_result.detected_faces,
                    "matched": face_result.matched,
                    "matched_name": face_result.best_match_name,
                    "similarity": face_result.best_similarity,
                    "threshold": face_result.threshold,
                    "is_stranger": face_result.detected_faces > 0 and not face_result.matched,
                    "error": face_result.error,
                }

            raw_output = self.reasoning_model.infer(vision_payload, similar_cases, reasoning_prompt)
            if config.DEBUG:
                logger.debug("推理模型原始响应: %s", raw_output)

            if isinstance(raw_output, ReasoningResult):
                return raw_output
            elif isinstance(raw_output, dict):
                reasoning_dict = raw_output
            else:
                try:
                    reasoning_dict = JSONFixer.safe_parse(raw_output)
                except Exception:
                    reasoning_dict = json.loads(raw_output)

            if not reasoning_dict:
                logger.warning("推理模型返回空结果，使用默认值")
                reasoning_dict = {
                    "final_decision": {
                        "is_alarm": "否",
                        "alarm_level": "无",
                        "alarm_reason": "推理系统异常",
                        "confidence": 0.0
                    },
                    "analysis": {
                        "risk_assessment": "推理系统故障",
                        "recommendation": "请检查推理模型",
                        "rules_applied": ["错误处理"]
                    },
                    "metadata": {
                        "model": "fallback",
                        "timestamp": datetime.now().isoformat(),
                        "error": "empty_reasoning_output"
                    }
                }

            reasoning_result = ReasoningResult(**reasoning_dict)
            logger.debug(f"推理完成: {reasoning_result.dict()}")
            return reasoning_result

        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            return ReasoningResult(
                final_decision=AlarmDecision(
                    is_alarm="否",
                    alarm_level="无",
                    alarm_reason="推理系统异常",
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
    
    def infer(
        self,
        frame,
        camera_id: Optional[str] = None,
        frame_timestamp: Optional[str] = None,
        broadcast: bool = True,
    ) -> Optional[RecognitionRecord]:
        """完整推理流程"""
        # 诊断：记录推理请求
        logger.debug(f"[diagnostic] 请求推理 camera={camera_id}")

        # 获取摄像头独立推理锁（允许不同摄像头真正并发）
        if not state.acquire_camera_lock(camera_id, timeout=2.0):
            logger.debug(f"[diagnostic] 推理锁获取失败（摄像头可能前一个推理还未完成）camera={camera_id}")
            return None
        logger.debug(f"[diagnostic] 推理锁已获取 camera={camera_id}")
        
        # 标记推理开始
        state.mark_inference_start(camera_id)
        active_count = state.get_active_inferences_count()
        active_info = state.get_active_inferences_info()
        logger.debug(f"[diagnostic][并发监控] {camera_id} 推理开始 | 当前活跃={active_count} | {active_info}")
        
        try:
            start_time = time.time()
            
            # 第一阶段：视觉分析
            logger.debug("[diagnostic] 开始视觉分析...")
            vision_facts = self.analyze_vision(frame)
            if vision_facts is None:
                return None

            # 第一阶段补充：人脸识别
            face_result = None
            if vision_facts.has_person:
                face_result = self.face_service.recognize(frame)
                logger.debug(
                    "[diagnostic] 人脸识别 camera=%s faces=%s matched=%s best=%s score=%.3f",
                    camera_id,
                    face_result.detected_faces,
                    face_result.matched,
                    face_result.best_match_name,
                    face_result.best_similarity,
                )
            
            # 第二阶段：知识库查询
            similar_cases = []
            if not config.FAST_RULE_ONLY_MODE:
                logger.debug("[diagnostic] 查询知识库...")
                similar_cases = self.get_similar_cases(vision_facts, face_result)
            
            # 第三阶段：推理分析
            if config.FAST_RULE_ONLY_MODE:
                reasoning_result = self._build_fast_reasoning_result(vision_facts)
            else:
                logger.debug("[diagnostic] 执行推理...")
                reasoning_result = self.reasoning_inference(vision_facts, similar_cases, face_result)
                if reasoning_result is None:
                    return None

            # 第三阶段补充：按“人脸+工牌”矩阵做一致性修正
            self._apply_badge_face_matrix(reasoning_result, vision_facts, face_result)
            
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
                frame_timestamp=frame_timestamp,
                is_alarm=final_decision.is_alarm,
                alarm_level=final_decision.alarm_level,
                alarm_reason=final_decision.alarm_reason,
                confidence=final_decision.confidence,
                image_path=image_path,
                camera_id=camera_id,
                vision_facts=vision_facts,
                analysis=reasoning_result.analysis,
                face_recognition=face_result,
                model_version=getattr(self.reasoning_model, "model", "未知"),
            )
            
            # 记录和广播
            self.alarm_service.record_alarm(record)
            if broadcast:
                self.alarm_service.broadcast_alarm(record)
            
            # 写入待审核案例池（报警/正常都落盘，不自动重建索引）
            try:
                self._save_to_knowledge_base(record, vision_facts, reasoning_result, similar_cases)
            except Exception as e:
                logger.warning(f"写入待审核案例失败: {e}")
            
            # 记录推理耗时
            self.alarm_service.record_inference_time(final_decision, time.time() - start_time, camera_id=camera_id)
            
            elapsed = time.time() - start_time
            logger.info(f"推理完成 ({elapsed:.2f}s): {final_decision.alarm_level}级警报 (置信度: {final_decision.confidence:.2f})")
            logger.debug(f"[diagnostic] 推理返回 camera={camera_id} elapsed={elapsed:.2f}s alarm_level={final_decision.alarm_level} confidence={final_decision.confidence:.2f}")
            
            return record
            
        except Exception as e:
            logger.error(f"推理流程异常: {e}", exc_info=True)
            return None
        finally:
            state.release_camera_lock(camera_id)
            # 注意：不在这里更新 last_infer_time，采样时间由 worker 在调度时设置（避免与推理耗时耦合）
            state.mark_inference_end(camera_id)
            active_count = state.get_active_inferences_count()
            active_info = state.get_active_inferences_info()
            logger.debug(f"[diagnostic][并发监控] {camera_id} 推理结束 | 剩余活跃={active_count} | {active_info}")
    
    def _save_to_knowledge_base(self, record: RecognitionRecord, vision_facts: VisionFacts,
                                 reasoning_result: ReasoningResult, similar_cases: list) -> None:
        """将推理案例保存到待审核案例池（封装成 case_data 并调用 kb.add_case）"""
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
            "final_decision": reasoning_result.final_decision.dict() if hasattr(reasoning_result, "final_decision") else {},
            "analysis": reasoning_result.analysis.dict() if hasattr(reasoning_result, "analysis") else {},
            "vision_facts": vision_facts.dict() if hasattr(vision_facts, "dict") else {},
            "face_recognition": record.face_recognition.dict() if getattr(record, "face_recognition", None) else {},
            "metadata": {
                "reasoning_model": getattr(self.reasoning_model, "model", "未知"),
                "vision_model": getattr(self.vision_model, "model", "未知"),
                "kb_cases_used": len(similar_cases) if similar_cases else 0,
                "kb_total_references": len(similar_cases) if similar_cases else 0,
            }
        }

        try:
            self.kb.add_case(case_data)
            logger.debug(f"案例已写入待审核池: is_alarm={record.is_alarm} level={record.alarm_level}")
        except Exception as e:
            logger.warning(f"写入知识库失败（内部）: {e}")

    def _apply_badge_face_matrix(
        self,
        reasoning_result: ReasoningResult,
        vision_facts: VisionFacts,
        face_result: Optional[FaceRecognitionResult],
    ) -> None:
        """按知识库中的人脸与工牌规则统一修正结论。"""
        if not vision_facts.has_person or face_result is None or not face_result.enabled:
            return

        rule = select_first_rule(get_badge_face_rules(), vision_facts, face_result)
        if rule is None:
            return

        decision = reasoning_result.final_decision
        analysis = reasoning_result.analysis

        decision_cfg = rule.get("decision", {})
        analysis_cfg = rule.get("analysis", {})

        if "is_alarm" in decision_cfg:
            decision.is_alarm = decision_cfg["is_alarm"]
        if "alarm_level" in decision_cfg:
            decision.alarm_level = decision_cfg["alarm_level"]
        if "alarm_reason" in decision_cfg:
            decision.alarm_reason = decision_cfg["alarm_reason"]
        if "confidence" in decision_cfg:
            decision.confidence = max(decision.confidence, float(decision_cfg["confidence"]))

        if "recommendation" in analysis_cfg:
            analysis.recommendation = analysis_cfg["recommendation"]
        for item in analysis_cfg.get("rules_applied", []) or []:
            if item not in analysis.rules_applied:
                analysis.rules_applied.append(item)

    def _build_fast_reasoning_result(self, vision_facts: VisionFacts) -> ReasoningResult:
        """极速模式：完全由知识库规则驱动。"""
        rule = select_first_rule(get_fast_rules(), vision_facts, None)
        if rule is None:
            return ReasoningResult(
                final_decision=AlarmDecision(
                    is_alarm="否",
                    alarm_level="无",
                    alarm_reason="规则未加载",
                    confidence=0.0,
                ),
                analysis=Analysis(
                    risk_assessment="规则未加载",
                    recommendation="请检查 kb/source/rules.json",
                    rules_applied=["默认兜底"],
                ),
                metadata={"mode": "fast_rule_only", "fallback": True},
            )

        decision_cfg = rule.get("decision", {})
        analysis_cfg = rule.get("analysis", {})
        metadata = dict(rule.get("metadata", {}) or {})
        metadata.setdefault("mode", "fast_rule_only")

        return ReasoningResult(
            final_decision=AlarmDecision(
                is_alarm=decision_cfg.get("is_alarm", "否"),
                alarm_level=decision_cfg.get("alarm_level", "无"),
                alarm_reason=decision_cfg.get("alarm_reason", ""),
                confidence=float(decision_cfg.get("confidence", 0.0) or 0.0),
            ),
            analysis=Analysis(
                risk_assessment=analysis_cfg.get("risk_assessment", ""),
                recommendation=analysis_cfg.get("recommendation", ""),
                rules_applied=list(analysis_cfg.get("rules_applied", []) or []),
            ),
            metadata=metadata,
        )