"""
数据模型和类型定义
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class ObjectDetails(BaseModel):
    """对象详情"""
    person_count: int = 0
    person_positions: List[str] = []
    environment_status: str = ""
    
    @field_validator("person_count", mode="before")
    @classmethod
    def default_person_count(cls, v):
        return v if v is not None else 0
    
    @field_validator("person_positions", mode="before")
    @classmethod
    def default_person_positions(cls, v):
        return v if v is not None else []
    
    @field_validator("environment_status", mode="before")
    @classmethod
    def default_environment_status(cls, v):
        return v if v is not None else ""


class VisionFacts(BaseModel):
    """视觉模型分析结果"""
    has_person: bool
    badge_status: str = Field(default="不适用", description="佩戴/未佩戴/无法确认/不适用")
    enter_restricted_area: bool = False
    has_fire_or_smoke: bool = False
    has_electric_risk: bool = False
    scene_summary: str = ""
    object_details: Optional[ObjectDetails] = None
    
    @field_validator("has_person", mode="before")
    @classmethod
    def default_has_person(cls, v):
        return v if v is not None else False
    
    @field_validator("badge_status", mode="before")
    @classmethod
    def default_badge_status(cls, v):
        return v if v is not None else "不适用"
    
    class Config:
        json_encoders = {
            bool: lambda v: v,
        }


class AlarmDecision(BaseModel):
    """报警决策"""
    is_alarm: str = Field(default="否", description="是/否")
    alarm_level: str = Field(default="无", description="无/一般/严重/紧急")
    alarm_reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @field_validator("is_alarm", mode="before")
    @classmethod
    def default_is_alarm(cls, v):
        return v if v is not None else "否"
    
    @field_validator("alarm_level", mode="before")
    @classmethod
    def default_alarm_level(cls, v):
        return v if v is not None else "无"
    
    @field_validator("confidence", mode="before")
    @classmethod
    def default_confidence(cls, v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class Analysis(BaseModel):
    """分析信息"""
    risk_assessment: str = ""
    recommendation: str = ""
    rules_applied: List[str] = []


class ReasoningResult(BaseModel):
    """推理模型结果"""
    final_decision: AlarmDecision
    analysis: Analysis
    metadata: Dict[str, Any] = {}


class RecognitionRecord(BaseModel):
    """识别记录"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    case_id: Optional[str] = None
    camera_id: Optional[str] = None
    is_alarm: str
    alarm_level: str
    alarm_reason: str
    confidence: float
    image_path: Optional[str] = None
    vision_facts: Optional[VisionFacts] = None
    analysis: Optional[Analysis] = None
    model_version: Optional[str] = None
    
    class Config:
        extra = "allow"


class CameraStats(BaseModel):
    """摄像头统计信息"""
    is_connected: bool
    frames_received: int
    connection_errors: int
    fps: float
    uptime_seconds: float
    last_frame_time: Optional[str] = None
    last_error_time: Optional[str] = None
    last_error_message: Optional[str] = None