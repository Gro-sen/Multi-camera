"""
自定义异常类
"""

class AppException(Exception):
    """应用基异常"""
    pass

class CameraException(AppException):
    """摄像头异常"""
    pass

class InferenceException(AppException):
    """推理异常"""
    pass

class KnowledgeBaseException(AppException):
    """知识库异常"""
    pass

class ModelException(AppException):
    """模型异常"""
    pass