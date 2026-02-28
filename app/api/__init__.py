from fastapi import APIRouter
from .routes import video, alarms, system, knowledge

def create_api_router() -> APIRouter:
    """创建API路由器"""
    router = APIRouter(prefix="/api")
    
    # 注册各个路由模块
    router.include_router(video.router)
    router.include_router(alarms.router)
    router.include_router(system.router)
    router.include_router(knowledge.router)
    
    return router

__all__ = ["create_api_router"]