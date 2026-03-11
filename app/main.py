"""
FastAPI主应用
"""
import os  
import asyncio
import threading
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.core import get_logger, config, state
from app.api import create_api_router
from app.api.websocket import websocket_handler, broadcast_worker
from app.services.camera import CameraService
from app.worker import InferenceWorker
logger = get_logger(__name__)

camera_service = CameraService()

class AppLifecycle:
    """应用生命周期管理"""
    
    def __init__(self):
        self.camera_service: CameraService = None
        self.inference_worker: InferenceWorker = None
    
    async def startup(self) -> None:
        """应用启动"""
        logger.info("="*60)
        logger.info("系统启动中...")
        logger.info("="*60)
        
        try:
            # 初始化 state
            self.state = state

            # 初始化知识库（修复 AttributeError）
            try:
                from kb import kb
                self.state.kb = kb
                logger.info("✓ 知识库已初始化")
            except Exception as e:
                logger.error(f"知识库初始化失败: {e}")
                self.state.kb = None

            # 初始化摄像头服务
            logger.info("初始化摄像头服务...")
            self.camera_service = CameraService()
            self.camera_service.start()
            logger.info("✓ 摄像头服务已启动")
            
            # 等待摄像头初始化
            logger.info("等待摄像头初始化...")
            await asyncio.sleep(2)
            logger.info("✓ 摄像头初始化完成")
            
            # 初始化推理工作线程
            logger.info("初始化推理工作线程...")
            self.inference_worker = InferenceWorker()
            self.inference_worker.start()
            logger.info("✓ 推理工作线程已启动")
            
            # 启动广播工作线程
            logger.info("启动广播工作线程...")
            asyncio.create_task(broadcast_worker())
            logger.info("✓ 广播工作线程已启动")  
            logger.info("="*60)
            logger.info("系统启动完成")
            logger.info("="*60 + "\n")
        
        except Exception as e:
            logger.error(f"系统启动失败: {e}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """应用关闭"""
        logger.info("\n" + "="*60)
        logger.info("系统关闭中...")
        logger.info("="*60)
        
        try:
            if self.inference_worker:
                self.inference_worker.stop()
                logger.info("✓ 推理工作线程已停止")
            
            if self.camera_service:
                self.camera_service.stop()
                logger.info("✓ 摄像头服务已关闭")
            
            state.is_running = False
            logger.info("✓ 全局状态已清理")
            
            logger.info("="*60)
            logger.info("系统已完全关闭")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"关闭过程异常: {e}", exc_info=True)

# 生命周期管理实例
lifecycle = AppLifecycle()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("【DEBUG】进入 lifespan 启动阶段")
    # 启动
    await lifecycle.startup()
    logger.info("【DEBUG】startup() 完成，准备 yield")
    yield
    logger.info("【DEBUG】进入 lifespan 关闭阶段")
    # 关闭
    await lifecycle.shutdown()
    logger.info("【DEBUG】shutdown() 完成")

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# 创建FastAPI应用
app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    lifespan=lifespan,
)

# 挂载静态文件
try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"✓ 静态文件已挂载: {STATIC_DIR}")
except Exception as e:
    logger.warning(f"⚠ 静态文件挂载失败: {e}")

# 初始化模板
try:
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    logger.info(f"✓ 模板已加载: {TEMPLATES_DIR}")
except Exception as e:
    logger.warning(f"⚠ 模板加载失败: {e}")

# ===== 基础路由 =====
@app.get("/")
async def index(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    """视频流（兼容旧接口）"""
    from fastapi.responses import StreamingResponse
    from app.services.stream import StreamService
    from app.core import config
    
    default_camera = config.CAMERA_SOURCES[0]["id"] if config.CAMERA_SOURCES else "cam1"
    return StreamingResponse(
        StreamService.generate_frames(default_camera),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ===== WebSocket路由 =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    await websocket_handler(websocket)

# ===== API路由 =====
api_router = create_api_router()
app.include_router(api_router)

# ===== 错误处理 =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return {
        "status": "error",
        "message": str(exc),
        "type": type(exc).__name__
    }

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    # 信号处理函数
    def signal_handler(sig, frame):
        logger.info(f"\n接收到信号 {sig}，正在优雅关闭...")
        # 设置停止标志
        if hasattr(state, 'is_running'):
            state.is_running = False
        
        # 等待一小段时间让进程正常退出
        import time
        time.sleep(0.5)
        
        # 强制退出
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # CTRL+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    logger.info("\n" + "="*60)
    logger.info("启动FastAPI服务器...")
    logger.info(f"访问地址: http://localhost:8000")
    logger.info(f"WebSocket地址: ws://localhost:8000/ws")
    logger.info(f"API文档: http://localhost:8000/docs")
    logger.info("按 CTRL+C 停止")
    logger.info("="*60 + "\n")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_config=None
        )
    except KeyboardInterrupt:
        logger.info("服务器被中断")
    except Exception as e:
        logger.error(f"服务器运行异常: {e}")