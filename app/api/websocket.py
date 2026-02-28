"""
WebSocket连接管理
"""
import json
import asyncio
import queue
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

from app.core import get_logger, state

logger = get_logger(__name__)


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        """接受新连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        state.register_ws_connection(websocket)
        logger.info(f"客户端已连接，当前连接数: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """断开连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        state.unregister_ws_connection(websocket)
        logger.info(f"客户端已断开，剩余连接数: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict) -> None:
        """广播消息到所有连接"""
        disconnected = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.debug(f"发送消息失败: {e}")
                disconnected.append(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        """发送消息到特定连接"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"发送个人消息失败: {e}")
            self.disconnect(websocket)


# 全局连接管理器
manager = ConnectionManager()


async def websocket_handler(websocket: WebSocket) -> None:
    """WebSocket连接处理"""
    await manager.connect(websocket)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "unknown")
                
                if message_type == "ping":
                    # 心跳检测
                    await manager.send_personal(websocket, {
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                elif message_type == "subscribe":
                    # 订阅特定事件
                    event_type = message.get("event", "all")
                    logger.debug(f"客户端订阅事件: {event_type}")
                else:
                    logger.debug(f"未知消息类型: {message_type}")
            
            except json.JSONDecodeError:
                logger.debug(f"无效的JSON消息")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket异常: {e}", exc_info=True)
        manager.disconnect(websocket)


async def broadcast_worker() -> None:
    """WebSocket广播工作线程"""
    logger.info("广播工作线程已启动")
    
    while True:
        try:
            # 使用非阻塞方式获取消息，避免阻塞事件循环
            try:
                message = state.broadcast_queue.get_nowait()
            except queue.Empty:
                message = None
            
            if message:
                await manager.broadcast(message)
                state.broadcast_queue_task_done()
                logger.debug(f"已广播消息: {message.get('type', 'unknown')}")
            else:
                # 没有消息时，让出控制权给其他任务
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"广播异常: {e}")
            await asyncio.sleep(0.1)