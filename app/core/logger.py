"""
统一日志系统
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}[{record.levelname}]{self.RESET}"
        return super().format(record)

import time
import threading

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器 - 高精度时间戳 + 线程ID"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'
    
    # 线程名称映射（便于识别）
    _thread_counter = {}
    _counter_lock = threading.Lock()
    
    @classmethod
    def get_thread_abbr(cls):
        """获取线程简称"""
        tid = threading.current_thread().ident
        with cls._counter_lock:
            if tid not in cls._thread_counter:
                cls._thread_counter[tid] = f"T{len(cls._thread_counter) + 1}"
        return cls._thread_counter[tid]
    
    def format(self, record):
        # 添加高精度时间戳（毫秒）
        ct = time.localtime(record.created)
        msecs = int((record.created - int(record.created)) * 1000)
        record.msecs = msecs
        
        # 添加线程简称
        record.thread_abbr = self.get_thread_abbr()
        
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}[{record.levelname}]{self.RESET}"
        return super().format(record)

def get_logger(name: str, log_file: Path = None, level: str = "INFO") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器
    if logger.hasHandlers():
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter(
            '%(levelname)s [%(asctime)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    )
    logger.addHandler(console_handler)
    # 自定义format以显示毫秒和线程ID
    formatter = console_handler.formatter
    formatter._fmt = '%(levelname)s [%(asctime)s.%(msecs)03d][%(thread_abbr)s] %(name)s: %(message)s'

    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter(
                '%(levelname)s [%(asctime)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(file_handler)
    
    return logger