"""
JSON相关实用工具
"""
import json
from typing import Any, Dict
from app.utils.json_fixer import JSONFixer

def parse_json_safe(text: str, default: Any = None) -> Any:
    """
    安全解析JSON文本
    
    Args:
        text: JSON文本
        default: 如果解析失败，返回的默认值
    
    Returns:
        解析结果或默认值
    """
    try:
        return JSONFixer.safe_parse(text)
    except:
        return default if default is not None else {}


def to_json_string(obj: Any, ensure_ascii: bool = False, indent: int = 2) -> str:
    """
    转换对象为JSON字符串
    
    Args:
        obj: 要转换的对象
        ensure_ascii: 是否确保ASCII
        indent: 缩进空格数
    
    Returns:
        JSON字符串
    """
    try:
        return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=ensure_ascii)