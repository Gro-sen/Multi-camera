"""
JSON修复工具 - 修复LLM输出的格式问题
"""
import json
import re
import math
from typing import Any, Dict

class JSONFixer:
    """JSON修复工具类"""

    @staticmethod
    def remove_trailing_commas(text: str) -> str:
        """移除尾部逗号"""
        return re.sub(r',\s*(?=[}\]])', '', text)

    @staticmethod
    def fix_broken_strings(text: str) -> str:
        """
        修复破碎的字符串，特别是时间戳格式
        示例："2024-01-"01T00":"00":00" → "2024-01-01T00:00:00"
        """
        # 先修复时间戳格式（这是一个常见问题）
        timestamp_pattern = r'"(\d{4}-\d{2}-)"(\d{2}T\d{2})"(:)"(\d{2})"(:)"(\d{2})"'
        
        def fix_timestamp(match):
            year_month = match.group(1).replace('"', '')[:-1]
            day_time = match.group(2).replace('"', '')
            hour_min = match.group(4).replace('"', '')
            sec = match.group(6).replace('"', '')
            return f'"{year_month}-{day_time}:{hour_min}:{sec}"'
        
        text = re.sub(timestamp_pattern, fix_timestamp, text)
        
        # 然后修复一般的破碎字符串
        prev = None
        while prev != text:
            prev = text
            text = re.sub(
                r'"([^"]*)"\s*"([^"]*)"',
                lambda m: '"' + m.group(1) + m.group(2) + '"',
                text
            )
        return text

    @staticmethod
    def eval_numeric_expressions(text: str) -> str:
        """
        仅处理 confidence 字段
        支持：
          1) a+b=c → 取右边数字
          2) a+b-c → 直接计算
        """
        def replace_confidence(match):
            field = match.group(1)
            value = match.group(2).strip()
            
            # 如果包含 "="，取右边部分
            if '=' in value:
                value = value.split('=')[1].strip()
            
            # 尝试计算表达式（包含+ - * /）
            try:
                # 移除非数字字符（除了 . 和 -）
                clean_value = re.sub(r'[^\d.\-+*/()\s]', '', value)
                if any(op in clean_value for op in ['+', '-', '*', '/']):
                    result = eval(clean_value)
                    value = str(result)
                else:
                    # 直接转换为float再转回string
                    value = str(float(clean_value))
            except:
                pass
            
            # 确保在0-1之间
            try:
                num = float(value)
                if num > 1:
                    num = num / 100  # 假设是百分比
                value = str(max(0, min(1, num)))
            except:
                value = "0.0"
            
            return f'"{field}": {value}'
        
        # 匹配 "confidence": "..." 格式
        text = re.sub(
            r'"(confidence)"\s*:\s*"([^"]*)"',
            replace_confidence,
            text
        )
        return text

    @staticmethod
    def fix_unquoted_values(text: str) -> str:
        """修复未引用的值"""
        # 在冒号后面如果是 true/false/null，转换为JSON格式
        text = re.sub(r':\s*\btrue\b', ': true', text, flags=re.IGNORECASE)
        text = re.sub(r':\s*\bfalse\b', ': false', text, flags=re.IGNORECASE)
        text = re.sub(r':\s*\bnull\b', ': null', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def safe_parse(text: str) -> Dict[str, Any]:
        """
        安全解析JSON文本
        
        Args:
            text: 可能有格式问题的JSON文本
        
        Returns:
            解析后的字典
        """
        if not text or not isinstance(text, str):
            return {}
        
        # 清理文本
        text = text.strip()
        
        # 去掉可能的代码块标记
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\n?', '', text)
        if text.endswith('```'):
            text = re.sub(r'\n?```$', '', text)
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 应用修复策略
        fixes = [
            ('移除尾部逗号', JSONFixer.remove_trailing_commas),
            ('修复破碎字符串', JSONFixer.fix_broken_strings),
            ('计算数值表达式', JSONFixer.eval_numeric_expressions),
            ('修复未引用值', JSONFixer.fix_unquoted_values),
        ]
        
        for fix_name, fix_func in fixes:
            try:
                text = fix_func(text)
                result = json.loads(text)
                return result
            except json.JSONDecodeError:
                continue
        
        # 如果所有修复都失败，返回空字典
        return {}

    @staticmethod
    def safe_parse_list(text: str) -> list:
        """安全解析JSON列表"""
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        
        # 去掉代码块标记
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\n?', '', text)
        if text.endswith('```'):
            text = re.sub(r'\n?```$', '', text)
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 应用修复策略
        fixes = [
            ('移除尾部逗号', JSONFixer.remove_trailing_commas),
            ('修复破碎字符串', JSONFixer.fix_broken_strings),
        ]
        
        for fix_name, fix_func in fixes:
            try:
                text = fix_func(text)
                result = json.loads(text)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
        
        return []