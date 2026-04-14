import json
from typing import Dict, Any, List

from kb.rule_source import get_prompt

STRICT_JSON_SUFFIX = get_prompt("strict_json_suffix")

def build_vision_prompt(base_prompt: str) -> str:
    return f"{base_prompt}\n\n{STRICT_JSON_SUFFIX}"

def build_reasoning_prompt(base_prompt: str, facts: Dict[str, Any], cases: List[Dict[str, Any]]) -> str:
    facts_json = json.dumps(facts or {}, ensure_ascii=False)
    cases_json = json.dumps((cases or [])[:5], ensure_ascii=False)
    return (
        f"{base_prompt}\n\n"
        f"## 当前画面分析结果(JSON)\n{facts_json}\n\n"
        f"## 参考案例(JSON)\n{cases_json}\n\n"
        f"{STRICT_JSON_SUFFIX}"
    )