"""
知识库规则源读取器。

只从 kb/source 读取规则数据，不在代码里写业务规则。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


SOURCE_DIR = Path(__file__).resolve().parent / "source"
RULE_FILE = SOURCE_DIR / "rules.json"


@lru_cache(maxsize=1)
def load_rules() -> Dict[str, Any]:
    if not RULE_FILE.exists():
        return {}
    try:
        data = json.loads(RULE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_prompt(name: str, default: str = "") -> str:
    prompts = load_rules().get("prompts", {})
    value = prompts.get(name, default)
    return str(value if value is not None else default)


def get_fast_rules() -> List[Dict[str, Any]]:
    return list(load_rules().get("fast_rules", []) or [])


def get_badge_face_rules() -> List[Dict[str, Any]]:
    return list(load_rules().get("badge_face_rules", []) or [])


def build_query_text(vision_facts: Any, face_result: Optional[Any] = None) -> str:
    query_map = load_rules().get("query_tokens", {})
    query_parts = list(query_map.get("base", []) or [])

    def add_tokens(key: str) -> None:
        for token in query_map.get(key, []) or []:
            if token not in query_parts:
                query_parts.append(token)

    if bool(getattr(vision_facts, "has_person", False)):
        add_tokens("has_person")
    else:
        add_tokens("no_person")

    if getattr(vision_facts, "badge_status", "不适用") in ["未佩戴", "无法确认"]:
        add_tokens("badge_issue")

    if face_result is not None and bool(getattr(face_result, "enabled", False)):
        detected_faces = int(getattr(face_result, "detected_faces", 0) or 0)
        matched = bool(getattr(face_result, "matched", False))
        if detected_faces <= 0:
            add_tokens("face_missing")
        elif matched:
            add_tokens("face_matched")
        else:
            add_tokens("face_unknown")

    if bool(getattr(vision_facts, "enter_restricted_area", False)):
        add_tokens("restricted_area")
    if bool(getattr(vision_facts, "has_fire_or_smoke", False)):
        add_tokens("fire")
    if bool(getattr(vision_facts, "has_electric_risk", False)):
        add_tokens("electric")

    return " ".join(query_parts)


def _matches_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def select_first_rule(rules: List[Dict[str, Any]], vision_facts: Any, face_result: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    context = {
        "has_person": bool(getattr(vision_facts, "has_person", False)),
        "badge_status": getattr(vision_facts, "badge_status", "不适用"),
        "enter_restricted_area": bool(getattr(vision_facts, "enter_restricted_area", False)),
        "has_fire_or_smoke": bool(getattr(vision_facts, "has_fire_or_smoke", False)),
        "has_electric_risk": bool(getattr(vision_facts, "has_electric_risk", False)),
        "face_known": False,
        "face_unknown_or_missing": False,
        "face_unknown": False,
        "face_missing": False,
    }

    if face_result is not None:
        detected_faces = int(getattr(face_result, "detected_faces", 0) or 0)
        matched = bool(getattr(face_result, "matched", False))
        context["face_known"] = detected_faces > 0 and matched
        context["face_unknown_or_missing"] = detected_faces <= 0 or (detected_faces > 0 and not matched)
        context["face_unknown"] = detected_faces > 0 and not matched
        context["face_missing"] = detected_faces <= 0

    for rule in rules or []:
        skip_flags = rule.get("skip_if_any", []) or []
        if any(context.get(flag) for flag in skip_flags):
            continue

        conditions = rule.get("conditions", {}) or {}
        if all(_matches_expected(context.get(key), expected) for key, expected in conditions.items()):
            return rule

    return None
