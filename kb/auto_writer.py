import os
import hashlib
from datetime import datetime
import json

KB_PEND_DIR = "kb/pend"


def write_case_to_pend(case: dict):
    """将推理案例写入待审核目录（Markdown格式，不触发索引重建）"""
    os.makedirs(KB_PEND_DIR, exist_ok=True)

    # 调试输出（保留简洁日志）
    if 'metadata' in case:
        try:
            print(f"【AUTO_WRITER】metadata: {json.dumps(case['metadata'], ensure_ascii=False)}")
        except Exception:
            pass

    alarm_level = case.get('alarm_level', '无')
    scene_summary = case.get('scene_summary', '')
    alarm_reason = case.get('alarm_reason', '无')

    case_id = case.get('case_id')
    if not case_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        scene_hash = hashlib.md5(scene_summary.encode('utf-8', errors='ignore')).hexdigest()[:8]
        case_id = f"{timestamp}_{scene_hash}"

    metadata = case.get('metadata', {})
    kb_total = metadata.get('kb_total_references', 0)
    kb_rules = metadata.get('kb_rule_files', 0)
    kb_history = metadata.get('kb_history_cases', metadata.get('kb_cases_used', 0))
    reasoning_model = metadata.get('reasoning_model', metadata.get('model', '未知'))
    vision_model = metadata.get('vision_model', '未知')
    model_used = f"推理模型: {reasoning_model} | 视觉模型: {vision_model}"
    kb_cases_used = metadata.get('kb_cases_used', 0)

    if model_used == '推理模型: 未知 | 视觉模型: 未知':
        model_used = case.get('model_used', case.get('model', '未知'))

    if kb_cases_used == 0:
        kb_cases_used = case.get('kb_cases_used', 0)

    final_decision = case.get('final_decision', {})
    is_alarm = final_decision.get('is_alarm', case.get('is_alarm', '未知'))
    confidence = final_decision.get('confidence', case.get('confidence', 0.0))

    analysis = case.get('analysis', {})
    risk_assessment = analysis.get('risk_assessment', case.get('risk_assessment', '无'))
    recommendation = analysis.get('recommendation', case.get('recommendation', '无'))

    is_alarm_text = str(is_alarm).strip()
    case_type = "alarm" if is_alarm_text == "是" else "normal"
    filename = f"{case_type}_case_{case_id}.md"
    path = os.path.join(KB_PEND_DIR, filename)

    if os.path.exists(path):
        counter = 1
        while os.path.exists(path):
            filename = f"{case_type}_case_{case_id}_v{counter}.md"
            path = os.path.join(KB_PEND_DIR, filename)
            counter += 1

    kb_reference_text = ""
    if kb_total > 0:
        if kb_history > 0 and kb_rules > 0:
            kb_reference_text = f"参考了 {kb_history} 个历史案例和 {kb_rules} 个规则文件"
        elif kb_history > 0:
            kb_reference_text = f"参考了 {kb_history} 个历史案例"
        elif kb_rules > 0:
            kb_reference_text = f"参考了 {kb_rules} 个规则文件"
    else:
        kb_reference_text = "未参考知识库"

    title = "报警案例" if case_type == "alarm" else "正常案例"
    content = f"""# {title}：{alarm_level}

## 案例信息
- **案例ID**: {case_id}
- **触发时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **案例类型**: {case_type}
- **报警级别**: {alarm_level}
- **是否报警**: {is_alarm}
- **置信度**: {confidence:.4f}

## 知识库参考
{kb_reference_text}

## 场景概述
{scene_summary}

## 报警原因
{alarm_reason}

## 最终决策
{json.dumps(final_decision, ensure_ascii=False, indent=2)}

## 系统信息
*使用模型: {model_used}

*知识库参考: 参考了 {kb_cases_used} 个历史案例

*图片路径: {case.get('image_path', '无')}

## 时间线:
视觉分析: {case.get('timestamp', '未知')}

案例生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

##关键词
{alarm_level}级报警
{scene_summary[:50].replace(',', '')}
{alarm_reason[:50].replace(',', '')}

*案例ID: {case_id}
*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"【待审核】案例已保存：{path}")
    print(f"【知识库】模型: {model_used}, 参考案例数: {kb_cases_used}")
    print("【待审核】未自动重建索引，请人工筛选后再导入 kb/source")


def write_alarm_case_to_kb(case: dict):
    """兼容旧调用：统一写入待审核目录"""
    write_case_to_pend(case)