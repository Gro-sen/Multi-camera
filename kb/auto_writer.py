import os
import hashlib
from datetime import datetime
import json
import threading
import queue
import time

KB_SOURCE_DIR = "kb/source"
KB_INDEX_DIR = "kb/index"

# 从环境读取阈值（可通过 app.core.config.KB_INDEX_UPDATE_THRESHOLD 覆盖）
KB_INDEX_UPDATE_THRESHOLD = int(os.getenv("KB_INDEX_UPDATE_THRESHOLD", "20"))

# 内部通知队列与守护线程
_notify_queue = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def write_alarm_case_to_kb(case: dict):
    """将报警案例写入知识库（Markdown格式）"""
    os.makedirs(KB_SOURCE_DIR, exist_ok=True)
    os.makedirs(KB_INDEX_DIR, exist_ok=True)

    # 调试输出（保留简洁日志）
    if 'metadata' in case:
        try:
            print(f"【AUTO_WRITER】metadata: {json.dumps(case['metadata'], ensure_ascii=False)}")
        except Exception:
            pass

    alarm_level = case.get('alarm_level', '一般')
    scene_summary = case.get('scene_summary', '')
    alarm_reason = case.get('alarm_reason', '')

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

    filename = f"case_{case_id}.md"
    path = os.path.join(KB_SOURCE_DIR, filename)

    if os.path.exists(path):
        counter = 1
        while os.path.exists(path):
            filename = f"case_{case_id}_v{counter}.md"
            path = os.path.join(KB_SOURCE_DIR, filename)
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

    content = f"""# 报警案例：{alarm_level}级报警

## 案例信息
- **案例ID**: {case_id}
- **触发时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
    print(f"【知识库】案例已保存：{path}")
    print(f"【知识库】模型: {model_used}, 参考案例数: {kb_cases_used}")

    # 只发通知给后台 worker（非阻塞）
    _enqueue_index_update_notification()


def _enqueue_index_update_notification():
    """向后台 worker 发出通知（非阻塞）。worker 负责计数与触发重建。"""
    try:
        _notify_queue.put_nowait(1)
        _ensure_worker_running()
    except Exception:
        # 不应因队列失败影响主流程
        pass


def _ensure_worker_running():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_index_worker, name="kb-index-worker", daemon=True)
            t.start()
            _worker_started = True
            print("【知识库】索引更新后台 worker 已启动")


def _index_worker():
    """后台 worker：累积通知并在达到阈值时重建索引。"""
    counter = 0
    last_notify = 0.0
    debounce_seconds = 2  # 收集短时间内的多次写入
    while True:
        try:
            # 阻塞等待一次通知（最长等待 60 秒，避免空转）
            try:
                _notify_queue.get(timeout=60)
            except Exception:
                # 超时，继续循环以支持守护线程长期运行
                continue

            counter += 1
            last_notify = time.time()

            # 等待短的抖动窗口，合并多次写入
            time.sleep(debounce_seconds)
            # 尝试快速清空队列并累积
            while True:
                try:
                    _notify_queue.get_nowait()
                    counter += 1
                except Exception:
                    break

            if counter < KB_INDEX_UPDATE_THRESHOLD:
                print(f"【知识库】索引更新计数: {counter}/{KB_INDEX_UPDATE_THRESHOLD}（暂不重建）")
                # 继续等待更多通知
                continue

            # 达到阈值：执行重建
            print("【知识库】计数达到阈值，开始重建索引（后台）...")
            try:
                # 小延迟以降低与当前写入的竞争
                time.sleep(1)
                from kb.indexing import build_index
                result = build_index(
                    data_dir=KB_SOURCE_DIR,
                    index_path=os.path.join(KB_INDEX_DIR, 'faiss_bge.index'),
                    meta_path=os.path.join(KB_INDEX_DIR, 'docs_bge.pkl'),
                    model_name=os.getenv('KB_EMBED_MODEL_PATH', 'D:/code/python/git/Multi-camera/bge-small-zh-v1.5')
                )
                if result.get('status') == 'success':
                    print(f"✅ 索引重建成功，块数量: {result.get('chunks_count')}")
                    # 刷新检索器缓存，让新索引生效
                    try:
                        from kb.retriever import refresh_cache
                        refresh_cache()
                        print("✅ 检索器缓存已刷新")
                    except Exception as e:
                        print(f"⚠️ 刷新检索缓存失败: {e}")
                else:
                    print(f"❌ 索引重建失败: {result.get('message')}")
            except Exception as e:
                print(f"【ERROR】后台重建索引时发生异常: {e}")
            finally:
                # 重置计数
                counter = 0

        except Exception as e:
            # 捕获并记录异常，继续运行
            print(f"【ERROR】索引后台线程异常: {e}")
            time.sleep(2)