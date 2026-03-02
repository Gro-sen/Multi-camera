import os
import hashlib
from datetime import datetime
import json
import threading
KB_SOURCE_DIR = "kb/source"
KB_INDEX_DIR = "kb/index"

def write_alarm_case_to_kb(case: dict):
    """将报警案例写入知识库（Markdown格式）"""
    os.makedirs(KB_SOURCE_DIR, exist_ok=True)
    os.makedirs(KB_INDEX_DIR, exist_ok=True)

    # 调试：打印case的所有键
    print(f"【DEBUG】case字典的键: {list(case.keys())}")
    if 'metadata' in case:
        print(f"【DEBUG】metadata内容: {json.dumps(case['metadata'], ensure_ascii=False)}")
    
    # 提取关键信息 - 修复字段提取路径
    alarm_level = case.get('alarm_level', '一般')
    scene_summary = case.get('scene_summary', '')
    alarm_reason = case.get('alarm_reason', '')
    
    # 获取case_id
    case_id = case.get('case_id')
    if not case_id:
        # 生成唯一case_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        scene_hash = hashlib.md5(scene_summary.encode()).hexdigest()[:8]
        case_id = f"{timestamp}_{scene_hash}"
    
    # 修复：从正确的路径提取metadata信息
    # 方法1：直接从metadata字段获取
    metadata = case.get('metadata', {})
    kb_total = metadata.get('kb_total_references', 0)
    kb_rules = metadata.get('kb_rule_files', 0)
    kb_history = metadata.get('kb_history_cases', metadata.get('kb_cases_used', 0))
    reasoning_model = metadata.get('reasoning_model', metadata.get('model', '未知'))
    vision_model = metadata.get('vision_model', '未知')
    model_used = f"推理模型: {reasoning_model} | 视觉模型: {vision_model}"
    kb_cases_used = metadata.get('kb_cases_used', 0)
    print(f"【AUTO_WRITER】知识库使用详情:")
    print(f"  总参考文档: {kb_total}")
    print(f"  规则文件: {kb_rules}")
    print(f"  历史案例: {kb_history}")

    # 方法2：如果metadata中没有，尝试从case的其他位置获取
    if model_used == '未知':
        model_used = case.get('model_used', case.get('model', '未知'))
    
    if kb_cases_used == 0:
        kb_cases_used = case.get('kb_cases_used', 0)
    
    # 修复：从final_decision中提取决策信息
    final_decision = case.get('final_decision', {})
    is_alarm = final_decision.get('is_alarm', case.get('is_alarm', '未知'))
    confidence = final_decision.get('confidence', case.get('confidence', 0.0))
    
    # 修复：从analysis中提取分析信息
    analysis = case.get('analysis', {})
    risk_assessment = analysis.get('risk_assessment', case.get('risk_assessment', '无'))
    recommendation = analysis.get('recommendation', case.get('recommendation', '无'))
    
    # 生成文件名（使用case_id确保唯一）
    filename = f"case_{case_id}.md"
    path = os.path.join(KB_SOURCE_DIR, filename)
    
    # 检查文件是否已存在（防止重复写入）
    if os.path.exists(path):
        # 如果文件已存在，添加计数器后缀
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

    # 构建更详细的Markdown内容
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
    trigger_index_update() 
    return path

def trigger_index_update():
    """触发知识库索引更新（异步）"""
    def _update_index():
        try:
            # 延迟5秒，避免频繁重建
            import time
            time.sleep(5)
            
            # 检查是否有kb模块
            try:
                from kb.indexing import build_index
                
                print("【知识库】开始重建索引...")
                
                # 重建索引
                result = build_index(
                    data_dir='kb/source',
                    index_path='kb/index/faiss_bge.index',
                    meta_path='kb/index/docs_bge.pkl',
                    model_name='D:/code/python/git/Multi-camera/bge-small-zh-v1.5'
                )
                
                if result['status'] == 'success':
                    print(f"✅ 索引重建成功！文档块数量: {result['chunks_count']}")
                    
                    # 🔥 关键修改：延迟刷新缓存，避免影响正在进行的查询
                    time.sleep(2)  # 等待2秒，让当前查询完成
                    
                    # 刷新检索器缓存，让新索引立即生效
                    try:
                        from kb.retriever import refresh_cache
                        refresh_cache()
                        print("✅ 检索器缓存已刷新，新索引立即生效")
                    except ImportError as e:
                        print(f"⚠️ 无法刷新缓存: {e}")
                        
                else:
                    print(f"❌ 索引重建失败: {result.get('message', '未知错误')}")
                    
            except ImportError as e:
                print(f"【WARN】无法导入索引模块: {e}")
                
        except Exception as e:
            print(f"【ERROR】索引更新线程异常: {e}")
        
    # 启动异步线程更新索引
    threading.Thread(target=_update_index, daemon=True).start()