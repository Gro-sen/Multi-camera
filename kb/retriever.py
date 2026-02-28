# retriever.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import threading
import time

# 全局缓存
_cached_model = None
_cached_index = None
_cached_meta = None

# 使用RLock（可重入锁）避免死锁
_cache_lock = threading.RLock()
_loading_in_progress = False  # 标记是否正在加载

def load_index(index_path='kb/index/faiss_bge.index', 
               meta_path='kb/index/docs_bge.pkl', 
               model_name='BAAI/bge-small-zh-v1.5'):
    """加载索引、元数据和模型（线程安全）"""
    global _cached_model, _cached_index, _cached_meta, _loading_in_progress
    
    with _cache_lock:
        # 双重检查锁定模式
        if _cached_index is not None and _cached_meta is not None and _cached_model is not None:
            return _cached_index, _cached_meta, _cached_model
        
        if _loading_in_progress:
            # 如果正在加载，等待
            while _loading_in_progress:
                time.sleep(0.1)
            return _cached_index, _cached_meta, _cached_model
        
        _loading_in_progress = True
        
        try:
            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                raise FileNotFoundError('Index or metadata not found. Run indexing first.')
            
            print("【检索器】开始加载索引和模型...")
            
            # 1. 加载模型（最耗时）
            print("  - 加载BGE模型...")
            start = time.time()
            _cached_model = SentenceTransformer(model_name)
            print(f"    ✅ 模型加载完成，耗时: {time.time()-start:.1f}秒")
            
            # 2. 加载索引
            print("  - 加载FAISS索引...")
            start = time.time()
            _cached_index = faiss.read_index(index_path)
            print(f"    ✅ 索引加载完成，耗时: {time.time()-start:.1f}秒")
            print(f"      索引大小: {_cached_index.ntotal}")
            
            # 3. 加载元数据
            print("  - 加载元数据...")
            start = time.time()
            with open(meta_path, 'rb') as f:
                _cached_meta = pickle.load(f)
            print(f"    ✅ 元数据加载完成，耗时: {time.time()-start:.1f}秒")
            print(f"      元数据数量: {len(_cached_meta)}")
            
            print("✅ 所有资源加载完成")
            return _cached_index, _cached_meta, _cached_model
            
        finally:
            _loading_in_progress = False

def query(query_text: str, top_k=5, similarity_threshold=0.3):
    """查询相似文档（线程安全）"""
    try:
        with _cache_lock:
            # 确保索引已加载
            if _cached_index is None or _cached_meta is None or _cached_model is None:
                load_index()
            
            # 复制引用到局部变量，避免在锁释放后被修改
            index = _cached_index
            meta = _cached_meta
            model = _cached_model
        
        # 检查复制后的引用是否有效
        if index is None or meta is None or model is None:
            print("【检索器】索引未加载，返回空结果")
            return []
        
        # BGE模型建议的查询格式
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_with_instruction = instruction + query_text
        
        # 计算查询向量
        q_emb = model.encode([query_with_instruction], 
                            convert_to_numpy=True,
                            normalize_embeddings=True).astype('float32')
        
        # 检索（使用内积距离）
        distances, indices = index.search(q_emb, top_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(meta):
                continue
            
            # 将内积距离转换为相似度（内积范围[-1,1]，转换为[0,1]）
            similarity = (distance + 1) / 2.0
            
            if similarity >= similarity_threshold:
                results.append({
                    'score': float(similarity),
                    'source': meta[idx]['source'],
                    'text': meta[idx]['text'].strip(),
                    'distance': float(distance)
                })
        
        print(f"【检索器】查询 '{query_text[:30]}...' 返回 {len(results)} 个结果")
        return results
        
    except Exception as e:
        print(f"【检索器】查询过程中出错: {e}")
        # 返回空结果而不是抛出异常
        return []

def refresh_cache():
    """刷新缓存，强制重新加载索引"""
    global _cached_model, _cached_index, _cached_meta
    with _cache_lock:
        _cached_model = None
        _cached_index = None
        _cached_meta = None
    print("【检索器】缓存已刷新")