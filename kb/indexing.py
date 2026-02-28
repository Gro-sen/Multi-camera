import os
import glob
import pickle
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def smart_chunk_text(text, source_file, max_chars=400):
    """智能分块文本，保持语义完整性"""
    chunks = []
    
    # 按标题分割（## 标题）
    title_sections = re.split(r'(?=\n## )', text.strip())
    
    for section in title_sections:
        if not section.strip():
            continue
        
        # 提取标题
        title_match = re.match(r'^(#+\s+.+?)\n', section)
        title = title_match.group(1) if title_match else "无标题"
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', section)
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            para_length = len(para)
            
            # 如果段落本身就很大，需要再分割
            if para_length > max_chars:
                # 按句子分割
                sentences = re.split(r'[。！？.!?]\s*', para)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    sent_length = len(sentence)
                    if current_length + sent_length <= max_chars:
                        current_chunk.append(sentence)
                        current_length += sent_length
                    else:
                        # 保存当前块
                        if current_chunk:
                            chunk_text = '。'.join(current_chunk) + '。'
                            chunks.append({
                                'text': chunk_text,
                                'source': source_file,
                                'type': 'paragraph',
                                'title': title
                            })
                        
                        # 开始新块
                        current_chunk = [sentence]
                        current_length = sent_length
            else:
                # 段落适合当前块
                if current_length + para_length <= max_chars:
                    current_chunk.append(para)
                    current_length += para_length
                else:
                    # 保存当前块
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'source': source_file,
                            'type': 'paragraph_group',
                            'title': title
                        })
                    
                    # 开始新块
                    current_chunk = [para]
                    current_length = para_length
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'source': source_file,
                'type': 'paragraph_group',
                'title': title
            })
    
    # 如果没有分块，则按固定长度分割
    if not chunks:
        for i in range(0, len(text), max_chars):
            chunk_text = text[i:i+max_chars]
            chunks.append({
                'text': chunk_text,
                'source': source_file,
                'type': 'fixed_length',
                'title': '未分块内容'
            })
    
    return chunks

def build_index(data_dir='kb/source', 
                index_path='kb/index/faiss_bge.index',
                meta_path='kb/index/docs_bge.pkl',
                model_name='BAAI/bge-small-zh-v1.5'):
    """构建知识库索引"""
    
    print("🔨 开始构建知识库索引...")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
    
    # 加载模型
    print(f"📥 加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"   模型维度: {dim}")
    
    # 查找所有Markdown文件
    files = glob.glob(os.path.join(data_dir, '*.md'))
    
    if not files:
        print("⚠️  没有找到知识库文件")
        return {'status': 'error', 'message': 'No source files found'}
    
    print(f"📚 找到 {len(files)} 个知识库文件")
    
    # 分块处理
    all_chunks = []
    
    for filepath in files:
        filename = Path(filepath).name
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            # 智能分块
            chunks = smart_chunk_text(content, filename, max_chars=500)
            
            print(f"  {filename}: {len(chunks)} 个块")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"  处理文件 {filename} 失败: {e}")
    
    if not all_chunks:
        print("❌ 没有生成有效的文档块")
        return {'status': 'error', 'message': 'No chunks generated'}
    
    print(f"📊 总共生成 {len(all_chunks)} 个文档块")
    
    # 准备文本用于编码
    texts = [chunk['text'] for chunk in all_chunks]
    
    # 创建内积索引（余弦相似度）
    index = faiss.IndexFlatIP(dim)
    
    # 分批处理向量化
    batch_size = 32
    print("⚡ 生成向量嵌入...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 编码并归一化
        embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 关键：归一化向量
            show_progress_bar=False
        ).astype('float32')
        
        index.add(embeddings)
        
        progress = min(i + batch_size, len(texts)) / len(texts) * 100
        print(f"  进度: {progress:.1f}%", end='\r')
    
    print(f"\n✅ 向量嵌入完成")
    
    # 保存索引
    faiss.write_index(index, index_path)
    print(f"💾 索引已保存: {index_path}")
    
    # 保存元数据
    with open(meta_path, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"💾 元数据已保存: {meta_path}")
    
    # 统计信息
    chunk_types = {}
    for chunk in all_chunks:
        chunk_type = chunk.get('type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    print(f"\n📈 统计信息:")
    print(f"  总文档块: {len(all_chunks)}")
    print(f"  索引维度: {dim}")
    print(f"  索引类型: 内积索引 (余弦相似度)")
    print(f"  块类型分布:")
    for t, c in chunk_types.items():
        print(f"    {t}: {c}")
    
    return {
        'status': 'success',
        'chunks_count': len(all_chunks),
        'dimension': dim,
        'index_path': index_path,
        'meta_path': meta_path,
        'model': model_name
    }

def rebuild_index():
    """重建索引（主入口函数）"""
    print("="*60)
    print("知识库索引重建工具")
    print("="*60)
    
    result = build_index()
    
    if result['status'] == 'success':
        print(f"\n✅ 索引重建成功！")
        print(f"   文档块数量: {result['chunks_count']}")
        print(f"   模型: {result['model']}")
        print(f"   索引文件: {result['index_path']}")
        print(f"\n⚠️  请重启应用以使用新索引")
    else:
        print(f"\n❌ 索引重建失败: {result.get('message', '未知错误')}")
    
    return result

if __name__ == "__main__":
    rebuild_index()