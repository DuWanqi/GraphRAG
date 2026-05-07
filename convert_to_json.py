#!/usr/bin/env python3
"""
将 input 目录下的所有文件（txt 和 json）统一转换为 JSON 格式
GraphRAG 要求的 JSON 格式: [{"id": "1", "text": "内容"}, ...]
"""

import json
import os
from pathlib import Path

input_dir = Path("data/graphrag_output/input")
output_dir = Path("data/graphrag_output/input_json")
output_dir.mkdir(exist_ok=True)

all_documents = []
doc_id = 0

# 处理所有文件
for file_path in sorted(input_dir.iterdir()):
    if file_path.is_file():
        print(f"处理: {file_path.name}")
        
        if file_path.suffix.lower() == '.json':
            # JSON 文件直接读取
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 如果是列表，遍历每个元素
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # 提取文本内容
                            text = item.get('content', item.get('text', item.get('description', '')))
                            # 同时包含日期和标题信息，确保年份被保留
                            date = item.get('date', '')
                            title = item.get('title', '')
                            location = item.get('location', '')
                            
                            # 组合完整文本：标题 + 日期 + 地点 + 内容
                            full_text_parts = []
                            if title:
                                full_text_parts.append(f"{title}")
                            if date:
                                full_text_parts.append(f"({date})")
                            if location:
                                full_text_parts.append(f"地点：{location}")
                            if text:
                                full_text_parts.append(text)
                            
                            full_text = "\n".join(full_text_parts)
                            
                            if full_text.strip():
                                all_documents.append({
                                    "id": str(doc_id),
                                    "text": full_text,
                                    "source": file_path.name
                                })
                                doc_id += 1
                # 如果是字典，直接处理
                elif isinstance(data, dict):
                    text = data.get('content', data.get('text', data.get('description', '')))
                    date = data.get('date', '')
                    title = data.get('title', '')
                    location = data.get('location', '')
                    
                    full_text_parts = []
                    if title:
                        full_text_parts.append(f"{title}")
                    if date:
                        full_text_parts.append(f"({date})")
                    if location:
                        full_text_parts.append(f"地点：{location}")
                    if text:
                        full_text_parts.append(text)
                    
                    full_text = "\n".join(full_text_parts)
                    
                    if full_text.strip():
                        all_documents.append({
                            "id": str(doc_id),
                            "text": full_text,
                            "source": file_path.name
                        })
                        doc_id += 1
            except Exception as e:
                print(f"  错误: {e}")
                
        elif file_path.suffix.lower() == '.txt':
            # TXT 文件读取内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if content.strip():
                    all_documents.append({
                        "id": str(doc_id),
                        "text": content,
                        "source": file_path.name
                    })
                    doc_id += 1
            except Exception as e:
                print(f"  错误: {e}")

print(f"\n总共处理了 {len(all_documents)} 个文档")

# 保存为统一的 JSON 文件
output_file = output_dir / "all_documents.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_documents, f, ensure_ascii=False, indent=2)

print(f"已保存到: {output_file}")
