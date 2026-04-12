#!/usr/bin/env python3
"""
测试数标标 API 的实体提取返回格式 - 使用 GraphRAG 期望的分隔符格式
"""

import os
import requests

# 从环境变量获取 API 密钥
api_key = os.environ.get('SHUBIAOBIAO_API_KEY', 'sk-6IPAckVNGiQCP8Q9O2sAl4TV7KzGmNiGS8YuGsPHF0t406av')
api_base = os.environ.get('SHUBIAOBIAO_API_BASE', 'https://hk.n1n.ai/v1')

print(f"Testing API: {api_base}")

# 读取修改后的提示词
with open('data/graphrag_output/prompts/extract_graph.txt', 'r', encoding='utf-8') as f:
    prompt_template = f.read()

# 替换变量
entity_types = "organization, person, geo, event"
input_text = "1978年12月，中国共产党十一届三中全会在北京召开。这次会议确立了\"解放思想、实事求是\"的思想路线，做出了把工作重点转移到社会主义现代化建设上来的战略决策。邓小平是这次会议的主要推动者。"

prompt = prompt_template.replace('{entity_types}', entity_types).replace('{input_text}', input_text)

print("\n=== Prompt (first 800 chars) ===")
print(prompt[:800] + "...")

data = {
    "model": "gemini-2.5-flash-nothinking",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0,
    "max_tokens": 2000
}

try:
    url = f"{api_base}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    print(f"\nSending request...")
    response = requests.post(url, headers=headers, json=data, timeout=60)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"\n=== Response Content ===")
        print(content)
        print(f"\n=== Usage ===")
        print(f"Total tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
        
        # 使用 GraphRAG 的方式解析
        TUPLE_DELIMITER = "<|>"
        RECORD_DELIMITER = "##"
        COMPLETION_DELIMITER = "<|COMPLETE|>"
        
        records = [r.strip() for r in content.split(RECORD_DELIMITER)]
        print(f"\n=== Analysis ===")
        print(f"Number of records: {len(records)}")
        
        entity_count = 0
        relationship_count = 0
        for raw_record in records:
            record = raw_record.strip()
            if not record or record == COMPLETION_DELIMITER:
                continue
            # Remove parentheses
            record = record.strip('()')
            record_attributes = record.split(TUPLE_DELIMITER)
            record_type = record_attributes[0] if record_attributes else ""
            
            if record_type == '"entity"' and len(record_attributes) >= 4:
                entity_count += 1
                print(f"  Entity: {record_attributes[1]} ({record_attributes[2]})")
            elif record_type == '"relationship"' and len(record_attributes) >= 5:
                relationship_count += 1
                print(f"  Relationship: {record_attributes[1]} -> {record_attributes[2]}")
        
        print(f"\nEntities found: {entity_count}")
        print(f"Relationships found: {relationship_count}")
        
        if entity_count > 0:
            print("\n✓ Response contains entities - GraphRAG should be able to parse this!")
        else:
            print("\n✗ No entities found - GraphRAG will fail!")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
