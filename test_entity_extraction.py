#!/usr/bin/env python3
"""
测试数标标 API 的实体提取返回格式
"""

import os
import requests
import json

# 从环境变量获取 API 密钥
api_key = os.environ.get('SHUBIAOBIAO_API_KEY', 'sk-6IPAckVNGiQCP8Q9O2sAl4TV7KzGmNiGS8YuGsPHF0t406av')
api_base = os.environ.get('SHUBIAOBIAO_API_BASE', 'https://hk.n1n.ai/v1')

print(f"Testing API: {api_base}")
print(f"API Key: {api_key[:4]}...{api_key[-4:]}")

# 测试实体提取 - 使用 GraphRAG 的提示词格式
def test_entity_extraction():
    print("\n=== Testing entity extraction with GraphRAG prompt format ===")
    url = f"{api_base}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # 使用 GraphRAG 的提示词格式
    prompt = """-Goal-
Given a text document that is relevant to this activity, identify all entities of the specified types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [organization, person, geo, event]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"|entity_name|entity_type|entity_description)

2. From the entities identified in step 1, identify all *clearly related* (source_entity, target_entity) pairs.
For each pair, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relationship_description: Explanation of why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score indicating the strength of the relationship between the source entity and the target entity
Format each relationship as ("relationship"|source_entity|target_entity|relationship_description|relationship_strength)

3. Return output as a single list of all entities and relationships identified in steps 1 and 2, using **
** as the list delimiter.

4. When finished, output 


######################
-Real Data-
######################
Entity_types: organization, person, geo, event
Text: 1978年12月，中国共产党十一届三中全会在北京召开。这次会议确立了"解放思想、实事求是"的思想路线，做出了把工作重点转移到社会主义现代化建设上来的战略决策。邓小平是这次会议的主要推动者。
######################
Output:
"""
    
    data = {
        "model": "gemini-2.5-flash-nothinking",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 2000
    }
    
    try:
        print(f"Sending request to {url}...")
        print(f"Model: gemini-2.5-flash-nothinking")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"\n=== Response Content ===")
            print(content)
            print(f"\n=== Usage ===")
            print(f"Prompt tokens: {result.get('usage', {}).get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
            print(f"Total tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            
            # 检查返回格式
            if '("entity"' in content or '("relationship"' in content:
                print("\n✓ Response contains entity/relationship format")
            else:
                print("\n✗ Response does NOT contain expected entity/relationship format")
                print("This might be why GraphRAG cannot detect entities!")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置环境变量
    os.environ['SHUBIAOBIAO_API_KEY'] = api_key
    os.environ['SHUBIAOBIAO_API_BASE'] = api_base
    
    print("Testing Shubiaobiao API with GraphRAG format...")
    test_entity_extraction()
