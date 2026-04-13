#!/usr/bin/env python3
"""
测试数标标 API 是否正常工作
"""

import os
import requests

# 从环境变量获取 API 密钥
api_key = os.environ.get('SHUBIAOBIAO_API_KEY', 'sk-6IPAckVNGiQCP8Q9O2sAl4TV7KzGmNiGS8YuGsPHF0t406av')
api_base = os.environ.get('SHUBIAOBIAO_API_BASE', 'https://hk.n1n.ai/v1')

print(f"Testing API: {api_base}")
print(f"API Key: {api_key[:4]}...{api_key[-4:]}")

# 测试 chat/completions 端点
def test_chat_completions():
    print("\n=== Testing chat/completions ===")
    url = f"{api_base}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

# 测试实体提取
def test_entity_extraction():
    print("\n=== Testing entity extraction ===")
    url = f"{api_base}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    prompt = """
请从以下文本中提取实体和关系：

文本：1978年12月，中国共产党十一届三中全会在北京召开。这次会议确立了"解放思想、实事求是"的思想路线，做出了把工作重点转移到社会主义现代化建设上来的战略决策。

实体类型：历史事件, 人物, 地点, 时间, 组织

请按照以下格式输出：
("entity"|实体名称|实体类型|实体描述)
("relationship"|源实体|目标实体|关系描述|关系强度)
    """
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    # 设置环境变量
    os.environ['SHUBIAOBIAO_API_KEY'] = api_key
    os.environ['SHUBIAOBIAO_API_BASE'] = api_base
    
    print("Testing Shubiaobiao API...")
    
    # 测试聊天完成
    chat_ok = test_chat_completions()
    
    # 测试实体提取
    entity_ok = test_entity_extraction()
    
    print("\n=== Test Results ===")
    print(f"Chat completions: {'✓' if chat_ok else '✗'}")
    print(f"Entity extraction: {'✓' if entity_ok else '✗'}")
