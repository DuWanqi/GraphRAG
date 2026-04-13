#!/usr/bin/env python3
"""
测试文件读取脚本
"""

import os
from pathlib import Path

def test_file_read():
    """测试文件读取"""
    test_files = [
        r'd:\projects\Capstone\GraphRAG\data\output\input\shenzhen.txt',
        r'd:\projects\Capstone\GraphRAG\data\output\input\guangdong.txt',
        r'd:\projects\Capstone\GraphRAG\data\output\input\zhongguo.txt'
    ]
    
    for file_path in test_files:
        print(f"\n测试文件：{file_path}")
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            print(f"文件存在")
            
            # 检查文件大小
            size = os.path.getsize(file_path)
            print(f"文件大小：{size} 字节")
            
            # 尝试读取文件
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"读取成功，内容长度：{len(content)} 字符")
                    print(f"前100字符：{content[:100]}...")
            except Exception as e:
                print(f"读取失败：{e}")
        else:
            print(f"文件不存在")

if __name__ == "__main__":
    test_file_read()
