#!/usr/bin/env python3
"""
通用历史事件文件处理脚本
处理隔一行一个事件的txt文件格式
"""

import os
from pathlib import Path
import re

def process_generic_file(input_path: str, output_path: str, location: str, category: str):
    """处理通用历史事件文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        location: 事件地点
        category: 事件类别
    """
    events = []
    current_event_lines = []
    
    print(f"开始处理文件：{input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    # 空行，保存当前事件
                    if current_event_lines:
                        event_text = ' '.join(current_event_lines)
                        
                        # 提取年份
                        year_match = re.search(r'(\d{4}年)', event_text)
                        if year_match:
                            year = year_match.group(1)
                            title = f"{category} - {year}"
                            date = year
                        else:
                            title = category
                            date = "未知"
                        
                        events.append({
                            'title': title,
                            'date': date,
                            'location': location,
                            'content': event_text
                        })
                        current_event_lines = []
                else:
                    # 非空行，添加到当前事件
                    current_event_lines.append(line)
        
        # 处理最后一个事件
        if current_event_lines:
            event_text = ' '.join(current_event_lines)
            
            # 提取年份
            year_match = re.search(r'(\d{4}年)', event_text)
            if year_match:
                year = year_match.group(1)
                title = f"{category} - {year}"
                date = year
            else:
                title = category
                date = "未知"
            
            events.append({
                'title': title,
                'date': date,
                'location': location,
                'content': event_text
            })
        
        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, event in enumerate(events):
                if i > 0:
                    f.write('---\n\n')
                f.write(f"事件标题：{event['title']}\n")
                f.write(f"时间：{event['date']}\n")
                f.write(f"地点：{event['location']}\n\n")
                f.write(f"{event['content']}\n")
        
        print(f"处理完成：{input_path} -> {output_path}")
        print(f"生成事件数：{len(events)}")
        
    except Exception as e:
        print(f"处理文件时出错：{e}")

def main():
    """主函数"""
    input_dir = Path(r'd:\projects\Capstone\GraphRAG\data\output\input')
    output_dir = Path(r'd:\projects\Capstone\GraphRAG\data\input')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理深圳文件
    shenzhen_input = input_dir / 'shenzhen.txt'
    shenzhen_output = output_dir / 'shenzhen_processed.txt'
    if shenzhen_input.exists():
        process_generic_file(shenzhen_input, shenzhen_output, '深圳', '深圳经济特区发展')
    
    # 处理广东文件
    guangdong_input = input_dir / 'guangdong.txt'
    guangdong_output = output_dir / 'guangdong_processed.txt'
    if guangdong_input.exists():
        process_generic_file(guangdong_input, guangdong_output, '广东', '广东改革开放')
    
    # 处理中国文件
    zhongguo_input = input_dir / 'zhongguo.txt'
    zhongguo_output = output_dir / 'zhongguo_processed.txt'
    if zhongguo_input.exists():
        process_generic_file(zhongguo_input, zhongguo_output, '中国', '中国现代史')
    
    print("\n所有文件处理完成！")
    print(f"处理后的文件保存在：{output_dir}")

if __name__ == "__main__":
    main()
