#!/usr/bin/env python3
"""
历史数据文件处理脚本
将不同格式的历史事件文件转换为系统要求的格式
"""

import os
from pathlib import Path
import re

def process_shenzhen_file(input_path: str, output_path: str):
    """处理深圳历史事件文件"""
    events = []
    current_event = None
    line_count = 0
    
    print(f"开始处理深圳文件：{input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line_content = line.rstrip()  # 保留换行符之外的空白
            print(f"第{line_count}行：'{line_content}'")
            
            line = line.strip()
            if not line:
                print("  跳过空行")
                continue
            
            # 检查是否以●开头（处理不同的编码）
            if line.startswith('●') or line.startswith('\u25cf'):
                print(f"  找到事件开始：{line}")
                # 保存当前事件
                if current_event:
                    events.append(current_event)
                    print(f"  保存事件：{current_event['title']}")
                
                # 开始新事件
                event_text = line[1:].strip()
                
                # 提取年份
                year_match = re.search(r'(\d{4}年)', event_text)
                if year_match:
                    year = year_match.group(1)
                    title = f"深圳经济特区发展 - {year}"
                    date = year
                else:
                    title = "深圳经济特区发展"
                    date = "1980年-2020年"
                
                current_event = {
                    'title': title,
                    'date': date,
                    'location': '深圳',
                    'content': event_text
                }
                print(f"  开始新事件：{title}")
            elif current_event:
                # 继续当前事件的内容
                current_event['content'] += '\n' + line
                print(f"  添加内容到当前事件")
            else:
                print(f"  跳过行（无当前事件）：{line}")
    
    # 保存最后一个事件
    if current_event:
        events.append(current_event)
        print(f"  保存最后事件：{current_event['title']}")
    
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

def process_guangdong_file(input_path: str, output_path: str):
    """处理广东历史事件文件"""
    events = []
    
    print(f"开始处理广东文件：{input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 提取序号和内容
            match = re.match(r'\s*\d+\.(.*)', line)
            if match:
                event_text = match.group(1).strip()
                
                # 提取年份
                year_match = re.search(r'(\d{4}年)', event_text)
                if year_match:
                    year = year_match.group(1)
                    title = f"广东改革开放 - {year}"
                    date = year
                else:
                    title = "广东改革开放"
                    date = "1977年-1988年"
                
                events.append({
                    'title': title,
                    'date': date,
                    'location': '广东',
                    'content': event_text
                })
            else:
                print(f"跳过行：{line[:50]}...")
    
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

def process_zhongguo_file(input_path: str, output_path: str):
    """处理中国历史事件文件"""
    events = []
    current_year = ""
    
    print(f"开始处理中国文件：{input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是年份行
            if re.match(r'^[一|\d]+年$', line):
                current_year = line
                print(f"找到年份：{current_year}")
                continue
            
            # 处理事件行
            if line and current_year:
                # 提取日期
                date_match = re.match(r'(\d+月\d+日)', line)
                if date_match:
                    date = f"{current_year} {date_match.group(1)}"
                else:
                    date = current_year
                
                # 生成标题
                title_start = line[:50].strip()
                title = f"中国现代史 - {current_year} - {title_start}"
                
                events.append({
                    'title': title,
                    'date': date,
                    'location': '中国',
                    'content': line
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

def main():
    """主函数"""
    input_dir = Path(r'd:\projects\Capstone\GraphRAG\data\output\input')
    output_dir = Path(r'd:\projects\Capstone\GraphRAG\data\input')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理深圳文件
    shenzhen_input = input_dir / 'shenzhen.txt'
    shenzhen_output = output_dir / 'shenzhen_processed.txt'
    if shenzhen_input.exists():
        process_shenzhen_file(shenzhen_input, shenzhen_output)
    
    # 处理广东文件
    guangdong_input = input_dir / 'guangdong.txt'
    guangdong_output = output_dir / 'guangdong_processed.txt'
    if guangdong_input.exists():
        process_guangdong_file(guangdong_input, guangdong_output)
    
    # 处理中国文件
    zhongguo_input = input_dir / 'zhongguo.txt'
    zhongguo_output = output_dir / 'zhongguo_processed.txt'
    if zhongguo_input.exists():
        process_zhongguo_file(zhongguo_input, zhongguo_output)
    
    print("\n所有文件处理完成！")
    print(f"处理后的文件保存在：{output_dir}")

if __name__ == "__main__":
    main()
