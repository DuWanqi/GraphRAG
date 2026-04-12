#!/usr/bin/env python3
"""
处理改革开放历史事件文件
格式：年份作为标题，下面列出该年份的事件
"""

from pathlib import Path
import re


def process_gaigekaifang_file(input_path: str, output_path: str):
    """
    处理改革开放历史事件文件
    
    格式示例：
    一九七八年
    
    12月18日—22日 中共十一届三中全会举行...
    
    1月1日 中国同美国正式建立外交关系...
    """
    print(f"处理文件：{input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按年份分割
    # 匹配模式：一九七八年、一九七九年等
    year_pattern = r'([一二三四五六七八九〇]{4}年)\s*\n'
    
    # 找到所有年份位置
    matches = list(re.finditer(year_pattern, content))
    
    events = []
    
    for i, match in enumerate(matches):
        year = match.group(1)
        start_pos = match.end()
        
        # 确定当前年份内容的结束位置
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        year_content = content[start_pos:end_pos].strip()
        
        # 分割该年份下的各个事件（按空行分割）
        event_blocks = year_content.split('\n\n')
        
        for block in event_blocks:
            block = block.strip()
            if not block:
                continue
            
            # 提取日期（如果有）
            date_match = re.match(r'(\d{1,2}月\d{1,2}日[—\-~～]?\d{0,2}日?)\s+', block)
            if date_match:
                date = date_match.group(1)
                event_text = block[date_match.end():].strip()
            else:
                date = ""
                event_text = block
            
            # 构建完整事件文本
            if date:
                full_event = f"{year} {date} {event_text}"
            else:
                full_event = f"{year} {event_text}"
            
            events.append(full_event)
    
    print(f"  找到 {len(events)} 个事件")
    
    # 写入输出文件，用---分隔
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, event in enumerate(events):
            if i > 0:
                f.write('\n---\n\n')
            f.write(event)
            f.write('\n')
    
    print(f"  输出到：{output_path}")
    print(f"  完成！\n")


def main():
    """主函数"""
    input_dir = Path(r'd:\projects\Capstone\GraphRAG\data\output\input')
    output_dir = Path(r'd:\projects\Capstone\GraphRAG\data\input')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理改革开放文件
    gaige_input = input_dir / 'gaigekaifang.txt'
    gaige_output = output_dir / 'gaigekaifang_processed.txt'
    if gaige_input.exists():
        process_gaigekaifang_file(gaige_input, gaige_output)
    
    print("所有文件处理完成！")
    print(f"处理后的文件保存在：{output_dir}")


if __name__ == "__main__":
    main()
