#!/usr/bin/env python3
"""
基于 guangdong_2004_2025_470.json 生成2009年广州外贸回忆录
"""

import json

# 读取广东历史事件
with open('data/graphrag_output/input/guangdong_2004_2025_470.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

# 筛选2008-2010年的事件
period_events = []
for event in events:
    date = event.get('date', '')
    if '2008' in date or '2009' in date or '2010' in date:
        period_events.append(event)

print(f"找到 {len(period_events)} 个2008-2010年的事件\n")
print("=" * 60)

# 分类展示
categories = {
    '经济/外贸': [],
    '交通/基建': [],
    '文化/教育': [],
    '社会/民生': [],
    '科技/产业': [],
    '其他': []
}

for event in period_events:
    title = event.get('title', '')
    content = event.get('content', '')
    date = event.get('date', '')
    keywords = event.get('keywords', [])
    
    # 根据关键词分类
    if any(k in str(keywords) + title + content for k in ['外贸', '经济', 'GDP', '金融', '出口', '贸易']):
        categories['经济/外贸'].append(event)
    elif any(k in str(keywords) + title + content for k in ['地铁', '交通', '高铁', '机场', '道路']):
        categories['交通/基建'].append(event)
    elif any(k in str(keywords) + title + content for k in ['文化', '教育', '图书馆', '博物馆', '艺术']):
        categories['文化/教育'].append(event)
    elif any(k in str(keywords) + title + content for k in ['社保', '医疗', '民生', '就业', '扶贫']):
        categories['社会/民生'].append(event)
    elif any(k in str(keywords) + title + content for k in ['科技', '创新', '产业', '高交会', '研发']):
        categories['科技/产业'].append(event)
    else:
        categories['其他'].append(event)

# 打印分类结果
for category, events_list in categories.items():
    if events_list:
        print(f"\n【{category}】({len(events_list)}个事件)")
        print("-" * 60)
        for i, event in enumerate(events_list[:5], 1):  # 只显示前5个
            print(f"{i}. {event.get('date')} - {event.get('title')}")
            print(f"   {event.get('content')[:60]}...")
            print(f"   关键词: {', '.join(event.get('keywords', []))}")
            print()

# 生成回忆录文本
print("\n" + "=" * 60)
print("生成的回忆录文本")
print("=" * 60)

memoir_text = """
2009年广州外贸回忆录

2009年春天，我来到广州，在白云区一家外贸公司找到了工作。
那正是金融危机最严重的时候，很多外贸企业都在裁员甚至倒闭。
但广州作为外贸重镇，依然有着顽强的生命力。

每天早上，我挤着地铁三号线去公司。这条线路是2006年底才开通的，
连接了天河区和白云区，对我们这些在白云区上班、住在天河区的人来说，
简直是救星。虽然挤，但比坐公交快多了。

公司主要做纺织品和服装出口，客户主要在欧美。2009年的广交会，
虽然来的外商明显少了，但我们还是通过老客户介绍，签下了几个意向单。
那时候真是每天都在打电话、发邮件，生怕丢了一个客户。

周末的时候，我喜欢去上下九步行街逛逛，吃一碗银记肠粉。
有时候也会去广州图书馆新馆看书学习，那里环境特别好，
是2009年才正式开放的，成了我周末常去的地方。

那年印象最深的是公司年会，老板带我们去长隆旅游度假区玩了一天。
看了精彩的大马戏表演，那是我第一次看这么震撼的演出。
虽然2009年外贸形势不好，但公司还是尽力给我们这些员工一些温暖。

年底的时候，听说广东GDP突破了3.9万亿，虽然受了金融危机影响，
但广东经济依然在全国领先。作为千千万万外贸从业者中的一员，
我也为这个数字贡献了自己的一份力量。

2010年，广州还要举办亚运会，整个城市都在为这个大事件做准备。
看着珠江新城一座座高楼拔地而起，我知道广州的未来会更好，
我的外贸生涯也才刚刚开始。
"""

print(memoir_text)

# 保存到文件
with open('memoir_2009_guangzhou_foreign_trade.txt', 'w', encoding='utf-8') as f:
    f.write(memoir_text)

print("\n回忆录已保存到: memoir_2009_guangzhou_foreign_trade.txt")
