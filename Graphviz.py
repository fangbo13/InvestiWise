from graphviz import Digraph

# 创建一个有向图对象
dot = Digraph()

# 设置整体的方向为从左到右
dot.attr(rankdir='LR')

# BERT模型子图
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='BERT 模型', style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white', shape='record')
    c.node('input_text', '输入文本')
    c.node('embedding', '词嵌入 + 文本嵌入 + 位置嵌入')
    c.node('encoder', '''<encoder> 编码器层 | 多头自注意力机制 | 前馈神经网络 | Add&Norm''')

    c.edge('input_text', 'embedding')
    c.edge('embedding', 'encoder')

# 特征提取节点
dot.node('feature_extraction', '特征提取')

# 情绪分类子图
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='情绪分类', style='filled', color='lightblue')
    c.node_attr.update(style='filled', color='white', shape='record')
    c.node('lstm', 'LSTM 层')
    c.node('bi_lstm', '双向LSTM 层')
    c.node('fc', '全连接层')

    c.edge('lstm', 'bi_lstm')
    c.edge('bi_lstm', 'fc')

# 连接BERT模型与情绪分类
dot.edge('encoder', 'feature_extraction')
dot.edge('feature_extraction', 'lstm')

# 保存并展示图像
file_path = '/mnt/data/bert_lstm_model_professional_fixed.png'
dot.render(file_path, format='png', cleanup=True)
