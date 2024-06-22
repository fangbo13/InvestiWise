from graphviz import Digraph


def create_neural_network_diagram():
    dot = Digraph()

    # 设置图的标题
    dot.attr(label='Neural Network Structure', fontsize='20')

    # 输入层
    dot.node('Input', 'Input Layer\n(shape=(60, 1))')

    # 第一层LSTM
    dot.node('LSTM1', 'LSTM Layer\n(units=200, return_sequences=True)')

    # 第一层Dropout
    dot.node('Dropout1', 'Dropout Layer\n(rate=0.031)')

    # 第二层LSTM
    dot.node('LSTM2', 'LSTM Layer\n(units=200, return_sequences=False)')

    # 第二层Dropout
    dot.node('Dropout2', 'Dropout Layer\n(rate=0.031)')

    # 输出层
    dot.node('Output', 'Dense Layer\n(units=1)')

    # 添加节点之间的连线
    dot.edge('Input', 'LSTM1')
    dot.edge('LSTM1', 'Dropout1')
    dot.edge('Dropout1', 'LSTM2')
    dot.edge('LSTM2', 'Dropout2')
    dot.edge('Dropout2', 'Output')

    # 保存并展示图
    dot.render('neural_network_structure', format='png', cleanup=False)

create_neural_network_diagram()
