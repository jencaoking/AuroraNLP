import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP.dependency_parser import (
    DependencyNode,
    DependencyTree,
    DependencyParser,
    ArcEagerOracle,
    ParserState,
    Transition,
    DEPENDENCY_RELATIONS,
    create_sample_dependency_corpus,
)


def demo_basic_usage():
    print("=" * 60)
    print("依存句法分析基础用法演示")
    print("=" * 60)
    
    print("\n1. 创建依存树")
    tree = DependencyTree()
    
    tree.add_node(DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj'))
    tree.add_node(DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root'))
    tree.add_node(DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj'))
    
    print(f"   依存树: {tree}")
    print(f"   节点数: {len(tree)}")
    print(f"   弧数: {len(tree.get_arcs())}")
    
    print("\n2. 查询依存关系")
    root = tree.get_root()
    print(f"   根节点: {root}")
    
    children = tree.get_children(2)
    print(f"   '爱'的子节点: {children}")
    
    head = tree.get_head(1)
    print(f"   '我'的父节点: {head}")
    
    print("\n3. CoNLL-U格式输出")
    print(tree.to_conllu())


def demo_parser_state():
    print("\n" + "=" * 60)
    print("解析器状态演示")
    print("=" * 60)
    
    nodes = [
        DependencyNode(id=0, form='我', pos='r'),
        DependencyNode(id=1, form='爱', pos='v'),
        DependencyNode(id=2, form='中国', pos='ns'),
    ]
    
    state = ParserState(nodes)
    
    print("\n初始状态:")
    print(f"   栈: {state.stack}")
    print(f"   缓冲区: {state.buffer}")
    print(f"   弧: {state.arcs}")
    
    print("\n执行 SHIFT 操作:")
    state.shift()
    print(f"   栈: {state.stack}")
    print(f"   缓冲区: {state.buffer}")
    
    print("\n执行 LEFT-ARC(nsubj) 操作:")
    state.left_arc('nsubj')
    print(f"   栈: {state.stack}")
    print(f"   弧: {state.arcs}")
    
    print("\n执行 SHIFT 操作:")
    state.shift()
    print(f"   栈: {state.stack}")
    print(f"   缓冲区: {state.buffer}")
    
    print("\n执行 RIGHT-ARC(obj) 操作:")
    state.right_arc('obj')
    print(f"   栈: {state.stack}")
    print(f"   弧: {state.arcs}")


def demo_oracle():
    print("\n" + "=" * 60)
    print("Oracle演示 (用于训练)")
    print("=" * 60)
    
    gold_tree = DependencyTree()
    gold_tree.add_node(DependencyNode(id=1, form='我', pos='r', head=2, deprel='nsubj'))
    gold_tree.add_node(DependencyNode(id=2, form='爱', pos='v', head=0, deprel='root'))
    gold_tree.add_node(DependencyNode(id=3, form='中国', pos='ns', head=2, deprel='obj'))
    
    oracle = ArcEagerOracle(gold_tree)
    
    nodes = [
        DependencyNode(id=0, form='我', pos='r'),
        DependencyNode(id=1, form='爱', pos='v'),
        DependencyNode(id=2, form='中国', pos='ns'),
    ]
    state = ParserState(nodes)
    
    print("\n逐步解析:")
    step = 1
    while not state.is_terminal():
        action, relation = oracle.get_next_action(state)
        print(f"   步骤 {step}: {action.name}" + (f"({relation})" if relation else ""))
        
        if action == Transition.SHIFT:
            state.shift()
        elif action == Transition.LEFT_ARC:
            state.left_arc(relation)
        elif action == Transition.RIGHT_ARC:
            state.right_arc(relation)
        elif action == Transition.REDUCE:
            state.reduce()
        
        step += 1
    
    print("\n最终解析结果:")
    result_tree = state.to_tree()
    print(result_tree.to_conllu())


def demo_parser_training():
    print("\n" + "=" * 60)
    print("依存解析器训练演示")
    print("=" * 60)
    
    corpus = create_sample_dependency_corpus()
    print(f"\n训练语料: {len(corpus)} 棵依存树")
    
    for i, tree in enumerate(corpus):
        print(f"\n   语料 {i+1}:")
        for node in tree:
            print(f"      {node.form}({node.pos}) -> head={node.head}, rel={node.deprel}")
    
    parser = DependencyParser()
    
    print("\n开始训练...")
    parser.train(corpus, max_iter=10, verbose=True)
    
    print("\n训练完成!")
    info = parser.get_model_info()
    print(f"   特征数量: {info['num_features']}")
    print(f"   关系类型数: {info['num_relations']}")


def demo_parser_inference():
    print("\n" + "=" * 60)
    print("依存解析器推理演示")
    print("=" * 60)
    
    corpus = create_sample_dependency_corpus()
    parser = DependencyParser()
    parser.train(corpus, max_iter=15, verbose=False)
    
    test_sentences = [
        (['我', '爱', '中国'], ['r', 'v', 'ns']),
        (['他', '吃', '了', '一个', '苹果'], ['r', 'v', 'u', 'm', 'n']),
        (['北京', '是', '中国', '首都'], ['ns', 'v', 'ns', 'n']),
    ]
    
    for words, pos_tags in test_sentences:
        print(f"\n输入: {' '.join(words)}")
        print(f"词性: {' '.join(pos_tags)}")
        
        tree = parser.parse(words, pos_tags)
        
        print("解析结果:")
        for node in tree:
            head_form = tree.get_node(node.head).form if node.head > 0 else 'ROOT'
            print(f"   {node.form}({node.pos}) -> {head_form} [{node.deprel}]")


def demo_dependency_relations():
    print("\n" + "=" * 60)
    print("依存关系类型")
    print("=" * 60)
    
    print("\n支持的依存关系类型:")
    for rel, desc in list(DEPENDENCY_RELATIONS.items())[:15]:
        print(f"   {rel}: {desc}")
    print(f"   ... 共 {len(DEPENDENCY_RELATIONS)} 种关系类型")


def demo_save_load():
    print("\n" + "=" * 60)
    print("模型保存与加载演示")
    print("=" * 60)
    
    corpus = create_sample_dependency_corpus()
    parser = DependencyParser()
    parser.train(corpus, max_iter=10, verbose=False)
    
    model_path = os.path.join(os.path.dirname(__file__), 'dependency_model.pkl')
    
    print(f"\n保存模型到: {model_path}")
    parser.save_model(model_path)
    
    new_parser = DependencyParser()
    print("加载模型...")
    new_parser.load_model(model_path)
    
    print("使用加载的模型进行解析:")
    tree = new_parser.parse(['我', '爱', '中国'])
    print(tree.to_conllu())
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print("\n清理临时模型文件")


def demo_conllu_format():
    print("\n" + "=" * 60)
    print("CoNLL-U格式处理演示")
    print("=" * 60)
    
    conllu_str = """# text = 我爱中国
1\t我\t我\tPRON\tr\t_\t2\tnsubj\t_\t_
2\t爱\t爱\tVERB\tv\t_\t0\troot\t_\t_
3\t中国\t中国\tPROPN\tns\t_\t2\tobj\t_\t_"""
    
    print("\n输入CoNLL-U格式:")
    print(conllu_str)
    
    tree = DependencyTree.from_conllu(conllu_str)
    
    print("\n解析后的依存树:")
    for node in tree:
        print(f"   ID={node.id}, FORM={node.form}, POS={node.pos}, HEAD={node.head}, DEPREL={node.deprel}")
    
    print("\n重新输出CoNLL-U格式:")
    print(tree.to_conllu())


def main():
    print("\n" + "#" * 60)
    print("#  AuroraNLP 依存句法分析演示")
    print("#" * 60)
    
    demo_basic_usage()
    demo_parser_state()
    demo_oracle()
    demo_parser_training()
    demo_parser_inference()
    demo_dependency_relations()
    demo_conllu_format()
    demo_save_load()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
