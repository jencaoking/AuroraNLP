import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AuroraNLP import PerceptronSegmentor, Segmentor


def test_perceptron_segmentor_basic():
    print("=" * 60)
    print("测试1: 基本分词功能")
    print("=" * 60)
    
    segmentor = PerceptronSegmentor()
    
    corpus = [
        ['我', '爱', '自然语言', '处理'],
        ['今天', '天气', '很', '好'],
        ['我', '喜欢', '学习', '中文', '分词'],
        ['北京', '是', '中国', '的', '首都'],
        ['自然语言', '处理', '是', '人工智能', '的', '重要', '分支'],
    ]
    
    segmentor.train(corpus, max_iter=5, verbose=True)
    
    test_texts = [
        "我爱自然语言处理",
        "今天天气很好",
        "我喜欢学习中文分词",
        "北京是中国的首都",
    ]
    
    for text in test_texts:
        result = segmentor.segment(text)
        print(f"原文: {text}")
        print(f"分词: {' / '.join(result)}")
        print()
    
    print("测试1通过!\n")


def test_online_learning():
    print("=" * 60)
    print("测试2: 在线学习功能")
    print("=" * 60)
    
    segmentor = PerceptronSegmentor()
    
    initial_corpus = [
        ['我', '爱', '编程'],
        ['学习', '很', '重要'],
    ]
    segmentor.train(initial_corpus, max_iter=3, verbose=False)
    
    text = "我爱编程"
    result = segmentor.segment(text)
    print(f"初始训练后分词: {' / '.join(result)}")
    
    print("\n开始在线学习...")
    new_samples = [
        ['我', '爱', '自然语言', '处理'],
        ['深度', '学习', '很', '有趣'],
        ['机器', '学习', '是', 'AI', '的', '核心'],
    ]
    
    for i, sample in enumerate(new_samples):
        is_correct, accuracy = segmentor.train_online(sample, update_weights=True)
        print(f"样本 {i+1}: {'正确' if is_correct else '已更新'}, 准确率: {accuracy:.2f}")
    
    text = "我爱自然语言处理"
    result = segmentor.segment(text)
    print(f"\n在线学习后分词: {' / '.join(result)}")
    
    print("\n测试2通过!\n")


def test_incremental_update():
    print("=" * 60)
    print("测试3: 增量更新功能")
    print("=" * 60)
    
    segmentor = PerceptronSegmentor()
    
    corpus1 = [
        ['今天', '天气', '不错'],
        ['我', '去', '公园', '散步'],
    ]
    segmentor.train(corpus1, max_iter=3, verbose=False)
    
    text = "今天天气不错"
    result = segmentor.segment(text)
    print(f"第一批数据训练后: {' / '.join(result)}")
    
    corpus2 = [
        ['明天', '天气', '也', '不错'],
        ['我', '想', '去', '爬山'],
    ]
    segmentor.partial_fit(corpus2, verbose=True)
    
    text = "明天天气也不错"
    result = segmentor.segment(text)
    print(f"增量更新后: {' / '.join(result)}")
    
    print("\n测试3通过!\n")


def test_model_save_load():
    print("=" * 60)
    print("测试4: 模型保存和加载")
    print("=" * 60)
    
    segmentor = PerceptronSegmentor()
    
    corpus = [
        ['我', '爱', '自然语言', '处理'],
        ['今天', '天气', '很', '好'],
        ['北京', '是', '中国', '的', '首都'],
    ]
    segmentor.train(corpus, max_iter=5, verbose=False)
    
    model_path = "test_perceptron_model.pkl"
    segmentor.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    model_info = segmentor.get_model_info()
    print(f"模型信息: {model_info}")
    
    new_segmentor = PerceptronSegmentor()
    new_segmentor.load_model(model_path)
    print("模型已加载")
    
    text = "我爱自然语言处理"
    result1 = segmentor.segment(text)
    result2 = new_segmentor.segment(text)
    
    print(f"原模型分词: {' / '.join(result1)}")
    print(f"加载模型分词: {' / '.join(result2)}")
    
    assert result1 == result2, "模型保存和加载结果不一致!"
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"已清理测试文件: {model_path}")
    
    print("\n测试4通过!\n")


def test_segmentor_integration():
    print("=" * 60)
    print("测试5: Segmentor 集成测试")
    print("=" * 60)
    
    segmentor = Segmentor(use_perceptron=True)
    
    corpus = [
        ['我', '爱', '自然语言', '处理'],
        ['今天', '天气', '很', '好'],
        ['北京', '是', '中国', '的', '首都'],
        ['人工智能', '改变', '世界'],
    ]
    
    segmentor.train_perceptron(corpus, max_iter=5, verbose=True)
    
    text = "我爱自然语言处理"
    result = segmentor.segment(text, mode='perceptron')
    print(f"分词结果: {' / '.join(result)}")
    
    states = segmentor.segment_with_perceptron_states(text)
    print(f"状态序列: {states}")
    
    model_info = segmentor.get_perceptron_model_info()
    print(f"模型信息: {model_info}")
    
    is_correct, accuracy = segmentor.train_perceptron_online(['深度', '学习', '很', '强大'])
    print(f"在线学习: 正确={is_correct}, 准确率={accuracy:.2f}")
    
    print("\n测试5通过!\n")


def test_segment_with_states():
    print("=" * 60)
    print("测试6: 状态序列分词")
    print("=" * 60)
    
    segmentor = PerceptronSegmentor()
    
    corpus = [
        ['我', '爱', '编程'],
        ['今天', '天气', '好'],
    ]
    segmentor.train(corpus, max_iter=3, verbose=False)
    
    text = "我爱编程"
    states = segmentor.segment_with_states(text)
    
    print(f"文本: {text}")
    print(f"状态序列: {states}")
    
    state_meanings = {'B': '词首', 'M': '词中', 'E': '词尾', 'S': '单字词'}
    for char, state in states:
        print(f"  {char} -> {state} ({state_meanings.get(state, '未知')})")
    
    print("\n测试6通过!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("感知器分词器测试套件")
    print("=" * 60 + "\n")
    
    test_perceptron_segmentor_basic()
    test_online_learning()
    test_incremental_update()
    test_model_save_load()
    test_segmentor_integration()
    test_segment_with_states()
    
    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)
