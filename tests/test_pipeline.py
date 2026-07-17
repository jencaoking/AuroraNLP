"""
AuroraNLP pipeline.py 核心类测试

覆盖 pipeline.py 中所有核心模块的单元测试。
使用 pytest 框架，中文测试函数名和注释。
"""

import json
import os
import asyncio
import tempfile
import pytest

from AuroraNLP.pipeline.pipeline import (
    StringStore,
    Doc,
    Span,
    Token,
    PipelineComponent,
    ConditionalBranch,
    Pipeline,
    ComponentRegistry,
    PipelineConfig,
    ConfigSchema,
    FreezableParams,
    ModelVersion,
    ModelLifecycle,
    LRUCache,
    ModelCache,
    APIRequest,
    APIResponse,
    RequestValidator,
    Route,
    APIServer,
    RPCMessage,
    RPCError,
    RPCStatusCode,
    RPCService,
    RPCServer,
    RPCClient,
    AsyncPipeline,
    StreamProcessor,
    ProgressCallback,
    PluginState,
    PluginInfo,
    Plugin,
    PluginDependency,
    PluginManager,
    SimpleTokenizerComponent,
    POSTaggerComponent,
    ComponentState,
)


# ============================================================
# 辅助：创建简单的测试用组件
# ============================================================

class _EchoComponent(PipelineComponent):
    """测试用回声组件 - 直接返回原文档"""

    def __init__(self, name="echo"):
        super().__init__(name=name)

    def process(self, doc: Doc) -> Doc:
        doc.set_attr("echoed", True)
        return doc


class _AppendComponent(PipelineComponent):
    """测试用追加组件 - 在文档属性中追加标记"""

    def __init__(self, name="appender", tag="processed"):
        super().__init__(name=name)
        self._tag = tag

    def process(self, doc: Doc) -> Doc:
        tags = doc.get_attr("tags", [])
        tags.append(self._tag)
        doc.set_attr("tags", tags)
        return doc


class _ErrorComponent(PipelineComponent):
    """测试用错误组件 - 总是抛出异常"""

    def __init__(self, name="error"):
        super().__init__(name=name)

    def process(self, doc: Doc) -> Doc:
        raise RuntimeError("测试异常")


# ============================================================
# 1. StringStore 测试
# ============================================================

class TestStringStore:
    """StringStore - 字符串存储池测试"""

    def test_添加字符串并获取ID(self):
        """测试添加字符串后能正确获取ID"""
        store = StringStore()
        id_a = store.add("hello")
        id_b = store.add("world")
        assert id_a == 0
        assert id_b == 1
        assert len(store) == 2

    def test_重复添加返回相同ID(self):
        """测试重复添加同一字符串返回相同ID"""
        store = StringStore()
        id1 = store.add("test")
        id2 = store.add("test")
        assert id1 == id2
        assert len(store) == 1

    def test_通过字符串获取ID和通过ID获取字符串(self):
        """测试双向映射的正确性"""
        store = StringStore()
        store.add("alpha")
        store.add("beta")
        assert store["alpha"] == 0
        assert store["beta"] == 1
        assert store[0] == "alpha"
        assert store[1] == "beta"

    def test_get_id和get_str不存在时返回None(self):
        """测试查询不存在的字符串/ID返回None"""
        store = StringStore()
        assert store.get_id("not_exist") is None
        assert store.get_str(999) is None

    def test_序列化和反序列化(self):
        """测试JSON序列化与反序列化的往返一致性"""
        store = StringStore()
        store.add("你好")
        store.add("世界")
        json_str = store.to_json()
        restored = StringStore.from_json(json_str)
        assert restored.get_id("你好") == 0
        assert restored.get_id("世界") == 1
        assert len(restored) == 2

    def test_contains检查字符串和ID(self):
        """测试in操作符对字符串和ID的支持"""
        store = StringStore()
        store.add("exists")
        assert "exists" in store
        assert 0 in store
        assert "nope" not in store
        assert 99 not in store

    def test_批量添加(self):
        """测试批量添加多个字符串"""
        store = StringStore()
        ids = store.batch_add(["a", "b", "c"])
        assert ids == [0, 1, 2]
        assert len(store) == 3

    def test_clear清空存储池(self):
        """测试清空后存储池为空"""
        store = StringStore()
        store.add("x")
        store.clear()
        assert len(store) == 0


# ============================================================
# 2. Doc 测试
# ============================================================

class TestDoc:
    """Doc - 文档对象测试"""

    def test_创建文档并获取文本(self):
        """测试创建文档后能正确获取原始文本"""
        doc = Doc(text="自然语言处理")
        assert doc.text == "自然语言处理"
        assert len(doc) == 6

    def test_设置和获取自定义属性(self):
        """测试文档的自定义属性存取"""
        doc = Doc(text="test")
        doc.set_attr("key1", "value1")
        assert doc.get_attr("key1") == "value1"
        assert doc.get_attr("missing", "default") == "default"

    def test_添加token和span(self):
        """测试向文档添加词元和文本片段"""
        doc = Doc(text="hello world")
        token = Token(doc=doc, start=0, end=5)
        doc.add_token(token)
        assert len(doc.tokens) == 1
        assert doc.tokens[0].text == "hello"

        span = Span(doc=doc, start=0, end=5, label="WORD")
        doc.add_span(span)
        assert len(doc.spans) == 1
        assert doc.spans[0].label == "WORD"

    def test_添加实体(self):
        """测试向文档添加命名实体"""
        doc = Doc(text="北京是中国的首都")
        entity = Span(doc=doc, start=0, end=2, label="LOC")
        doc.add_entity(entity)
        assert len(doc.entities) == 1
        assert doc.entities[0].text == "北京"

    def test_char_span创建片段(self):
        """测试通过字符偏移创建文本片段"""
        doc = Doc(text="abcdef")
        span = doc.char_span(1, 4)
        assert span.text == "bcd"
        assert span.start == 1
        assert span.end == 4

    def test_文档拷贝(self):
        """测试文档浅拷贝的独立性"""
        doc = Doc(text="original")
        doc.set_attr("k", "v")
        copied = doc.copy()
        assert copied.text == "original"
        assert copied.get_attr("k") == "v"
        # 修改拷贝不影响原文档
        copied.set_attr("k", "changed")
        assert doc.get_attr("k") == "v"

    def test_has_attr检查属性(self):
        """测试属性存在性检查"""
        doc = Doc(text="test")
        assert not doc.has_attr("missing")
        doc.set_attr("exists", True)
        assert doc.has_attr("exists")

    def test_事件回调注册与触发(self):
        """测试文档事件回调机制"""
        doc = Doc(text="test")
        results = []
        doc.register_callback("test_event", lambda d, **kw: results.append(kw.get("val")))
        doc.trigger_event("test_event", val=42)
        assert results == [42]


# ============================================================
# 3. Span 测试
# ============================================================

class TestSpan:
    """Span - 文本片段测试"""

    def test_创建span并获取文本(self):
        """测试创建Span后能正确获取对应文本"""
        doc = Doc(text="自然语言处理")
        span = Span(doc=doc, start=0, end=4)
        assert span.text == "自然语言"
        assert len(span) == 4

    def test_span切片操作(self):
        """测试在Span内创建子Span"""
        doc = Doc(text="abcdefgh")
        parent = Span(doc=doc, start=0, end=8)
        child = parent.slice(2, 5)
        assert child.text == "cde"
        assert child.start == 2
        assert child.end == 5

    def test_span属性继承(self):
        """测试Span的标签继承"""
        doc = Doc(text="hello")
        parent = Span(doc=doc, start=0, end=5, label="PARENT")
        child = parent.slice(1, 4)
        assert child.label == "PARENT"

    def test_span重叠检测(self):
        """测试两个Span的重叠判断"""
        doc = Doc(text="0123456789")
        s1 = Span(doc=doc, start=2, end=6)
        s2 = Span(doc=doc, start=4, end=8)
        s3 = Span(doc=doc, start=8, end=10)
        assert s1.overlaps(s2) is True
        assert s1.overlaps(s3) is False

    def test_span包含检测(self):
        """测试Span的包含关系判断"""
        doc = Doc(text="0123456789")
        outer = Span(doc=doc, start=1, end=8)
        inner = Span(doc=doc, start=3, end=6)
        assert outer.contains(inner) is True
        assert inner.contains(outer) is False

    def test_span关系链接(self):
        """测试Span间的关系添加与获取"""
        doc = Doc(text="A B C")
        s1 = Span(doc=doc, start=0, end=1)
        s2 = Span(doc=doc, start=2, end=3)
        s1.add_relation("ref", s2)
        related = s1.get_relations("ref")
        assert len(related) == 1
        assert related[0] is s2


# ============================================================
# 4. Token 测试
# ============================================================

class TestToken:
    """Token - 词元对象测试"""

    def test_创建token并获取文本(self):
        """测试创建Token后能正确获取文本"""
        doc = Doc(text="自然语言")
        token = Token(doc=doc, start=0, end=4)
        assert token.text == "自然语言"
        assert len(token) == 4

    def test_token属性设置与访问(self):
        """测试Token的词性、词元等属性"""
        doc = Doc(text="running")
        token = Token(doc=doc, start=0, end=7)
        token.pos = "v"
        token.lemma = "run"
        token.ner_label = "O"
        assert token.pos == "v"
        assert token.lemma == "run"
        assert token.ner_label == "O"

    def test_token通过字典接口访问属性(self):
        """测试通过getitem/setitem访问Token属性"""
        doc = Doc(text="test")
        token = Token(doc=doc, start=0, end=4)
        token["pos"] = "n"
        token["custom"] = "data"
        assert token["pos"] == "n"
        assert token["custom"] == "data"

    def test_token转换为span(self):
        """测试Token转Span功能"""
        doc = Doc(text="hello")
        token = Token(doc=doc, start=0, end=5)
        token.ner_label = "ENTITY"
        span = token.as_span()
        assert span.text == "hello"
        assert span.label == "ENTITY"

    def test_token自定义属性(self):
        """测试Token的自定义属性存取"""
        doc = Doc(text="x")
        token = Token(doc=doc, start=0, end=1)
        token.set_attr("freq", 100)
        assert token.get_attr("freq") == 100
        assert token.has_attr("freq") is True
        assert token.has_attr("none") is False


# ============================================================
# 5. PipelineComponent 测试
# ============================================================

class TestPipelineComponent:
    """PipelineComponent - 流水线组件基类测试"""

    def test_组件创建与名称(self):
        """测试组件创建后名称正确"""
        comp = _EchoComponent(name="my_echo")
        assert comp.name == "my_echo"
        assert comp.state == "uninitialized"

    def test_组件初始化(self):
        """测试组件初始化后状态变更"""
        comp = _EchoComponent()
        pipeline = Pipeline()
        comp.initialize(pipeline)
        assert comp.state == "initialized"
        assert comp.pipeline is pipeline

    def test_组件冻结与解冻(self):
        """测试组件冻结和解冻机制"""
        comp = _EchoComponent()
        pipeline = Pipeline()
        comp.initialize(pipeline)
        comp.freeze()
        assert comp.frozen is True
        assert comp.state == "frozen"
        comp.unfreeze()
        assert comp.frozen is False
        assert comp.state == "initialized"

    def test_组件禁用标记(self):
        """测试组件的禁用状态"""
        comp = _EchoComponent()
        comp._disabled = True
        assert comp.disabled is True

    def test_组件前置依赖声明(self):
        """测试组件的require/provide声明"""
        comp = _EchoComponent()
        comp.require("tokens", "pos_tags")
        comp.provide("ner_tags")
        assert "tokens" in comp.get_requirements()
        assert "ner_tags" in comp.get_provides()

    def test_组件序列化为配置(self):
        """测试组件的to_config方法"""
        comp = _EchoComponent(name="test_comp")
        config = comp.to_config()
        assert config["name"] == "test_comp"
        assert "class" in config


# ============================================================
# 6. ConditionalBranch 测试
# ============================================================

class TestConditionalBranch:
    """ConditionalBranch - 条件分支测试"""

    def test_条件为真时选择true组件(self):
        """测试条件为真时返回true分支组件"""
        doc = Doc(text="long text here")
        true_comp = _EchoComponent(name="true_comp")
        branch = ConditionalBranch(
            condition=lambda d: len(d.text) > 5,
            true_components=[true_comp],
        )
        active = branch.get_active_components(doc)
        assert len(active) == 1
        assert active[0].name == "true_comp"

    def test_条件为假时选择false组件(self):
        """测试条件为假时返回false分支组件"""
        doc = Doc(text="hi")
        false_comp = _EchoComponent(name="false_comp")
        branch = ConditionalBranch(
            condition=lambda d: len(d.text) > 5,
            false_components=[false_comp],
        )
        active = branch.get_active_components(doc)
        assert len(active) == 1
        assert active[0].name == "false_comp"

    def test_条件评估(self):
        """测试evaluate方法直接返回布尔值"""
        doc = Doc(text="test")
        branch = ConditionalBranch(condition=lambda d: "test" in d.text)
        assert branch.evaluate(doc) is True


# ============================================================
# 7. Pipeline 测试
# ============================================================

class TestPipeline:
    """Pipeline - 流水线测试"""

    def test_创建流水线并添加组件(self):
        """测试流水线创建和组件添加"""
        pipeline = Pipeline(name="test_pipeline")
        comp = _EchoComponent(name="echo1")
        pipeline.add_component(comp)
        assert "echo1" in pipeline.component_names
        assert pipeline.get_component("echo1") is comp

    def test_流水线执行组件(self):
        """测试流水线处理文本时组件被正确执行"""
        pipeline = Pipeline(name="exec_test")
        pipeline.add_component(_AppendComponent(name="step1", tag="A"))
        pipeline.add_component(_AppendComponent(name="step2", tag="B"))
        doc = pipeline.process("hello")
        assert doc.get_attr("tags") == ["A", "B"]

    def test_流水线批量处理(self):
        """测试批量处理多个文本"""
        pipeline = Pipeline(name="batch_test")
        pipeline.add_component(_EchoComponent())
        results = pipeline.process_batch(["text1", "text2", "text3"])
        assert len(results) == 3
        for doc in results:
            assert doc.get_attr("echoed") is True

    def test_流水线禁用组件(self):
        """测试禁用组件后不参与处理"""
        pipeline = Pipeline(name="disable_test")
        pipeline.add_component(_AppendComponent(name="step1", tag="A"))
        pipeline.add_component(_AppendComponent(name="step2", tag="B"))
        pipeline.disable_component("step2")
        doc = pipeline.process("test")
        assert doc.get_attr("tags") == ["A"]

    def test_流水线启用组件(self):
        """测试启用已禁用的组件"""
        pipeline = Pipeline(name="enable_test")
        comp = _AppendComponent(name="step1", tag="A")
        pipeline.add_component(comp)
        pipeline.disable_component("step1")
        pipeline.enable_component("step1")
        doc = pipeline.process("test")
        assert doc.get_attr("tags") == ["A"]

    def test_流水线移除组件(self):
        """测试从流水线中移除组件"""
        pipeline = Pipeline(name="remove_test")
        pipeline.add_component(_EchoComponent(name="to_remove"))
        assert pipeline.remove_component("to_remove") is True
        assert pipeline.get_component("to_remove") is None

    def test_流水线条件分支执行(self):
        """测试条件分支在流水线中的执行"""
        pipeline = Pipeline(name="branch_test")
        pipeline.add_component(_AppendComponent(name="always", tag="X"))
        branch = ConditionalBranch(
            condition=lambda d: len(d.text) > 3,
            true_components=[_AppendComponent(name="long", tag="LONG")],
            false_components=[_AppendComponent(name="short", tag="SHORT")],
        )
        pipeline.add_branch(branch)
        doc_long = pipeline.process("hello")
        doc_short = pipeline.process("hi")
        assert "LONG" in doc_long.get_attr("tags")
        assert "SHORT" in doc_short.get_attr("tags")

    def test_流水线前处理器和后处理器(self):
        """测试前处理和后处理器的调用"""
        pipeline = Pipeline(name="prepost_test")
        pipeline.add_pre_processor(lambda d: (d.set_attr("pre", True), d)[1])
        pipeline.add_post_processor(lambda d: (d.set_attr("post", True), d)[1])
        doc = pipeline.process("test")
        assert doc.get_attr("pre") is True
        assert doc.get_attr("post") is True

    def test_流水线错误处理器(self):
        """测试错误处理器捕获组件异常"""
        pipeline = Pipeline(name="error_test")
        pipeline.add_component(_ErrorComponent(name="fail"))
        pipeline.add_error_handler(lambda d, e: (d.set_attr("error_handled", str(e)), d)[1])
        doc = pipeline.process("test")
        assert doc.get_attr("error_handled") == "测试异常"

    def test_流水线冻结组件(self):
        """测试通过Pipeline冻结指定组件"""
        pipeline = Pipeline(name="freeze_test")
        pipeline.add_component(_EchoComponent(name="comp1"))
        assert pipeline.freeze_component("comp1") is True
        assert pipeline.get_component("comp1").frozen is True

    def test_流水线使用SimpleTokenizerComponent(self):
        """测试使用内置的分词组件"""
        pipeline = Pipeline(name="tokenizer_test")
        pipeline.add_component(SimpleTokenizerComponent())
        doc = pipeline.process("中国人工智能技术")
        assert len(doc.tokens) > 0


# ============================================================
# 8. ComponentRegistry 测试
# ============================================================

class TestComponentRegistry:
    """ComponentRegistry - 组件注册表测试"""

    def test_装饰器注册组件(self):
        """测试使用装饰器注册组件类"""
        registry = ComponentRegistry()

        @registry.register("my_comp", category="test")
        class MyComp(PipelineComponent):
            def process(self, doc):
                return doc

        assert registry.get("my_comp") is MyComp
        assert "my_comp" in registry.list_components(category="test")

    def test_直接注册组件类(self):
        """测试使用register_class直接注册"""
        registry = ComponentRegistry()
        registry.register_class(_EchoComponent, name="echo_reg", category="demo")
        assert registry.get("echo_reg") is _EchoComponent

    def test_通过别名查找组件(self):
        """测试通过别名查找已注册组件"""
        registry = ComponentRegistry()
        registry.register_class(_EchoComponent, name="echo", aliases=["e", "echo_alias"])
        assert registry.get("e") is _EchoComponent
        assert registry.get("echo_alias") is _EchoComponent

    def test_创建组件实例(self):
        """测试通过注册表创建组件实例"""
        registry = ComponentRegistry()
        registry.register_class(_EchoComponent, name="echo")
        instance = registry.create("echo")
        assert isinstance(instance, _EchoComponent)

    def test_取消注册组件(self):
        """测试取消注册后组件不再可用"""
        registry = ComponentRegistry()
        registry.register_class(_EchoComponent, name="to_remove")
        assert registry.unregister("to_remove") is True
        assert registry.get("to_remove") is None

    def test_列出所有类别(self):
        """测试列出所有已注册的组件类别"""
        registry = ComponentRegistry()
        registry.register_class(_EchoComponent, name="a", category="cat1")
        registry.register_class(_AppendComponent, name="b", category="cat2")
        categories = registry.list_categories()
        assert "cat1" in categories
        assert "cat2" in categories


# ============================================================
# 9. PipelineConfig 测试
# ============================================================

class TestPipelineConfig:
    """PipelineConfig - 配置管理器测试"""

    def test_从JSON字符串加载配置(self):
        """测试从JSON字符串创建配置"""
        json_str = '{"name": "test", "version": 1}'
        config = PipelineConfig.from_json(json_str)
        assert config["name"] == "test"
        assert config["version"] == 1

    def test_配置导出为JSON(self):
        """测试配置导出为JSON字符串"""
        config = PipelineConfig({"key": "value"})
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["key"] == "value"

    def test_配置嵌套键访问(self):
        """测试使用点号分隔访问嵌套配置"""
        config = PipelineConfig({"a": {"b": {"c": 42}}})
        assert config["a.b.c"] == 42

    def test_配置嵌套键设置(self):
        """测试使用点号分隔设置嵌套配置"""
        config = PipelineConfig()
        config["x.y.z"] = "deep"
        assert config["x"]["y"]["z"] == "deep"

    def test_配置合并(self):
        """测试两个配置的深度合并"""
        base = PipelineConfig({"a": 1, "b": {"c": 2}})
        override = PipelineConfig({"b": {"d": 3}, "e": 4})
        base.merge(override)
        assert base["a"] == 1
        assert base["b"]["c"] == 2
        assert base["b"]["d"] == 3
        assert base["e"] == 4

    def test_配置验证通过模式(self):
        """测试使用ConfigSchema验证配置"""
        schema = ConfigSchema()
        schema.add_field("name", str, required=True)
        schema.add_field("count", int, min_value=0, max_value=100)

        config = PipelineConfig({"name": "test", "count": 50})
        config.add_schema("basic", schema)
        valid, errors = config.validate("basic")
        assert valid is True
        assert len(errors) == 0

    def test_配置验证失败(self):
        """测试配置验证失败时返回错误信息"""
        schema = ConfigSchema()
        schema.add_field("name", str, required=True)
        schema.add_field("count", int, min_value=0, max_value=10)

        config = PipelineConfig({"count": 50})
        config.add_schema("strict", schema)
        valid, errors = config.validate("strict")
        assert valid is False
        assert any("name" in e for e in errors)

    def test_配置应用默认值(self):
        """测试配置应用模式中的默认值"""
        schema = ConfigSchema()
        schema.add_field("timeout", int, default=30)
        schema.add_field("retries", int, default=3)

        config = PipelineConfig({})
        config.add_schema("defaults", schema)
        config.apply_defaults("defaults")
        assert config["timeout"] == 30
        assert config["retries"] == 3

    def test_配置get带默认值(self):
        """测试get方法在键不存在时返回默认值"""
        config = PipelineConfig({"a": 1})
        assert config.get("missing", "fallback") == "fallback"
        assert config.get("a") == 1


# ============================================================
# 10. FreezableParams 测试
# ============================================================

class TestFreezableParams:
    """FreezableParams - 可冻结参数测试"""

    def test_设置和获取参数(self):
        """测试参数的基本存取"""
        params = FreezableParams({"lr": 0.001, "epochs": 10})
        assert params["lr"] == 0.001
        assert params.get("epochs") == 10

    def test_冻结后修改参数抛出异常(self):
        """测试冻结参数后修改会抛出RuntimeError"""
        params = FreezableParams({"weight": 1.0})
        params.freeze("weight")
        assert params.is_frozen("weight") is True
        with pytest.raises(RuntimeError, match="已冻结"):
            params["weight"] = 2.0

    def test_解冻后可以修改参数(self):
        """测试解冻后参数可以正常修改"""
        params = FreezableParams({"bias": 0.5})
        params.freeze("bias")
        params.unfreeze("bias")
        params["bias"] = 1.0
        assert params["bias"] == 1.0

    def test_批量冻结和批量解冻(self):
        """测试批量冻结和解冻多个参数"""
        params = FreezableParams({"a": 1, "b": 2, "c": 3})
        params.batch_freeze(["a", "b"])
        assert params.is_frozen("a") is True
        assert params.is_frozen("b") is True
        assert params.is_frozen("c") is False
        params.batch_unfreeze(["a"])
        assert params.is_frozen("a") is False

    def test_冻结所有参数(self):
        """测试冻结全部参数"""
        params = FreezableParams({"x": 1, "y": 2})
        params.freeze_all()
        assert len(params.frozen_keys) == 2

    def test_快照创建与恢复(self):
        """测试参数快照的创建和恢复"""
        params = FreezableParams({"lr": 0.01})
        params.take_snapshot()
        params["lr"] = 0.001
        assert params["lr"] == 0.001
        assert params.restore_snapshot() is True
        assert params["lr"] == 0.01

    def test_无快照时恢复返回False(self):
        """测试没有快照时恢复返回False"""
        params = FreezableParams({"a": 1})
        assert params.restore_snapshot() is False

    def test_update跳过冻结参数(self):
        """测试批量更新时跳过冻结参数"""
        params = FreezableParams({"a": 1, "b": 2})
        params.freeze("a")
        params.update({"a": 10, "b": 20})
        assert params["a"] == 1  # 被跳过
        assert params["b"] == 20

    def test_to_dict导出参数(self):
        """测试导出参数为字典"""
        params = FreezableParams({"k": "v"})
        d = params.to_dict()
        assert d == {"k": "v"}


# ============================================================
# 11. ModelVersion 测试
# ============================================================

class TestModelVersion:
    """ModelVersion - 模型版本管理测试"""

    def test_创建版本并获取版本字符串(self):
        """测试创建版本后版本字符串格式正确"""
        v = ModelVersion(1, 2, 3)
        assert v.version_string == "1.2.3"
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_带预发布和构建元数据的版本(self):
        """测试预发布标识和构建元数据"""
        v = ModelVersion(2, 0, 0, prerelease="alpha", build="20240101")
        assert v.version_string == "2.0.0-alpha+20240101"
        assert v.prerelease == "alpha"
        assert v.build == "20240101"

    def test_版本比较(self):
        """测试版本大小比较"""
        v1 = ModelVersion(1, 0, 0)
        v2 = ModelVersion(2, 0, 0)
        v3 = ModelVersion(1, 1, 0)
        assert v1 < v2
        assert v1 < v3
        assert v3 < v2
        assert v1 == ModelVersion(1, 0, 0)

    def test_版本兼容性检查(self):
        """测试主版本号相同则兼容"""
        v1 = ModelVersion(1, 5, 0)
        v2 = ModelVersion(1, 9, 9)
        v3 = ModelVersion(2, 0, 0)
        assert v1.is_compatible(v2) is True
        assert v1.is_compatible(v3) is False

    def test_破坏性变更检查(self):
        """测试主版本号不同则为破坏性变更"""
        v1 = ModelVersion(1, 0, 0)
        v2 = ModelVersion(2, 0, 0)
        assert v1.is_breaking_change(v2) is True
        assert v1.is_breaking_change(v1) is False

    def test_版本号递增(self):
        """测试bump方法返回递增后的新版本"""
        v = ModelVersion(1, 2, 3)
        assert v.bump_major().version_string == "2.0.0"
        assert v.bump_minor().version_string == "1.3.0"
        assert v.bump_patch().version_string == "1.2.4"

    def test_从字符串解析版本(self):
        """测试从版本字符串解析ModelVersion"""
        v = ModelVersion.parse("3.4.5")
        assert v.major == 3
        assert v.minor == 4
        assert v.patch == 5

    def test_解析带预发布的版本字符串(self):
        """测试解析带预发布标识的版本字符串"""
        v = ModelVersion.parse("1.0.0-beta")
        assert v.prerelease == "beta"
        assert v.version_string == "1.0.0-beta"

    def test_解析无效版本字符串抛出异常(self):
        """测试解析无效版本字符串抛出ValueError"""
        with pytest.raises(ValueError):
            ModelVersion.parse("invalid")
        with pytest.raises(ValueError):
            ModelVersion.parse("1.2")

    def test_版本导出为字典(self):
        """测试版本信息导出为字典"""
        v = ModelVersion(1, 0, 0)
        d = v.to_dict()
        assert d["version"] == "1.0.0"
        assert d["major"] == 1


# ============================================================
# 12. ModelLifecycle 测试
# ============================================================

class TestModelLifecycle:
    """ModelLifecycle - 模型生命周期测试"""

    def test_创建生命周期对象(self):
        """测试创建生命周期对象后初始阶段为DEVELOPMENT"""
        v = ModelVersion(1, 0, 0)
        lc = ModelLifecycle("test_model", v)
        assert lc.model_name == "test_model"
        assert lc.stage == ModelLifecycle.Stage.DEVELOPMENT

    def test_自动推进到下一阶段(self):
        """测试自动推进生命周期阶段"""
        v = ModelVersion(1, 0, 0)
        lc = ModelLifecycle("model", v)
        lc.advance_stage()
        assert lc.stage == ModelLifecycle.Stage.ALPHA
        lc.advance_stage()
        assert lc.stage == ModelLifecycle.Stage.BETA

    def test_手动指定推进到特定阶段(self):
        """测试手动指定推进到特定阶段"""
        v = ModelVersion(1, 0, 0)
        lc = ModelLifecycle("model", v)
        lc.advance_stage(ModelLifecycle.Stage.STABLE)
        assert lc.stage == ModelLifecycle.Stage.STABLE

    def test_阶段变更历史记录(self):
        """测试阶段变更历史被正确记录"""
        v = ModelVersion(1, 0, 0)
        lc = ModelLifecycle("model", v)
        lc.advance_stage()
        lc.advance_stage(ModelLifecycle.Stage.STABLE)
        history = lc.get_stage_history()
        assert len(history) == 3  # 初始 + 两次推进
        assert history[0][0] == "development"
        assert history[2][0] == "stable"

    def test_导出为字典(self):
        """测试生命周期信息导出为字典"""
        v = ModelVersion(1, 0, 0)
        lc = ModelLifecycle("model", v)
        lc.set_file_info("/path/model.bin", 1024)
        d = lc.to_dict()
        assert d["model_name"] == "model"
        assert d["size_bytes"] == 1024


# ============================================================
# 13. LRUCache 测试
# ============================================================

class TestLRUCache:
    """LRUCache - LRU缓存测试"""

    def test_基本存取(self):
        """测试缓存的put和get操作"""
        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None

    def test_缓存命中和未命中统计(self):
        """测试缓存命中/未命中计数"""
        cache = LRUCache(capacity=3)
        cache.put("x", 10)
        cache.get("x")  # 命中
        cache.get("x")  # 命中
        cache.get("y")  # 未命中
        assert cache.hits == 2
        assert cache.misses == 1

    def test_缓存淘汰策略(self):
        """测试容量满时淘汰最近最少使用的条目"""
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # 应淘汰 "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_访问后更新使用时间(self):
        """测试get操作更新条目的使用时间防止淘汰"""
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # 访问a，使其变为最近使用
        cache.put("c", 3)  # 应淘汰 "b" 而不是 "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_删除缓存条目(self):
        """测试从缓存中删除指定条目"""
        cache = LRUCache(capacity=5)
        cache.put("x", 10)
        assert cache.delete("x") is True
        assert cache.get("x") is None
        assert cache.delete("x") is False

    def test_清空缓存(self):
        """测试清空缓存后所有数据被清除"""
        cache = LRUCache(capacity=5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_调整缓存容量(self):
        """测试调整容量后多余的条目被淘汰"""
        cache = LRUCache(capacity=5)
        for i in range(5):
            cache.put(i, i)
        cache.resize(2)
        assert cache.size == 2

    def test_命中率计算(self):
        """测试命中率计算"""
        cache = LRUCache(capacity=5)
        cache.put("k", "v")
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("no")  # miss
        assert abs(cache.hit_rate - 2 / 3) < 1e-9


# ============================================================
# 14. ModelCache 测试
# ============================================================

class TestModelCache:
    """ModelCache - 模型缓存测试"""

    def test_基本存取模型(self):
        """测试模型的存入和获取"""
        cache = ModelCache(max_models=3)
        cache.put("model_a", {"data": "A"})
        result = cache.get("model_a")
        assert result == {"data": "A"}

    def test_带版本的模型缓存(self):
        """测试带版本号的模型缓存"""
        cache = ModelCache(max_models=5)
        v1 = ModelVersion(1, 0, 0)
        v2 = ModelVersion(2, 0, 0)
        cache.put("model", "v1_data", version=v1)
        cache.put("model", "v2_data", version=v2)
        assert cache.get("model", version=v1) == "v1_data"
        assert cache.get("model", version=v2) == "v2_data"

    def test_使缓存失效(self):
        """测试使指定模型缓存失效"""
        cache = ModelCache(max_models=5)
        cache.put("model_x", "data")
        assert cache.invalidate("model_x") is True
        assert cache.get("model_x") is None

    def test_清空模型缓存(self):
        """测试清空所有模型缓存"""
        cache = ModelCache(max_models=5)
        cache.put("m1", "d1")
        cache.put("m2", "d2")
        cache.clear()
        assert cache.get("m1") is None
        assert cache.get("m2") is None

    def test_记录和获取加载时间(self):
        """测试记录模型加载时间"""
        cache = ModelCache()
        cache.record_load_time("model_a", 1.5)
        assert cache.get_load_time("model_a") == 1.5
        assert cache.get_load_time("model_b") is None

    def test_缓存统计信息(self):
        """测试获取缓存统计信息"""
        cache = ModelCache(max_models=2, max_memory_mb=100)
        cache.put("m1", "d1", memory_mb=10)
        stats = cache.stats()
        assert "cache_stats" in stats
        assert "memory_usage_mb" in stats


# ============================================================
# 15. APIServer 测试
# ============================================================

class TestAPIServer:
    """APIServer - API服务器测试"""

    def test_添加路由并处理请求(self):
        """测试添加路由后能正确处理请求"""
        server = APIServer()

        def hello_handler(req: APIRequest) -> APIResponse:
            return APIResponse.ok({"message": "hello"})

        server.add_route("GET", "/hello", hello_handler)
        request = APIRequest(method="GET", path="/hello")
        response = server.handle_request(request)
        assert response.status_code == 200
        assert response.body["data"]["message"] == "hello"

    def test_未找到路由返回404(self):
        """测试请求不存在的路径返回404"""
        server = APIServer()
        request = APIRequest(method="GET", path="/not_exist")
        response = server.handle_request(request)
        assert response.status_code == 404

    def test_路由参数匹配(self):
        """测试带路径参数的路由匹配"""
        server = APIServer()

        def user_handler(req: APIRequest) -> APIResponse:
            return APIResponse.ok({"id": "matched"})

        server.add_route("GET", "/users/{id}", user_handler)
        request = APIRequest(method="GET", path="/users/123")
        response = server.handle_request(request)
        assert response.status_code == 200

    def test_请求验证失败返回400(self):
        """测试请求验证失败返回400"""
        server = APIServer()
        validator = RequestValidator()
        validator.add_rule("text", str, required=True, min_length=1)

        def handler(req: APIRequest) -> APIResponse:
            return APIResponse.ok()

        server.add_route("POST", "/process", handler, validator)
        # 不带body的请求
        request = APIRequest(method="POST", path="/process", body="{}")
        response = server.handle_request(request)
        assert response.status_code == 400

    def test_APIResponse工厂方法(self):
        """测试APIResponse的各种工厂方法"""
        ok_resp = APIResponse.ok(data="ok")
        assert ok_resp.status_code == 200

        created_resp = APIResponse.created(data="new")
        assert created_resp.status_code == 201

        bad_resp = APIResponse.bad_request("error")
        assert bad_resp.status_code == 400

        not_found_resp = APIResponse.not_found()
        assert not_found_resp.status_code == 404

        err_resp = APIResponse.server_error("fail")
        assert err_resp.status_code == 500

    def test_RequestValidator验证规则(self):
        """测试RequestValidator的验证规则"""
        validator = RequestValidator()
        validator.add_rule("name", str, required=True, min_length=2, max_length=20)
        validator.add_rule("age", int, required=False)

        # 缺少必填字段
        valid, errors = validator.validate({})
        assert valid is False

        # 字段过短
        valid, errors = validator.validate({"name": "A"})
        assert valid is False

        # 有效数据
        valid, errors = validator.validate({"name": "Alice", "age": 30})
        assert valid is True


# ============================================================
# 16. RPCServer/RPCClient 测试
# ============================================================

class TestRPCFramework:
    """RPC框架 - 服务端和客户端测试"""

    def test_RPCMessage序列化与反序列化(self):
        """测试RPC消息的JSON序列化往返"""
        msg = RPCMessage(payload={"text": "hello"}, metadata={"service": "test"})
        json_str = msg.to_json()
        restored = RPCMessage.from_json(json_str)
        assert restored.payload["text"] == "hello"
        assert restored.metadata["service"] == "test"

    def test_RPCService注册和调用方法(self):
        """测试RPC服务的注册和调用"""
        service = RPCService("TestService", version="1.0")

        def echo_handler(request: RPCMessage) -> RPCMessage:
            return RPCMessage(payload={"echo": request.payload.get("msg", "")})

        service.register_method("Echo", echo_handler)
        assert "Echo" in service.list_methods()

        request = RPCMessage(payload={"msg": "hello"})
        response = service.call_method("Echo", request)
        assert response.payload["echo"] == "hello"

    def test_RPCService调用不存在的方法抛出异常(self):
        """测试调用不存在的方法抛出RPCError"""
        service = RPCService("EmptyService")
        request = RPCMessage(payload={})
        with pytest.raises(RPCError, match="不存在"):
            service.call_method("NonExist", request)

    def test_RPCService输入验证(self):
        """测试RPC方法的输入验证"""
        service = RPCService("ValidService")

        def strict_handler(request: RPCMessage) -> RPCMessage:
            return RPCMessage(payload={"ok": True})

        def validator(request: RPCMessage):
            if "key" not in request.payload:
                return False, "缺少key字段"
            return True, ""

        service.register_method("Strict", strict_handler, input_validator=validator)
        bad_request = RPCMessage(payload={})
        with pytest.raises(RPCError, match="缺少key字段"):
            service.call_method("Strict", bad_request)

    def test_RPCServer_dispatch分发(self):
        """测试RPCServer的请求分发（不启动网络）"""
        server = RPCServer(port=0)  # 使用随机端口避免冲突
        service = RPCService("MathService")

        def add_handler(request: RPCMessage) -> RPCMessage:
            a = request.payload.get("a", 0)
            b = request.payload.get("b", 0)
            return RPCMessage(payload={"result": a + b})

        service.register_method("Add", add_handler)
        server.register_service(service)

        request = RPCMessage(
            payload={"a": 3, "b": 5},
            metadata={"service": "MathService", "method": "Add"}
        )
        response = server._dispatch(request)
        assert response.payload["result"] == 8

    def test_RPCError属性(self):
        """测试RPCError的属性"""
        err = RPCError(code=RPCStatusCode.NOT_FOUND, message="未找到", details="详情")
        assert err.code == RPCStatusCode.NOT_FOUND
        assert err.message == "未找到"
        assert err.details == "详情"

    def test_RPCClient上下文管理器(self):
        """测试RPCClient的上下文管理器接口（不实际连接）"""
        client = RPCClient(port=0)
        # 不调用connect，只测试接口存在
        assert client.connected is False
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")


# ============================================================
# 17. AsyncPipeline 测试
# ============================================================

class TestAsyncPipeline:
    """AsyncPipeline - 异步处理测试"""

    def test_异步处理单个文本(self):
        """测试异步处理单个文本返回正确结果"""
        pipeline = Pipeline(name="async_test")
        pipeline.add_component(_EchoComponent())
        async_pipeline = AsyncPipeline(pipeline)

        async def _run():
            return await async_pipeline.process("hello")

        result = asyncio.run(_run())
        assert result.text == "hello"
        assert result.get_attr("echoed") is True
        async_pipeline.shutdown()

    def test_异步批量处理(self):
        """测试异步批量处理多个文本"""
        pipeline = Pipeline(name="async_batch")
        pipeline.add_component(_AppendComponent(name="tag", tag="done"))
        async_pipeline = AsyncPipeline(pipeline)

        async def _run():
            return await async_pipeline.process_batch(["a", "b", "c"])

        results = asyncio.run(_run())
        assert len(results) == 3
        for doc in results:
            assert doc.get_attr("tags") == ["done"]
        async_pipeline.shutdown()

    def test_异步处理带回调(self):
        """测试异步处理完成后执行回调"""
        pipeline = Pipeline(name="async_cb")
        pipeline.add_component(_EchoComponent())
        async_pipeline = AsyncPipeline(pipeline)
        callback_results = []

        def callback(doc):
            callback_results.append(doc.text)

        async def _run():
            return await async_pipeline.process_with_callback("test_text", callback)

        result = asyncio.run(_run())
        assert result.text == "test_text"
        assert callback_results == ["test_text"]
        async_pipeline.shutdown()


# ============================================================
# 18. StreamProcessor 测试
# ============================================================

class TestStreamProcessor:
    """StreamProcessor - 流式处理测试"""

    def test_流式处理文本行列表(self):
        """测试process_lines处理文本行列表"""
        pipeline = Pipeline(name="stream_test")
        pipeline.add_component(_EchoComponent())
        processor = StreamProcessor(pipeline, chunk_size=2)
        results = processor.process_lines(["hello", "world", "test"])
        assert len(results) == 3
        for doc in results:
            assert doc.get_attr("echoed") is True

    def test_流式处理可迭代对象(self):
        """测试process_iterable处理可迭代文本"""
        pipeline = Pipeline(name="stream_iter")
        pipeline.add_component(_EchoComponent())
        processor = StreamProcessor(pipeline, chunk_size=2)
        texts = ["line1", "line2", "line3"]
        results = list(processor.process_iterable(texts))
        assert len(results) == 3

    def test_流式处理文件(self):
        """测试process_file处理文本文件"""
        pipeline = Pipeline(name="stream_file")
        pipeline.add_component(_EchoComponent())
        processor = StreamProcessor(pipeline, chunk_size=2)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("第一行\n第二行\n第三行\n")
            temp_path = f.name

        try:
            results = processor.process_file(temp_path)
            assert len(results) == 3
        finally:
            os.unlink(temp_path)

    def test_流式处理带行过滤(self):
        """测试process_file带行过滤功能"""
        pipeline = Pipeline(name="stream_filter")
        pipeline.add_component(_EchoComponent())
        processor = StreamProcessor(pipeline, chunk_size=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("keep\nskip\nkeep2\n")
            temp_path = f.name

        try:
            results = processor.process_file(
                temp_path,
                line_filter=lambda line: line.startswith("keep")
            )
            assert len(results) == 2
        finally:
            os.unlink(temp_path)


# ============================================================
# 19. Plugin/PluginManager 测试
# ============================================================

class TestPluginSystem:
    """Plugin/PluginManager - 插件系统测试"""

    def _create_concrete_plugin(self, name="test_plugin"):
        """创建一个具体的插件实现"""

        class ConcretePlugin(Plugin):
            def get_info(self):
                return PluginInfo(
                    name=name,
                    version="1.0.0",
                    description="测试插件",
                )

        return ConcretePlugin()

    def test_插件信息创建(self):
        """测试PluginInfo的创建和属性"""
        info = PluginInfo(
            name="demo",
            version="0.1.0",
            description="演示插件",
            author="tester",
        )
        assert info.name == "demo"
        assert info.version == "0.1.0"
        assert info.state == PluginState.DISCOVERED

    def test_插件生命周期状态转换(self):
        """测试插件从加载到启用的状态转换"""
        plugin = self._create_concrete_plugin()
        assert plugin.state == PluginState.DISCOVERED

        # 模拟加载
        info = plugin.get_info()
        info.state = PluginState.LOADED
        plugin._info = info
        plugin.on_load()
        assert plugin.state == PluginState.DISCOVERED  # on_load不改变state

        # 初始化
        pipeline = Pipeline()
        plugin.on_initialize(pipeline, {"key": "val"})
        assert plugin.state == PluginState.INITIALIZED

        # 启用
        plugin.on_enable()
        assert plugin.state == PluginState.ENABLED

        # 禁用
        plugin.on_disable()
        assert plugin.state == PluginState.DISABLED

    def test_插件信息导出为字典(self):
        """测试PluginInfo导出为字典"""
        info = PluginInfo(name="p1", version="1.0")
        d = info.to_dict()
        assert d["name"] == "p1"
        assert d["version"] == "1.0"
        assert d["state"] == "discovered"

    def test_PluginManager注册和获取插件(self):
        """测试PluginManager直接注册和获取插件"""
        manager = PluginManager()
        plugin = self._create_concrete_plugin("manual_plugin")
        # 直接放入内部字典模拟加载
        plugin._info = plugin.get_info()
        plugin._info.state = PluginState.LOADED
        manager._plugins["manual_plugin"] = plugin

        retrieved = manager.get_plugin("manual_plugin")
        assert retrieved is plugin
        assert retrieved.name == "manual_plugin"

    def test_PluginManager启用和禁用插件(self):
        """测试PluginManager启用和禁用插件"""
        manager = PluginManager()
        plugin = self._create_concrete_plugin("toggle_plugin")
        plugin._info = plugin.get_info()
        plugin._info.state = PluginState.LOADED
        plugin.on_initialize(None)
        manager._plugins["toggle_plugin"] = plugin

        assert manager.enable_plugin("toggle_plugin") is True
        assert plugin.state == PluginState.ENABLED

        assert manager.disable_plugin("toggle_plugin") is True
        assert plugin.state == PluginState.DISABLED

    def test_PluginManager卸载插件(self):
        """测试PluginManager卸载插件"""
        manager = PluginManager()
        plugin = self._create_concrete_plugin("unload_plugin")
        plugin._info = plugin.get_info()
        plugin._info.state = PluginState.LOADED
        manager._plugins["unload_plugin"] = plugin

        assert manager.unload_plugin("unload_plugin") is True
        assert manager.get_plugin("unload_plugin") is None

    def test_PluginManager列出插件(self):
        """测试PluginManager列出所有已加载插件"""
        manager = PluginManager()
        p1 = self._create_concrete_plugin("p1")
        p1._info = p1.get_info()
        p1._info.state = PluginState.LOADED
        manager._plugins["p1"] = p1

        p2 = self._create_concrete_plugin("p2")
        p2._info = p2.get_info()
        p2._info.state = PluginState.LOADED
        manager._plugins["p2"] = p2

        infos = manager.list_plugins()
        assert len(infos) == 2
        names = {i.name for i in infos}
        assert names == {"p1", "p2"}

    def test_PluginManager统计信息(self):
        """测试PluginManager的统计信息"""
        manager = PluginManager()
        manager.add_plugin_dir("/tmp/plugins")
        stats = manager.stats()
        assert "total_discovered" in stats
        assert "total_loaded" in stats

    def test_PluginDependency版本兼容性检查(self):
        """测试PluginDependency的版本兼容性检查"""
        dep = PluginDependency(name="core", min_version="1.0.0", max_version="2.0.0")
        assert dep.is_compatible("1.5.0") is True
        assert dep.is_compatible("0.9.0") is False
        assert dep.is_compatible("2.1.0") is False

    def test_PluginDependency无版本约束(self):
        """测试无版本约束时任何版本都兼容"""
        dep = PluginDependency(name="any")
        assert dep.is_compatible("99.99.99") is True

    def test_PluginManager检查依赖(self):
        """测试PluginManager检查插件依赖"""
        manager = PluginManager()
        info = PluginInfo(name="plugin_a", dependencies=["core", "utils"])
        manager._plugin_infos["plugin_a"] = info

        # 没有加载任何依赖
        ok, missing = manager.check_dependencies("plugin_a")
        assert ok is False
        assert set(missing) == {"core", "utils"}

        # 加载一个依赖
        p = self._create_concrete_plugin("core")
        p._info = p.get_info()
        manager._plugins["core"] = p
        ok, missing = manager.check_dependencies("plugin_a")
        assert ok is False
        assert "utils" in missing

    def test_PluginManager事件钩子(self):
        """测试PluginManager的事件钩子机制"""
        manager = PluginManager()
        hook_results = []
        manager.register_hook("test_event", lambda **kw: hook_results.append(kw))
        manager._trigger_hook("test_event", key="value")
        assert len(hook_results) == 1
        assert hook_results[0]["key"] == "value"


# ============================================================
# 附加：ProgressCallback 测试
# ============================================================

class TestProgressCallback:
    """ProgressCallback - 进度回调测试"""

    def test_更新进度计数(self):
        """测试进度更新后计数正确"""
        callback = ProgressCallback(total=100)
        callback.update(10)
        callback.update(20)
        assert callback.processed == 30

    def test_回调函数被调用(self):
        """测试进度更新时回调函数被触发"""
        reports = []
        callback = ProgressCallback(
            total=100,
            callback=lambda p, t, e: reports.append((p, t)),
            report_interval=0.0  # 立即报告
        )
        callback.update(10)
        assert len(reports) >= 1
        assert reports[0][0] == 10


# ============================================================
# 附加：Route 测试
# ============================================================

class TestRoute:
    """Route - 路由定义测试"""

    def test_路由匹配成功(self):
        """测试路由匹配成功返回参数"""
        route = Route("GET", "/items/{id}", lambda r: APIResponse.ok())
        params = route.match("GET", "/items/42")
        assert params is not None
        assert params["id"] == "42"

    def test_路由方法不匹配(self):
        """测试HTTP方法不匹配时返回None"""
        route = Route("POST", "/items", lambda r: APIResponse.ok())
        assert route.match("GET", "/items") is None

    def test_路由路径不匹配(self):
        """测试路径不匹配时返回None"""
        route = Route("GET", "/items/{id}", lambda r: APIResponse.ok())
        assert route.match("GET", "/users/1") is None
