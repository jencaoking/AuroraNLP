#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
import tempfile
import json
import time
import threading

from AuroraNLP import (
    LogLevel,
    LogFormat,
    StructuredLogRecord,
    ConsoleLogHandler,
    FileLogHandler,
    TimeRotatingFileLogHandler,
    MemoryLogHandler,
    LevelFilter,
    KeywordFilter,
    ModuleFilter,
    Logger,
    LogManager,
    get_logger,
    HealthStatus,
    HealthCheck,
    MemoryHealthCheck,
    DiskHealthCheck,
    HealthChecker,
    MetricType,
    PrometheusMetric,
    PrometheusCounter,
    PrometheusGauge,
    PrometheusRegistry,
    generate_dockerfile_content,
    generate_k8s_deployment_content,
    generate_k8s_service_content,
    generate_k8s_ingress_content,
)


class TestLogLevel(unittest.TestCase):
    """测试日志级别"""

    def test_level_values(self):
        self.assertEqual(LogLevel.TRACE, 5)
        self.assertEqual(LogLevel.DEBUG, 10)
        self.assertEqual(LogLevel.INFO, 20)
        self.assertEqual(LogLevel.WARN, 30)
        self.assertEqual(LogLevel.WARNING, 30)
        self.assertEqual(LogLevel.ERROR, 40)
        self.assertEqual(LogLevel.FATAL, 50)
        self.assertEqual(LogLevel.CRITICAL, 50)


class TestStructuredLogRecord(unittest.TestCase):
    """测试结构化日志记录"""

    def test_to_dict_basic(self):
        record = StructuredLogRecord(
            level=LogLevel.INFO,
            message="hello world",
            logger_name="test",
        )
        d = record.to_dict()
        self.assertEqual(d["level"], "INFO")
        self.assertEqual(d["message"], "hello world")
        self.assertEqual(d["logger"], "test")
        self.assertIn("timestamp", d)

    def test_to_dict_with_extra(self):
        record = StructuredLogRecord(
            level=LogLevel.INFO,
            message="test",
            extra={"key": "value", "count": 42},
        )
        d = record.to_dict()
        self.assertEqual(d["extra"]["key"], "value")
        self.assertEqual(d["extra"]["count"], 42)

    def test_to_dict_with_exception(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            record = StructuredLogRecord(
                level=LogLevel.ERROR,
                message="出错了",
                exception=e,
            )
        d = record.to_dict()
        self.assertEqual(d["exception"]["type"], "ValueError")
        self.assertIn("traceback", d["exception"])

    def test_to_dict_with_request_id(self):
        record = StructuredLogRecord(
            level=LogLevel.INFO,
            message="test",
            request_id="req-123",
            trace_id="trace-456",
            span_id="span-789",
        )
        d = record.to_dict()
        self.assertEqual(d["request_id"], "req-123")
        self.assertEqual(d["trace_id"], "trace-456")
        self.assertEqual(d["span_id"], "span-789")

    def test_to_json(self):
        record = StructuredLogRecord(
            level=LogLevel.INFO,
            message="测试中文",
        )
        j = record.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["message"], "测试中文")
        self.assertEqual(parsed["level"], "INFO")

    def test_to_text(self):
        record = StructuredLogRecord(
            level=LogLevel.INFO,
            message="hello",
            logger_name="myapp",
        )
        text = record.to_text()
        self.assertIn("INFO", text)
        self.assertIn("myapp", text)
        self.assertIn("hello", text)

    def test_to_text_with_exception(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            record = StructuredLogRecord(
                level=LogLevel.ERROR,
                message="fail",
                exception=e,
            )
        text = record.to_text()
        self.assertIn("RuntimeError", text)
        self.assertIn("boom", text)

    def test_to_text_with_request_id(self):
        record = StructuredLogRecord(
            level=LogLevel.INFO,
            message="test",
            request_id="req-abc",
        )
        text = record.to_text()
        self.assertIn("req=req-abc", text)


class TestMemoryLogHandler(unittest.TestCase):
    """测试内存日志处理器"""

    def setUp(self):
        self.handler = MemoryLogHandler(max_records=100)

    def test_emit_and_get(self):
        record = StructuredLogRecord(level=LogLevel.INFO, message="test")
        self.handler.emit(record)
        self.assertEqual(self.handler.count, 1)
        records = self.handler.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].message, "test")

    def test_level_filter(self):
        self.handler.level = LogLevel.ERROR
        self.handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="info"))
        self.handler.emit(StructuredLogRecord(level=LogLevel.ERROR, message="error"))
        self.assertEqual(self.handler.count, 1)

    def test_max_records(self):
        handler = MemoryLogHandler(max_records=5)
        for i in range(10):
            handler.emit(StructuredLogRecord(level=LogLevel.INFO, message=f"msg{i}"))
        self.assertEqual(handler.count, 5)
        records = handler.get_records()
        self.assertEqual(records[-1].message, "msg9")

    def test_clear(self):
        self.handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="test"))
        self.handler.clear()
        self.assertEqual(self.handler.count, 0)


class TestConsoleLogHandler(unittest.TestCase):
    """测试控制台日志处理器"""

    def test_level_filter(self):
        handler = ConsoleLogHandler()
        handler.level = LogLevel.ERROR
        # 不应抛出异常
        handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="info"))
        handler.emit(StructuredLogRecord(level=LogLevel.ERROR, message="error"))
        handler.close()


class TestFileLogHandler(unittest.TestCase):
    """测试文件日志处理器"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test.log")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_and_read(self):
        handler = FileLogHandler(
            filepath=self.log_path,
            format_type=LogFormat.TEXT,
        )
        handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="hello"))
        handler.close()

        with open(self.log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("hello", content)
        self.assertIn("INFO", content)

    def test_json_format(self):
        handler = FileLogHandler(
            filepath=self.log_path,
            format_type=LogFormat.JSON,
        )
        handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="测试"))
        handler.close()

        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["message"], "测试")

    def test_rotation(self):
        handler = FileLogHandler(
            filepath=self.log_path,
            max_bytes=200,
            backup_count=3,
        )
        for i in range(50):
            handler.emit(StructuredLogRecord(
                level=LogLevel.INFO,
                message=f"msg_{i:04d}_padding_data",
            ))
        handler.close()

        # 应该有轮转文件
        self.assertTrue(os.path.exists(self.log_path))
        rotated = f"{self.log_path}.1"
        self.assertTrue(os.path.exists(rotated))

    def test_level_filter(self):
        handler = FileLogHandler(
            filepath=self.log_path,
            level=LogLevel.ERROR,
        )
        handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="info"))
        handler.emit(StructuredLogRecord(level=LogLevel.ERROR, message="error"))
        handler.close()

        with open(self.log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("info", content)
        self.assertIn("error", content)


class TestTimeRotatingFileLogHandler(unittest.TestCase):
    """测试基于时间的日志轮转处理器"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "timed.log")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write(self):
        handler = TimeRotatingFileLogHandler(
            filepath=self.log_path,
            interval=TimeRotatingFileLogHandler.RotationInterval.DAILY,
        )
        handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="hello"))
        handler.close()

        # 应该创建了带日期后缀的文件
        files = os.listdir(self.temp_dir)
        dated_files = [f for f in files if f.startswith("timed.log.")]
        self.assertGreaterEqual(len(dated_files), 1)

    def test_json_format(self):
        handler = TimeRotatingFileLogHandler(
            filepath=self.log_path,
            format_type=LogFormat.JSON,
        )
        handler.emit(StructuredLogRecord(level=LogLevel.INFO, message="json_test"))
        handler.close()

        files = os.listdir(self.temp_dir)
        dated_files = [f for f in files if f.startswith("timed.log.")]
        self.assertGreaterEqual(len(dated_files), 1)

        with open(os.path.join(self.temp_dir, dated_files[0]), "r", encoding="utf-8") as f:
            lines = f.readlines()
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["message"], "json_test")


class TestLogFilters(unittest.TestCase):
    """测试日志过滤器"""

    def test_level_filter(self):
        f = LevelFilter(min_level=LogLevel.ERROR)
        record_info = StructuredLogRecord(level=LogLevel.INFO, message="info")
        record_error = StructuredLogRecord(level=LogLevel.ERROR, message="error")
        self.assertFalse(f.filter(record_info))
        self.assertTrue(f.filter(record_error))

    def test_keyword_filter_include(self):
        f = KeywordFilter(keyword="ERROR")
        self.assertTrue(f.filter(StructuredLogRecord(level=LogLevel.INFO, message="ERROR found")))
        self.assertFalse(f.filter(StructuredLogRecord(level=LogLevel.INFO, message="normal msg")))

    def test_keyword_filter_exclude(self):
        f = KeywordFilter(keyword="DEBUG", exclude=True)
        self.assertFalse(f.filter(StructuredLogRecord(level=LogLevel.INFO, message="DEBUG skip")))
        self.assertTrue(f.filter(StructuredLogRecord(level=LogLevel.INFO, message="normal msg")))

    def test_module_filter(self):
        f = ModuleFilter(module_name="my_module")
        self.assertTrue(f.filter(StructuredLogRecord(level=LogLevel.INFO, message="test", logger_name="my_module")))
        self.assertFalse(f.filter(StructuredLogRecord(level=LogLevel.INFO, message="test", logger_name="other")))


class TestLogger(unittest.TestCase):
    """测试Logger"""

    def setUp(self):
        self.memory_handler = MemoryLogHandler()
        self.logger = Logger(
            name="test_logger",
            level=LogLevel.TRACE,
            handlers=[self.memory_handler],
        )

    def tearDown(self):
        self.logger.close()

    def test_log_levels(self):
        self.logger.trace("trace msg")
        self.logger.debug("debug msg")
        self.logger.info("info msg")
        self.logger.warn("warn msg")
        self.logger.warning("warning msg")
        self.logger.error("error msg")
        self.logger.fatal("fatal msg")
        self.logger.critical("critical msg")
        self.assertEqual(self.memory_handler.count, 8)

    def test_logger_level_filter(self):
        self.logger.level = LogLevel.ERROR
        self.logger.info("should not appear")
        self.logger.error("should appear")
        self.assertEqual(self.memory_handler.count, 1)
        self.assertEqual(self.memory_handler.get_records()[0].message, "should appear")

    def test_extra_fields(self):
        self.logger.info("test", key1="val1", key2=42)
        record = self.memory_handler.get_records()[0]
        self.assertEqual(record.extra["key1"], "val1")
        self.assertEqual(record.extra["key2"], 42)

    def test_exception_logging(self):
        try:
            raise ValueError("test exception")
        except ValueError as e:
            self.logger.error("出错了", exception=e)
        record = self.memory_handler.get_records()[0]
        self.assertIsNotNone(record.exception)
        self.assertEqual(str(record.exception), "test exception")

    def test_context(self):
        self.logger.set_context(service="nlp", version="1.0")
        self.logger.info("with context")
        record = self.memory_handler.get_records()[0]
        self.assertEqual(record.context["service"], "nlp")
        self.assertEqual(record.context["version"], "1.0")

    def test_global_context(self):
        Logger.set_global_context(env="test", host="localhost")
        try:
            logger2 = Logger(
                name="logger2",
                handlers=[MemoryLogHandler()],
            )
            logger2.info("global ctx")
            record = logger2._handlers[0].get_records()[0]
            self.assertEqual(record.context["env"], "test")
            self.assertEqual(record.context["host"], "localhost")
        finally:
            Logger.clear_global_context()

    def test_add_remove_handler(self):
        h2 = MemoryLogHandler()
        self.logger.add_handler(h2)
        self.logger.info("both")
        self.assertEqual(self.memory_handler.count, 1)
        self.assertEqual(h2.count, 1)
        self.logger.remove_handler(h2)
        self.logger.info("only first")
        self.assertEqual(self.memory_handler.count, 2)
        self.assertEqual(h2.count, 1)

    def test_add_filter(self):
        self.logger.add_filter(LevelFilter(LogLevel.ERROR))
        self.logger.info("filtered")
        self.logger.error("passed")
        self.assertEqual(self.memory_handler.count, 1)

    def test_structured_record_content(self):
        self.logger.info("structured test", count=99)
        record = self.memory_handler.get_records()[0]
        d = record.to_dict()
        self.assertEqual(d["message"], "structured test")
        self.assertEqual(d["extra"]["count"], 99)
        self.assertEqual(d["logger"], "test_logger")
        self.assertIn("timestamp", d)


class TestLogManager(unittest.TestCase):
    """测试LogManager单例"""

    def setUp(self):
        # 重置单例
        LogManager._instance = None
        LogManager._instance_lock = threading.RLock()

    def tearDown(self):
        if LogManager._instance:
            LogManager._instance.close_all()
        LogManager._instance = None

    def test_singleton(self):
        m1 = LogManager()
        m2 = LogManager()
        self.assertIs(m1, m2)

    def test_get_logger(self):
        mgr = LogManager()
        l1 = mgr.get_logger("mod1")
        l2 = mgr.get_logger("mod1")
        self.assertIs(l1, l2)
        l3 = mgr.get_logger("mod2")
        self.assertIsNot(l1, l3)

    def test_global_handler(self):
        mgr = LogManager()
        handler = MemoryLogHandler()
        mgr.add_handler(handler)
        logger = mgr.get_logger("test")
        logger.info("global handler test")
        self.assertEqual(handler.count, 1)

    def test_global_level(self):
        mgr = LogManager()
        handler = MemoryLogHandler()
        mgr.add_handler(handler)
        mgr.set_level(LogLevel.ERROR)
        logger = mgr.get_logger("test")
        logger.info("filtered")
        logger.error("passed")
        self.assertEqual(handler.count, 1)

    def test_global_filter(self):
        mgr = LogManager()
        handler = MemoryLogHandler()
        mgr.add_handler(handler)
        mgr.add_filter(KeywordFilter(keyword="SECRET", exclude=True))
        logger = mgr.get_logger("test")
        logger.info("normal message")
        logger.info("SECRET message")
        self.assertEqual(handler.count, 1)

    def test_close_all(self):
        mgr = LogManager()
        mgr.get_logger("a")
        mgr.get_logger("b")
        mgr.close_all()
        self.assertEqual(len(mgr._loggers), 0)


class TestGetLogger(unittest.TestCase):
    """测试get_logger快捷方法"""

    def setUp(self):
        LogManager._instance = None
        LogManager._instance_lock = threading.RLock()

    def tearDown(self):
        if LogManager._instance:
            LogManager._instance.close_all()
        LogManager._instance = None

    def test_get_logger_returns_logger(self):
        logger = get_logger("quick_test")
        self.assertIsInstance(logger, Logger)
        self.assertEqual(logger.name, "quick_test")


class TestHealthCheck(unittest.TestCase):
    """测试健康检查功能 (步骤82)"""

    def test_health_status_enum(self):
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.UNHEALTHY.value, "unhealthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.UNKNOWN.value, "unknown")

    def test_memory_health_check(self):
        check = MemoryHealthCheck()
        self.assertEqual(check.name, "memory")
        status, msg, metadata = check.check()
        # 即使没有resource模块，也应该返回HEALTHY或UNKNOWN
        self.assertIn(status, [HealthStatus.HEALTHY, HealthStatus.UNKNOWN])

    def test_disk_health_check(self):
        check = DiskHealthCheck()
        self.assertEqual(check.name, "disk")
        status, msg, metadata = check.check()
        self.assertIsNotNone(status)

    def test_health_checker_basic(self):
        checker = HealthChecker(version="1.0.0")
        self.assertEqual(checker.version, "1.0.0")
        checker.add_check(MemoryHealthCheck())
        checker.add_check(DiskHealthCheck())
        result = checker.check_liveness()
        self.assertIsInstance(result, HealthChecker.AggregatedResult)
        self.assertIn(result.overall_status, list(HealthStatus))
        self.assertIsInstance(result.to_dict(), dict)
        self.assertIsInstance(result.to_json(), str)
        self.assertGreaterEqual(len(result.checks), 2)

    def test_health_checker_compute_overall(self):
        checker = HealthChecker()
        result_healthy = checker._compute_overall([])
        self.assertEqual(result_healthy, HealthStatus.HEALTHY)


class TestPrometheusMetrics(unittest.TestCase):
    """测试Prometheus指标功能 (步骤83)"""

    def test_counter_basic(self):
        counter = PrometheusCounter("test_counter", "Test counter")
        counter.inc()
        counter.inc(5)
        exposition = counter.to_exposition()
        self.assertIn("test_counter", exposition)
        self.assertIn("Test counter", exposition)
        self.assertIn("6.0", exposition)

    def test_counter_with_labels(self):
        counter = PrometheusCounter("requests_total", "Total requests")
        counter.inc(1, {"endpoint": "/api", "method": "GET"})
        counter.inc(2, {"endpoint": "/api", "method": "POST"})
        exposition = counter.to_exposition()
        self.assertIn('endpoint="/api"', exposition)
        self.assertIn('method="GET"', exposition)

    def test_counter_negative_error(self):
        counter = PrometheusCounter("test", "Test")
        with self.assertRaises(ValueError):
            counter.inc(-1)

    def test_gauge_basic(self):
        gauge = PrometheusGauge("active_connections", "Active connections")
        gauge.set(100)
        gauge.inc(20)
        gauge.dec(10)
        exposition = gauge.to_exposition()
        self.assertIn("110", exposition)

    def test_registry(self):
        registry = PrometheusRegistry()
        counter = PrometheusCounter("req", "Requests")
        gauge = PrometheusGauge("conn", "Connections")
        registry.register(counter)
        registry.register(gauge)
        counter.inc(42)
        gauge.set(100)
        exposition = registry.get_exposition()
        self.assertIn("req", exposition)
        self.assertIn("conn", exposition)
        self.assertIn("42", exposition)
        self.assertIn("100", exposition)


class TestDockerAndK8s(unittest.TestCase):
    """测试Docker和K8s配置生成 (步骤84-85)"""

    def test_dockerfile_generation(self):
        content = generate_dockerfile_content()
        self.assertIn("FROM", content)
        self.assertIn("WORKDIR", content)
        self.assertIn("COPY requirements.txt", content)
        self.assertIn("EXPOSE 8000", content)
        self.assertIn("HEALTHCHECK", content)

    def test_k8s_deployment(self):
        content = generate_k8s_deployment_content()
        self.assertIn("apiVersion: apps/v1", content)
        self.assertIn("kind: Deployment", content)
        self.assertIn("livenessProbe", content)
        self.assertIn("readinessProbe", content)
        self.assertIn("resources", content)

    def test_k8s_service(self):
        content = generate_k8s_service_content()
        self.assertIn("apiVersion: v1", content)
        self.assertIn("kind: Service", content)
        self.assertIn("selector", content)

    def test_k8s_ingress(self):
        content = generate_k8s_ingress_content()
        self.assertIn("apiVersion: networking.k8s.io/v1", content)
        self.assertIn("kind: Ingress", content)


if __name__ == "__main__":
    unittest.main()
