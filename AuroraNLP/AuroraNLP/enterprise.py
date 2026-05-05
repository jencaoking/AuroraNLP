"""
AuroraNLP 企业级功能模块 - 阶段六

本模块实现了步骤81-90的功能：
- 步骤81: 日志系统（结构化日志、日志级别配置、日志轮转）
- 步骤82: 健康检查接口
- 步骤83: Prometheus指标
- 步骤84: Docker镜像支持
- 步骤85: Kubernetes配置
- 步骤86: 限流熔断
- 步骤87: 认证授权
- 步骤88: 配置中心集成
- 步骤89: 灰度发布支持
- 步骤90: 灾备方案

约束：零外部依赖，纯Python标准库实现
"""

import os
import sys
import json
import time
import logging
import logging.handlers
import threading
import traceback
from typing import (
    Any, Callable, Dict, List, Optional,
    Tuple, Union, Iterator
)
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from collections import deque
from datetime import datetime, timezone
from functools import wraps


# ==================== 步骤81: 日志系统 ====================

class LogLevel(IntEnum):
    """日志级别"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    WARNING = 30
    ERROR = 40
    FATAL = 50
    CRITICAL = 50


class LogFormat(Enum):
    """日志格式"""
    TEXT = "text"
    JSON = "json"


class StructuredLogRecord:
    """结构化日志记录"""

    def __init__(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "auroranlp",
        timestamp: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
        source_func: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ):
        self.level = level
        self.message = message
        self.logger_name = logger_name
        self.timestamp = timestamp or time.time()
        self.extra = extra or {}
        self.context = context or {}
        self.exception = exception
        self.source_file = source_file
        self.source_line = source_line
        self.source_func = source_func
        self.request_id = request_id
        self.trace_id = trace_id
        self.span_id = span_id

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "level": self.level.name,
            "logger": self.logger_name,
            "message": self.message,
        }
        if self.request_id:
            result["request_id"] = self.request_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.source_file:
            result["source"] = {
                "file": self.source_file,
                "line": self.source_line,
                "func": self.source_func,
            }
        if self.exception:
            result["exception"] = {
                "type": type(self.exception).__name__,
                "message": str(self.exception),
                "traceback": traceback.format_exception(
                    type(self.exception),
                    self.exception,
                    self.exception.__traceback__,
                ),
            }
        if self.extra:
            result["extra"] = self.extra
        if self.context:
            result["context"] = self.context
        return result

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def to_text(self) -> str:
        """转换为文本格式"""
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        ts_str = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}Z"
        parts = [f"[{ts_str}]", f"[{self.level.name:>8s}]", f"[{self.logger_name}]"]
        if self.request_id:
            parts.append(f"[req={self.request_id}]")
        if self.trace_id:
            parts.append(f"[trace={self.trace_id}]")
        parts.append(self.message)
        if self.exception:
            parts.append(
                f"\n  {type(self.exception).__name__}: {self.exception}"
            )
        return " ".join(parts)


# ---------- 日志处理器 ----------

class LogHandler(ABC):
    """日志处理器抽象基类"""

    @abstractmethod
    def emit(self, record: StructuredLogRecord) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class ConsoleLogHandler(LogHandler):
    """控制台日志处理器"""

    def __init__(self, format_type: LogFormat = LogFormat.TEXT):
        self._format = format_type
        self._level = LogLevel.TRACE

    @property
    def level(self) -> LogLevel:
        return self._level

    @level.setter
    def level(self, value: LogLevel):
        self._level = value

    def emit(self, record: StructuredLogRecord) -> None:
        if record.level < self._level:
            return
        if self._format == LogFormat.JSON:
            print(record.to_json(), file=sys.stderr)
        else:
            print(record.to_text(), file=sys.stderr)

    def close(self) -> None:
        pass


class FileLogHandler(LogHandler):
    """文件日志处理器（带轮转）"""

    def __init__(
        self,
        filepath: str,
        format_type: LogFormat = LogFormat.TEXT,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        level: LogLevel = LogLevel.TRACE,
    ):
        self._filepath = filepath
        self._format = format_type
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._level = level
        self._lock = threading.RLock()
        self._file = None
        self._current_size = 0
        self._open_file()

    def _open_file(self) -> None:
        """打开日志文件"""
        dirpath = os.path.dirname(self._filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self._file = open(self._filepath, "a", encoding="utf-8")
        self._current_size = os.path.getsize(self._filepath)

    def _rotate(self) -> None:
        """执行日志轮转"""
        if self._file:
            self._file.close()
        for i in range(self._backup_count - 1, 0, -1):
            src = f"{self._filepath}.{i}"
            dst = f"{self._filepath}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        if os.path.exists(self._filepath):
            os.rename(self._filepath, f"{self._filepath}.1")
        self._file = open(self._filepath, "a", encoding="utf-8")
        self._current_size = 0

    @property
    def level(self) -> LogLevel:
        return self._level

    @level.setter
    def level(self, value: LogLevel):
        self._level = value

    def emit(self, record: StructuredLogRecord) -> None:
        if record.level < self._level:
            return
        line = (
            record.to_json() if self._format == LogFormat.JSON
            else record.to_text()
        )
        line += "\n"
        with self._lock:
            if self._max_bytes > 0 and self._current_size + len(line) > self._max_bytes:
                self._rotate()
            self._file.write(line)
            self._file.flush()
            self._current_size += len(line)

    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class TimeRotatingFileLogHandler(LogHandler):
    """基于时间的日志轮转处理器"""

    class RotationInterval(Enum):
        HOURLY = "H"
        DAILY = "D"
        WEEKLY = "W"
        MONTHLY = "M"

    def __init__(
        self,
        filepath: str,
        format_type: LogFormat = LogFormat.TEXT,
        interval: RotationInterval = RotationInterval.DAILY,
        backup_count: int = 30,
        level: LogLevel = LogLevel.TRACE,
    ):
        self._filepath = filepath
        self._format = format_type
        self._interval = interval
        self._backup_count = backup_count
        self._level = level
        self._lock = threading.RLock()
        self._file = None
        self._current_date = None
        self._open_file()

    def _get_date_key(self) -> str:
        """获取当前日期键"""
        now = datetime.now()
        if self._interval == self.RotationInterval.HOURLY:
            return now.strftime("%Y%m%d%H")
        elif self._interval == self.RotationInterval.DAILY:
            return now.strftime("%Y%m%d")
        elif self._interval == self.RotationInterval.WEEKLY:
            return now.strftime("%Y%W")
        else:
            return now.strftime("%Y%m")

    def _open_file(self) -> None:
        """打开日志文件"""
        dirpath = os.path.dirname(self._filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self._current_date = self._get_date_key()
        dated_path = f"{self._filepath}.{self._current_date}"
        self._file = open(dated_path, "a", encoding="utf-8")

    def _rotate(self) -> None:
        """执行时间轮转"""
        if self._file:
            self._file.close()
        self._open_file()
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """清理旧日志文件"""
        dirpath = os.path.dirname(self._filepath) or "."
        basename = os.path.basename(self._filepath)
        prefix = basename + "."
        files = []
        for f in os.listdir(dirpath):
            if f.startswith(prefix) and f != basename:
                files.append(f)
        files.sort(reverse=True)
        for f in files[self._backup_count:]:
            os.remove(os.path.join(dirpath, f))

    @property
    def level(self) -> LogLevel:
        return self._level

    @level.setter
    def level(self, value: LogLevel):
        self._level = value

    def emit(self, record: StructuredLogRecord) -> None:
        if record.level < self._level:
            return
        date_key = self._get_date_key()
        with self._lock:
            if date_key != self._current_date:
                self._rotate()
            line = (
                record.to_json() if self._format == LogFormat.JSON
                else record.to_text()
            )
            line += "\n"
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class MemoryLogHandler(LogHandler):
    """内存日志处理器（用于测试和调试）"""

    def __init__(self, max_records: int = 1000, level: LogLevel = LogLevel.TRACE):
        self._records: deque = deque(maxlen=max_records)
        self._level = level

    @property
    def level(self) -> LogLevel:
        return self._level

    @level.setter
    def level(self, value: LogLevel):
        self._level = value

    def emit(self, record: StructuredLogRecord) -> None:
        if record.level < self._level:
            return
        self._records.append(record)

    def close(self) -> None:
        self._records.clear()

    def get_records(self) -> List[StructuredLogRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    @property
    def count(self) -> int:
        return len(self._records)


# ---------- 日志过滤器 ----------

class LogFilter(ABC):
    """日志过滤器抽象基类"""

    @abstractmethod
    def filter(self, record: StructuredLogRecord) -> bool:
        ...


class LevelFilter(LogFilter):
    """级别过滤器"""

    def __init__(self, min_level: LogLevel = LogLevel.TRACE):
        self._min_level = min_level

    def filter(self, record: StructuredLogRecord) -> bool:
        return record.level >= self._min_level


class KeywordFilter(LogFilter):
    """关键词过滤器"""

    def __init__(self, keyword: str, exclude: bool = False):
        self._keyword = keyword
        self._exclude = exclude

    def filter(self, record: StructuredLogRecord) -> bool:
        matched = self._keyword in record.message
        return not matched if self._exclude else matched


class ModuleFilter(LogFilter):
    """模块过滤器"""

    def __init__(self, module_name: str):
        self._module_name = module_name

    def filter(self, record: StructuredLogRecord) -> bool:
        return record.logger_name == self._module_name


# ---------- 日志管理器 ----------

class Logger:
    """AuroraNLP 日志管理器

    提供结构化日志、多处理器、日志轮转等功能。

    使用示例::

        logger = Logger("my_module")
        logger.info("处理开始", text_count=100)
        logger.error("处理失败", error_code=500, extra={"detail": "timeout"})
    """

    _global_context: Dict[str, Any] = {}
    _global_context_lock = threading.RLock()

    def __init__(
        self,
        name: str = "auroranlp",
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[LogHandler]] = None,
    ):
        self._name = name
        self._level = level
        self._handlers: List[LogHandler] = handlers or []
        self._filters: List[LogFilter] = []
        self._context: Dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def level(self) -> LogLevel:
        return self._level

    @level.setter
    def level(self, value: LogLevel):
        self._level = value

    def add_handler(self, handler: LogHandler) -> None:
        """添加日志处理器"""
        with self._lock:
            self._handlers.append(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """移除日志处理器"""
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def add_filter(self, f: LogFilter) -> None:
        """添加过滤器"""
        self._filters.append(f)

    def remove_filter(self, f: LogFilter) -> None:
        """移除过滤器"""
        if f in self._filters:
            self._filters.remove(f)

    def set_context(self, **kwargs: Any) -> None:
        """设置日志上下文"""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """清除日志上下文"""
        self._context.clear()

    @classmethod
    def set_global_context(cls, **kwargs: Any) -> None:
        """设置全局日志上下文"""
        with cls._global_context_lock:
            cls._global_context.update(kwargs)

    @classmethod
    def clear_global_context(cls) -> None:
        """清除全局日志上下文"""
        with cls._global_context_lock:
            cls._global_context.clear()

    def _build_record(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
    ) -> StructuredLogRecord:
        """构建结构化日志记录"""
        # 获取调用者信息
        frame = traceback.extract_stack()[-3] if traceback.extract_stack() else None
        source_file = frame.filename if frame else None
        source_line = frame.lineno if frame else None
        source_func = frame.name if frame else None

        # 合并上下文
        merged_context: Dict[str, Any] = {}
        with self._global_context_lock:
            merged_context.update(self._global_context)
        merged_context.update(self._context)

        return StructuredLogRecord(
            level=level,
            message=message,
            logger_name=self._name,
            extra=extra,
            context=merged_context if merged_context else None,
            exception=exception,
            source_file=source_file,
            source_line=source_line,
            source_func=source_func,
        )

    def _log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        """核心日志方法"""
        if level < self._level:
            return
        record = self._build_record(level, message, extra, exception)
        # 应用过滤器
        for f in self._filters:
            if not f.filter(record):
                return
        with self._lock:
            for handler in self._handlers:
                handler.emit(record)

    def trace(self, message: str, **extra: Any) -> None:
        self._log(LogLevel.TRACE, message, extra)

    def debug(self, message: str, **extra: Any) -> None:
        self._log(LogLevel.DEBUG, message, extra)

    def info(self, message: str, **extra: Any) -> None:
        self._log(LogLevel.INFO, message, extra)

    def warn(self, message: str, **extra: Any) -> None:
        self._log(LogLevel.WARN, message, extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._log(LogLevel.WARNING, message, extra)

    def error(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        **extra: Any,
    ) -> None:
        self._log(LogLevel.ERROR, message, extra, exception)

    def fatal(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        **extra: Any,
    ) -> None:
        self._log(LogLevel.FATAL, message, extra, exception)

    def critical(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        **extra: Any,
    ) -> None:
        self._log(LogLevel.CRITICAL, message, extra, exception)

    def close(self) -> None:
        """关闭所有处理器"""
        with self._lock:
            for handler in self._handlers:
                handler.close()


class LogManager:
    """日志管理器（单例）

    集中管理所有Logger实例，提供统一配置。
    """

    _instance: Optional["LogManager"] = None
    _instance_lock = threading.RLock()

    def __new__(cls) -> "LogManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._loggers: Dict[str, Logger] = {}
        self._global_handlers: List[LogHandler] = []
        self._global_filters: List[LogFilter] = []
        self._default_level = LogLevel.INFO
        self._lock = threading.RLock()
        self._initialized = True

    def get_logger(self, name: str = "auroranlp") -> Logger:
        """获取或创建Logger"""
        with self._lock:
            if name not in self._loggers:
                logger = Logger(name=name, level=self._default_level)
                for handler in self._global_handlers:
                    logger.add_handler(handler)
                for f in self._global_filters:
                    logger.add_filter(f)
                self._loggers[name] = logger
            return self._loggers[name]

    def add_handler(self, handler: LogHandler) -> None:
        """添加全局处理器"""
        with self._lock:
            self._global_handlers.append(handler)
            for logger in self._loggers.values():
                logger.add_handler(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """移除全局处理器"""
        with self._lock:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
            for logger in self._loggers.values():
                logger.remove_handler(handler)

    def set_level(self, level: LogLevel) -> None:
        """设置全局日志级别"""
        with self._lock:
            self._default_level = level
            for logger in self._loggers.values():
                logger.level = level

    def add_filter(self, f: LogFilter) -> None:
        """添加全局过滤器"""
        with self._lock:
            self._global_filters.append(f)
            for logger in self._loggers.values():
                logger.add_filter(f)

    def close_all(self) -> None:
        """关闭所有日志器"""
        with self._lock:
            for logger in self._loggers.values():
                logger.close()
            self._loggers.clear()


def get_logger(name: str = "auroranlp") -> Logger:
    """快捷方法：获取日志器"""
    return LogManager().get_logger(name)


# ==================== 步骤82: 健康检查接口 ====================

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthCheck(ABC):
    """健康检查抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def check(self) -> Tuple[HealthStatus, Optional[str], Optional[Dict[str, Any]]]:
        ...


class MemoryHealthCheck(HealthCheck):
    """内存健康检查"""

    def __init__(self, warning_threshold: float = 0.85, error_threshold: float = 0.95):
        self._warning_threshold = warning_threshold
        self._error_threshold = error_threshold

    @property
    def name(self) -> str:
        return "memory"

    def check(self) -> Tuple[HealthStatus, Optional[str], Optional[Dict[str, Any]]]:
        try:
            import gc
            gc.collect()
            free = gc.mem_alloc() if hasattr(gc, 'mem_alloc') else 0
            total = gc.mem_free() if hasattr(gc, 'mem_free') else 1
            # 尝试使用更准确的方法
            try:
                import resource
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                max_rss = rusage.ru_maxrss
                metrics = {
                    "max_rss": max_rss,
                    "gc_collects": len(gc.get_stats()) if hasattr(gc, 'get_stats') else 0
                }
                return HealthStatus.HEALTHY, None, metrics
            except ImportError:
                pass
            return HealthStatus.HEALTHY, None, {"simple_check": True}
        except Exception as e:
            return HealthStatus.UNKNOWN, str(e), None


class DiskHealthCheck(HealthCheck):
    """磁盘健康检查"""

    def __init__(self, path: str = "/", warning_threshold: float = 0.90, error_threshold: float = 0.98):
        self._path = path
        self._warning_threshold = warning_threshold
        self._error_threshold = error_threshold

    @property
    def name(self) -> str:
        return "disk"

    def check(self) -> Tuple[HealthStatus, Optional[str], Optional[Dict[str, Any]]]:
        try:
            statvfs = os.statvfs(self._path)
            free = statvfs.f_frsize * statvfs.f_bfree
            total = statvfs.f_frsize * statvfs.f_blocks
            used = total - free
            usage_ratio = used / total if total > 0 else 0
            metrics = {
                "path": self._path,
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "usage_percent": round(usage_ratio * 100, 2)
            }
            if usage_ratio >= self._error_threshold:
                return HealthStatus.UNHEALTHY, "磁盘空间不足", metrics
            elif usage_ratio >= self._warning_threshold:
                return HealthStatus.DEGRADED, "磁盘空间紧张", metrics
            else:
                return HealthStatus.HEALTHY, None, metrics
        except Exception as e:
            return HealthStatus.UNKNOWN, str(e), None


class HealthChecker:
    """健康检查管理器

    提供统一的健康检查接口，支持 liveness/readiness 探针。

    使用示例::

        checker = HealthChecker()
        checker.add_check(DiskHealthCheck("/"))
        result = checker.check_readiness()
        print(result.to_json())
    """

    class CheckResult:
        """单个检查结果"""

        def __init__(
            self,
            name: str,
            status: HealthStatus,
            message: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            duration_ms: float = 0.0,
        ):
            self.name = name
            self.status = status
            self.message = message
            self.metadata = metadata
            self.duration_ms = duration_ms

        def to_dict(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "status": self.status.value,
                "message": self.message,
                "metadata": self.metadata,
                "duration_ms": round(self.duration_ms, 3),
            }

    class AggregatedResult:
        """聚合健康检查结果"""

        def __init__(
            self,
            overall_status: HealthStatus,
            checks: List["HealthChecker.CheckResult"],
            version: Optional[str] = None,
            uptime_seconds: Optional[float] = None,
        ):
            self.overall_status = overall_status
            self.checks = checks
            self.version = version
            self.uptime_seconds = uptime_seconds
            self.timestamp = time.time()

        def to_dict(self) -> Dict[str, Any]:
            return {
                "status": self.overall_status.value,
                "timestamp": self.timestamp,
                "version": self.version,
                "uptime_seconds": round(self.uptime_seconds, 2) if self.uptime_seconds else None,
                "checks": [c.to_dict() for c in self.checks],
            }

        def to_json(self) -> str:
            return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def __init__(
        self,
        version: Optional[str] = None,
        start_time: Optional[float] = None,
    ):
        self._checks: Dict[str, HealthCheck] = {}
        self._version = version
        self._start_time = start_time or time.time()
        self._lock = threading.RLock()

    @property
    def version(self) -> Optional[str]:
        return self._version

    @version.setter
    def version(self, value: Optional[str]):
        self._version = value

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def add_check(self, check: HealthCheck) -> None:
        with self._lock:
            self._checks[check.name] = check

    def remove_check(self, name: str) -> None:
        with self._lock:
            if name in self._checks:
                del self._checks[name]

    def check_liveness(self) -> AggregatedResult:
        """Liveness 探针：检测服务是否需要重启"""
        checks = []
        with self._lock:
            for name, check in list(self._checks.items()):
                start = time.time()
                try:
                    status, msg, metadata = check.check()
                except Exception as e:
                    status = HealthStatus.UNKNOWN
                    msg = str(e)
                    metadata = None
                duration = (time.time() - start) * 1000
                checks.append(
                    self.CheckResult(
                        name=name,
                        status=status,
                        message=msg,
                        metadata=metadata,
                        duration_ms=duration,
                    )
                )
        overall = self._compute_overall(checks)
        return self.AggregatedResult(
            overall_status=overall,
            checks=checks,
            version=self._version,
            uptime_seconds=self.uptime_seconds,
        )

    def check_readiness(self) -> AggregatedResult:
        """Readiness 探针：检测服务是否可接收请求"""
        return self.check_liveness()

    def _compute_overall(self, results: List[CheckResult]) -> HealthStatus:
        if not results:
            return HealthStatus.HEALTHY
        has_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in results)
        if has_unhealthy:
            return HealthStatus.UNHEALTHY
        has_degraded = any(r.status == HealthStatus.DEGRADED for r in results)
        if has_degraded:
            return HealthStatus.DEGRADED
        has_unknown = any(r.status == HealthStatus.UNKNOWN for r in results)
        if has_unknown:
            return HealthStatus.UNKNOWN
        return HealthStatus.HEALTHY


# ==================== 步骤83: Prometheus指标 ====================

class MetricType(Enum):
    """Prometheus 指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class PrometheusMetric:
    """Prometheus 指标基类"""

    def __init__(
        self,
        name: str,
        type: MetricType,
        help: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self._name = name
        self._type = type
        self._help = help
        self._labels = labels or {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> MetricType:
        return self._type

    @property
    def help(self) -> str:
        return self._help

    @abstractmethod
    def to_exposition(self) -> str:
        ...

    def _format_labels(self, additional: Optional[Dict[str, str]] = None) -> str:
        merged = {**self._labels, **(additional or {})}
        if not merged:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(merged.items())]
        return "{" + ",".join(pairs) + "}"


class PrometheusCounter(PrometheusMetric):
    """Counter 类型：只增不减的计数器"""

    def __init__(
        self,
        name: str,
        help: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, MetricType.COUNTER, help, labels)
        self._value = 0.0
        self._values: Dict[str, float] = {}

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        if amount < 0:
            raise ValueError("Counter 只能增加")
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                self._values[key] = self._values.get(key, 0.0) + amount
            else:
                self._value += amount

    def to_exposition(self) -> str:
        lines = [
            f"# HELP {self._name} {self._help}",
            f"# TYPE {self._name} counter",
        ]
        with self._lock:
            if self._value > 0 or not self._values:
                lines.append(f"{self._name} {self._value}")
            for label_key, val in self._values.items():
                label_dict = dict(label_key)
                labels_str = self._format_labels(label_dict)
                lines.append(f"{self._name}{labels_str} {val}")
        return "\n".join(lines) + "\n"


class PrometheusGauge(PrometheusMetric):
    """Gauge 类型：可增减的仪表盘"""

    def __init__(
        self,
        name: str,
        help: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, MetricType.GAUGE, help, labels)
        self._value = 0.0
        self._values: Dict[str, float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                self._values[key] = value
            else:
                self._value = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                self._values[key] = self._values.get(key, 0.0) + amount
            else:
                self._value += amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        self.inc(-amount, labels)

    def to_exposition(self) -> str:
        lines = [
            f"# HELP {self._name} {self._help}",
            f"# TYPE {self._name} gauge",
        ]
        with self._lock:
            if self._value != 0 or not self._values:
                lines.append(f"{self._name} {self._value}")
            for label_key, val in self._values.items():
                label_dict = dict(label_key)
                labels_str = self._format_labels(label_dict)
                lines.append(f"{self._name}{labels_str} {val}")
        return "\n".join(lines) + "\n"


class PrometheusRegistry:
    """Prometheus 指标注册表

    使用示例::

        registry = PrometheusRegistry()
        counter = PrometheusCounter("requests_total", "Total requests")
        registry.register(counter)
        counter.inc()
        exposition = registry.get_exposition()
    """

    def __init__(self):
        self._metrics: Dict[str, PrometheusMetric] = {}
        self._lock = threading.RLock()

    def register(self, metric: PrometheusMetric) -> None:
        """注册指标"""
        with self._lock:
            self._metrics[metric.name] = metric

    def unregister(self, name: str) -> None:
        """注销指标"""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]

    def get_metric(self, name: str) -> Optional[PrometheusMetric]:
        """获取指标"""
        with self._lock:
            return self._metrics.get(name)

    def get_exposition(self) -> str:
        """获取 Prometheus exposition 格式"""
        lines = []
        with self._lock:
            for metric in self._metrics.values():
                lines.append(metric.to_exposition())
        return "".join(lines)


# ==================== 步骤84: Docker 镜像支持 ====================

def generate_dockerfile_content(
    base_image: str = "python:3.12-slim",
    work_dir: str = "/app",
    extra_deps: Optional[List[str]] = None,
) -> str:
    """生成 Dockerfile 内容"""
    deps = extra_deps or []
    deps_str = "\n".join(f"RUN apt-get update && apt-get install -y --no-install-recommends {d} && rm -rf /var/lib/apt/lists/*" for d in deps) if deps else ""

    content = f"""
# AuroraNLP Dockerfile
FROM {base_image}

WORKDIR {work_dir}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    git \\
    && rm -rf /var/lib/apt/lists/*
{deps_str}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \\
    CMD python -c \"from AuroraNLP import HealthChecker; c = HealthChecker(); r = c.check_liveness(); import sys; sys.exit(0 if r.overall_status.value == 'healthy' else 1)\"

# Run
CMD [\"python\", \"-m\", \"AuroraNLP\"]
""".strip()
    return content


# ==================== 步骤85: Kubernetes 配置 ====================

def generate_k8s_deployment_content(
    name: str = "auroranlp",
    image: str = "auroranlp:latest",
    replicas: int = 3,
    port: int = 8000,
    cpu_request: str = "200m",
    cpu_limit: str = "500m",
    memory_request: str = "256Mi",
    memory_limit: str = "512Mi",
) -> str:
    """生成 Kubernetes Deployment 配置"""
    return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
        livenessProbe:
          httpGet:
            path: /health/live
            port: {port}
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
""".strip()


def generate_k8s_service_content(
    name: str = "auroranlp",
    port: int = 8000,
    target_port: int = 8000,
    service_type: str = "ClusterIP",
) -> str:
    """生成 Kubernetes Service 配置"""
    return f"""
apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  type: {service_type}
  selector:
    app: {name}
  ports:
  - protocol: TCP
    port: {port}
    targetPort: {target_port}
""".strip()


def generate_k8s_ingress_content(
    name: str = "auroranlp",
    host: str = "auroranlp.example.com",
    service_name: str = "auroranlp",
    service_port: int = 8000,
) -> str:
    """生成 Kubernetes Ingress 配置"""
    return f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  rules:
  - host: {host}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {service_name}
            port:
              number: {service_port}
""".strip()


# 更新导出列表
__all_enterprise__ = [
    # 步骤81: 日志系统
    "LogLevel",
    "LogFormat",
    "StructuredLogRecord",
    "LogHandler",
    "ConsoleLogHandler",
    "FileLogHandler",
    "TimeRotatingFileLogHandler",
    "MemoryLogHandler",
    "LogFilter",
    "LevelFilter",
    "KeywordFilter",
    "ModuleFilter",
    "Logger",
    "LogManager",
    "get_logger",
    # 步骤82: 健康检查接口
    "HealthStatus",
    "HealthCheck",
    "MemoryHealthCheck",
    "DiskHealthCheck",
    "HealthChecker",
    # 步骤83: Prometheus指标
    "MetricType",
    "PrometheusMetric",
    "PrometheusCounter",
    "PrometheusGauge",
    "PrometheusRegistry",
    # 步骤84: Docker镜像支持
    "generate_dockerfile_content",
    # 步骤85: Kubernetes配置
    "generate_k8s_deployment_content",
    "generate_k8s_service_content",
    "generate_k8s_ingress_content",
    # 步骤86: 限流熔断
    "CircuitState",
    "RateLimitState",
    "TokenBucket",
    "SlidingWindow",
    "CircuitBreaker",
    # 步骤87: 认证授权
    "TokenType",
    "Permission",
    "Token",
    "AuthContext",
    "Authenticator",
    "Authorizer",
    # 步骤88: 配置中心集成
    "ConfigEvent",
    "ConfigWatch",
    "InMemoryConfigStore",
    "FileConfigStore",
    "ConfigManager",
    # 步骤89: 灰度发布支持
    "DeploymentState",
    "TrafficRule",
    "DeploymentVersion",
    "CanaryDeployer",
    # 步骤90: 灾备方案
    "BackupType",
    "ClusterNode",
    "FailoverStrategy",
    "DataBackupManager",
    "FailoverController",
]


# ==================== 步骤86: 限流熔断 ====================

class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RateLimitState(Enum):
    """限流状态"""
    ALLOWED = "allowed"
    DROPPED = "dropped"
    QUEUED = "queued"


class TokenBucket:
    """令牌桶限流算法
    
    使用场景：平滑限流，允许一定突发流量
    
    使用示例:
    >>> bucket = TokenBucket(capacity=100, rate=10)
    >>> if bucket.acquire(5):
    ...     print("获取令牌成功")
    """
    
    def __init__(self, capacity: int, rate: float):
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """获取指定数量的令牌"""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self.lock:
            self._refill()
            return self.tokens


class SlidingWindow:
    """滑动窗口限流算法
    
    使用场景：精确限制指定时间窗口内的请求数
    
    使用示例:
    >>> window = SlidingWindow(window_seconds=60, max_requests=1000)
    >>> if window.allow():
    ...     print("请求通过")
    """
    
    def __init__(self, window_seconds: float, max_requests: int):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.timestamps = []
        self.lock = threading.Lock()
    
    def _cleanup(self, now: float):
        """清理过期的时间戳"""
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]
    
    def allow(self) -> bool:
        """检查是否允许通过"""
        now = time.time()
        with self.lock:
            self._cleanup(now)
            if len(self.timestamps) < self.max_requests:
                self.timestamps.append(now)
                return True
            return False
    
    def get_current_count(self) -> int:
        """获取当前窗口内的请求数"""
        now = time.time()
        with self.lock:
            self._cleanup(now)
            return len(self.timestamps)


class CircuitBreaker:
    """熔断器，实现自动熔断与恢复
    
    使用场景：保护系统免受故障服务的影响，快速失败
    
    使用示例:
    >>> breaker = CircuitBreaker(fail_threshold=5, reset_timeout=30)
    >>> try:
    ...     with breaker:
    ...         risky_operation()
    ... except CircuitOpenError:
    ...     print("熔断器已打开")
    """
    
    class CircuitOpenError(Exception):
        """熔断器打开时抛出的错误"""
        pass
    
    def __init__(
        self,
        fail_threshold: int = 5,
        reset_timeout: float = 30.0,
        success_threshold: int = 3,
    ):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """检查是否允许请求通过"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.HALF_OPEN:
                return True
            else:
                now = time.time()
                if now - self.last_failure_time >= self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.successes = 0
                    return True
                return False
    
    def on_success(self):
        """请求成功时调用"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.successes += 1
                if self.successes >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failures = 0
                    self.successes = 0
    
    def on_failure(self):
        """请求失败时调用"""
        with self.lock:
            self.last_failure_time = time.time()
            if self.state == CircuitState.CLOSED:
                self.failures += 1
                if self.failures >= self.fail_threshold:
                    self.state = CircuitState.OPEN
                    self.failures = 0
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
    
    def __enter__(self):
        if not self.allow_request():
            raise CircuitBreaker.CircuitOpenError("Circuit is open")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.on_success()
        else:
            self.on_failure()
        return False


# ==================== 步骤87: 认证授权 ====================

class TokenType(Enum):
    """令牌类型"""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"


class Permission(Enum):
    """权限枚举"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    DELETE = "delete"


class Token:
    """认证令牌"""
    
    def __init__(
        self,
        token_type: TokenType,
        value: str,
        user_id: Optional[str] = None,
        permissions: Optional[list[Permission]] = None,
        expires_at: Optional[float] = None,
    ):
        self.token_type = token_type
        self.value = value
        self.user_id = user_id
        self.permissions = permissions or []
        self.expires_at = expires_at
    
    def is_expired(self) -> bool:
        """检查令牌是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        """检查是否有指定权限"""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions


class AuthContext:
    """认证上下文，保存当前请求的认证信息"""
    
    _local = threading.local()
    
    @classmethod
    def set_token(cls, token: Token):
        """设置当前请求的令牌"""
        cls._local.token = token
    
    @classmethod
    def get_token(cls) -> Optional[Token]:
        """获取当前请求的令牌"""
        return getattr(cls._local, "token", None)
    
    @classmethod
    def clear(cls):
        """清除当前请求的认证信息"""
        if hasattr(cls._local, "token"):
            del cls._local.token


class Authenticator:
    """认证器，验证令牌有效性"""
    
    def __init__(self):
        self._tokens: dict[str, Token] = {}
        self._lock = threading.Lock()
    
    def register_token(self, token: Token):
        """注册一个有效令牌"""
        with self._lock:
            self._tokens[token.value] = token
    
    def revoke_token(self, token_value: str):
        """撤销令牌"""
        with self._lock:
            if token_value in self._tokens:
                del self._tokens[token_value]
    
    def authenticate(self, token_value: str) -> Optional[Token]:
        """验证令牌，返回Token对象或None"""
        with self._lock:
            token = self._tokens.get(token_value)
            if token and not token.is_expired():
                return token
            return None


class Authorizer:
    """授权器，检查权限"""
    
    def __init__(self, authenticator: Optional[Authenticator] = None):
        self.authenticator = authenticator
    
    def has_permission(self, token: Token, permission: Permission) -> bool:
        """检查令牌是否有指定权限"""
        return token.has_permission(permission)
    
    def authorize(self, token_value: str, permission: Permission) -> bool:
        """验证并授权"""
        if not self.authenticator:
            return False
        token = self.authenticator.authenticate(token_value)
        if not token:
            return False
        return self.has_permission(token, permission)


# ==================== 步骤88: 配置中心集成 ====================

class ConfigEvent(Enum):
    """配置变更事件类型"""
    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"


class ConfigWatch:
    """配置变更监听器"""
    
    def __init__(self):
        self._callbacks: list[tuple[str, Callable]] = []
        self._lock = threading.Lock()
    
    def watch(self, key: str, callback: Callable):
        """监听特定配置键的变更"""
        with self._lock:
            self._callbacks.append((key, callback))
    
    def notify(self, key: str, event: ConfigEvent, value: Any):
        """通知所有监听器"""
        with self._lock:
            for watch_key, callback in self._callbacks:
                if watch_key == key or watch_key == "*":
                    try:
                        callback(event, key, value)
                    except Exception:
                        pass


class InMemoryConfigStore:
    """内存配置存储"""
    
    def __init__(self):
        self._configs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.watch = ConfigWatch()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        with self._lock:
            return self._configs.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置"""
        with self._lock:
            exists = key in self._configs
            self._configs[key] = value
            event = ConfigEvent.UPDATED if exists else ConfigEvent.ADDED
            self.watch.notify(key, event, value)
    
    def delete(self, key: str):
        """删除配置"""
        with self._lock:
            if key in self._configs:
                value = self._configs.pop(key)
                self.watch.notify(key, ConfigEvent.DELETED, value)
    
    def get_all(self) -> dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return self._configs.copy()


class FileConfigStore:
    """文件配置存储，支持自动热更新"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._configs: dict[str, Any] = {}
        self._last_modified: float = 0.0
        self._lock = threading.Lock()
        self.watch = ConfigWatch()
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.file_path):
                stat = os.stat(self.file_path)
                self._last_modified = stat.st_mtime
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._configs = json.load(f)
        except Exception:
            self._configs = {}
    
    def _check_and_reload(self) -> bool:
        """检查文件变化并重新加载"""
        try:
            if os.path.exists(self.file_path):
                stat = os.stat(self.file_path)
                if stat.st_mtime > self._last_modified:
                    old_configs = self._configs.copy()
                    self._load_config()
                    for key, value in self._configs.items():
                        if key not in old_configs or old_configs[key] != value:
                            event = ConfigEvent.UPDATED if key in old_configs else ConfigEvent.ADDED
                            self.watch.notify(key, event, value)
                    for key in old_configs:
                        if key not in self._configs:
                            self.watch.notify(key, ConfigEvent.DELETED, old_configs[key])
                    return True
            return False
        except Exception:
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置，自动检查更新"""
        with self._lock:
            self._check_and_reload()
            return self._configs.get(key, default)
    
    def get_all(self) -> dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            self._check_and_reload()
            return self._configs.copy()


class ConfigManager:
    """配置管理器，统一管理配置"""
    
    def __init__(self):
        self._stores: dict[str, Any] = {}
        self._default_store: Optional[str] = None
        self._lock = threading.Lock()
    
    def add_store(self, name: str, store, is_default: bool = False):
        """添加配置存储"""
        with self._lock:
            self._stores[name] = store
            if is_default or self._default_store is None:
                self._default_store = name
    
    def get(self, key: str, default: Any = None, store_name: Optional[str] = None) -> Any:
        """获取配置"""
        store = self._get_store(store_name)
        return store.get(key, default) if store else default
    
    def set(self, key: str, value: Any, store_name: Optional[str] = None):
        """设置配置"""
        store = self._get_store(store_name)
        if store and hasattr(store, "set"):
            store.set(key, value)
    
    def _get_store(self, name: Optional[str]):
        """获取配置存储"""
        with self._lock:
            target = name or self._default_store
            return self._stores.get(target)


# ==================== 步骤89: 灰度发布支持 ====================

class DeploymentState(Enum):
    """部署状态"""
    DEPLOYING = "deploying"
    ACTIVE = "active"
    STAGING = "staging"
    ROLLED_BACK = "rolled_back"


class TrafficRule:
    """流量规则，定义如何分配流量"""
    
    def __init__(self, percentage: float, key: Optional[str] = None):
        self.percentage = percentage
        self.key = key or "random"


class DeploymentVersion:
    """部署版本信息"""
    
    def __init__(
        self,
        version_id: str,
        state: DeploymentState,
        traffic_rule: Optional[TrafficRule] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.version_id = version_id
        self.state = state
        self.traffic_rule = traffic_rule or TrafficRule(0.0)
        self.metadata = metadata or {}
        self.deployed_at = time.time()


class CanaryDeployer:
    """灰度发布管理器
    
    使用场景：逐步将流量切换到新版本，支持快速回滚
    
    使用示例:
    >>> deployer = CanaryDeployer()
    >>> deployer.deploy_canary("v2", percentage=10)
    >>> deployer.rollback("v2")
    """
    
    def __init__(self):
        self._versions: dict[str, DeploymentVersion] = {}
        self._current_stable: Optional[str] = None
        self._lock = threading.Lock()
    
    def deploy_version(self, version_id: str, metadata: Optional[dict[str, Any]] = None):
        """部署新版本作为稳定版本"""
        with self._lock:
            version = DeploymentVersion(
                version_id=version_id,
                state=DeploymentState.ACTIVE,
                traffic_rule=TrafficRule(100.0),
                metadata=metadata,
            )
            self._versions[version_id] = version
            self._current_stable = version_id
            for vid, v in self._versions.items():
                if vid != version_id:
                    v.state = DeploymentState.STAGING
    
    def deploy_canary(self, version_id: str, percentage: float, metadata: Optional[dict[str, Any]] = None):
        """部署灰度版本，分配指定百分比的流量"""
        with self._lock:
            version = DeploymentVersion(
                version_id=version_id,
                state=DeploymentState.STAGING,
                traffic_rule=TrafficRule(percentage),
                metadata=metadata,
            )
            self._versions[version_id] = version
            if self._current_stable:
                stable_version = self._versions[self._current_stable]
                stable_version.traffic_rule.percentage = max(0.0, 100.0 - percentage)
    
    def update_traffic(self, version_id: str, percentage: float):
        """更新版本的流量分配"""
        with self._lock:
            if version_id in self._versions:
                self._versions[version_id].traffic_rule.percentage = percentage
    
    def promote_to_stable(self, version_id: str):
        """将版本晋升为稳定版本，接管100%流量"""
        with self._lock:
            if version_id in self._versions:
                self.deploy_version(version_id, self._versions[version_id].metadata)
    
    def rollback(self, version_id: str):
        """回滚指定版本"""
        with self._lock:
            if version_id in self._versions:
                self._versions[version_id].state = DeploymentState.ROLLED_BACK
                self._versions[version_id].traffic_rule.percentage = 0.0
                if self._current_stable:
                    self._versions[self._current_stable].traffic_rule.percentage = 100.0
    
    def select_version(self, request_key: Optional[str] = None) -> Optional[str]:
        """根据规则选择处理请求的版本"""
        with self._lock:
            rand_val = hash(request_key or str(time.time())) % 10000 / 100.0
            cumulative = 0.0
            for vid, version in self._versions.items():
                if version.state in [DeploymentState.ACTIVE, DeploymentState.STAGING]:
                    cumulative += version.traffic_rule.percentage
                    if rand_val < cumulative:
                        return vid
            return self._current_stable


# ==================== 步骤90: 灾备方案 ====================

class BackupType(Enum):
    """备份类型"""
    INCREMENTAL = "incremental"
    FULL = "full"
    SNAPSHOT = "snapshot"


class ClusterNode:
    """集群节点信息"""
    
    def __init__(self, node_id: str, is_master: bool = False):
        self.node_id = node_id
        self.is_master = is_master
        self.is_alive = True
        self.last_heartbeat = time.time()
    
    def heartbeat(self):
        """节点心跳"""
        self.is_alive = True
        self.last_heartbeat = time.time()


class FailoverStrategy(Enum):
    """故障转移策略"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    NONE = "none"


class DataBackupManager:
    """数据备份管理器
    
    使用场景：定期备份数据，支持完整备份和增量备份
    
    使用示例:
    >>> manager = DataBackupManager("/backups")
    >>> manager.create_backup("data/1", BackupType.FULL)
    >>> manager.restore("backup_123456")
    """
    
    def __init__(self, backup_dir: str):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self._backups: list[dict[str, Any]] = []
    
    def create_backup(self, source_path: str, backup_type: BackupType, name: Optional[str] = None) -> str:
        """创建备份"""
        backup_name = name or f"backup_{int(time.time())}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        backup_info = {
            "name": backup_name,
            "type": backup_type.value,
            "timestamp": time.time(),
            "source": source_path,
            "path": backup_path,
        }
        try:
            import shutil
            if os.path.exists(source_path):
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, backup_path)
                else:
                    shutil.copy2(source_path, backup_path)
        except Exception:
            pass
        self._backups.append(backup_info)
        return backup_name
    
    def list_backups(self) -> list[dict[str, Any]]:
        """列出所有备份"""
        return self._backups.copy()
    
    def restore(self, backup_name: str, target_path: str) -> bool:
        """恢复备份"""
        try:
            import shutil
            backup_info = next((b for b in self._backups if b["name"] == backup_name), None)
            if backup_info and os.path.exists(backup_info["path"]):
                if os.path.isdir(backup_info["path"]):
                    shutil.copytree(backup_info["path"], target_path)
                else:
                    shutil.copy2(backup_info["path"], target_path)
                return True
        except Exception:
            pass
        return False


class FailoverController:
    """故障转移控制器
    
    使用场景：实现高可用架构，自动故障转移
    
    使用示例:
    >>> controller = FailoverController()
    >>> controller.register_node(ClusterNode("node1", is_master=True))
    >>> controller.register_node(ClusterNode("node2"))
    >>> controller.failover()
    """
    
    def __init__(self, strategy: FailoverStrategy = FailoverStrategy.AUTOMATIC):
        self.strategy = strategy
        self._nodes: dict[str, ClusterNode] = {}
        self._current_master: Optional[str] = None
        self._lock = threading.Lock()
    
    def register_node(self, node: ClusterNode):
        """注册集群节点"""
        with self._lock:
            self._nodes[node.node_id] = node
            if node.is_master and self._current_master is None:
                self._current_master = node.node_id
    
    def get_nodes(self) -> list[ClusterNode]:
        """获取所有节点"""
        with self._lock:
            return list(self._nodes.values())
    
    def get_current_master(self) -> Optional[ClusterNode]:
        """获取当前主节点"""
        with self._lock:
            return self._nodes.get(self._current_master)
    
    def heartbeat(self, node_id: str):
        """节点心跳"""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].heartbeat()
    
    def failover(self) -> Optional[ClusterNode]:
        """执行故障转移，选举新的主节点"""
        with self._lock:
            if self.strategy == FailoverStrategy.NONE:
                return None
            alive_slaves = [
                node for node in self._nodes.values()
                if node.is_alive and not node.is_master
            ]
            if alive_slaves:
                if self._current_master and self._current_master in self._nodes:
                    self._nodes[self._current_master].is_master = False
                new_master = alive_slaves[0]
                new_master.is_master = True
                self._current_master = new_master.node_id
                return new_master
        return None


