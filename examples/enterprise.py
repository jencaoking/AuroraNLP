#!/usr/bin/env python3
"""
企业级功能示例
演示 AuroraNLP 的企业级功能
"""


def logging_example():
    """日志示例"""
    print("=" * 60)
    print("1. 日志系统")
    print("=" * 60)
    
    from AuroraNLP import LogManager, LogLevel, LogFormat
    from AuroraNLP import ConsoleLogHandler, FileLogHandler
    
    logger = LogManager.get_logger("enterprise_example")
    logger.set_level(LogLevel.INFO)
    logger.set_format(LogFormat.TEXT)
    
    # 添加处理器
    logger.add_handler(ConsoleLogHandler())
    
    # 记录日志
    logger.debug("这是一条 debug 日志")
    logger.info("这是一条 info 日志")
    logger.warning("这是一条 warning 日志")
    logger.error("这是一条 error 日志")
    
    # 结构化日志
    logger.info("请求处理完成", extra={
        "text_length": 100,
        "processing_time": 0.001,
        "success": True
    })
    print()


def health_check_example():
    """健康检查示例"""
    print("=" * 60)
    print("2. 健康检查")
    print("=" * 60)
    
    from AuroraNLP import HealthChecker, MemoryHealthCheck, DiskHealthCheck
    
    checker = HealthChecker()
    checker.add_check(MemoryHealthCheck(max_usage_percent=90))
    checker.add_check(DiskHealthCheck(path="/", max_usage_percent=95))
    
    status = checker.check()
    print(f"整体状态: {status.status}")
    print(f"状态详情:")
    for check in status.checks:
        print(f"  {check.name}: {check.status} - {check.message}")
    print(f"JSON 格式:\n{status.to_json(indent=2)}")
    print()


def prometheus_example():
    """Prometheus 指标示例"""
    print("=" * 60)
    print("3. Prometheus 指标")
    print("=" * 60)
    
    from AuroraNLP import PrometheusRegistry, PrometheusCounter, PrometheusGauge
    
    registry = PrometheusRegistry()
    
    # Counter
    requests_counter = PrometheusCounter(
        "requests_total",
        "总请求数",
        labels=["method", "endpoint"]
    )
    registry.register(requests_counter)
    
    requests_counter.inc(labels={"method": "GET", "endpoint": "/api"})
    requests_counter.inc(labels={"method": "GET", "endpoint": "/api"})
    requests_counter.inc(labels={"method": "POST", "endpoint": "/api"})
    
    # Gauge
    active_users = PrometheusGauge(
        "active_users",
        "活跃用户数"
    )
    registry.register(active_users)
    
    active_users.set(100)
    active_users.inc()
    active_users.dec(20)
    
    # 导出指标
    metrics = registry.export()
    print("Prometheus 指标:")
    print(metrics)
    print()


def rate_limit_example():
    """限流熔断示例"""
    print("=" * 60)
    print("4. 限流和熔断")
    print("=" * 60)
    
    from AuroraNLP import TokenBucket, CircuitBreaker
    
    # 令牌桶限流
    bucket = TokenBucket(capacity=10, rate=2)
    
    print("令牌桶限流:")
    for i in range(15):
        success = bucket.try_consume()
        print(f"  请求 {i+1}: {'通过' if success else '被限流'}")
    print()
    
    # 熔断器
    print("熔断器:")
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5,
        half_open_max_calls=2
    )
    
    print(f"初始状态: {breaker.state}")
    
    # 模拟失败
    for i in range(4):
        try:
            with breaker:
                raise Exception(f"错误 {i+1}")
        except Exception as e:
            print(f"  请求 {i+1}: 失败 - 熔断器状态: {breaker.state}")
    
    print()


def config_example():
    """配置管理示例"""
    print("=" * 60)
    print("5. 配置管理")
    print("=" * 60)
    
    from AuroraNLP import ConfigManager, InMemoryConfigStore
    
    config_store = InMemoryConfigStore()
    manager = ConfigManager(config_store)
    
    # 设置配置
    manager.set("database.host", "localhost")
    manager.set("database.port", 5432)
    manager.set("api.timeout", 30)
    
    # 获取配置
    host = manager.get("database.host")
    port = manager.get("database.port")
    timeout = manager.get("api.timeout")
    print(f"database.host: {host}")
    print(f"database.port: {port}")
    print(f"api.timeout: {timeout}")
    
    # 获取所有配置
    all_config = manager.get_all()
    print(f"所有配置: {all_config}")
    print()


def docker_k8s_example():
    """Docker/K8s 示例"""
    print("=" * 60)
    print("6. Docker/K8s 配置生成")
    print("=" * 60)
    
    from AuroraNLP import (
        generate_dockerfile_content,
        generate_k8s_deployment_content,
        generate_k8s_service_content,
        generate_k8s_ingress_content
    )
    
    # Dockerfile
    dockerfile = generate_dockerfile_content(
        base_image="python:3.10-slim",
        expose_ports=[8000],
        healthcheck=True
    )
    print("Dockerfile:")
    print(dockerfile)
    print("\n" + "=" * 60 + "\n")
    
    # K8s Deployment
    deployment = generate_k8s_deployment_content(
        name="aurora-nlp",
        image="aurora-nlp:latest",
        replicas=3,
        ports=[8000],
        liveness_probe=True,
        readiness_probe=True
    )
    print("K8s Deployment:")
    print(deployment)
    print()


def main():
    print("\n")
    logging_example()
    health_check_example()
    prometheus_example()
    rate_limit_example()
    config_example()
    docker_k8s_example()
    print("\n")


if __name__ == "__main__":
    main()
