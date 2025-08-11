# Monitoring and Observability Guide

This guide covers the comprehensive monitoring and observability system for Graph Hypernetwork Forge, including health checks, metrics collection, alerting, and dashboard visualization.

## Overview

The monitoring system provides:

- **Health Checks**: Model, memory, GPU, and dependency health monitoring
- **Metrics Collection**: Performance, resource utilization, and accuracy metrics
- **Alerting System**: Configurable alerts with email, Slack, and webhook notifications
- **Observability Dashboard**: Real-time monitoring with web and console interfaces
- **Monitoring Server**: HTTP endpoints for integration with external monitoring systems

## Quick Start

### Basic Setup

```python
from graph_hypernetwork_forge.utils import (
    setup_default_health_checks,
    setup_metrics_collection,
    setup_monitoring_dashboard
)

# Setup monitoring components
health_registry = setup_default_health_checks()
metrics_aggregator = setup_metrics_collection("/tmp/metrics.db")
dashboard = setup_monitoring_dashboard(dashboard_type="web", port=8080)

# Check system health
health_summary = health_registry.get_health_summary()
print(f"System health: {health_summary['overall_status']}")

# Start dashboard
dashboard.run()
```

### Using the Startup Script

```bash
# Start basic monitoring
python scripts/start_monitoring.py --mode basic

# Start web dashboard only
python scripts/start_monitoring.py --mode dashboard --port 8080

# Start monitoring server only  
python scripts/start_monitoring.py --mode server --port 8000

# Start console dashboard
python scripts/start_monitoring.py --mode console

# Start production monitoring
python scripts/start_monitoring.py --mode production --config configs/monitoring
```

## Health Checks

### Available Health Checks

1. **Model Health Check**: Validates model loading and inference capability
2. **Memory Health Check**: Monitors system and GPU memory usage
3. **GPU Health Check**: Verifies GPU availability and functionality
4. **Data Pipeline Health Check**: Tests data loading and processing
5. **Dependencies Health Check**: Validates required packages and versions
6. **External Services Health Check**: Monitors external API dependencies

### Custom Health Checks

```python
from graph_hypernetwork_forge.utils.health_checks import BaseHealthCheck, HealthStatus

class CustomHealthCheck(BaseHealthCheck):
    def __init__(self):
        super().__init__("custom_check", timeout=10.0, critical=True)
        
    def _execute(self):
        # Your health check logic here
        if some_condition:
            return HealthStatus.HEALTHY, "All good", {"detail": "value"}
        else:
            return HealthStatus.UNHEALTHY, "Problem detected", {"error": "details"}

# Register custom health check
health_registry = get_health_registry()
health_registry.register(CustomHealthCheck())
```

### Health Check Endpoints

When using the monitoring server, health checks are available via HTTP:

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health information
- `GET /health/ready` - Readiness probe (for Kubernetes)
- `GET /health/live` - Liveness probe (for Kubernetes)

## Metrics Collection

### Automatic Metrics

The system automatically collects:

- **System Resources**: CPU, memory, disk usage
- **GPU Metrics**: Memory usage, utilization
- **Model Performance**: Inference latency, throughput
- **Training Progress**: Loss, accuracy, epoch times

### Manual Metrics

```python
from graph_hypernetwork_forge.utils import get_metrics_aggregator

metrics_aggregator = get_metrics_aggregator()

# Record training metrics
model_collector = metrics_aggregator.model_collector
model_collector.record_training_metric("loss", 0.5, epoch=10, step=1000)
model_collector.record_validation_metric("accuracy", 0.92, epoch=10)

# Record performance metrics
perf_collector = metrics_aggregator.performance_collector
perf_collector.record_latency("inference", 150.0)  # ms
perf_collector.record_throughput("batch_processing", 32)
```

### Metrics Storage

Metrics are stored in SQLite by default but can be configured for other backends:

```python
# SQLite (default)
setup_metrics_collection("/path/to/metrics.db")

# In-memory (for testing)
setup_metrics_collection(":memory:")
```

## Alerting System

### Configuration

Create `configs/monitoring/alerting.yml`:

```yaml
rules:
  - name: "high_memory_usage"
    condition: "memory_usage_percent"
    severity: "warning"
    threshold: 85.0
    comparison: ">"
    duration_minutes: 5
    enabled: true
    description: "System memory usage is high"
    
channels:
  - name: "email_alerts"
    type: "email"
    enabled: true
    severities: ["critical", "warning"]
    config:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "alerts@company.com"
      password: "app-password"
      from_email: "alerts@company.com"
      recipients: ["team@company.com"]
      
  - name: "slack_alerts"
    type: "slack"
    enabled: true
    severities: ["critical", "warning"]
    config:
      webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
      channel: "#ml-alerts"
```

### Programmatic Alerting

```python
from graph_hypernetwork_forge.utils.alerting import (
    AlertRule, AlertSeverity, get_alert_manager
)

# Create alert rule
rule = AlertRule(
    name="custom_metric_alert",
    condition="custom_metric",
    severity=AlertSeverity.WARNING,
    threshold=100.0,
    comparison=">",
    duration_minutes=5
)

# Add to alert manager
alert_manager = get_alert_manager()
alert_manager.add_rule(rule)

# Evaluate metrics against rules
alerts = alert_manager.evaluate_metric("custom_metric", 150.0)
```

### Alert Types

- **Threshold Alerts**: Triggered when metrics cross thresholds
- **Health-based Alerts**: Based on health check results
- **Trend Alerts**: Detect patterns in metric changes
- **Composite Alerts**: Combine multiple conditions

## Dashboard and Visualization

### Web Dashboard

The web dashboard provides real-time monitoring with:

- System overview with key metrics
- Resource utilization charts
- Training progress visualization
- Performance metrics tracking
- Alert timeline and logs

```python
from graph_hypernetwork_forge.utils import setup_monitoring_dashboard

dashboard = setup_monitoring_dashboard(
    dashboard_type="web",
    host="0.0.0.0",
    port=8080
)

dashboard.run()
```

### Console Dashboard

For environments without web access:

```python
dashboard = setup_monitoring_dashboard(dashboard_type="console")
dashboard.run()  # Updates every 5 seconds
```

### Dashboard Configuration

Create `configs/monitoring/dashboard.yml`:

```yaml
dashboard:
  type: "web"
  web:
    host: "0.0.0.0"
    port: 8080
    auto_refresh_interval: 5

charts:
  resource_utilization:
    enabled: true
    time_range_hours: 1
    refresh_interval: 10
    
  performance:
    enabled: true
    metrics:
      - "inference_latency_ms"
      - "inference_throughput"
      - "model_accuracy"
```

## Monitoring Server

### HTTP Endpoints

The monitoring server provides REST API endpoints:

| Endpoint | Description |
|----------|-------------|
| `/health` | Basic health status |
| `/health/detailed` | Comprehensive health info |
| `/metrics` | Prometheus-compatible metrics |
| `/metrics/<name>` | Specific metric timeseries |
| `/alerts` | Current firing alerts |
| `/dashboard/data` | Complete dashboard data |
| `/training/status` | Training progress |
| `/system/info` | System information |

### Starting the Server

```python
from graph_hypernetwork_forge.utils import create_monitoring_server

server = create_monitoring_server(host="0.0.0.0", port=8000)
server.start()
```

Or use the command line:

```bash
python scripts/start_monitoring.py --mode server --port 8000
```

## Integration Examples

### Training Loop Integration

```python
from graph_hypernetwork_forge.utils import (
    get_metrics_aggregator, get_health_registry
)

def train_model_with_monitoring():
    # Setup monitoring
    metrics_aggregator = get_metrics_aggregator()
    health_registry = get_health_registry()
    
    model_collector = metrics_aggregator.model_collector
    
    for epoch in range(num_epochs):
        # Training step
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)
        
        # Record metrics
        model_collector.record_training_metric("loss", train_loss, epoch, step)
        model_collector.record_validation_metric("loss", val_loss, epoch)
        model_collector.record_validation_metric("accuracy", val_acc, epoch)
        
        # Periodic health checks
        if epoch % 10 == 0:
            health_summary = health_registry.get_health_summary()
            if health_summary['overall_status'] != 'healthy':
                logger.warning(f"Health issue detected: {health_summary}")
```

### Inference Server Integration

```python
def inference_with_monitoring(model, input_data):
    metrics_aggregator = get_metrics_aggregator()
    perf_collector = metrics_aggregator.performance_collector
    
    # Time inference
    start_time = time.time()
    result = model(input_data)
    latency_ms = (time.time() - start_time) * 1000
    
    # Record metrics
    perf_collector.record_latency("inference", latency_ms)
    perf_collector.record_throughput("inference", len(input_data))
    
    return result
```

## Production Deployment

### Docker Integration

```dockerfile
FROM python:3.9

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose monitoring ports
EXPOSE 8000 8080

# Start monitoring
CMD ["python", "scripts/start_monitoring.py", "--mode", "production", "--config", "configs/monitoring"]
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ghf-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ghf-monitoring
  template:
    metadata:
      labels:
        app: ghf-monitoring
    spec:
      containers:
      - name: monitoring
        image: ghf:latest
        ports:
        - containerPort: 8000
          name: monitoring
        - containerPort: 8080
          name: dashboard
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ghf-monitoring-service
spec:
  selector:
    app: ghf-monitoring
  ports:
  - name: monitoring
    port: 8000
    targetPort: 8000
  - name: dashboard
    port: 8080
    targetPort: 8080
```

### Environment Variables

Configure monitoring through environment variables:

```bash
# Metrics storage
export GHF_METRICS_DB_PATH="/data/metrics.db"

# Alert configuration
export GHF_ALERTING_CONFIG="/config/alerting.yml"

# Server settings
export GHF_MONITORING_HOST="0.0.0.0"
export GHF_MONITORING_PORT="8000"

# Dashboard settings
export GHF_DASHBOARD_HOST="0.0.0.0"
export GHF_DASHBOARD_PORT="8080"
```

## Configuration Reference

### Health Check Configuration (`health-checks.yml`)

```yaml
health_checks:
  api:
    endpoint: "/health"
    timeout: 5
    interval: 30
    retries: 3
    
  model_ready:
    endpoint: "/health/model"
    timeout: 10
    interval: 60

custom_checks:
  gpu_availability:
    command: "python -c 'import torch; print(torch.cuda.is_available())'"
    expected_output: "True"
    timeout: 10
```

### Alerting Configuration (`alerting.yml`)

```yaml
rules:
  - name: "alert_name"
    condition: "metric_name"
    severity: "warning|critical|info"
    threshold: 85.0
    comparison: ">|<|>=|<=|==|!="
    duration_minutes: 5
    enabled: true
    description: "Alert description"
    tags:
      service: "ghf"
      
channels:
  - name: "channel_name"
    type: "email|slack|webhook|console"
    enabled: true
    severities: ["critical", "warning"]
    config:
      # Channel-specific configuration
```

### Dashboard Configuration (`dashboard.yml`)

```yaml
dashboard:
  type: "web|console"
  web:
    host: "0.0.0.0"
    port: 8080
    debug: false
    
data:
  metrics_storage:
    type: "sqlite"
    path: "/path/to/metrics.db"
    retention_hours: 168
    
charts:
  resource_utilization:
    enabled: true
    time_range_hours: 1
    metrics:
      - "cpu_usage_percent"
      - "memory_usage_percent"
```

## Troubleshooting

### Common Issues

1. **Dashboard not accessible**: Check firewall settings and port binding
2. **Metrics not collected**: Verify database permissions and disk space
3. **Alerts not sent**: Check notification channel configuration
4. **High resource usage**: Adjust collection intervals and retention policies

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check component status
health_summary = health_registry.get_health_summary()
print(f"Health checks: {health_summary}")

# Verify metrics collection
current_metrics = metrics_aggregator.resource_collector.get_current_resource_metrics()
print(f"Current metrics: {current_metrics}")

# Test alerting
alert_summary = alert_manager.get_alert_summary()
print(f"Alerts: {alert_summary}")
```

### Performance Optimization

- **Reduce collection frequency** for non-critical metrics
- **Limit retention period** for historical data
- **Use external storage** for large-scale deployments
- **Batch metric writes** to reduce I/O overhead

## Best Practices

1. **Start Simple**: Begin with basic monitoring and add complexity gradually
2. **Monitor What Matters**: Focus on metrics that impact your specific use case
3. **Set Appropriate Thresholds**: Avoid alert fatigue with well-tuned thresholds
4. **Test Alerting**: Regularly test notification channels
5. **Regular Review**: Periodically review and update monitoring configuration
6. **Documentation**: Document custom health checks and alert procedures
7. **Security**: Secure monitoring endpoints and credentials in production

## API Reference

For detailed API documentation of monitoring components, see:

- [Health Checks API](../source/api_reference.rst#health-checks)
- [Metrics Collection API](../source/api_reference.rst#metrics-collection)
- [Alerting API](../source/api_reference.rst#alerting)
- [Dashboard API](../source/api_reference.rst#dashboard)

## Contributing

To contribute to the monitoring system:

1. Follow existing patterns for new health checks
2. Add comprehensive tests for new components
3. Update configuration schemas
4. Document new features and examples
5. Consider backward compatibility

For more information, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Key Metrics

### Application Metrics

- **Model Performance**
  - Inference latency (p50, p95, p99)
  - Throughput (requests/second)
  - Memory usage during operations
  - GPU utilization (if applicable)
  - Model accuracy metrics

- **System Resource Usage**
  - CPU utilization
  - Memory consumption
  - Disk I/O
  - Network traffic
  - Container resource limits

### Business Metrics

- **Usage Patterns**
  - Graph sizes processed
  - Node/edge counts
  - Text encoding frequency
  - Zero-shot inference requests

- **Quality Metrics**
  - Prediction confidence scores
  - Error rates by operation type
  - User satisfaction indicators

## Monitoring Stack

### Core Components

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.enable-lifecycle'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - monitoring

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring:
    driver: bridge
```

### Application Instrumentation

```python
# monitoring/instrumentation.py
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Prometheus metrics
REQUEST_COUNT = Counter('hypergnn_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('hypergnn_request_duration_seconds', 'Request latency')
MODEL_INFERENCE_TIME = Histogram('hypergnn_inference_duration_seconds', 'Model inference time')
GPU_MEMORY_USAGE = Gauge('hypergnn_gpu_memory_bytes', 'GPU memory usage')
CPU_USAGE = Gauge('hypergnn_cpu_usage_percent', 'CPU usage percentage')

class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        start_http_server(port)
        
    def record_request(self, method: str, endpoint: str, duration: float):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_LATENCY.observe(duration)
        
    def record_inference(self, duration: float, graph_size: int):
        """Record model inference metrics."""
        MODEL_INFERENCE_TIME.observe(duration)
        
    def update_system_metrics(self):
        """Update system resource metrics."""
        CPU_USAGE.set(psutil.cpu_percent())
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            GPU_MEMORY_USAGE.set(gpu_memory)

# OpenTelemetry tracing setup
def setup_tracing(service_name: str = "hypergnn-forge"):
    """Setup distributed tracing."""
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=14268,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument requests
    RequestsInstrumentor().instrument()
    
    return tracer
```

## Configuration Files

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'hypergnn-app'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['localhost:8080']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
# monitoring/grafana/dashboards/hypergnn-dashboard.json
{
  "dashboard": {
    "id": null,
    "title": "Graph Hypernetwork Forge Metrics",
    "tags": ["hypergnn", "ml", "performance"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(hypergnn_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(hypergnn_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(hypergnn_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "singlestat",
        "targets": [
          {
            "expr": "hypergnn_gpu_memory_bytes / 1024 / 1024 / 1024",
            "legendFormat": "GPU Memory (GB)"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

## Alerting Rules

### Prometheus Alerting

```yaml
# monitoring/rules/alerts.yml
groups:
  - name: hypergnn.rules
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(hypergnn_inference_duration_seconds_bucket[5m])) > 2.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile inference latency is {{ $value }}s"
          
      - alert: HighErrorRate
        expr: rate(hypergnn_requests_total{status=~"5.."}[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
          
      - alert: GPUMemoryHigh
        expr: hypergnn_gpu_memory_bytes / hypergnn_gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
          
      - alert: ModelServiceDown
        expr: up{job="hypergnn-app"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Model service is down"
          description: "The HyperGNN model service has been down for more than 30 seconds"
```

## Health Checks

### Application Health Endpoint

```python
# monitoring/health.py
from flask import Flask, jsonify
import torch
import psutil
import time

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '0.1.0'
    })

@app.route('/health/deep')
def deep_health_check():
    """Comprehensive health check."""
    checks = {
        'pytorch': check_pytorch(),
        'memory': check_memory(),
        'gpu': check_gpu(),
        'dependencies': check_dependencies()
    }
    
    overall_status = 'healthy' if all(check['status'] == 'ok' for check in checks.values()) else 'unhealthy'
    
    return jsonify({
        'status': overall_status,
        'checks': checks,
        'timestamp': time.time()
    })

def check_pytorch():
    """Check PyTorch availability."""
    try:
        torch.randn(2, 2)
        return {'status': 'ok', 'version': torch.__version__}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def check_memory():
    """Check memory usage."""
    memory_info = psutil.virtual_memory()
    if memory_info.percent > 90:
        return {'status': 'warning', 'usage_percent': memory_info.percent}
    return {'status': 'ok', 'usage_percent': memory_info.percent}

def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        return {'status': 'ok', 'message': 'GPU not required'}
    
    try:
        memory_allocated = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        usage_percent = (memory_allocated / memory_total) * 100
        
        if usage_percent > 90:
            return {'status': 'warning', 'gpu_memory_percent': usage_percent}
        return {'status': 'ok', 'gpu_memory_percent': usage_percent}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def check_dependencies():
    """Check critical dependencies."""
    try:
        import transformers
        import torch_geometric
        import sentence_transformers
        return {'status': 'ok', 'message': 'All dependencies available'}
    except ImportError as e:
        return {'status': 'error', 'message': f'Missing dependency: {e}'}
```

## Logging Configuration

### Structured Logging

```python
# monitoring/logging_config.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service information
        log_record['service'] = 'hypergnn-forge'
        log_record['version'] = '0.1.0'
        
        # Add trace information if available
        if hasattr(record, 'trace_id'):
            log_record['trace_id'] = record.trace_id
        if hasattr(record, 'span_id'):
            log_record['span_id'] = record.span_id

def setup_logging(log_level: str = 'INFO'):
    """Setup structured logging configuration."""
    
    # Create formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    # Setup handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[console_handler]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
```

## Deployment Integration

### Docker Compose Override

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  app:
    environment:
      - PROMETHEUS_ENABLED=true
      - JAEGER_ENABLED=true
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"  # Metrics endpoint
    depends_on:
      - prometheus
      - jaeger
    networks:
      - hypergnn-net
      - monitoring

  prometheus:
    extends:
      file: docker-compose.monitoring.yml
      service: prometheus
      
  grafana:
    extends:
      file: docker-compose.monitoring.yml
      service: grafana
      
  jaeger:
    extends:
      file: docker-compose.monitoring.yml
      service: jaeger

networks:
  monitoring:
    external: false
```

## Usage Examples

### Basic Monitoring Setup

```bash
# Start monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
open http://localhost:16686 # Jaeger

# Check application health
curl http://localhost:8080/health
curl http://localhost:8080/health/deep
```

### Custom Metrics Integration

```python
# In your application code
from monitoring.instrumentation import MetricsCollector

metrics = MetricsCollector()

# Record custom metrics
with metrics.time_inference():
    result = model.forward(graph_data)
    
metrics.record_graph_size(num_nodes, num_edges)
metrics.update_system_metrics()
```

## Best Practices

1. **Metric Naming**: Use consistent naming conventions
2. **Label Cardinality**: Keep label cardinality low to avoid metric explosion
3. **Sampling**: Use sampling for high-frequency events
4. **Retention**: Configure appropriate data retention policies
5. **Alerting**: Create actionable alerts with clear runbooks
6. **Documentation**: Document all custom metrics and their purposes

## Troubleshooting

### Common Issues

- **High Memory Usage**: Check for metric cardinality explosion
- **Missing Metrics**: Verify scrape configuration and network connectivity
- **Slow Queries**: Optimize PromQL queries and add recording rules
- **Alert Fatigue**: Review alert thresholds and add proper grouping

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Validate metrics endpoint
curl http://localhost:8000/metrics

# Test alerting rules
promtool check rules monitoring/rules/*.yml

# Validate Prometheus config
promtool check config monitoring/prometheus.yml
```