# Monitoring and Observability

## Overview

Comprehensive monitoring setup for Graph Hypernetwork Forge, covering performance metrics, error tracking, and operational insights.

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