"""Real-time observability dashboard with monitoring visualization."""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from collections import defaultdict, deque

try:
    from flask import Flask, render_template_string, jsonify, request, Response
    import plotly.graph_objs as go
    import plotly.utils
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .logging_utils import get_logger
from .health_checks import get_health_registry
from .metrics_collector import get_metrics_aggregator
from .alerting import get_alert_manager

logger = get_logger(__name__)


class DashboardData:
    """Data aggregator for dashboard."""
    
    def __init__(self, metrics_aggregator=None, health_registry=None, alert_manager=None):
        self.metrics_aggregator = metrics_aggregator
        self.health_registry = health_registry  
        self.alert_manager = alert_manager
        self.update_interval = 5.0  # seconds
        self.data_cache = {}
        self.cache_timestamp = None
        
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview data."""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'unknown',
            'active_alerts': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'gpu_usage': 0,
            'model_status': 'unknown'
        }
        
        try:
            # Health status
            if self.health_registry:
                health_summary = self.health_registry.get_health_summary()
                overview['system_health'] = health_summary['overall_status']
                overview['health_checks'] = health_summary['summary']
                
            # Alerts
            if self.alert_manager:
                firing_alerts = self.alert_manager.get_firing_alerts()
                overview['active_alerts'] = len(firing_alerts)
                overview['alert_breakdown'] = {
                    'critical': sum(1 for a in firing_alerts if a.severity.value == 'critical'),
                    'warning': sum(1 for a in firing_alerts if a.severity.value == 'warning'),
                    'info': sum(1 for a in firing_alerts if a.severity.value == 'info')
                }
                
            # Resource metrics
            if self.metrics_aggregator:
                resource_metrics = self.metrics_aggregator.resource_collector.get_current_resource_metrics()
                overview['cpu_usage'] = resource_metrics.get('cpu_percent', 0)
                overview['memory_usage'] = resource_metrics.get('memory_percent', 0)
                
                gpu_metrics = resource_metrics.get('gpu', {})
                if gpu_metrics:
                    gpu_usage_values = [v for k, v in gpu_metrics.items() if 'memory_percent' in k]
                    overview['gpu_usage'] = max(gpu_usage_values) if gpu_usage_values else 0
                    
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            
        return overview
        
    def get_metrics_timeseries(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get time series data for a metric."""
        if not self.metrics_aggregator or not self.metrics_aggregator.db_connection:
            return {'timestamps': [], 'values': [], 'error': 'No metrics database available'}
            
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = self.metrics_aggregator.db_connection.execute(
                "SELECT timestamp, value FROM metrics WHERE name = ? AND timestamp >= ? ORDER BY timestamp",
                (metric_name, cutoff)
            )
            
            rows = cursor.fetchall()
            timestamps = [row[0] for row in rows]
            values = [row[1] for row in rows]
            
            return {
                'metric_name': metric_name,
                'timestamps': timestamps,
                'values': values,
                'count': len(values)
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics timeseries: {e}")
            return {'timestamps': [], 'values': [], 'error': str(e)}
            
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress data."""
        training_data = {
            'epochs': [],
            'training_loss': [],
            'validation_loss': [], 
            'accuracy': [],
            'current_epoch': 0,
            'training_active': False
        }
        
        try:
            if self.metrics_aggregator:
                model_collector = self.metrics_aggregator.model_collector
                
                # Get training metrics
                for metric_name, metrics_list in model_collector.training_metrics.items():
                    if 'loss' in metric_name.lower():
                        recent_metrics = sorted(metrics_list, key=lambda x: x.timestamp)[-100:]
                        
                        if 'validation' in metric_name or 'val_' in metric_name:
                            training_data['validation_loss'] = [
                                {'epoch': m.metadata.get('epoch', 0), 'value': m.value, 'timestamp': m.timestamp.isoformat()}
                                for m in recent_metrics
                            ]
                        else:
                            training_data['training_loss'] = [
                                {'epoch': m.metadata.get('epoch', 0), 'step': m.metadata.get('step', 0), 
                                 'value': m.value, 'timestamp': m.timestamp.isoformat()}
                                for m in recent_metrics
                            ]
                            
                    elif 'accuracy' in metric_name.lower():
                        recent_metrics = sorted(metrics_list, key=lambda x: x.timestamp)[-100:]
                        training_data['accuracy'] = [
                            {'epoch': m.metadata.get('epoch', 0), 'value': m.value, 'timestamp': m.timestamp.isoformat()}
                            for m in recent_metrics
                        ]
                        
                # Determine current epoch and if training is active
                all_metrics = []
                for metrics_list in model_collector.training_metrics.values():
                    all_metrics.extend(metrics_list)
                    
                if all_metrics:
                    latest_metric = max(all_metrics, key=lambda x: x.timestamp)
                    training_data['current_epoch'] = latest_metric.metadata.get('epoch', 0)
                    
                    # Consider training active if we have metrics from the last 5 minutes
                    training_data['training_active'] = (
                        datetime.now() - latest_metric.timestamp
                    ).total_seconds() < 300
                    
        except Exception as e:
            logger.error(f"Failed to get training progress: {e}")
            training_data['error'] = str(e)
            
        return training_data
        
    def get_resource_utilization(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource utilization data."""
        resource_data = {
            'cpu': {'timestamps': [], 'values': []},
            'memory': {'timestamps': [], 'values': []},
            'gpu_memory': {'timestamps': [], 'values': []},
            'disk': {'timestamps': [], 'values': []}
        }
        
        try:
            # CPU utilization
            cpu_data = self.get_metrics_timeseries('cpu_usage_percent', hours)
            resource_data['cpu']['timestamps'] = cpu_data['timestamps']
            resource_data['cpu']['values'] = cpu_data['values']
            
            # Memory utilization
            memory_data = self.get_metrics_timeseries('memory_usage_percent', hours)
            resource_data['memory']['timestamps'] = memory_data['timestamps']
            resource_data['memory']['values'] = memory_data['values']
            
            # GPU memory utilization
            gpu_data = self.get_metrics_timeseries('gpu_memory_usage_percent', hours)
            resource_data['gpu_memory']['timestamps'] = gpu_data['timestamps']
            resource_data['gpu_memory']['values'] = gpu_data['values']
            
            # Disk utilization
            disk_data = self.get_metrics_timeseries('disk_usage_percent', hours)
            resource_data['disk']['timestamps'] = disk_data['timestamps']
            resource_data['disk']['values'] = disk_data['values']
            
        except Exception as e:
            logger.error(f"Failed to get resource utilization: {e}")
            resource_data['error'] = str(e)
            
        return resource_data
        
    def get_error_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent error logs."""
        error_logs = []
        
        try:
            # Get errors from alert history
            if self.alert_manager:
                recent_alerts = self.alert_manager.get_alert_history(limit)
                error_logs = [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'level': alert.severity.value,
                        'message': alert.message,
                        'source': alert.source,
                        'details': alert.metadata
                    }
                    for alert in recent_alerts
                ]
                
        except Exception as e:
            logger.error(f"Failed to get error logs: {e}")
            
        return error_logs
        
    def get_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics summary."""
        performance = {
            'inference_latency': {'mean': 0, 'p95': 0, 'p99': 0},
            'throughput': {'current': 0, 'mean': 0},
            'error_rate': 0,
            'requests_per_minute': 0
        }
        
        try:
            if self.metrics_aggregator:
                # Inference latency
                latency_summary = self.metrics_aggregator.get_metric_summary('inference_latency_ms', hours)
                if latency_summary:
                    performance['inference_latency'] = {
                        'mean': latency_summary.mean,
                        'p95': latency_summary.p95,
                        'p99': latency_summary.p99,
                        'latest': latency_summary.latest_value
                    }
                    
                # Throughput
                throughput_summary = self.metrics_aggregator.get_metric_summary('inference_throughput', hours)
                if throughput_summary:
                    performance['throughput'] = {
                        'current': throughput_summary.latest_value,
                        'mean': throughput_summary.mean,
                        'max': throughput_summary.max_value
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            
        return performance
        
    def get_cached_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached dashboard data."""
        now = datetime.now()
        
        if (not force_refresh and self.cache_timestamp and 
            (now - self.cache_timestamp).total_seconds() < self.update_interval):
            return self.data_cache
            
        # Refresh cache
        self.data_cache = {
            'timestamp': now.isoformat(),
            'system_overview': self.get_system_overview(),
            'training_progress': self.get_training_progress(),
            'resource_utilization': self.get_resource_utilization(hours=1),
            'performance_metrics': self.get_performance_metrics(hours=1),
            'error_logs': self.get_error_logs(limit=50)
        }
        
        self.cache_timestamp = now
        return self.data_cache


class WebDashboard:
    """Web-based dashboard using Flask."""
    
    def __init__(self, data_aggregator: DashboardData, host: str = "0.0.0.0", port: int = 8080):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and plotly are required for web dashboard. Install with: pip install flask plotly")
            
        self.app = Flask(__name__)
        self.data_aggregator = data_aggregator
        self.host = host
        self.port = port
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(self._get_dashboard_template())
            
        @self.app.route('/api/overview')
        def api_overview():
            """API endpoint for system overview."""
            return jsonify(self.data_aggregator.get_system_overview())
            
        @self.app.route('/api/metrics/<metric_name>')
        def api_metrics(metric_name):
            """API endpoint for metric timeseries."""
            hours = request.args.get('hours', 1, type=int)
            return jsonify(self.data_aggregator.get_metrics_timeseries(metric_name, hours))
            
        @self.app.route('/api/training')
        def api_training():
            """API endpoint for training progress."""
            return jsonify(self.data_aggregator.get_training_progress())
            
        @self.app.route('/api/resources')
        def api_resources():
            """API endpoint for resource utilization."""
            hours = request.args.get('hours', 1, type=int)
            return jsonify(self.data_aggregator.get_resource_utilization(hours))
            
        @self.app.route('/api/performance')
        def api_performance():
            """API endpoint for performance metrics."""
            hours = request.args.get('hours', 1, type=int)
            return jsonify(self.data_aggregator.get_performance_metrics(hours))
            
        @self.app.route('/api/errors')
        def api_errors():
            """API endpoint for error logs."""
            limit = request.args.get('limit', 100, type=int)
            return jsonify(self.data_aggregator.get_error_logs(limit))
            
        @self.app.route('/api/alerts')
        def api_alerts():
            """API endpoint for alerts."""
            if self.data_aggregator.alert_manager:
                firing_alerts = self.data_aggregator.alert_manager.get_firing_alerts()
                return jsonify([alert.to_dict() for alert in firing_alerts])
            return jsonify([])
            
        @self.app.route('/api/dashboard-data')
        def api_dashboard_data():
            """API endpoint for all dashboard data."""
            return jsonify(self.data_aggregator.get_cached_data())
            
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            if self.data_aggregator.health_registry:
                health_summary = self.data_aggregator.health_registry.get_health_summary()
                return jsonify(health_summary)
            return jsonify({'status': 'unknown'})
            
    def _get_dashboard_template(self) -> str:
        """Get HTML template for dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Hypernetwork Forge - Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card {
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .chart-container {
            height: 400px;
        }
        .log-entry {
            border-left: 4px solid #ddd;
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        .log-error { border-color: #e74c3c; }
        .log-warning { border-color: #f39c12; }
        .log-info { border-color: #3498db; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Graph Hypernetwork Forge - Monitoring Dashboard</h1>
        <p>Real-time monitoring and observability</p>
    </div>
    
    <div class="grid">
        <div class="card metric-card">
            <div class="metric-label">System Health</div>
            <div id="system-health" class="metric-value">Loading...</div>
        </div>
        <div class="card metric-card">
            <div class="metric-label">Active Alerts</div>
            <div id="active-alerts" class="metric-value">Loading...</div>
        </div>
        <div class="card metric-card">
            <div class="metric-label">CPU Usage</div>
            <div id="cpu-usage" class="metric-value">Loading...</div>
        </div>
        <div class="card metric-card">
            <div class="metric-label">Memory Usage</div>
            <div id="memory-usage" class="metric-value">Loading...</div>
        </div>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>Resource Utilization</h3>
            <div id="resource-chart" class="chart-container"></div>
        </div>
        <div class="card">
            <h3>Training Progress</h3>
            <div id="training-chart" class="chart-container"></div>
        </div>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>Performance Metrics</h3>
            <div id="performance-chart" class="chart-container"></div>
        </div>
        <div class="card">
            <h3>Recent Alerts & Errors</h3>
            <div id="error-logs" style="max-height: 400px; overflow-y: auto;"></div>
        </div>
    </div>

    <script>
        // Auto-refresh dashboard
        function refreshDashboard() {
            fetch('/api/dashboard-data')
                .then(response => response.json())
                .then(data => {
                    updateOverview(data.system_overview);
                    updateResourceChart(data.resource_utilization);
                    updateTrainingChart(data.training_progress);
                    updatePerformanceChart(data.performance_metrics);
                    updateErrorLogs(data.error_logs);
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateOverview(overview) {
            document.getElementById('system-health').textContent = overview.system_health || 'Unknown';
            document.getElementById('system-health').className = 'metric-value status-' + (overview.system_health || 'unknown');
            
            document.getElementById('active-alerts').textContent = overview.active_alerts || 0;
            document.getElementById('cpu-usage').textContent = (overview.cpu_usage || 0).toFixed(1) + '%';
            document.getElementById('memory-usage').textContent = (overview.memory_usage || 0).toFixed(1) + '%';
        }
        
        function updateResourceChart(resourceData) {
            if (!resourceData || !resourceData.cpu) return;
            
            const traces = [
                {
                    x: resourceData.cpu.timestamps,
                    y: resourceData.cpu.values,
                    name: 'CPU %',
                    type: 'scatter',
                    mode: 'lines'
                },
                {
                    x: resourceData.memory.timestamps,
                    y: resourceData.memory.values,
                    name: 'Memory %',
                    type: 'scatter',
                    mode: 'lines'
                }
            ];
            
            if (resourceData.gpu_memory && resourceData.gpu_memory.values.length > 0) {
                traces.push({
                    x: resourceData.gpu_memory.timestamps,
                    y: resourceData.gpu_memory.values,
                    name: 'GPU Memory %',
                    type: 'scatter',
                    mode: 'lines'
                });
            }
            
            Plotly.newPlot('resource-chart', traces, {
                title: 'Resource Utilization Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Percentage' }
            });
        }
        
        function updateTrainingChart(trainingData) {
            if (!trainingData || !trainingData.training_loss) return;
            
            const traces = [];
            
            if (trainingData.training_loss.length > 0) {
                traces.push({
                    x: trainingData.training_loss.map(d => d.epoch),
                    y: trainingData.training_loss.map(d => d.value),
                    name: 'Training Loss',
                    type: 'scatter',
                    mode: 'lines+markers'
                });
            }
            
            if (trainingData.validation_loss.length > 0) {
                traces.push({
                    x: trainingData.validation_loss.map(d => d.epoch),
                    y: trainingData.validation_loss.map(d => d.value),
                    name: 'Validation Loss',
                    type: 'scatter',
                    mode: 'lines+markers'
                });
            }
            
            Plotly.newPlot('training-chart', traces, {
                title: 'Training Progress',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss' }
            });
        }
        
        function updatePerformanceChart(performanceData) {
            const data = [{
                values: [
                    performanceData.inference_latency?.mean || 0,
                    performanceData.throughput?.mean || 0,
                    performanceData.error_rate || 0
                ],
                labels: ['Avg Latency (ms)', 'Avg Throughput', 'Error Rate %'],
                type: 'pie'
            }];
            
            Plotly.newPlot('performance-chart', data, {
                title: 'Performance Overview'
            });
        }
        
        function updateErrorLogs(errorLogs) {
            const container = document.getElementById('error-logs');
            container.innerHTML = '';
            
            errorLogs.slice(-10).reverse().forEach(log => {
                const logDiv = document.createElement('div');
                logDiv.className = `log-entry log-${log.level}`;
                logDiv.innerHTML = `
                    <div><strong>${new Date(log.timestamp).toLocaleString()}</strong> [${log.level.toUpperCase()}]</div>
                    <div>${log.message}</div>
                    <div style="font-size: 0.8em; color: #666;">Source: ${log.source}</div>
                `;
                container.appendChild(logDiv);
            });
        }
        
        // Initial load and setup auto-refresh
        refreshDashboard();
        setInterval(refreshDashboard, 5000); // Refresh every 5 seconds
    </script>
</body>
</html>
        """
        
    def run(self, debug: bool = False):
        """Run the web dashboard."""
        logger.info(f"Starting web dashboard on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


class ConsoleDashboard:
    """Console-based dashboard for environments without web interface."""
    
    def __init__(self, data_aggregator: DashboardData, update_interval: float = 5.0):
        self.data_aggregator = data_aggregator
        self.update_interval = update_interval
        self.running = False
        
    def display_overview(self, overview: Dict[str, Any]):
        """Display system overview."""
        print("\n" + "="*60)
        print("  GRAPH HYPERNETWORK FORGE - MONITORING DASHBOARD")
        print("="*60)
        
        print(f"System Health: {overview.get('system_health', 'Unknown').upper()}")
        print(f"Active Alerts: {overview.get('active_alerts', 0)}")
        print(f"CPU Usage: {overview.get('cpu_usage', 0):.1f}%")
        print(f"Memory Usage: {overview.get('memory_usage', 0):.1f}%")
        print(f"GPU Usage: {overview.get('gpu_usage', 0):.1f}%")
        
        # Health checks summary
        if 'health_checks' in overview:
            hc = overview['health_checks']
            print(f"Health Checks: {hc.get('healthy', 0)} healthy, {hc.get('degraded', 0)} degraded, {hc.get('unhealthy', 0)} unhealthy")
            
    def display_alerts(self, alerts: List[Dict[str, Any]]):
        """Display active alerts."""
        if not alerts:
            return
            
        print("\n" + "-"*40)
        print("  ACTIVE ALERTS")
        print("-"*40)
        
        for alert in alerts[-5:]:  # Show last 5 alerts
            severity = alert.get('severity', 'unknown').upper()
            print(f"[{severity}] {alert.get('name', 'Unknown')}")
            print(f"  {alert.get('message', 'No message')}")
            print(f"  Time: {alert.get('timestamp', 'Unknown')}")
            print()
            
    def display_performance(self, performance: Dict[str, Any]):
        """Display performance metrics."""
        print("\n" + "-"*40)
        print("  PERFORMANCE METRICS")
        print("-"*40)
        
        latency = performance.get('inference_latency', {})
        throughput = performance.get('throughput', {})
        
        print(f"Inference Latency:")
        print(f"  Mean: {latency.get('mean', 0):.2f}ms")
        print(f"  P95: {latency.get('p95', 0):.2f}ms")
        print(f"  P99: {latency.get('p99', 0):.2f}ms")
        
        print(f"Throughput:")
        print(f"  Current: {throughput.get('current', 0):.2f} req/s")
        print(f"  Mean: {throughput.get('mean', 0):.2f} req/s")
        
        print(f"Error Rate: {performance.get('error_rate', 0):.2f}%")
        
    def run_once(self):
        """Display dashboard once."""
        try:
            data = self.data_aggregator.get_cached_data(force_refresh=True)
            
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            self.display_overview(data.get('system_overview', {}))
            
            if self.data_aggregator.alert_manager:
                alerts = [alert.to_dict() for alert in self.data_aggregator.alert_manager.get_firing_alerts()]
                self.display_alerts(alerts)
                
            self.display_performance(data.get('performance_metrics', {}))
            
            print(f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Press Ctrl+C to exit")
            
        except Exception as e:
            print(f"Error displaying dashboard: {e}")
            
    def run(self):
        """Run console dashboard with auto-refresh."""
        self.running = True
        
        try:
            while self.running:
                self.run_once()
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")
        finally:
            self.running = False


def create_dashboard(dashboard_type: str = "web", 
                    metrics_storage_path: Optional[str] = None,
                    alerting_config_path: Optional[str] = None,
                    host: str = "0.0.0.0", 
                    port: int = 8080) -> Union[WebDashboard, ConsoleDashboard]:
    """Create and configure dashboard."""
    
    # Initialize components
    metrics_aggregator = get_metrics_aggregator(metrics_storage_path)
    health_registry = get_health_registry()
    alert_manager = get_alert_manager(alerting_config_path)
    
    # Create data aggregator
    data_aggregator = DashboardData(
        metrics_aggregator=metrics_aggregator,
        health_registry=health_registry,
        alert_manager=alert_manager
    )
    
    # Create appropriate dashboard
    if dashboard_type.lower() == "web":
        return WebDashboard(data_aggregator, host, port)
    else:
        return ConsoleDashboard(data_aggregator)


def setup_monitoring_dashboard(dashboard_type: str = "web",
                             metrics_storage_path: str = "/tmp/ghf_metrics.db",
                             alerting_config_path: Optional[str] = None,
                             host: str = "0.0.0.0",
                             port: int = 8080):
    """Setup complete monitoring dashboard with all components."""
    
    # Setup metrics collection
    from .metrics_collector import setup_metrics_collection
    setup_metrics_collection(metrics_storage_path)
    
    # Setup health checks
    from .health_checks import setup_default_health_checks
    setup_default_health_checks()
    
    # Setup alerting if config provided
    if alerting_config_path:
        from .alerting import setup_alerting
        setup_alerting(alerting_config_path)
        
    # Create dashboard
    dashboard = create_dashboard(
        dashboard_type=dashboard_type,
        metrics_storage_path=metrics_storage_path, 
        alerting_config_path=alerting_config_path,
        host=host,
        port=port
    )
    
    logger.info(f"Monitoring dashboard setup complete ({dashboard_type})")
    return dashboard