"""HTTP server for monitoring endpoints and health checks."""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import yaml

try:
    from flask import Flask, jsonify, request, Response
    from werkzeug.serving import make_server
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .logging_utils import get_logger
from .health_checks import get_health_registry, setup_default_health_checks
from .metrics_collector import get_metrics_aggregator, setup_metrics_collection
from .alerting import get_alert_manager, setup_alerting
from .dashboard import DashboardData

logger = get_logger(__name__)


class MonitoringServer:
    """HTTP server providing monitoring endpoints."""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 config_path: Optional[str] = None):
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for monitoring server. Install with: pip install flask")
            
        self.host = host
        self.port = port
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.health_registry = get_health_registry()
        self.metrics_aggregator = get_metrics_aggregator()
        self.alert_manager = get_alert_manager()
        self.dashboard_data = DashboardData(
            metrics_aggregator=self.metrics_aggregator,
            health_registry=self.health_registry,
            alert_manager=self.alert_manager
        )
        
        # Flask app
        self.app = Flask(__name__)
        self.server = None
        self.server_thread = None
        
        self._setup_routes()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring server configuration."""
        default_config = {
            'api': {
                'rate_limiting': {
                    'enabled': False,
                    'requests_per_minute': 100
                }
            },
            'security': {
                'cors': {
                    'enabled': True,
                    'allowed_origins': ['*']
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
        
    def _setup_routes(self):
        """Setup Flask routes for monitoring endpoints."""
        
        # Enable CORS if configured
        if self.config.get('security', {}).get('cors', {}).get('enabled', True):
            @self.app.after_request
            def after_request(response):
                allowed_origins = self.config.get('security', {}).get('cors', {}).get('allowed_origins', ['*'])
                if '*' in allowed_origins:
                    response.headers.add('Access-Control-Allow-Origin', '*')
                else:
                    origin = request.headers.get('Origin')
                    if origin in allowed_origins:
                        response.headers.add('Access-Control-Allow-Origin', origin)
                        
                response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
                response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
                return response
                
        @self.app.route('/health', methods=['GET'])
        def health_endpoint():
            """Basic health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'graph-hypernetwork-forge'
            })
            
        @self.app.route('/health/detailed', methods=['GET'])
        def detailed_health_endpoint():
            """Detailed health check endpoint."""
            try:
                health_summary = self.health_registry.get_health_summary()
                return jsonify(health_summary)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return jsonify({
                    'overall_status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        @self.app.route('/health/ready', methods=['GET'])
        def readiness_endpoint():
            """Readiness probe endpoint."""
            try:
                # Check if critical components are ready
                health_results = self.health_registry.run_all_checks(include_non_critical=False)
                
                # Consider ready if no critical health checks are unhealthy
                unhealthy_critical = [
                    name for name, result in health_results.items()
                    if self.health_registry.health_checks.get(name, None) and
                    self.health_registry.health_checks[name].critical and
                    result.status.value == 'unhealthy'
                ]
                
                if unhealthy_critical:
                    return jsonify({
                        'ready': False,
                        'reason': f"Critical health checks failing: {unhealthy_critical}",
                        'timestamp': datetime.now().isoformat()
                    }), 503
                    
                return jsonify({
                    'ready': True,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Readiness check failed: {e}")
                return jsonify({
                    'ready': False,
                    'reason': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 503
                
        @self.app.route('/health/live', methods=['GET'])
        def liveness_endpoint():
            """Liveness probe endpoint."""
            # Simple liveness check - if we can respond, we're alive
            return jsonify({
                'alive': True,
                'timestamp': datetime.now().isoformat()
            })
            
        @self.app.route('/metrics', methods=['GET'])
        def metrics_endpoint():
            """Prometheus-style metrics endpoint."""
            try:
                # Get current metrics
                current_metrics = self.metrics_aggregator.resource_collector.get_current_resource_metrics()
                
                # Format as Prometheus metrics
                metrics_lines = []
                metrics_lines.append('# HELP ghf_system_info System information')
                metrics_lines.append('# TYPE ghf_system_info gauge')
                metrics_lines.append(f'ghf_system_info{{version="0.1.0"}} 1')
                
                # CPU metrics
                if 'cpu_percent' in current_metrics:
                    metrics_lines.append('# HELP ghf_cpu_usage_percent CPU usage percentage')
                    metrics_lines.append('# TYPE ghf_cpu_usage_percent gauge')
                    metrics_lines.append(f'ghf_cpu_usage_percent {current_metrics["cpu_percent"]}')
                    
                # Memory metrics
                if 'memory_percent' in current_metrics:
                    metrics_lines.append('# HELP ghf_memory_usage_percent Memory usage percentage')
                    metrics_lines.append('# TYPE ghf_memory_usage_percent gauge')
                    metrics_lines.append(f'ghf_memory_usage_percent {current_metrics["memory_percent"]}')
                    
                # GPU metrics
                gpu_metrics = current_metrics.get('gpu', {})
                for gpu_metric, value in gpu_metrics.items():
                    if 'memory_percent' in gpu_metric:
                        gpu_id = gpu_metric.split('_')[1]
                        metrics_lines.append('# HELP ghf_gpu_memory_usage_percent GPU memory usage percentage')
                        metrics_lines.append('# TYPE ghf_gpu_memory_usage_percent gauge')
                        metrics_lines.append(f'ghf_gpu_memory_usage_percent{{gpu="{gpu_id}"}} {value}')
                        
                # Health status
                health_summary = self.health_registry.get_health_summary()
                overall_status = health_summary.get('overall_status', 'unknown')
                status_value = {'healthy': 1, 'degraded': 0.5, 'unhealthy': 0, 'unknown': -1}.get(overall_status, -1)
                
                metrics_lines.append('# HELP ghf_health_status Overall health status (1=healthy, 0.5=degraded, 0=unhealthy, -1=unknown)')
                metrics_lines.append('# TYPE ghf_health_status gauge')
                metrics_lines.append(f'ghf_health_status {status_value}')
                
                return Response('\n'.join(metrics_lines) + '\n', mimetype='text/plain')
                
            except Exception as e:
                logger.error(f"Metrics endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/metrics/<metric_name>', methods=['GET'])
        def specific_metric_endpoint(metric_name):
            """Get specific metric data."""
            try:
                hours = request.args.get('hours', 1, type=int)
                metric_data = self.dashboard_data.get_metrics_timeseries(metric_name, hours)
                return jsonify(metric_data)
            except Exception as e:
                logger.error(f"Failed to get metric {metric_name}: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/alerts', methods=['GET'])
        def alerts_endpoint():
            """Get current alerts."""
            try:
                firing_alerts = self.alert_manager.get_firing_alerts()
                return jsonify({
                    'alerts': [alert.to_dict() for alert in firing_alerts],
                    'count': len(firing_alerts),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Alerts endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/alerts/history', methods=['GET'])
        def alert_history_endpoint():
            """Get alert history."""
            try:
                limit = request.args.get('limit', 50, type=int)
                alert_history = self.alert_manager.get_alert_history(limit)
                return jsonify({
                    'alerts': [alert.to_dict() for alert in alert_history],
                    'count': len(alert_history),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Alert history endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/dashboard/data', methods=['GET'])
        def dashboard_data_endpoint():
            """Get comprehensive dashboard data."""
            try:
                dashboard_data = self.dashboard_data.get_cached_data()
                return jsonify(dashboard_data)
            except Exception as e:
                logger.error(f"Dashboard data endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/training/status', methods=['GET'])
        def training_status_endpoint():
            """Get training status and progress."""
            try:
                training_data = self.dashboard_data.get_training_progress()
                return jsonify(training_data)
            except Exception as e:
                logger.error(f"Training status endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/system/info', methods=['GET'])
        def system_info_endpoint():
            """Get system information."""
            try:
                import platform
                import psutil
                import torch
                
                system_info = {
                    'platform': {
                        'system': platform.system(),
                        'release': platform.release(),
                        'version': platform.version(),
                        'machine': platform.machine(),
                        'processor': platform.processor()
                    },
                    'python': {
                        'version': platform.python_version(),
                        'implementation': platform.python_implementation()
                    },
                    'resources': {
                        'cpu_count': psutil.cpu_count(),
                        'cpu_count_logical': psutil.cpu_count(logical=True),
                        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                        'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
                    },
                    'pytorch': {
                        'version': torch.__version__,
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                if torch.cuda.is_available():
                    gpu_info = []
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        gpu_info.append({
                            'id': i,
                            'name': props.name,
                            'compute_capability': f"{props.major}.{props.minor}",
                            'total_memory_gb': props.total_memory / (1024**3),
                            'multi_processor_count': props.multi_processor_count
                        })
                    system_info['gpu_info'] = gpu_info
                    
                return jsonify(system_info)
                
            except Exception as e:
                logger.error(f"System info endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/config', methods=['GET'])
        def config_endpoint():
            """Get monitoring configuration."""
            try:
                # Return sanitized config (remove sensitive information)
                safe_config = {}
                for key, value in self.config.items():
                    if key not in ['security', 'auth']:  # Skip sensitive sections
                        safe_config[key] = value
                        
                return jsonify({
                    'config': safe_config,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Config endpoint failed: {e}")
                return jsonify({'error': str(e)}), 500
                
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested endpoint does not exist',
                'timestamp': datetime.now().isoformat()
            }), 404
            
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An internal server error occurred',
                'timestamp': datetime.now().isoformat()
            }), 500
            
    def start(self, threaded: bool = True):
        """Start the monitoring server."""
        try:
            if threaded:
                self.server = make_server(self.host, self.port, self.app, threaded=True)
                self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.server_thread.start()
                logger.info(f"Monitoring server started on http://{self.host}:{self.port} (threaded)")
            else:
                logger.info(f"Starting monitoring server on http://{self.host}:{self.port}")
                self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
                
        except Exception as e:
            logger.error(f"Failed to start monitoring server: {e}")
            raise
            
    def stop(self):
        """Stop the monitoring server."""
        if self.server:
            self.server.shutdown()
            if self.server_thread:
                self.server_thread.join()
            logger.info("Monitoring server stopped")
            
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.server_thread is not None and self.server_thread.is_alive()


def create_monitoring_server(host: str = "0.0.0.0", 
                           port: int = 8000,
                           config_path: Optional[str] = None,
                           setup_components: bool = True) -> MonitoringServer:
    """Create and configure monitoring server."""
    
    if setup_components:
        # Setup monitoring components
        setup_default_health_checks()
        setup_metrics_collection()
        
        # Setup alerting if config available
        if config_path:
            alerting_config = Path(config_path).parent / "alerting.yml"
            if alerting_config.exists():
                setup_alerting(str(alerting_config))
                
    return MonitoringServer(host=host, port=port, config_path=config_path)


def run_monitoring_server(host: str = "0.0.0.0",
                         port: int = 8000, 
                         config_path: Optional[str] = None,
                         setup_components: bool = True):
    """Run monitoring server with all components."""
    
    server = create_monitoring_server(
        host=host,
        port=port, 
        config_path=config_path,
        setup_components=setup_components
    )
    
    try:
        server.start(threaded=False)
    except KeyboardInterrupt:
        logger.info("Monitoring server stopped by user")
    except Exception as e:
        logger.error(f"Monitoring server error: {e}")
    finally:
        server.stop()