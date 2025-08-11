#!/usr/bin/env python3
"""
Production monitoring setup example for Graph Hypernetwork Forge.

This example demonstrates a production-ready monitoring setup with:
- Comprehensive health checks
- Metrics collection with persistence
- Alert management with multiple notification channels
- Web dashboard for real-time monitoring
- Monitoring server with proper endpoints
- Integration with training and inference pipelines
"""

import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path
from contextlib import contextmanager

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_hypernetwork_forge.utils import (
    setup_default_health_checks, get_health_registry,
    setup_metrics_collection, get_metrics_aggregator,
    setup_alerting, get_alert_manager,
    create_monitoring_server,
    setup_monitoring_dashboard,
    get_logger, setup_logging
)


class ProductionMonitoringSetup:
    """Production monitoring setup and management."""
    
    def __init__(self, config_dir: str = "./configs/monitoring"):
        self.config_dir = Path(config_dir)
        self.logger = get_logger(__name__)
        
        # Components
        self.health_registry = None
        self.metrics_aggregator = None
        self.alert_manager = None
        self.monitoring_server = None
        self.dashboard = None
        
        # Control flags
        self.shutdown_requested = False
        self.components_started = []
        
    def setup_logging(self):
        """Setup production logging."""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        setup_logging(
            log_level=logging.INFO,
            log_file=str(log_dir / "monitoring.log"),
            max_file_size=10 * 1024 * 1024,  # 10MB
            backup_count=5
        )
        
        self.logger.info("Production monitoring logging setup complete")
        
    def create_config_files(self):
        """Create default configuration files if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create alerting configuration
        alerting_config = self.config_dir / "alerting.yml"
        if not alerting_config.exists():
            self._create_alerting_config(alerting_config)
            
        # Create dashboard configuration  
        dashboard_config = self.config_dir / "dashboard.yml"
        if not dashboard_config.exists():
            self._create_dashboard_config(dashboard_config)
            
        self.logger.info("Configuration files created/verified")
        
    def _create_alerting_config(self, config_path: Path):
        """Create production alerting configuration."""
        config_content = """
# Production alerting configuration
rules:
  # Critical system alerts
  - name: "critical_memory_usage"
    condition: "memory_usage_percent"
    severity: "critical"
    threshold: 95.0
    comparison: ">"
    duration_minutes: 2
    enabled: true
    description: "System memory usage is critical"
    
  - name: "high_memory_usage"
    condition: "memory_usage_percent"
    severity: "warning"
    threshold: 85.0
    comparison: ">"
    duration_minutes: 5
    enabled: true
    description: "System memory usage is high"
    
  - name: "gpu_memory_critical"
    condition: "gpu_memory_usage_percent"
    severity: "critical"
    threshold: 98.0
    comparison: ">"
    duration_minutes: 1
    enabled: true
    description: "GPU memory usage is critical"
    
  - name: "high_inference_latency"
    condition: "inference_latency_ms"
    severity: "warning"
    threshold: 2000.0
    comparison: ">"
    duration_minutes: 5
    enabled: true
    description: "Model inference latency is high"
    
  - name: "low_model_accuracy"
    condition: "model_accuracy"
    severity: "warning"
    threshold: 0.85
    comparison: "<"
    duration_minutes: 10
    enabled: true
    description: "Model accuracy below threshold"
    
  - name: "training_stalled"
    condition: "training_loss"
    severity: "warning" 
    threshold: 0.001
    comparison: "<"
    duration_minutes: 30
    enabled: true
    description: "Training loss not decreasing"

channels:
  # Console logging (always enabled)
  - name: "console"
    type: "console"
    enabled: true
    severities: ["critical", "warning", "info"]
    config: {}
    
  # Email alerts (configure with your SMTP settings)
  - name: "email_production"
    type: "email"
    enabled: false  # Enable and configure for production
    severities: ["critical", "warning"]
    config:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "your-monitoring@company.com"
      password: "your-app-password"
      from_email: "your-monitoring@company.com"
      recipients:
        - "devops@company.com"
        - "ml-team@company.com"
      use_tls: true
      
  # Slack notifications (configure with your webhook)
  - name: "slack_production"
    type: "slack"
    enabled: false  # Enable and configure for production
    severities: ["critical", "warning"]
    config:
      webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
      channel: "#ml-alerts"
      
  # Webhook for external monitoring systems
  - name: "webhook_monitoring"
    type: "webhook"
    enabled: false  # Enable and configure for production
    severities: ["critical", "warning", "info"]
    config:
      webhook_url: "https://your-monitoring-system.com/webhook"
      headers:
        "Content-Type": "application/json"
        "Authorization": "Bearer your-token"
      timeout: 10

settings:
  evaluation_interval_seconds: 30
  history_retention_hours: 168  # 7 days
  max_alerts_per_hour: 50
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
            
    def _create_dashboard_config(self, config_path: Path):
        """Create production dashboard configuration."""
        config_content = """
# Production dashboard configuration
dashboard:
  type: "web"
  web:
    host: "0.0.0.0"
    port: 8080
    debug: false
    auto_refresh_interval: 5

data:
  metrics_storage:
    type: "sqlite"
    path: "./data/monitoring/metrics.db"
    retention_hours: 168  # 7 days
    
charts:
  resource_utilization:
    enabled: true
    time_range_hours: 2
    refresh_interval: 10
    
  performance:
    enabled: true
    time_range_hours: 1
    refresh_interval: 10
    
  training:
    enabled: true
    time_range_hours: 24
    refresh_interval: 30

api:
  enabled: true
  rate_limiting:
    enabled: true
    requests_per_minute: 1000

security:
  cors:
    enabled: true
    allowed_origins: ["*"]  # Restrict in production
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
            
    def setup_health_checks(self):
        """Setup comprehensive health checks for production."""
        self.logger.info("Setting up health checks...")
        
        # Setup default health checks
        self.health_registry = setup_default_health_checks()
        
        # Add custom health checks for production
        self._add_production_health_checks()
        
        self.components_started.append("health_checks")
        self.logger.info(f"Health checks setup complete ({len(self.health_registry.health_checks)} checks)")
        
    def _add_production_health_checks(self):
        """Add production-specific health checks."""
        from graph_hypernetwork_forge.utils.health_checks import ExternalServiceHealthCheck
        
        # Add external service health checks if needed
        # Example: database, API endpoints, etc.
        external_services = {
            # "database": "http://localhost:5432/health",
            # "api_gateway": "https://api.yourcompany.com/health"
        }
        
        if external_services:
            external_health_check = ExternalServiceHealthCheck(external_services)
            self.health_registry.register(external_health_check)
            
    def setup_metrics_collection(self):
        """Setup metrics collection with persistent storage."""
        self.logger.info("Setting up metrics collection...")
        
        # Ensure data directory exists
        data_dir = Path("./data/monitoring")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup metrics collection with database
        metrics_db_path = data_dir / "metrics.db"
        self.metrics_aggregator = setup_metrics_collection(str(metrics_db_path))
        
        self.components_started.append("metrics_collection")
        self.logger.info("Metrics collection setup complete")
        
    def setup_alerting(self):
        """Setup alerting system."""
        self.logger.info("Setting up alerting...")
        
        alerting_config = self.config_dir / "alerting.yml"
        self.alert_manager = setup_alerting(str(alerting_config))
        
        self.components_started.append("alerting")
        self.logger.info(f"Alerting setup complete ({len(self.alert_manager.rules)} rules)")
        
    def setup_monitoring_server(self):
        """Setup monitoring HTTP server."""
        self.logger.info("Setting up monitoring server...")
        
        self.monitoring_server = create_monitoring_server(
            host="0.0.0.0",
            port=8000,
            config_path=str(self.config_dir / "dashboard.yml"),
            setup_components=False  # Already set up
        )
        
        # Start server in background
        self.monitoring_server.start(threaded=True)
        
        self.components_started.append("monitoring_server")
        self.logger.info("Monitoring server started on http://0.0.0.0:8000")
        
    def setup_dashboard(self):
        """Setup monitoring dashboard."""
        self.logger.info("Setting up dashboard...")
        
        try:
            self.dashboard = setup_monitoring_dashboard(
                dashboard_type="web",
                metrics_storage_path=str(Path("./data/monitoring/metrics.db")),
                alerting_config_path=str(self.config_dir / "alerting.yml"),
                host="0.0.0.0",
                port=8080
            )
            
            self.components_started.append("dashboard")
            self.logger.info("Dashboard setup complete")
            
        except ImportError as e:
            self.logger.warning(f"Dashboard requires additional dependencies: {e}")
            self.logger.info("Install with: pip install flask plotly")
            
    def run_health_monitoring_loop(self):
        """Run continuous health monitoring."""
        self.logger.info("Starting health monitoring loop...")
        
        while not self.shutdown_requested:
            try:
                # Run health checks
                health_results = self.health_registry.run_all_checks()
                
                # Check for alerts
                for name, result in health_results.items():
                    if result.status.value in ['unhealthy', 'degraded']:
                        alert = self.alert_manager.evaluate_health_check(result)
                        if alert:
                            self.logger.warning(f"Health alert: {alert.name} - {alert.message}")
                            
                # Log summary periodically
                overall_status = self.health_registry.get_overall_health(health_results)
                self.logger.debug(f"System health: {overall_status.value}")
                
                # Sleep before next check
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
                time.sleep(10)
                
    def run_metrics_monitoring_loop(self):
        """Run continuous metrics monitoring for alerts."""
        self.logger.info("Starting metrics monitoring loop...")
        
        while not self.shutdown_requested:
            try:
                # Get current metrics
                current_metrics = self.metrics_aggregator.resource_collector.get_current_resource_metrics()
                
                # Evaluate alerts for key metrics
                key_metrics = [
                    ('memory_usage_percent', current_metrics.get('memory_percent', 0)),
                    ('cpu_usage_percent', current_metrics.get('cpu_percent', 0))
                ]
                
                for metric_name, value in key_metrics:
                    alerts = self.alert_manager.evaluate_metric(metric_name, value)
                    for alert in alerts:
                        if alert.status.value == 'firing':
                            self.logger.warning(f"Metric alert: {alert.name} - {alert.message}")
                            
                # Sleep before next check
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Metrics monitoring loop error: {e}")
                time.sleep(10)
                
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def shutdown(self):
        """Shutdown all monitoring components."""
        self.logger.info("Shutting down monitoring components...")
        
        self.shutdown_requested = True
        
        # Stop components in reverse order
        if "dashboard" in self.components_started and self.dashboard:
            try:
                # Dashboard shutdown depends on type
                pass
            except Exception as e:
                self.logger.error(f"Error shutting down dashboard: {e}")
                
        if "monitoring_server" in self.components_started and self.monitoring_server:
            try:
                self.monitoring_server.stop()
            except Exception as e:
                self.logger.error(f"Error shutting down monitoring server: {e}")
                
        if "metrics_collection" in self.components_started and self.metrics_aggregator:
            try:
                self.metrics_aggregator.stop_collection()
            except Exception as e:
                self.logger.error(f"Error shutting down metrics collection: {e}")
                
        self.logger.info("Monitoring shutdown complete")
        
    def run(self):
        """Run production monitoring system."""
        try:
            self.logger.info("Starting production monitoring system...")
            
            # Setup components
            self.setup_logging()
            self.create_config_files()
            self.setup_health_checks()
            self.setup_metrics_collection()
            self.setup_alerting()
            self.setup_monitoring_server()
            self.setup_dashboard()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Start monitoring loops
            health_thread = threading.Thread(
                target=self.run_health_monitoring_loop, 
                daemon=True
            )
            metrics_thread = threading.Thread(
                target=self.run_metrics_monitoring_loop,
                daemon=True
            )
            
            health_thread.start()
            metrics_thread.start()
            
            self.logger.info("Production monitoring system is running")
            self.logger.info("Available endpoints:")
            self.logger.info("  - Health checks: http://localhost:8000/health")
            self.logger.info("  - Metrics: http://localhost:8000/metrics")
            self.logger.info("  - Dashboard: http://localhost:8080")
            
            # Keep main thread alive
            while not self.shutdown_requested:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Production monitoring error: {e}")
            raise
        finally:
            self.shutdown()


@contextmanager
def production_monitoring():
    """Context manager for production monitoring."""
    monitoring = ProductionMonitoringSetup()
    try:
        # Setup components
        monitoring.setup_logging()
        monitoring.create_config_files()
        monitoring.setup_health_checks()
        monitoring.setup_metrics_collection()
        monitoring.setup_alerting()
        monitoring.setup_monitoring_server()
        
        yield monitoring
        
    finally:
        monitoring.shutdown()


def main():
    """Main entry point for production monitoring."""
    print("Starting Graph Hypernetwork Forge Production Monitoring")
    print("=" * 60)
    
    monitoring = ProductionMonitoringSetup()
    
    try:
        monitoring.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()