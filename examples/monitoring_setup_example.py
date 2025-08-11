#!/usr/bin/env python3
"""
Example script demonstrating how to set up comprehensive monitoring
for Graph Hypernetwork Forge systems.

This example shows how to:
1. Initialize monitoring components
2. Set up health checks
3. Configure metrics collection
4. Start alerting system
5. Launch monitoring dashboard
6. Integrate monitoring with training loops
"""

import time
import torch
import numpy as np
from pathlib import Path

# Import monitoring components
from graph_hypernetwork_forge.utils import (
    # Health checks
    setup_default_health_checks, get_health_registry,
    ModelHealthCheck, DataPipelineHealthCheck,
    
    # Metrics collection
    setup_metrics_collection, get_metrics_aggregator,
    
    # Alerting
    setup_alerting, get_alert_manager,
    
    # Dashboard
    setup_monitoring_dashboard, create_dashboard,
    
    # Monitoring server
    create_monitoring_server,
    
    # Core components
    get_logger
)

from graph_hypernetwork_forge.models import HyperGNN
from graph_hypernetwork_forge.data import TextualKnowledgeGraph

logger = get_logger(__name__)


def setup_basic_monitoring():
    """Set up basic monitoring without configuration files."""
    print("Setting up basic monitoring...")
    
    # 1. Setup health checks
    health_registry = setup_default_health_checks()
    print(f"Registered {len(health_registry.health_checks)} health checks")
    
    # 2. Setup metrics collection
    metrics_aggregator = setup_metrics_collection("/tmp/ghf_metrics_demo.db")
    print("Started metrics collection")
    
    # 3. Run initial health check
    health_summary = health_registry.get_health_summary()
    print(f"System health: {health_summary['overall_status']}")
    
    return health_registry, metrics_aggregator


def setup_advanced_monitoring():
    """Set up advanced monitoring with custom configurations."""
    print("Setting up advanced monitoring...")
    
    # Create sample config directory if it doesn't exist
    config_dir = Path("./monitoring_config")
    config_dir.mkdir(exist_ok=True)
    
    # Create a simple alerting configuration
    alerting_config = config_dir / "alerting.yml"
    if not alerting_config.exists():
        with open(alerting_config, 'w') as f:
            f.write("""
rules:
  - name: "high_memory_usage"
    condition: "memory_usage_percent"
    severity: "warning"
    threshold: 85.0
    comparison: ">"
    duration_minutes: 2
    enabled: true
    description: "Memory usage is high"
    
channels:
  - name: "console_alerts"
    type: "console"
    enabled: true
    severities: ["critical", "warning", "info"]
    config: {}
""")
    
    # Setup components with configuration
    health_registry = setup_default_health_checks()
    metrics_aggregator = setup_metrics_collection("/tmp/ghf_metrics_advanced.db")
    alert_manager = setup_alerting(str(alerting_config))
    
    print(f"Advanced monitoring setup complete:")
    print(f"- Health checks: {len(health_registry.health_checks)}")
    print(f"- Alert rules: {len(alert_manager.rules)}")
    print(f"- Notification channels: {len(alert_manager.channels)}")
    
    return health_registry, metrics_aggregator, alert_manager


def demonstrate_model_monitoring():
    """Demonstrate monitoring integration with model training."""
    print("\nDemonstrating model monitoring...")
    
    # Setup monitoring
    health_registry, metrics_aggregator = setup_basic_monitoring()
    
    # Create a sample model
    try:
        model = HyperGNN(
            text_encoder_name='distilbert-base-uncased',
            hypernetwork_hidden_dim=128,
            gnn_hidden_dim=64,
            gnn_output_dim=32
        )
        
        # Add custom model health check
        def sample_input_generator():
            """Generate sample input for model testing."""
            # Create dummy graph data
            import torch_geometric
            from torch_geometric.data import Data
            
            # Simple graph with 5 nodes and random features
            x = torch.randn(5, 10)  # 5 nodes, 10 features each
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
            
            # Create text metadata (simulated)
            text_metadata = ["node description"] * 5
            
            return {
                'graph_data': Data(x=x, edge_index=edge_index),
                'text_metadata': text_metadata
            }
        
        # Register custom model health check
        model_health_check = ModelHealthCheck(
            model_loader=lambda: model,
            sample_input_generator=sample_input_generator
        )
        health_registry.register(model_health_check)
        
        # Analyze model
        model_collector = metrics_aggregator.model_collector
        model_stats = model_collector.analyze_model(model, "HyperGNN")
        
        print(f"Model analysis:")
        print(f"- Total parameters: {model_stats['total_parameters']:,}")
        print(f"- Model size: {model_stats['model_size_mb']:.2f} MB")
        
        # Simulate some training metrics
        for epoch in range(5):
            # Simulate training loss
            loss = 1.0 - (epoch * 0.15) + np.random.normal(0, 0.05)
            model_collector.record_training_metric("loss", loss, epoch, epoch * 100)
            
            # Simulate validation metrics  
            val_loss = loss + 0.1 + np.random.normal(0, 0.02)
            accuracy = 0.7 + (epoch * 0.05) + np.random.normal(0, 0.02)
            
            model_collector.record_validation_metric("loss", val_loss, epoch)
            model_collector.record_validation_metric("accuracy", accuracy, epoch)
            
            print(f"Epoch {epoch}: loss={loss:.3f}, val_loss={val_loss:.3f}, acc={accuracy:.3f}")
        
        # Run health checks
        health_results = health_registry.run_all_checks()
        print(f"\nHealth check results:")
        for name, result in health_results.items():
            print(f"- {name}: {result.status.value} ({result.duration_ms:.1f}ms)")
        
        # Get training progress
        training_progress = model_collector.get_training_progress("loss")
        print(f"\nTraining progress: {training_progress}")
        
    except Exception as e:
        print(f"Error in model monitoring demo: {e}")
        logger.exception("Model monitoring demo failed")


def demonstrate_dashboard():
    """Demonstrate dashboard functionality."""
    print("\nDemonstrating dashboard...")
    
    try:
        # Setup monitoring with dashboard
        dashboard = setup_monitoring_dashboard(
            dashboard_type="console",  # Use console dashboard for demo
            metrics_storage_path="/tmp/ghf_dashboard_demo.db"
        )
        
        print("Console dashboard created. Running once...")
        dashboard.run_once()
        
        print("\nDashboard data structure:")
        data = dashboard.data_aggregator.get_cached_data()
        print(f"- System overview keys: {list(data['system_overview'].keys())}")
        print(f"- Resource utilization keys: {list(data['resource_utilization'].keys())}")
        print(f"- Performance metrics keys: {list(data['performance_metrics'].keys())}")
        
    except ImportError as e:
        print(f"Dashboard demo requires additional dependencies: {e}")
    except Exception as e:
        print(f"Error in dashboard demo: {e}")
        logger.exception("Dashboard demo failed")


def demonstrate_monitoring_server():
    """Demonstrate monitoring server setup."""
    print("\nDemonstrating monitoring server...")
    
    try:
        # Create monitoring server
        server = create_monitoring_server(
            host="127.0.0.1", 
            port=8001,  # Different port to avoid conflicts
            setup_components=True
        )
        
        # Start server in threaded mode for demo
        server.start(threaded=True)
        print("Monitoring server started on http://127.0.0.1:8001")
        
        # Give it a moment to start
        time.sleep(2)
        
        # Test endpoints
        import requests
        try:
            # Test health endpoint
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            print(f"Health endpoint: {response.status_code} - {response.json()}")
            
            # Test system info endpoint
            response = requests.get("http://127.0.0.1:8001/system/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"System info: {info['platform']['system']} - Python {info['python']['version']}")
            
        except requests.exceptions.RequestException as e:
            print(f"Could not test server endpoints: {e}")
        
        # Stop server
        server.stop()
        print("Monitoring server stopped")
        
    except ImportError as e:
        print(f"Monitoring server requires additional dependencies: {e}")
    except Exception as e:
        print(f"Error in monitoring server demo: {e}")
        logger.exception("Monitoring server demo failed")


def demonstrate_alerting():
    """Demonstrate alerting system."""
    print("\nDemonstrating alerting system...")
    
    try:
        # Setup advanced monitoring with alerting
        health_registry, metrics_aggregator, alert_manager = setup_advanced_monitoring()
        
        # Simulate high memory usage to trigger alert
        print("Simulating high memory usage alert...")
        
        # Manually trigger an alert evaluation
        alert = alert_manager.evaluate_metric("memory_usage_percent", 90.0)
        if alert:
            print(f"Alert triggered: {alert.name} - {alert.message}")
        else:
            print("No alert triggered (may need to build up history)")
        
        # Get current alerts
        firing_alerts = alert_manager.get_firing_alerts()
        print(f"Currently firing alerts: {len(firing_alerts)}")
        
        for alert in firing_alerts:
            print(f"- {alert.name}: {alert.severity.value} - {alert.message}")
        
        # Get alert summary
        summary = alert_manager.get_alert_summary()
        print(f"Alert summary: {summary}")
        
    except Exception as e:
        print(f"Error in alerting demo: {e}")
        logger.exception("Alerting demo failed")


def integration_example():
    """Example of integrating monitoring into existing training code."""
    print("\nIntegration example...")
    
    # This shows how to integrate monitoring into your existing training loops
    
    # Setup monitoring (typically done once at application startup)
    health_registry = setup_default_health_checks()
    metrics_aggregator = setup_metrics_collection("/tmp/ghf_integration.db")
    
    # In your training loop, you would add monitoring calls:
    
    # Start training monitoring
    training_monitor = metrics_aggregator.model_collector
    
    def train_with_monitoring():
        """Example training function with monitoring integration."""
        
        # Record training start
        print("Starting monitored training...")
        
        for epoch in range(3):
            epoch_start = time.time()
            
            # Your training code here...
            time.sleep(0.1)  # Simulate training time
            
            # Record training metrics
            train_loss = 1.0 - epoch * 0.2 + np.random.normal(0, 0.1)
            training_monitor.record_training_metric("loss", train_loss, epoch, epoch * 10)
            
            # Record performance metrics  
            epoch_time = time.time() - epoch_start
            metrics_aggregator.performance_collector.record_latency("epoch_training", epoch_time * 1000)
            
            # Periodic health checks (every few epochs)
            if epoch % 2 == 0:
                health_summary = health_registry.get_health_summary()
                print(f"Epoch {epoch} - Health: {health_summary['overall_status']}")
        
        print("Training with monitoring complete!")
    
    train_with_monitoring()
    
    # Get final metrics summary
    dashboard_data = metrics_aggregator.get_dashboard_data()
    print(f"Final system health score: {dashboard_data['system_health']:.2f}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Graph Hypernetwork Forge - Monitoring System Demo")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_model_monitoring()
    demonstrate_dashboard()
    demonstrate_monitoring_server()
    demonstrate_alerting()
    integration_example()
    
    print("\n" + "=" * 60)
    print("Monitoring demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Customize monitoring configuration files in configs/monitoring/")
    print("2. Set up alerting channels (email, Slack, etc.)")
    print("3. Deploy monitoring server for production use") 
    print("4. Integrate monitoring calls into your training scripts")
    print("5. Set up automated dashboard for your team")


if __name__ == "__main__":
    main()