#!/usr/bin/env python3
"""
Monitoring startup script for Graph Hypernetwork Forge.

This script provides a simple way to start the monitoring system
with various configurations and options.

Usage:
    python scripts/start_monitoring.py --mode basic
    python scripts/start_monitoring.py --mode production --config configs/monitoring
    python scripts/start_monitoring.py --mode dashboard-only --port 8080
    python scripts/start_monitoring.py --mode server-only --port 8000
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_hypernetwork_forge.utils import (
    setup_default_health_checks,
    setup_metrics_collection,
    setup_alerting,
    create_monitoring_server,
    setup_monitoring_dashboard,
    get_logger
)


def start_basic_monitoring():
    """Start basic monitoring setup."""
    print("Starting basic monitoring...")
    
    # Setup components
    health_registry = setup_default_health_checks()
    metrics_aggregator = setup_metrics_collection("/tmp/ghf_metrics.db")
    
    print(f"✓ Health checks initialized ({len(health_registry.health_checks)} checks)")
    print("✓ Metrics collection started")
    
    # Run initial health check
    health_summary = health_registry.get_health_summary()
    print(f"✓ System health: {health_summary['overall_status']}")
    
    # Keep running
    try:
        print("\nBasic monitoring is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            # Periodic health check
            health_summary = health_registry.get_health_summary()
            if health_summary['overall_status'] != 'healthy':
                print(f"⚠ System health: {health_summary['overall_status']}")
                
    except KeyboardInterrupt:
        print("\nStopping basic monitoring...")
        metrics_aggregator.stop_collection()
        

def start_server_only(port: int = 8000, config_dir: str = None):
    """Start monitoring server only."""
    print(f"Starting monitoring server on port {port}...")
    
    config_path = None
    if config_dir:
        config_path = Path(config_dir) / "dashboard.yml"
        
    server = create_monitoring_server(
        host="0.0.0.0",
        port=port,
        config_path=str(config_path) if config_path and config_path.exists() else None,
        setup_components=True
    )
    
    print(f"✓ Monitoring server started on http://0.0.0.0:{port}")
    print(f"  Available endpoints:")
    print(f"    - Health: http://localhost:{port}/health")
    print(f"    - Metrics: http://localhost:{port}/metrics")
    print(f"    - System Info: http://localhost:{port}/system/info")
    print(f"    - Alerts: http://localhost:{port}/alerts")
    
    try:
        server.start(threaded=False)
    except KeyboardInterrupt:
        print(f"\nStopping monitoring server...")
        server.stop()


def start_dashboard_only(port: int = 8080, config_dir: str = None):
    """Start dashboard only."""
    print(f"Starting dashboard on port {port}...")
    
    try:
        dashboard = setup_monitoring_dashboard(
            dashboard_type="web",
            metrics_storage_path="/tmp/ghf_dashboard_metrics.db",
            alerting_config_path=str(Path(config_dir) / "alerting.yml") if config_dir else None,
            host="0.0.0.0",
            port=port
        )
        
        print(f"✓ Dashboard started on http://0.0.0.0:{port}")
        print(f"  Open http://localhost:{port} in your browser")
        
        dashboard.run(debug=False)
        
    except ImportError as e:
        print(f"❌ Dashboard requires additional dependencies: {e}")
        print("Install with: pip install flask plotly")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nStopping dashboard...")


def start_console_dashboard():
    """Start console dashboard."""
    print("Starting console dashboard...")
    
    try:
        dashboard = setup_monitoring_dashboard(
            dashboard_type="console",
            metrics_storage_path="/tmp/ghf_console_metrics.db"
        )
        
        print("✓ Console dashboard started")
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\nStopping console dashboard...")


def start_production_monitoring(config_dir: str):
    """Start full production monitoring."""
    print("Starting production monitoring...")
    
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"❌ Configuration directory not found: {config_dir}")
        sys.exit(1)
        
    # Import production monitoring
    from examples.production_monitoring import ProductionMonitoringSetup
    
    monitoring = ProductionMonitoringSetup(config_dir=config_dir)
    
    try:
        monitoring.run()
    except Exception as e:
        print(f"❌ Production monitoring failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start Graph Hypernetwork Forge monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode basic
    Start basic monitoring with health checks and metrics

  %(prog)s --mode server --port 8000
    Start monitoring HTTP server only

  %(prog)s --mode dashboard --port 8080
    Start web dashboard only

  %(prog)s --mode console
    Start console dashboard
  
  %(prog)s --mode production --config configs/monitoring
    Start full production monitoring with configuration
"""
    )
    
    parser.add_argument(
        "--mode",
        choices=["basic", "server", "dashboard", "console", "production"],
        default="basic",
        help="Monitoring mode to start"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Port for server/dashboard (default: 8000 for server, 8080 for dashboard)"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration directory path (required for production mode)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Start appropriate monitoring mode
    try:
        if args.mode == "basic":
            start_basic_monitoring()
            
        elif args.mode == "server":
            port = args.port or 8000
            start_server_only(port, args.config)
            
        elif args.mode == "dashboard":
            port = args.port or 8080
            start_dashboard_only(port, args.config)
            
        elif args.mode == "console":
            start_console_dashboard()
            
        elif args.mode == "production":
            if not args.config:
                print("❌ Production mode requires --config argument")
                sys.exit(1)
            start_production_monitoring(args.config)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()