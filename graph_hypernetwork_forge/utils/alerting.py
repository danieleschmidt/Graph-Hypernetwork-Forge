"""Configurable alerting system with email/webhook notifications."""

import time
import json
import smtplib
import requests
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from pathlib import Path

from .logging_utils import get_logger
from .health_checks import HealthStatus, HealthCheckResult
from .metrics_collector import MetricPoint, MetricSummary

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    description: str
    timestamp: datetime
    source: str
    tags: Dict[str, str]
    metadata: Dict[str, Any]
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        if self.suppressed_until:
            result['suppressed_until'] = self.suppressed_until.isoformat()
        return result


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    comparison: str  # >, <, >=, <=, ==, !=
    duration_minutes: int
    enabled: bool = True
    description: str = ""
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass  
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: str  # email, webhook, slack
    config: Dict[str, Any]
    enabled: bool = True
    severities: List[AlertSeverity] = None
    
    def __post_init__(self):
        if self.severities is None:
            self.severities = list(AlertSeverity)


class AlertEvaluator:
    """Evaluate alert conditions."""
    
    def __init__(self):
        self.alert_history: Dict[str, List[float]] = {}
        self.firing_alerts: Dict[str, Alert] = {}
        
    def evaluate_metric_rule(self, rule: AlertRule, current_value: float) -> Optional[Alert]:
        """Evaluate an alert rule against a metric value."""
        if not rule.enabled:
            return None
            
        # Store value in history
        if rule.name not in self.alert_history:
            self.alert_history[rule.name] = []
        
        self.alert_history[rule.name].append(current_value)
        
        # Keep only recent history
        max_history_size = rule.duration_minutes * 2  # Assuming 30-second intervals
        if len(self.alert_history[rule.name]) > max_history_size:
            self.alert_history[rule.name] = self.alert_history[rule.name][-max_history_size:]
            
        # Check if we have enough history
        min_required_points = max(1, rule.duration_minutes // 2)
        if len(self.alert_history[rule.name]) < min_required_points:
            return None
            
        # Evaluate condition
        values = self.alert_history[rule.name][-min_required_points:]
        condition_met = self._evaluate_condition(values, rule.threshold, rule.comparison)
        
        alert_id = f"{rule.name}_{hash(rule.condition)}"
        
        if condition_met:
            if alert_id not in self.firing_alerts:
                # New alert
                alert = Alert(
                    id=alert_id,
                    name=rule.name,
                    severity=rule.severity,
                    status=AlertStatus.FIRING,
                    message=f"{rule.name}: {rule.comparison} {rule.threshold}",
                    description=rule.description or f"Alert condition {rule.condition} met",
                    timestamp=datetime.now(),
                    source="metric_evaluation",
                    tags=rule.tags,
                    metadata={
                        "current_value": current_value,
                        "threshold": rule.threshold,
                        "comparison": rule.comparison,
                        "duration_minutes": rule.duration_minutes,
                        "values": values
                    }
                )
                self.firing_alerts[alert_id] = alert
                return alert
        else:
            # Check if alert should be resolved
            if alert_id in self.firing_alerts:
                alert = self.firing_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                del self.firing_alerts[alert_id]
                return alert
                
        return None
        
    def evaluate_health_rule(self, health_result: HealthCheckResult) -> Optional[Alert]:
        """Evaluate alert rule for health check result."""
        if health_result.status == HealthStatus.HEALTHY:
            return None
            
        severity = AlertSeverity.WARNING
        if health_result.status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.CRITICAL
            
        alert_id = f"health_{health_result.name}"
        
        if alert_id not in self.firing_alerts:
            alert = Alert(
                id=alert_id,
                name=f"Health Check: {health_result.name}",
                severity=severity,
                status=AlertStatus.FIRING,
                message=health_result.message,
                description=f"Health check {health_result.name} is {health_result.status.value}",
                timestamp=datetime.now(),
                source="health_check",
                tags={"health_check": health_result.name},
                metadata={
                    "health_status": health_result.status.value,
                    "duration_ms": health_result.duration_ms,
                    "details": health_result.details
                }
            )
            self.firing_alerts[alert_id] = alert
            return alert
        
        return None
        
    def _evaluate_condition(self, values: List[float], threshold: float, comparison: str) -> bool:
        """Evaluate condition against values."""
        # For duration-based rules, all values in the period should meet the condition
        if comparison == ">":
            return all(v > threshold for v in values)
        elif comparison == "<":
            return all(v < threshold for v in values)
        elif comparison == ">=":
            return all(v >= threshold for v in values)
        elif comparison == "<=":
            return all(v <= threshold for v in values)
        elif comparison == "==":
            return all(v == threshold for v in values)
        elif comparison == "!=":
            return all(v != threshold for v in values)
        else:
            logger.warning(f"Unknown comparison operator: {comparison}")
            return False
            
    def get_firing_alerts(self) -> List[Alert]:
        """Get currently firing alerts."""
        return list(self.firing_alerts.values())
        
    def suppress_alert(self, alert_id: str, duration_minutes: int):
        """Suppress an alert for a specified duration."""
        if alert_id in self.firing_alerts:
            alert = self.firing_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)
            
    def clear_resolved_alerts(self, max_age_hours: int = 24):
        """Clear old resolved alerts from history."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        # This would typically be implemented with persistent storage
        pass


class EmailNotifier:
    """Email notification handler."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, use_tls: bool = True):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls
        
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert email sent to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
            
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        color = {
            AlertSeverity.CRITICAL: "#FF4444",
            AlertSeverity.WARNING: "#FFA500", 
            AlertSeverity.INFO: "#4444FF"
        }.get(alert.severity, "#888888")
        
        html = f"""
        <html>
        <body>
            <h2 style="color: {color}">{alert.name}</h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Status:</strong> {alert.status.value}</p>
            <p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
            <p><strong>Message:</strong> {alert.message}</p>
            <p><strong>Description:</strong> {alert.description}</p>
            
            <h3>Metadata</h3>
            <ul>
        """
        
        for key, value in alert.metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
            
        html += """
            </ul>
            
            <h3>Tags</h3>
            <ul>
        """
        
        for key, value in alert.tags.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
            
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html


class WebhookNotifier:
    """Webhook notification handler."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout
        
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            logger.info(f"Alert webhook sent successfully: {response.status_code}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert webhook: {e}")
            return False


class SlackNotifier:
    """Slack notification handler."""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel
        
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            color = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "good"
            }.get(alert.severity, "#888888")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"{alert.severity.value.upper()}: {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Description",
                            "value": alert.description,
                            "short": False
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        },
                        {
                            "title": "Source", 
                            "value": alert.source,
                            "short": True
                        }
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }]
            }
            
            if self.channel:
                payload["channel"] = self.channel
                
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Alert sent to Slack successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertManager:
    """Central alert management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.evaluator = AlertEvaluator()
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.notifiers: Dict[str, Any] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        
        if config_path:
            self.load_configuration(config_path)
            
    def load_configuration(self, config_path: str):
        """Load alert configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Load alert rules
            for rule_config in config.get('rules', []):
                rule = AlertRule(
                    name=rule_config['name'],
                    condition=rule_config['condition'],
                    severity=AlertSeverity(rule_config['severity']),
                    threshold=rule_config['threshold'],
                    comparison=rule_config['comparison'],
                    duration_minutes=rule_config.get('duration_minutes', 5),
                    enabled=rule_config.get('enabled', True),
                    description=rule_config.get('description', ''),
                    tags=rule_config.get('tags', {})
                )
                self.rules[rule.name] = rule
                
            # Load notification channels
            for channel_config in config.get('channels', []):
                severities = [AlertSeverity(s) for s in channel_config.get('severities', ['critical', 'warning', 'info'])]
                channel = NotificationChannel(
                    name=channel_config['name'],
                    type=channel_config['type'],
                    config=channel_config['config'],
                    enabled=channel_config.get('enabled', True),
                    severities=severities
                )
                self.channels[channel.name] = channel
                
                # Create notifier instance
                if channel.type == 'email' and channel.enabled:
                    self.notifiers[channel.name] = EmailNotifier(
                        smtp_server=channel.config['smtp_server'],
                        smtp_port=channel.config['smtp_port'],
                        username=channel.config['username'],
                        password=channel.config['password'],
                        from_email=channel.config['from_email'],
                        use_tls=channel.config.get('use_tls', True)
                    )
                elif channel.type == 'webhook' and channel.enabled:
                    self.notifiers[channel.name] = WebhookNotifier(
                        webhook_url=channel.config['webhook_url'],
                        headers=channel.config.get('headers'),
                        timeout=channel.config.get('timeout', 10)
                    )
                elif channel.type == 'slack' and channel.enabled:
                    self.notifiers[channel.name] = SlackNotifier(
                        webhook_url=channel.config['webhook_url'],
                        channel=channel.config.get('channel')
                    )
                    
            logger.info(f"Loaded {len(self.rules)} alert rules and {len(self.channels)} notification channels")
            
        except Exception as e:
            logger.error(f"Failed to load alert configuration: {e}")
            
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.rules.pop(rule_name, None)
        
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.channels[channel.name] = channel
        
    def evaluate_metric(self, metric_name: str, value: float) -> List[Alert]:
        """Evaluate metric against all applicable rules."""
        alerts = []
        
        for rule in self.rules.values():
            if rule.condition == metric_name or metric_name in rule.condition:
                alert = self.evaluator.evaluate_metric_rule(rule, value)
                if alert:
                    alerts.append(alert)
                    self._process_alert(alert)
                    
        return alerts
        
    def evaluate_health_check(self, health_result: HealthCheckResult) -> Optional[Alert]:
        """Evaluate health check result."""
        alert = self.evaluator.evaluate_health_rule(health_result)
        if alert:
            self._process_alert(alert)
        return alert
        
    def _process_alert(self, alert: Alert):
        """Process and send alert through appropriate channels."""
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
            
        # Send notifications
        for channel_name, channel in self.channels.items():
            if not channel.enabled or alert.severity not in channel.severities:
                continue
                
            notifier = self.notifiers.get(channel_name)
            if not notifier:
                continue
                
            try:
                if channel.type == 'email':
                    recipients = channel.config.get('recipients', [])
                    notifier.send_alert(alert, recipients)
                else:
                    notifier.send_alert(alert)
                    
            except Exception as e:
                logger.error(f"Failed to send alert through {channel_name}: {e}")
                
    def get_firing_alerts(self) -> List[Alert]:
        """Get currently firing alerts."""
        return self.evaluator.get_firing_alerts()
        
    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:] if limit else self.alert_history
        
    def suppress_alert(self, alert_id: str, duration_minutes: int):
        """Suppress an alert."""
        self.evaluator.suppress_alert(alert_id, duration_minutes)
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        firing_alerts = self.get_firing_alerts()
        recent_alerts = self.get_alert_history(100)
        
        summary = {
            'firing_alerts_count': len(firing_alerts),
            'recent_alerts_count': len(recent_alerts),
            'alerts_by_severity': {
                'critical': sum(1 for a in recent_alerts if a.severity == AlertSeverity.CRITICAL),
                'warning': sum(1 for a in recent_alerts if a.severity == AlertSeverity.WARNING),
                'info': sum(1 for a in recent_alerts if a.severity == AlertSeverity.INFO)
            },
            'active_rules_count': len([r for r in self.rules.values() if r.enabled]),
            'notification_channels_count': len([c for c in self.channels.values() if c.enabled])
        }
        
        return summary


# Global alert manager
alert_manager: Optional[AlertManager] = None


def get_alert_manager(config_path: Optional[str] = None) -> AlertManager:
    """Get or create global alert manager."""
    global alert_manager
    
    if alert_manager is None:
        alert_manager = AlertManager(config_path)
        
    return alert_manager


def setup_alerting(config_path: str = "/root/repo/configs/monitoring/alerting.yml") -> AlertManager:
    """Setup alerting system with configuration."""
    manager = get_alert_manager(config_path)
    logger.info("Alerting system initialized")
    return manager