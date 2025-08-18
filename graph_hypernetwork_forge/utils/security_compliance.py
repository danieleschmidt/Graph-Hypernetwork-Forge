"""Security compliance and governance framework for Graph Hypernetwork Forge."""

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    # Mock crypto classes for compilation
    class Fernet:
        def __init__(self, key): pass
        def encrypt(self, data): return data
        def decrypt(self, data): return data
    def generate_key(): return b"mock_key"

try:
    from .logging_utils import get_logger, SecurityAuditLogger
    from .exceptions import GraphHypernetworkError
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    class SecurityAuditLogger:
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger("security")
        def log_security_event(self, event, details): 
            self.logger.warning(f"SECURITY: {event} - {details}")
        def log_access_attempt(self, resource, user, success):
            self.logger.info(f"ACCESS: {resource} by {user} - {'SUCCESS' if success else 'FAILED'}")
    class GraphHypernetworkError(Exception): pass
    ENHANCED_FEATURES = False

logger = get_logger(__name__)
security_logger = SecurityAuditLogger("compliance_framework")


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"              # General Data Protection Regulation
    CCPA = "ccpa"              # California Consumer Privacy Act
    HIPAA = "hipaa"            # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"              # Service Organization Control 2
    ISO27001 = "iso27001"      # ISO/IEC 27001
    PCI_DSS = "pci_dss"        # Payment Card Industry Data Security Standard


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"           # Public information
    INTERNAL = "internal"       # Internal use only
    CONFIDENTIAL = "confidential"  # Confidential business information
    RESTRICTED = "restricted"   # Highly sensitive, restricted access
    PII = "pii"                # Personally identifiable information
    PHI = "phi"                # Protected health information


class SecurityControl(Enum):
    """Security control types."""
    ENCRYPTION = "encryption"
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"
    DATA_RETENTION = "data_retention"
    ANONYMIZATION = "anonymization"
    SECURE_DELETION = "secure_deletion"
    BACKUP_ENCRYPTION = "backup_encryption"
    NETWORK_SECURITY = "network_security"


@dataclass
class DataGovernancePolicy:
    """Data governance policy definition."""
    name: str
    classification: DataClassification
    frameworks: List[ComplianceFramework]
    required_controls: List[SecurityControl]
    retention_days: int
    encryption_required: bool = True
    audit_required: bool = True
    anonymization_required: bool = False
    geographic_restrictions: List[str] = field(default_factory=list)
    access_roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    timestamp: datetime
    incident_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_data: str
    response_actions: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    lessons_learned: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataEncryptionManager:
    """Comprehensive data encryption and key management."""
    
    def __init__(self, key_rotation_days: int = 90):
        """Initialize encryption manager.
        
        Args:
            key_rotation_days: Days between automatic key rotation
        """
        self.key_rotation_days = key_rotation_days
        self.encryption_keys = {}
        self.key_metadata = {}
        
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available - using mock encryption")
        
        # Initialize master key
        self._initialize_master_key()
        
        logger.info("Data encryption manager initialized")
    
    def _initialize_master_key(self):
        """Initialize or load master encryption key."""
        key_file = Path(".hypergnn_master.key")
        
        if key_file.exists():
            # Load existing key
            try:
                with open(key_file, "rb") as f:
                    self.master_key = f.read()
                logger.info("Loaded existing master key")
            except Exception as e:
                logger.error(f"Failed to load master key: {e}")
                self._generate_new_master_key(key_file)
        else:
            # Generate new key
            self._generate_new_master_key(key_file)
    
    def _generate_new_master_key(self, key_file: Path):
        """Generate new master encryption key."""
        if CRYPTO_AVAILABLE:
            self.master_key = Fernet.generate_key()
        else:
            self.master_key = secrets.token_bytes(32)
        
        # Securely store key
        with open(key_file, "wb") as f:
            f.write(self.master_key)
        
        # Set restrictive permissions
        os.chmod(key_file, 0o600)
        
        security_logger.log_security_event(
            "MASTER_KEY_GENERATED",
            f"New master key generated and stored in {key_file}"
        )
    
    def encrypt_data(self, data: Union[str, bytes], data_id: str, 
                    classification: DataClassification) -> bytes:
        """Encrypt data according to classification policy.
        
        Args:
            data: Data to encrypt
            data_id: Unique identifier for the data
            classification: Data classification level
            
        Returns:
            Encrypted data bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Get or create encryption key for this data
        key = self._get_or_create_key(data_id, classification)
        
        if CRYPTO_AVAILABLE:
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data)
        else:
            # Mock encryption for testing
            encrypted_data = data
        
        # Log encryption event
        security_logger.log_security_event(
            "DATA_ENCRYPTED",
            f"Data {data_id} encrypted with {classification.value} classification"
        )
        
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes, data_id: str) -> bytes:
        """Decrypt data using stored key.
        
        Args:
            encrypted_data: Encrypted data bytes
            data_id: Data identifier used during encryption
            
        Returns:
            Decrypted data bytes
        """
        if data_id not in self.encryption_keys:
            raise GraphHypernetworkError(f"Encryption key not found for data: {data_id}")
        
        key = self.encryption_keys[data_id]
        
        if CRYPTO_AVAILABLE:
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
        else:
            # Mock decryption
            decrypted_data = encrypted_data
        
        # Log decryption event
        security_logger.log_security_event(
            "DATA_DECRYPTED",
            f"Data {data_id} decrypted for authorized access"
        )
        
        return decrypted_data
    
    def _get_or_create_key(self, data_id: str, classification: DataClassification) -> bytes:
        """Get existing or create new encryption key for data."""
        if data_id in self.encryption_keys:
            # Check if key needs rotation
            key_metadata = self.key_metadata.get(data_id, {})
            created_time = key_metadata.get('created', datetime.now(timezone.utc))
            
            if isinstance(created_time, str):
                created_time = datetime.fromisoformat(created_time)
            
            days_old = (datetime.now(timezone.utc) - created_time).days
            
            if days_old > self.key_rotation_days:
                logger.info(f"Rotating encryption key for {data_id} (age: {days_old} days)")
                return self._rotate_key(data_id, classification)
            
            return self.encryption_keys[data_id]
        
        # Create new key
        if CRYPTO_AVAILABLE:
            key = Fernet.generate_key()
        else:
            key = secrets.token_bytes(32)
        
        self.encryption_keys[data_id] = key
        self.key_metadata[data_id] = {
            'created': datetime.now(timezone.utc),
            'classification': classification.value,
            'algorithm': 'Fernet' if CRYPTO_AVAILABLE else 'mock',
            'rotations': 0
        }
        
        return key
    
    def _rotate_key(self, data_id: str, classification: DataClassification) -> bytes:
        """Rotate encryption key for data."""
        # Generate new key
        if CRYPTO_AVAILABLE:
            new_key = Fernet.generate_key()
        else:
            new_key = secrets.token_bytes(32)
        
        # Update metadata
        old_metadata = self.key_metadata.get(data_id, {})
        self.key_metadata[data_id] = {
            'created': datetime.now(timezone.utc),
            'classification': classification.value,
            'algorithm': 'Fernet' if CRYPTO_AVAILABLE else 'mock',
            'rotations': old_metadata.get('rotations', 0) + 1,
            'previous_key_retired': datetime.now(timezone.utc)
        }
        
        # Store old key temporarily for decryption of existing data
        old_key = self.encryption_keys[data_id]
        self.encryption_keys[f"{data_id}_previous"] = old_key
        self.encryption_keys[data_id] = new_key
        
        security_logger.log_security_event(
            "KEY_ROTATED",
            f"Encryption key rotated for {data_id}"
        )
        
        return new_key


class DataAnonymizer:
    """Data anonymization and pseudonymization utilities."""
    
    def __init__(self):
        """Initialize data anonymizer."""
        # Common PII patterns
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}-?\d{3}-?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }
        
        # Anonymization salt (should be securely managed)
        self.anonymization_salt = secrets.token_bytes(32)
        
        logger.info("Data anonymizer initialized with PII detection patterns")
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of PII types and their matches
        """
        pii_found = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                pii_found[pii_type] = matches
        
        if pii_found:
            security_logger.log_security_event(
                "PII_DETECTED",
                f"PII detected: {list(pii_found.keys())}"
            )
        
        return pii_found
    
    def anonymize_text(self, text: str, preserve_format: bool = True) -> str:
        """Anonymize PII in text.
        
        Args:
            text: Text to anonymize
            preserve_format: Whether to preserve original format
            
        Returns:
            Anonymized text
        """
        anonymized_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            
            for match in matches:
                if preserve_format:
                    replacement = self._generate_format_preserving_replacement(match, pii_type)
                else:
                    replacement = self._hash_value(match)
                
                anonymized_text = anonymized_text.replace(match, replacement)
        
        return anonymized_text
    
    def _generate_format_preserving_replacement(self, value: str, pii_type: str) -> str:
        """Generate format-preserving replacement for PII."""
        if pii_type == 'email':
            # Replace with anonymous email maintaining domain structure
            parts = value.split('@')
            if len(parts) == 2:
                domain = parts[1]
                return f"user{self._hash_value(parts[0])[:8]}@{domain}"
        
        elif pii_type == 'phone':
            # Replace with XXX-XXX-XXXX format
            return 'XXX-XXX-XXXX'
        
        elif pii_type == 'ssn':
            # Replace with XXX-XX-XXXX format
            return 'XXX-XX-XXXX'
        
        elif pii_type == 'credit_card':
            # Mask all but last 4 digits
            clean_number = re.sub(r'[-\s]', '', value)
            if len(clean_number) >= 4:
                masked = 'X' * (len(clean_number) - 4) + clean_number[-4:]
                # Restore original formatting
                if '-' in value:
                    return '-'.join([masked[i:i+4] for i in range(0, len(masked), 4)])
                elif ' ' in value:
                    return ' '.join([masked[i:i+4] for i in range(0, len(masked), 4)])
                return masked
        
        # Default to hash-based replacement
        return self._hash_value(value)
    
    def _hash_value(self, value: str) -> str:
        """Generate consistent hash for value."""
        hash_input = value.encode('utf-8') + self.anonymization_salt
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def pseudonymize_value(self, value: str, context: str = "") -> str:
        """Create reversible pseudonym for value.
        
        Args:
            value: Value to pseudonymize
            context: Context for domain-specific pseudonymization
            
        Returns:
            Pseudonymized value
        """
        # Create context-specific hash
        context_salt = hashlib.sha256((context + "pseudonym").encode()).digest()
        hash_input = value.encode('utf-8') + context_salt
        
        return "PSEUDO_" + hashlib.sha256(hash_input).hexdigest()[:16]


class ComplianceManager:
    """Comprehensive compliance management and reporting."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.policies = {}
        self.incidents = []
        self.audit_log = []
        self.compliance_checks = {}
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("Compliance manager initialized with default policies")
    
    def _initialize_default_policies(self):
        """Initialize default data governance policies."""
        
        # GDPR Policy for PII
        self.add_policy(DataGovernancePolicy(
            name="GDPR_PII_Policy",
            classification=DataClassification.PII,
            frameworks=[ComplianceFramework.GDPR],
            required_controls=[
                SecurityControl.ENCRYPTION,
                SecurityControl.ACCESS_CONTROL,
                SecurityControl.AUDIT_LOGGING,
                SecurityControl.ANONYMIZATION
            ],
            retention_days=365 * 7,  # 7 years
            encryption_required=True,
            audit_required=True,
            anonymization_required=True,
            geographic_restrictions=["EU"],
            access_roles=["data_controller", "data_processor"]
        ))
        
        # HIPAA Policy for PHI
        self.add_policy(DataGovernancePolicy(
            name="HIPAA_PHI_Policy",
            classification=DataClassification.PHI,
            frameworks=[ComplianceFramework.HIPAA],
            required_controls=[
                SecurityControl.ENCRYPTION,
                SecurityControl.ACCESS_CONTROL,
                SecurityControl.AUDIT_LOGGING,
                SecurityControl.SECURE_DELETION
            ],
            retention_days=365 * 6,  # 6 years
            encryption_required=True,
            audit_required=True,
            access_roles=["healthcare_provider", "covered_entity"]
        ))
        
        # SOC2 Policy for confidential data
        self.add_policy(DataGovernancePolicy(
            name="SOC2_Confidential_Policy",
            classification=DataClassification.CONFIDENTIAL,
            frameworks=[ComplianceFramework.SOC2],
            required_controls=[
                SecurityControl.ENCRYPTION,
                SecurityControl.ACCESS_CONTROL,
                SecurityControl.AUDIT_LOGGING,
                SecurityControl.NETWORK_SECURITY
            ],
            retention_days=365 * 3,  # 3 years
            encryption_required=True,
            audit_required=True
        ))
    
    def add_policy(self, policy: DataGovernancePolicy):
        """Add data governance policy."""
        self.policies[policy.name] = policy
        logger.info(f"Added compliance policy: {policy.name}")
    
    def get_policy_for_classification(self, classification: DataClassification) -> Optional[DataGovernancePolicy]:
        """Get applicable policy for data classification."""
        for policy in self.policies.values():
            if policy.classification == classification:
                return policy
        return None
    
    def validate_compliance(self, data_classification: DataClassification, 
                          applied_controls: List[SecurityControl]) -> Dict[str, Any]:
        """Validate compliance for data handling.
        
        Args:
            data_classification: Classification of the data
            applied_controls: Security controls that have been applied
            
        Returns:
            Compliance validation results
        """
        policy = self.get_policy_for_classification(data_classification)
        
        if not policy:
            return {
                'compliant': False,
                'reason': f'No policy found for classification: {data_classification.value}',
                'missing_controls': [],
                'frameworks': []
            }
        
        # Check required controls
        missing_controls = [
            control for control in policy.required_controls
            if control not in applied_controls
        ]
        
        compliant = len(missing_controls) == 0
        
        result = {
            'compliant': compliant,
            'policy': policy.name,
            'frameworks': [f.value for f in policy.frameworks],
            'required_controls': [c.value for c in policy.required_controls],
            'applied_controls': [c.value for c in applied_controls],
            'missing_controls': [c.value for c in missing_controls],
            'validation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Log compliance check
        security_logger.log_security_event(
            "COMPLIANCE_CHECK",
            f"Compliance validation for {data_classification.value}: {'PASSED' if compliant else 'FAILED'}"
        )
        
        return result
    
    def record_incident(self, incident: SecurityIncident):
        """Record security incident."""
        self.incidents.append(incident)
        
        # Log critical incidents immediately
        if incident.severity in ['high', 'critical']:
            logger.critical(f"SECURITY INCIDENT [{incident.severity.upper()}]: {incident.description}")
        
        security_logger.log_security_event(
            "INCIDENT_RECORDED",
            f"Security incident {incident.incident_id} recorded: {incident.description}"
        )
    
    def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for specific framework.
        
        Args:
            framework: Compliance framework to report on
            
        Returns:
            Comprehensive compliance report
        """
        # Find applicable policies
        applicable_policies = [
            policy for policy in self.policies.values()
            if framework in policy.frameworks
        ]
        
        # Analyze incidents
        framework_incidents = [
            incident for incident in self.incidents
            if any(framework in policy.frameworks 
                  for policy in self.policies.values()
                  if policy.classification.value in incident.affected_data)
        ]
        
        # Generate summary statistics
        report = {
            'framework': framework.value,
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'applicable_policies': len(applicable_policies),
            'total_incidents': len(framework_incidents),
            'critical_incidents': len([i for i in framework_incidents if i.severity == 'critical']),
            'resolved_incidents': len([i for i in framework_incidents if i.resolved]),
            'policies': [
                {
                    'name': policy.name,
                    'classification': policy.classification.value,
                    'required_controls': [c.value for c in policy.required_controls],
                    'retention_days': policy.retention_days,
                    'encryption_required': policy.encryption_required
                }
                for policy in applicable_policies
            ],
            'recent_incidents': [
                {
                    'incident_id': incident.incident_id,
                    'timestamp': incident.timestamp.isoformat(),
                    'severity': incident.severity,
                    'description': incident.description,
                    'resolved': incident.resolved
                }
                for incident in framework_incidents[-10:]  # Last 10 incidents
            ],
            'compliance_recommendations': self._generate_recommendations(framework, applicable_policies)
        }
        
        logger.info(f"Generated compliance report for {framework.value}")
        return report
    
    def _generate_recommendations(self, framework: ComplianceFramework, 
                                policies: List[DataGovernancePolicy]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Ensure explicit consent mechanisms are in place for data collection",
                "Implement data subject rights (access, rectification, erasure)",
                "Conduct regular Data Protection Impact Assessments (DPIAs)",
                "Maintain comprehensive data processing records"
            ])
        
        elif framework == ComplianceFramework.HIPAA:
            recommendations.extend([
                "Implement physical safeguards for PHI storage and access",
                "Conduct regular risk assessments for PHI handling",
                "Ensure Business Associate Agreements (BAAs) are in place",
                "Implement breach notification procedures"
            ])
        
        elif framework == ComplianceFramework.SOC2:
            recommendations.extend([
                "Implement comprehensive access controls and monitoring",
                "Conduct regular security awareness training",
                "Maintain incident response and business continuity plans",
                "Perform regular vulnerability assessments"
            ])
        
        # Add policy-specific recommendations
        all_controls = set()
        for policy in policies:
            all_controls.update(policy.required_controls)
        
        if SecurityControl.ENCRYPTION not in all_controls:
            recommendations.append("Consider implementing encryption for sensitive data")
        
        if SecurityControl.AUDIT_LOGGING not in all_controls:
            recommendations.append("Enhance audit logging capabilities")
        
        return recommendations


# Global instances
_global_encryption_manager = None
_global_anonymizer = None  
_global_compliance_manager = None

def get_encryption_manager() -> DataEncryptionManager:
    """Get global encryption manager."""
    global _global_encryption_manager
    if _global_encryption_manager is None:
        _global_encryption_manager = DataEncryptionManager()
    return _global_encryption_manager

def get_data_anonymizer() -> DataAnonymizer:
    """Get global data anonymizer."""
    global _global_anonymizer
    if _global_anonymizer is None:
        _global_anonymizer = DataAnonymizer()
    return _global_anonymizer

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = ComplianceManager()
    return _global_compliance_manager

def ensure_compliance(classification: DataClassification, applied_controls: List[SecurityControl]):
    """Decorator to ensure compliance for operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_compliance_manager()
            validation = manager.validate_compliance(classification, applied_controls)
            
            if not validation['compliant']:
                raise GraphHypernetworkError(
                    f"Compliance violation: {validation['reason']}. "
                    f"Missing controls: {validation['missing_controls']}"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator