"""Global-first implementation with internationalization and multi-region support."""

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

try:
    from .logging_utils import get_logger
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"


class Region(Enum):
    """Supported regions for compliance and deployment."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"
    MIDDLE_EAST_AFRICA = "mea"


class ComplianceRegime(Enum):
    """Data compliance regimes by region."""
    GDPR = "gdpr"           # European Union
    CCPA = "ccpa"           # California
    LGPD = "lgpd"           # Brazil
    PDPA_SG = "pdpa_sg"     # Singapore
    PIPEDA = "pipeda"       # Canada
    PDPA_TH = "pdpa_th"     # Thailand
    POPI = "popi"           # South Africa


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage
    region: Region
    timezone: str
    date_format: str
    number_format: str
    currency: str
    compliance_regime: ComplianceRegime
    rtl_support: bool = False  # Right-to-left text support


class InternationalizationManager:
    """Comprehensive internationalization and localization manager."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        """Initialize i18n manager.
        
        Args:
            default_language: Default language for the system
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self.pluralization_rules = {}
        self._lock = threading.Lock()
        
        # Load default translations
        self._load_translations()
        self._load_pluralization_rules()
        
        logger.info(f"I18n manager initialized with default language: {default_language.value}")
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        # Default English translations
        self.translations[SupportedLanguage.ENGLISH] = {
            # System messages
            "system.startup": "System starting up",
            "system.shutdown": "System shutting down",
            "system.error": "System error occurred",
            "system.ready": "System ready",
            
            # Model messages
            "model.loading": "Loading model",
            "model.loaded": "Model loaded successfully",
            "model.inference.start": "Starting inference",
            "model.inference.complete": "Inference completed",
            "model.training.start": "Starting model training",
            "model.training.complete": "Training completed",
            "model.error.invalid_input": "Invalid input provided",
            "model.error.inference_failed": "Model inference failed",
            
            # Data messages
            "data.loading": "Loading data",
            "data.loaded": "Data loaded successfully",
            "data.processing": "Processing data",
            "data.encrypted": "Data encrypted",
            "data.decrypted": "Data decrypted",
            "data.anonymized": "Data anonymized",
            
            # Security messages
            "security.access_granted": "Access granted",
            "security.access_denied": "Access denied",
            "security.authentication_required": "Authentication required",
            "security.invalid_credentials": "Invalid credentials",
            "security.session_expired": "Session expired",
            
            # Validation messages
            "validation.required": "This field is required",
            "validation.invalid_format": "Invalid format",
            "validation.out_of_range": "Value out of range",
            "validation.too_large": "Value too large",
            "validation.too_small": "Value too small",
            
            # Performance messages
            "performance.high_latency": "High latency detected",
            "performance.low_throughput": "Low throughput detected",
            "performance.memory_warning": "High memory usage",
            "performance.optimization_applied": "Performance optimization applied"
        }
        
        # Spanish translations
        self.translations[SupportedLanguage.SPANISH] = {
            "system.startup": "Iniciando sistema",
            "system.shutdown": "Cerrando sistema",
            "system.error": "Error del sistema ocurrido",
            "system.ready": "Sistema listo",
            "model.loading": "Cargando modelo",
            "model.loaded": "Modelo cargado exitosamente",
            "model.inference.start": "Iniciando inferencia",
            "model.inference.complete": "Inferencia completada",
            "model.training.start": "Iniciando entrenamiento del modelo",
            "model.training.complete": "Entrenamiento completado",
            "model.error.invalid_input": "Entrada inválida proporcionada",
            "model.error.inference_failed": "Inferencia del modelo falló",
            "data.loading": "Cargando datos",
            "data.loaded": "Datos cargados exitosamente",
            "data.processing": "Procesando datos",
            "data.encrypted": "Datos encriptados",
            "data.decrypted": "Datos desencriptados",
            "data.anonymized": "Datos anonimizados",
            "security.access_granted": "Acceso concedido",
            "security.access_denied": "Acceso denegado",
            "security.authentication_required": "Autenticación requerida",
            "security.invalid_credentials": "Credenciales inválidas",
            "security.session_expired": "Sesión expirada",
            "validation.required": "Este campo es requerido",
            "validation.invalid_format": "Formato inválido",
            "validation.out_of_range": "Valor fuera de rango",
            "validation.too_large": "Valor demasiado grande",
            "validation.too_small": "Valor demasiado pequeño",
            "performance.high_latency": "Alta latencia detectada",
            "performance.low_throughput": "Baja capacidad de procesamiento detectada",
            "performance.memory_warning": "Alto uso de memoria",
            "performance.optimization_applied": "Optimización de rendimiento aplicada"
        }
        
        # French translations
        self.translations[SupportedLanguage.FRENCH] = {
            "system.startup": "Démarrage du système",
            "system.shutdown": "Arrêt du système",
            "system.error": "Erreur système survenue",
            "system.ready": "Système prêt",
            "model.loading": "Chargement du modèle",
            "model.loaded": "Modèle chargé avec succès",
            "model.inference.start": "Démarrage de l'inférence",
            "model.inference.complete": "Inférence terminée",
            "model.training.start": "Démarrage de l'entraînement du modèle",
            "model.training.complete": "Entraînement terminé",
            "model.error.invalid_input": "Entrée invalide fournie",
            "model.error.inference_failed": "Échec de l'inférence du modèle",
            "data.loading": "Chargement des données",
            "data.loaded": "Données chargées avec succès",
            "data.processing": "Traitement des données",
            "data.encrypted": "Données chiffrées",
            "data.decrypted": "Données déchiffrées",
            "data.anonymized": "Données anonymisées",
            "security.access_granted": "Accès accordé",
            "security.access_denied": "Accès refusé",
            "security.authentication_required": "Authentification requise",
            "security.invalid_credentials": "Identifiants invalides",
            "security.session_expired": "Session expirée",
            "validation.required": "Ce champ est requis",
            "validation.invalid_format": "Format invalide",
            "validation.out_of_range": "Valeur hors limites",
            "validation.too_large": "Valeur trop grande",
            "validation.too_small": "Valeur trop petite",
            "performance.high_latency": "Latence élevée détectée",
            "performance.low_throughput": "Faible débit détecté",
            "performance.memory_warning": "Utilisation mémoire élevée",
            "performance.optimization_applied": "Optimisation de performance appliquée"
        }
        
        # German translations
        self.translations[SupportedLanguage.GERMAN] = {
            "system.startup": "System wird gestartet",
            "system.shutdown": "System wird heruntergefahren",
            "system.error": "Systemfehler aufgetreten",
            "system.ready": "System bereit",
            "model.loading": "Modell wird geladen",
            "model.loaded": "Modell erfolgreich geladen",
            "model.inference.start": "Inferenz wird gestartet",
            "model.inference.complete": "Inferenz abgeschlossen",
            "model.training.start": "Modelltraining wird gestartet",
            "model.training.complete": "Training abgeschlossen",
            "model.error.invalid_input": "Ungültige Eingabe bereitgestellt",
            "model.error.inference_failed": "Modellinferenz fehlgeschlagen",
            "data.loading": "Daten werden geladen",
            "data.loaded": "Daten erfolgreich geladen",
            "data.processing": "Daten werden verarbeitet",
            "data.encrypted": "Daten verschlüsselt",
            "data.decrypted": "Daten entschlüsselt",
            "data.anonymized": "Daten anonymisiert",
            "security.access_granted": "Zugriff gewährt",
            "security.access_denied": "Zugriff verweigert",
            "security.authentication_required": "Authentifizierung erforderlich",
            "security.invalid_credentials": "Ungültige Anmeldedaten",
            "security.session_expired": "Sitzung abgelaufen",
            "validation.required": "Dieses Feld ist erforderlich",
            "validation.invalid_format": "Ungültiges Format",
            "validation.out_of_range": "Wert außerhalb des Bereichs",
            "validation.too_large": "Wert zu groß",
            "validation.too_small": "Wert zu klein",
            "performance.high_latency": "Hohe Latenz erkannt",
            "performance.low_throughput": "Niedriger Durchsatz erkannt",
            "performance.memory_warning": "Hohe Speichernutzung",
            "performance.optimization_applied": "Leistungsoptimierung angewendet"
        }
        
        # Japanese translations
        self.translations[SupportedLanguage.JAPANESE] = {
            "system.startup": "システムを起動中",
            "system.shutdown": "システムをシャットダウン中",
            "system.error": "システムエラーが発生しました",
            "system.ready": "システム準備完了",
            "model.loading": "モデルを読み込み中",
            "model.loaded": "モデルの読み込みが完了しました",
            "model.inference.start": "推論を開始中",
            "model.inference.complete": "推論が完了しました",
            "model.training.start": "モデルトレーニングを開始中",
            "model.training.complete": "トレーニングが完了しました",
            "model.error.invalid_input": "無効な入力が提供されました",
            "model.error.inference_failed": "モデル推論に失敗しました",
            "data.loading": "データを読み込み中",
            "data.loaded": "データの読み込みが完了しました",
            "data.processing": "データを処理中",
            "data.encrypted": "データが暗号化されました",
            "data.decrypted": "データが復号化されました",
            "data.anonymized": "データが匿名化されました",
            "security.access_granted": "アクセスが許可されました",
            "security.access_denied": "アクセスが拒否されました",
            "security.authentication_required": "認証が必要です",
            "security.invalid_credentials": "無効な資格情報",
            "security.session_expired": "セッションが期限切れです",
            "validation.required": "この項目は必須です",
            "validation.invalid_format": "無効な形式",
            "validation.out_of_range": "値が範囲外です",
            "validation.too_large": "値が大きすぎます",
            "validation.too_small": "値が小さすぎます",
            "performance.high_latency": "高レイテンシが検出されました",
            "performance.low_throughput": "低スループットが検出されました",
            "performance.memory_warning": "高メモリ使用量",
            "performance.optimization_applied": "パフォーマンス最適化が適用されました"
        }
    
    def _load_pluralization_rules(self):
        """Load pluralization rules for different languages."""
        # English pluralization (simple)
        self.pluralization_rules[SupportedLanguage.ENGLISH] = {
            "zero": lambda n: "no",
            "one": lambda n: "one" if n == 1 else "other",
            "other": lambda n: "other"
        }
        
        # Spanish pluralization
        self.pluralization_rules[SupportedLanguage.SPANISH] = {
            "zero": lambda n: "cero",
            "one": lambda n: "uno" if n == 1 else "otros",
            "other": lambda n: "otros"
        }
        
        # Add other language pluralization rules as needed
    
    def set_language(self, language: SupportedLanguage):
        """Set current language.
        
        Args:
            language: Language to set as current
        """
        with self._lock:
            self.current_language = language
            logger.info(f"Language set to: {language.value}")
    
    def get_text(self, key: str, language: Optional[SupportedLanguage] = None, 
                 **kwargs) -> str:
        """Get localized text for a key.
        
        Args:
            key: Translation key
            language: Optional language override
            **kwargs: Variables for string formatting
            
        Returns:
            Localized text
        """
        lang = language or self.current_language
        
        # Get translation
        translations = self.translations.get(lang, {})
        text = translations.get(key)
        
        # Fallback to English if not found
        if text is None and lang != self.default_language:
            translations = self.translations.get(self.default_language, {})
            text = translations.get(key)
        
        # Final fallback to key itself
        if text is None:
            text = key
            logger.warning(f"Translation not found for key: {key}")
        
        # Format with variables
        try:
            if kwargs:
                text = text.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"String formatting failed for key {key}: {e}")
        
        return text
    
    def get_plural_text(self, key: str, count: int, 
                       language: Optional[SupportedLanguage] = None) -> str:
        """Get pluralized text based on count.
        
        Args:
            key: Base translation key
            count: Count for pluralization
            language: Optional language override
            
        Returns:
            Pluralized text
        """
        lang = language or self.current_language
        
        # Determine plural form
        rules = self.pluralization_rules.get(lang, {})
        
        if count == 0 and "zero" in rules:
            plural_key = f"{key}.zero"
        elif count == 1:
            plural_key = f"{key}.one"
        else:
            plural_key = f"{key}.other"
        
        return self.get_text(plural_key, language, count=count)
    
    @contextmanager
    def use_language(self, language: SupportedLanguage):
        """Context manager for temporary language switching.
        
        Args:
            language: Language to use temporarily
        """
        original_language = self.current_language
        self.set_language(language)
        try:
            yield
        finally:
            self.set_language(original_language)
    
    def get_available_languages(self) -> List[SupportedLanguage]:
        """Get list of available languages."""
        return list(self.translations.keys())


class RegionalComplianceManager:
    """Manage regional compliance requirements and data governance."""
    
    def __init__(self):
        """Initialize regional compliance manager."""
        self.compliance_rules = self._load_compliance_rules()
        self.regional_configs = self._load_regional_configs()
        
        logger.info("Regional compliance manager initialized")
    
    def _load_compliance_rules(self) -> Dict[ComplianceRegime, Dict[str, Any]]:
        """Load compliance rules for different regimes."""
        return {
            ComplianceRegime.GDPR: {
                "data_retention_days": 365 * 7,  # 7 years
                "consent_required": True,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required": True,
                "breach_notification_hours": 72,
                "allowed_transfers": ["adequacy_decision", "bcr", "scc"],
                "prohibited_processing": ["automated_decision_making_without_consent"],
                "required_documentation": ["processing_records", "dpia", "consent_records"]
            },
            ComplianceRegime.CCPA: {
                "data_retention_days": 365 * 2,  # 2 years
                "opt_out_required": True,
                "data_sale_disclosure": True,
                "consumer_rights": ["know", "delete", "opt_out", "non_discrimination"],
                "verification_required": True,
                "service_provider_contracts": True,
                "privacy_policy_required": True,
                "training_required": True
            },
            ComplianceRegime.LGPD: {
                "data_retention_days": 365 * 5,  # 5 years
                "consent_required": True,
                "data_protection_officer": True,
                "impact_assessment": True,
                "international_transfer_safeguards": True,
                "breach_notification_hours": 72,
                "data_subject_rights": ["access", "correction", "anonymization", "portability"]
            },
            ComplianceRegime.PDPA_SG: {
                "consent_required": True,
                "purpose_limitation": True,
                "data_protection_officer": False,
                "breach_notification_days": 3,
                "do_not_call_registry": True,
                "data_portability": False
            }
        }
    
    def _load_regional_configs(self) -> Dict[Region, LocalizationConfig]:
        """Load regional configuration defaults."""
        return {
            Region.NORTH_AMERICA: LocalizationConfig(
                language=SupportedLanguage.ENGLISH,
                region=Region.NORTH_AMERICA,
                timezone="America/New_York",
                date_format="%m/%d/%Y",
                number_format="en_US",
                currency="USD",
                compliance_regime=ComplianceRegime.CCPA
            ),
            Region.EUROPE: LocalizationConfig(
                language=SupportedLanguage.ENGLISH,
                region=Region.EUROPE,
                timezone="Europe/London",
                date_format="%d/%m/%Y",
                number_format="en_GB",
                currency="EUR",
                compliance_regime=ComplianceRegime.GDPR
            ),
            Region.ASIA_PACIFIC: LocalizationConfig(
                language=SupportedLanguage.ENGLISH,
                region=Region.ASIA_PACIFIC,
                timezone="Asia/Singapore",
                date_format="%d/%m/%Y",
                number_format="en_SG",
                currency="SGD",
                compliance_regime=ComplianceRegime.PDPA_SG
            ),
            Region.LATIN_AMERICA: LocalizationConfig(
                language=SupportedLanguage.SPANISH,
                region=Region.LATIN_AMERICA,
                timezone="America/Mexico_City",
                date_format="%d/%m/%Y",
                number_format="es_MX",
                currency="MXN",
                compliance_regime=ComplianceRegime.LGPD
            )
        }
    
    def get_compliance_requirements(self, regime: ComplianceRegime) -> Dict[str, Any]:
        """Get compliance requirements for a regime.
        
        Args:
            regime: Compliance regime
            
        Returns:
            Compliance requirements
        """
        return self.compliance_rules.get(regime, {})
    
    def validate_data_transfer(self, source_region: Region, 
                             target_region: Region,
                             data_classification: str) -> Dict[str, Any]:
        """Validate data transfer between regions.
        
        Args:
            source_region: Source region
            target_region: Target region
            data_classification: Classification of data being transferred
            
        Returns:
            Transfer validation results
        """
        source_config = self.regional_configs.get(source_region)
        target_config = self.regional_configs.get(target_region)
        
        if not source_config or not target_config:
            return {
                "allowed": False,
                "reason": "Unknown region configuration"
            }
        
        source_regime = source_config.compliance_regime
        target_regime = target_config.compliance_regime
        
        # Check if transfer is allowed based on compliance regimes
        if source_regime == ComplianceRegime.GDPR:
            gdpr_rules = self.compliance_rules[ComplianceRegime.GDPR]
            allowed_transfers = gdpr_rules.get("allowed_transfers", [])
            
            # Simplified logic - in practice would be more complex
            if target_regime in [ComplianceRegime.CCPA, ComplianceRegime.PDPA_SG]:
                return {
                    "allowed": True,
                    "mechanism": "adequacy_decision",
                    "conditions": ["standard_contractual_clauses"]
                }
        
        # Default: allow transfers with appropriate safeguards
        return {
            "allowed": True,
            "mechanism": "standard_contractual_clauses",
            "conditions": ["data_encryption", "access_controls", "audit_logging"]
        }
    
    def get_data_retention_period(self, regime: ComplianceRegime, 
                                 data_type: str) -> int:
        """Get required data retention period.
        
        Args:
            regime: Compliance regime
            data_type: Type of data
            
        Returns:
            Retention period in days
        """
        rules = self.compliance_rules.get(regime, {})
        return rules.get("data_retention_days", 365)  # Default 1 year
    
    def check_consent_requirements(self, regime: ComplianceRegime) -> Dict[str, bool]:
        """Check consent requirements for a regime.
        
        Args:
            regime: Compliance regime
            
        Returns:
            Consent requirements
        """
        rules = self.compliance_rules.get(regime, {})
        
        return {
            "consent_required": rules.get("consent_required", False),
            "explicit_consent": regime == ComplianceRegime.GDPR,
            "opt_out_allowed": regime in [ComplianceRegime.CCPA],
            "withdrawal_allowed": rules.get("consent_required", False),
            "granular_consent": regime == ComplianceRegime.GDPR
        }


class MultiRegionDeploymentManager:
    """Manage multi-region deployment configurations and routing."""
    
    def __init__(self):
        """Initialize multi-region deployment manager."""
        self.deployment_configs = self._load_deployment_configs()
        self.routing_rules = {}
        self.active_regions = set()
        
        logger.info("Multi-region deployment manager initialized")
    
    def _load_deployment_configs(self) -> Dict[Region, Dict[str, Any]]:
        """Load deployment configurations for each region."""
        return {
            Region.NORTH_AMERICA: {
                "primary_datacenter": "us-east-1",
                "backup_datacenters": ["us-west-2", "ca-central-1"],
                "cdn_endpoints": ["cloudfront", "fastly"],
                "compliance_requirements": [ComplianceRegime.CCPA],
                "data_residency_required": False,
                "encryption_in_transit": True,
                "encryption_at_rest": True,
                "backup_frequency_hours": 6,
                "disaster_recovery_rto": 4,  # hours
                "disaster_recovery_rpo": 1   # hours
            },
            Region.EUROPE: {
                "primary_datacenter": "eu-west-1",
                "backup_datacenters": ["eu-central-1", "eu-north-1"],
                "cdn_endpoints": ["cloudflare", "akamai"],
                "compliance_requirements": [ComplianceRegime.GDPR],
                "data_residency_required": True,
                "encryption_in_transit": True,
                "encryption_at_rest": True,
                "backup_frequency_hours": 4,
                "disaster_recovery_rto": 2,
                "disaster_recovery_rpo": 0.5
            },
            Region.ASIA_PACIFIC: {
                "primary_datacenter": "ap-southeast-1",
                "backup_datacenters": ["ap-northeast-1", "ap-south-1"],
                "cdn_endpoints": ["cloudfront", "alibaba"],
                "compliance_requirements": [ComplianceRegime.PDPA_SG],
                "data_residency_required": True,
                "encryption_in_transit": True,
                "encryption_at_rest": True,
                "backup_frequency_hours": 8,
                "disaster_recovery_rto": 6,
                "disaster_recovery_rpo": 2
            }
        }
    
    def activate_region(self, region: Region) -> bool:
        """Activate a region for deployment.
        
        Args:
            region: Region to activate
            
        Returns:
            True if activation successful
        """
        try:
            config = self.deployment_configs.get(region)
            if not config:
                logger.error(f"No configuration found for region: {region}")
                return False
            
            # Validate region configuration
            required_fields = ["primary_datacenter", "compliance_requirements"]
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field {field} for region {region}")
                    return False
            
            self.active_regions.add(region)
            logger.info(f"Activated region: {region.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate region {region}: {e}")
            return False
    
    def get_optimal_region(self, user_location: Optional[str] = None,
                          data_classification: Optional[str] = None) -> Region:
        """Get optimal region for user or data.
        
        Args:
            user_location: User's geographic location (country code)
            data_classification: Classification of data being processed
            
        Returns:
            Optimal region
        """
        if not self.active_regions:
            return Region.NORTH_AMERICA  # Default fallback
        
        # Simple geographic routing
        if user_location:
            location_mapping = {
                "US": Region.NORTH_AMERICA,
                "CA": Region.NORTH_AMERICA,
                "MX": Region.LATIN_AMERICA,
                "BR": Region.LATIN_AMERICA,
                "GB": Region.EUROPE,
                "DE": Region.EUROPE,
                "FR": Region.EUROPE,
                "SG": Region.ASIA_PACIFIC,
                "JP": Region.ASIA_PACIFIC,
                "AU": Region.ASIA_PACIFIC
            }
            
            preferred_region = location_mapping.get(user_location.upper())
            if preferred_region and preferred_region in self.active_regions:
                return preferred_region
        
        # Data residency requirements
        if data_classification in ["PII", "PHI", "sensitive"]:
            # Prefer regions with strict data residency
            for region in [Region.EUROPE, Region.ASIA_PACIFIC]:
                if region in self.active_regions:
                    config = self.deployment_configs[region]
                    if config.get("data_residency_required"):
                        return region
        
        # Default to first active region
        return next(iter(self.active_regions))
    
    def get_deployment_config(self, region: Region) -> Optional[Dict[str, Any]]:
        """Get deployment configuration for region.
        
        Args:
            region: Target region
            
        Returns:
            Deployment configuration or None
        """
        return self.deployment_configs.get(region)
    
    def validate_cross_region_operation(self, source_region: Region,
                                      target_region: Region,
                                      operation_type: str) -> Dict[str, Any]:
        """Validate cross-region operation.
        
        Args:
            source_region: Source region
            target_region: Target region
            operation_type: Type of operation (data_sync, backup, etc.)
            
        Returns:
            Validation result
        """
        source_config = self.deployment_configs.get(source_region)
        target_config = self.deployment_configs.get(target_region)
        
        if not source_config or not target_config:
            return {
                "allowed": False,
                "reason": "Invalid region configuration"
            }
        
        # Check compliance compatibility
        source_compliance = set(source_config.get("compliance_requirements", []))
        target_compliance = set(target_config.get("compliance_requirements", []))
        
        # GDPR has strict requirements for data transfer
        if ComplianceRegime.GDPR in source_compliance:
            if ComplianceRegime.GDPR not in target_compliance:
                return {
                    "allowed": False,
                    "reason": "GDPR data cannot be transferred to non-GDPR region",
                    "suggested_action": "Use data processing agreement"
                }
        
        return {
            "allowed": True,
            "requirements": [
                "encryption_in_transit",
                "audit_logging",
                "compliance_monitoring"
            ]
        }


# Global instances
_i18n_manager = None
_compliance_manager = None
_deployment_manager = None

def get_i18n_manager() -> InternationalizationManager:
    """Get global i18n manager."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = InternationalizationManager()
    return _i18n_manager

def get_compliance_manager() -> RegionalComplianceManager:
    """Get global compliance manager."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = RegionalComplianceManager()
    return _compliance_manager

def get_deployment_manager() -> MultiRegionDeploymentManager:
    """Get global deployment manager."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = MultiRegionDeploymentManager()
    return _deployment_manager

def _(key: str, **kwargs) -> str:
    """Convenience function for getting localized text.
    
    Args:
        key: Translation key
        **kwargs: Variables for string formatting
        
    Returns:
        Localized text
    """
    return get_i18n_manager().get_text(key, **kwargs)

def _n(key: str, count: int) -> str:
    """Convenience function for getting pluralized text.
    
    Args:
        key: Base translation key
        count: Count for pluralization
        
    Returns:
        Pluralized text
    """
    return get_i18n_manager().get_plural_text(key, count)

def set_global_language(language: SupportedLanguage):
    """Set global language for the application.
    
    Args:
        language: Language to set globally
    """
    get_i18n_manager().set_language(language)
    logger.info(f"Global language set to: {language.value}")

def activate_region(region: Region) -> bool:
    """Activate a region for global deployment.
    
    Args:
        region: Region to activate
        
    Returns:
        True if activation successful
    """
    return get_deployment_manager().activate_region(region)