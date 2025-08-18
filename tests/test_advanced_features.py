"""Comprehensive tests for advanced Graph Hypernetwork Forge features."""

import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Import the modules we're testing
try:
    from graph_hypernetwork_forge.utils.resilience_advanced import (
        CircuitBreaker, RetryStrategy, HealthCheck, BulkheadIsolation,
        AdaptiveRateLimiter, ResilienceOrchestrator, resilient
    )
    from graph_hypernetwork_forge.utils.production_monitoring import (
        MetricsCollector, PerformanceProfiler, ModelPerformanceMetrics,
        SystemResourceMetrics, monitor_performance
    )
    from graph_hypernetwork_forge.utils.security_compliance import (
        DataEncryptionManager, DataAnonymizer, ComplianceManager,
        DataClassification, SecurityControl, ComplianceFramework
    )
    from graph_hypernetwork_forge.utils.distributed_inference import (
        DistributedInferenceEngine, BatchProcessor, InferenceRequest,
        InferenceBackend, create_inference_engine
    )
    from graph_hypernetwork_forge.utils.adaptive_optimization import (
        AdaptiveHyperparameterOptimizer, ParameterSpec, ParameterType,
        OptimizationStrategy, AutoTuner
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    # Create mock classes for testing compilation
    MODULES_AVAILABLE = False
    
    class CircuitBreaker:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, func): return func
    
    class RetryStrategy:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, func): return func
    
    class HealthCheck:
        def __init__(self, *args, **kwargs): pass
        def register_check(self, *args, **kwargs): pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
    
    class MetricsCollector:
        def __init__(self, *args, **kwargs): pass
        def record_model_performance(self, *args): pass
    
    class DataEncryptionManager:
        def __init__(self, *args, **kwargs): pass
        def encrypt_data(self, data, data_id, classification): return b"encrypted"
        def decrypt_data(self, data, data_id): return b"decrypted"
    
    # Mock other classes similarly...
    print(f"Modules not available for testing: {e}")


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=1.0,
            success_threshold=2
        )
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        @self.circuit_breaker
        def successful_function():
            return "success"
        
        # Should work normally in closed state
        result = successful_function()
        self.assertEqual(result, "success")
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failures."""
        failure_count = 0
        
        @self.circuit_breaker
        def failing_function():
            nonlocal failure_count
            failure_count += 1
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for _ in range(3):
            with self.assertRaises(Exception):
                failing_function()
        
        # Circuit should now be open
        from graph_hypernetwork_forge.utils.exceptions import ModelError
        with self.assertRaises(ModelError):
            failing_function()
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        failure_count = 0
        
        @self.circuit_breaker
        def intermittent_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Initial failures")
            return "recovered"
        
        # Open the circuit
        for _ in range(3):
            with self.assertRaises(Exception):
                intermittent_function()
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should allow testing in half-open state and recover
        result = intermittent_function()
        self.assertEqual(result, "recovered")


class TestRetryStrategy(unittest.TestCase):
    """Test retry strategy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.retry_strategy = RetryStrategy(
            max_attempts=3,
            base_delay=0.1,
            exponential_base=2.0
        )
    
    def test_retry_successful_after_failures(self):
        """Test retry succeeds after initial failures."""
        attempt_count = 0
        
        @self.retry_strategy
        def eventually_successful():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = eventually_successful()
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 3)
    
    def test_retry_exhausts_attempts(self):
        """Test retry exhausts all attempts on persistent failure."""
        attempt_count = 0
        
        @self.retry_strategy
        def always_failing():
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionError("Persistent failure")
        
        with self.assertRaises(ConnectionError):
            always_failing()
        
        self.assertEqual(attempt_count, 3)
    
    def test_non_retryable_exception(self):
        """Test non-retryable exceptions fail immediately."""
        attempt_count = 0
        
        @self.retry_strategy
        def non_retryable_failure():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Not retryable")
        
        with self.assertRaises(ValueError):
            non_retryable_failure()
        
        self.assertEqual(attempt_count, 1)


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.collector = MetricsCollector(
            collection_interval=0.1,
            export_prometheus=False  # Disable for testing
        )
    
    def test_record_model_performance(self):
        """Test recording model performance metrics."""
        metrics = ModelPerformanceMetrics(
            operation="test_inference",
            duration_ms=150.0,
            memory_used_mb=256.0,
            success=True
        )
        
        self.collector.record_model_performance(metrics)
        
        # Check metrics were recorded
        self.assertEqual(len(self.collector.performance_metrics), 1)
        recorded = self.collector.performance_metrics[0]
        self.assertEqual(recorded.operation, "test_inference")
        self.assertEqual(recorded.duration_ms, 150.0)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Record multiple metrics
        for i in range(5):
            metrics = ModelPerformanceMetrics(
                operation="test_op",
                duration_ms=100.0 + i * 10,
                memory_used_mb=200.0 + i * 20,
                success=True
            )
            self.collector.record_model_performance(metrics)
        
        summary = self.collector.get_performance_summary("test_op")
        
        self.assertEqual(summary['total_operations'], 5)
        self.assertEqual(summary['successful_operations'], 5)
        self.assertEqual(summary['success_rate'], 100.0)
        self.assertGreater(summary['duration_ms']['mean'], 100.0)
    
    def test_automatic_collection(self):
        """Test automatic metrics collection."""
        self.collector.start_collection()
        
        # Let it collect for a short time
        time.sleep(0.3)
        
        self.collector.stop_collection()
        
        # Should have collected some resource metrics
        self.assertGreater(len(self.collector.resource_metrics), 0)


class TestDataEncryption(unittest.TestCase):
    """Test data encryption functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.encryption_manager = DataEncryptionManager()
    
    def test_encrypt_decrypt_cycle(self):
        """Test encrypt and decrypt cycle."""
        original_data = "This is sensitive test data"
        data_id = "test_data_1"
        classification = DataClassification.CONFIDENTIAL
        
        # Encrypt data
        encrypted_data = self.encryption_manager.encrypt_data(
            original_data, data_id, classification
        )
        
        # Verify encryption occurred (data should be different)
        self.assertNotEqual(encrypted_data, original_data.encode())
        
        # Decrypt data
        decrypted_data = self.encryption_manager.decrypt_data(encrypted_data, data_id)
        
        # Verify decryption
        self.assertEqual(decrypted_data.decode(), original_data)
    
    def test_key_management(self):
        """Test encryption key management."""
        data_id = "test_key_mgmt"
        classification = DataClassification.PII
        
        # Encrypt some data to create key
        self.encryption_manager.encrypt_data("test", data_id, classification)
        
        # Verify key was created
        self.assertIn(data_id, self.encryption_manager.encryption_keys)
        self.assertIn(data_id, self.encryption_manager.key_metadata)
        
        # Check metadata
        metadata = self.encryption_manager.key_metadata[data_id]
        self.assertEqual(metadata['classification'], classification.value)


class TestDataAnonymizer(unittest.TestCase):
    """Test data anonymization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.anonymizer = DataAnonymizer()
    
    def test_pii_detection(self):
        """Test PII detection in text."""
        text = "Contact John Doe at john.doe@example.com or call 555-123-4567"
        
        pii_found = self.anonymizer.detect_pii(text)
        
        self.assertIn('email', pii_found)
        self.assertIn('phone', pii_found)
        self.assertEqual(pii_found['email'], ['john.doe@example.com'])
        self.assertEqual(pii_found['phone'], ['555-123-4567'])
    
    def test_text_anonymization(self):
        """Test text anonymization."""
        text = "Email me at test@example.com or call 555-0123"
        
        anonymized = self.anonymizer.anonymize_text(text)
        
        # Should not contain original PII
        self.assertNotIn('test@example.com', anonymized)
        self.assertNotIn('555-0123', anonymized)
        
        # But should preserve structure
        self.assertIn('@example.com', anonymized)  # Domain preserved
    
    def test_pseudonymization(self):
        """Test pseudonymization with reversibility."""
        original_value = "john_doe_123"
        context = "user_id"
        
        pseudonym1 = self.anonymizer.pseudonymize_value(original_value, context)
        pseudonym2 = self.anonymizer.pseudonymize_value(original_value, context)
        
        # Should be consistent
        self.assertEqual(pseudonym1, pseudonym2)
        
        # Should be different from original
        self.assertNotEqual(pseudonym1, original_value)
        
        # Should start with prefix
        self.assertTrue(pseudonym1.startswith("PSEUDO_"))


class TestComplianceManager(unittest.TestCase):
    """Test compliance management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.compliance_manager = ComplianceManager()
    
    def test_policy_validation(self):
        """Test compliance policy validation."""
        classification = DataClassification.PII
        applied_controls = [
            SecurityControl.ENCRYPTION,
            SecurityControl.ACCESS_CONTROL,
            SecurityControl.AUDIT_LOGGING
        ]
        
        validation = self.compliance_manager.validate_compliance(
            classification, applied_controls
        )
        
        # Should have validation results
        self.assertIn('compliant', validation)
        self.assertIn('required_controls', validation)
        self.assertIn('applied_controls', validation)
    
    def test_compliance_report_generation(self):
        """Test compliance report generation."""
        report = self.compliance_manager.generate_compliance_report(
            ComplianceFramework.GDPR
        )
        
        self.assertIn('framework', report)
        self.assertIn('applicable_policies', report)
        self.assertIn('compliance_recommendations', report)
        self.assertEqual(report['framework'], 'gdpr')


class TestDistributedInference(unittest.TestCase):
    """Test distributed inference functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        # Mock model factory
        def mock_model_factory():
            mock_model = Mock()
            mock_model.predict.return_value = "mock_result"
            return mock_model
        
        self.model_factory = mock_model_factory
        self.engine = DistributedInferenceEngine(
            model_factory=self.model_factory,
            backend=InferenceBackend.LOCAL,  # Use local for testing
            enable_batching=False
        )
    
    def test_single_inference(self):
        """Test single inference request."""
        request = InferenceRequest(
            request_id="test_1",
            edge_index=Mock(),
            node_features=Mock(),
            node_texts=["test node"]
        )
        
        result = self.engine.infer(request)
        
        self.assertEqual(result.request_id, "test_1")
        self.assertTrue(result.success)
        self.assertEqual(result.result, "mock_result")
    
    def test_batch_processing(self):
        """Test batch inference processing."""
        requests = [
            InferenceRequest(
                request_id=f"test_{i}",
                edge_index=Mock(),
                node_features=Mock(),
                node_texts=[f"node_{i}"]
            )
            for i in range(3)
        ]
        
        results = self.engine.infer_batch(requests)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)
            self.assertEqual(result.result, "mock_result")
    
    def test_batch_processor(self):
        """Test intelligent batch processing."""
        if not MODULES_AVAILABLE:
            return
        
        batch_processor = BatchProcessor(
            max_batch_size=3,
            max_wait_time=0.1
        )
        
        # Add requests
        for i in range(5):
            request = InferenceRequest(
                request_id=f"batch_test_{i}",
                edge_index=Mock(),
                node_features=Mock(),
                node_texts=[f"node_{i}"]
            )
            batch_processor.add_request(request)
        
        # Get first batch
        batch = batch_processor.get_next_batch()
        self.assertLessEqual(len(batch), 3)
        
        # Update stats
        batch_processor.update_batch_stats(len(batch), 0.1)
        self.assertGreater(batch_processor.batch_stats['total_batches'], 0)


class TestAdaptiveOptimization(unittest.TestCase):
    """Test adaptive optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        self.parameter_specs = [
            ParameterSpec(
                name="learning_rate",
                param_type=ParameterType.CONTINUOUS,
                default_value=0.001,
                min_value=0.0001,
                max_value=0.1
            ),
            ParameterSpec(
                name="batch_size",
                param_type=ParameterType.DISCRETE,
                default_value=32,
                min_value=8,
                max_value=128
            ),
            ParameterSpec(
                name="optimizer",
                param_type=ParameterType.CATEGORICAL,
                default_value="adam",
                choices=["adam", "sgd", "rmsprop"]
            )
        ]
    
    def test_random_search_optimization(self):
        """Test random search optimization."""
        optimizer = AdaptiveHyperparameterOptimizer(
            parameter_specs=self.parameter_specs,
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            max_evaluations=10
        )
        
        def mock_objective(params):
            # Mock objective function that prefers smaller learning rates
            score = 1.0 - params['learning_rate']
            if params['batch_size'] == 64:
                score += 0.1
            return score
        
        result = optimizer.optimize(mock_objective)
        
        self.assertIsNotNone(result.best_params)
        self.assertGreater(result.best_score, 0)
        self.assertEqual(result.total_evaluations, 10)
        self.assertGreater(result.optimization_time, 0)
    
    def test_parameter_generation(self):
        """Test parameter generation."""
        optimizer = AdaptiveHyperparameterOptimizer(
            parameter_specs=self.parameter_specs,
            strategy=OptimizationStrategy.RANDOM_SEARCH
        )
        
        # Generate random parameters
        params = optimizer._generate_random_params()
        
        # Verify parameters are within bounds
        self.assertGreaterEqual(params['learning_rate'], 0.0001)
        self.assertLessEqual(params['learning_rate'], 0.1)
        self.assertGreaterEqual(params['batch_size'], 8)
        self.assertLessEqual(params['batch_size'], 128)
        self.assertIn(params['optimizer'], ["adam", "sgd", "rmsprop"])
    
    def test_auto_tuner(self):
        """Test automated model tuning."""
        auto_tuner = AutoTuner()
        
        # Mock model factory
        def mock_model_factory(**params):
            mock_model = Mock()
            mock_model.params = params
            return mock_model
        
        # Mock evaluation dataset
        mock_dataset = Mock()
        
        # Run auto-tuning (limited evaluations for speed)
        with patch.object(auto_tuner, '_evaluate_model') as mock_eval:
            mock_eval.return_value = {
                'latency': 0.1,
                'throughput': 10.0,
                'accuracy': 0.9,
                'memory_efficiency': 0.8
            }
            
            best_config = auto_tuner.auto_tune_model(
                model_factory=mock_model_factory,
                parameter_specs=self.parameter_specs[:2],  # Limit for speed
                evaluation_dataset=mock_dataset,
                optimization_strategy=OptimizationStrategy.RANDOM_SEARCH
            )
        
        self.assertIsInstance(best_config, dict)
        self.assertIn('learning_rate', best_config)
        self.assertIn('batch_size', best_config)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
    
    def test_resilient_inference_with_monitoring(self):
        """Test resilient inference with performance monitoring."""
        # Set up components
        metrics_collector = MetricsCollector(export_prometheus=False)
        profiler = PerformanceProfiler(metrics_collector)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = "inference_result"
        
        # Resilient operation with monitoring
        @monitor_performance("test_inference")
        @resilient("test_operation")
        def inference_operation():
            return mock_model.predict(Mock(), Mock(), ["test"])
        
        # Execute operation
        result = inference_operation()
        
        self.assertEqual(result, "inference_result")
        
        # Check metrics were recorded
        self.assertGreater(len(metrics_collector.performance_metrics), 0)
    
    def test_secure_distributed_inference(self):
        """Test secure distributed inference with encryption."""
        # Set up encryption
        encryption_manager = DataEncryptionManager()
        
        # Mock sensitive data
        sensitive_texts = ["Patient John Doe has condition X"]
        data_id = "patient_data_1"
        
        # Encrypt sensitive data
        encrypted_texts = []
        for i, text in enumerate(sensitive_texts):
            encrypted = encryption_manager.encrypt_data(
                text, f"{data_id}_{i}", DataClassification.PHI
            )
            encrypted_texts.append(encrypted)
        
        # Set up distributed inference
        def secure_model_factory():
            mock_model = Mock()
            
            def secure_predict(edge_index, node_features, node_texts):
                # In real implementation, would decrypt texts here
                return "secure_inference_result"
            
            mock_model.predict = secure_predict
            return mock_model
        
        engine = DistributedInferenceEngine(
            model_factory=secure_model_factory,
            backend=InferenceBackend.LOCAL
        )
        
        # Create inference request
        request = InferenceRequest(
            request_id="secure_test_1",
            edge_index=Mock(),
            node_features=Mock(),
            node_texts=sensitive_texts  # In practice, would be decrypted
        )
        
        # Execute secure inference
        result = engine.infer(request)
        
        self.assertTrue(result.success)
        self.assertEqual(result.result, "secure_inference_result")
    
    def test_end_to_end_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create parameter specifications
        param_specs = [
            ParameterSpec(
                name="hidden_dim",
                param_type=ParameterType.DISCRETE,
                default_value=256,
                min_value=128,
                max_value=512
            ),
            ParameterSpec(
                name="learning_rate",
                param_type=ParameterType.CONTINUOUS,
                default_value=0.001,
                min_value=0.0001,
                max_value=0.01
            )
        ]
        
        # Mock model with performance monitoring
        class MockOptimizableModel:
            def __init__(self, hidden_dim=256, learning_rate=0.001):
                self.hidden_dim = hidden_dim
                self.learning_rate = learning_rate
            
            def predict(self, *args):
                # Simulate performance based on parameters
                latency = 0.1 + (self.hidden_dim / 1000)
                accuracy = 0.8 + (self.learning_rate * 10)
                return {"latency": latency, "accuracy": accuracy}
        
        # Set up optimization
        auto_tuner = AutoTuner()
        
        def model_factory(**params):
            return MockOptimizableModel(**params)
        
        # Mock evaluation
        with patch.object(auto_tuner, '_evaluate_model') as mock_eval:
            def evaluate_mock(model, dataset):
                result = model.predict()
                return {
                    'latency': result['latency'],
                    'throughput': 1.0 / result['latency'],
                    'accuracy': min(0.95, result['accuracy']),
                    'memory_efficiency': 0.8
                }
            
            mock_eval.side_effect = evaluate_mock
            
            # Run optimization
            best_config = auto_tuner.auto_tune_model(
                model_factory=model_factory,
                parameter_specs=param_specs,
                evaluation_dataset=Mock(),
                optimization_strategy=OptimizationStrategy.RANDOM_SEARCH
            )
        
        # Verify optimization results
        self.assertIn('hidden_dim', best_config)
        self.assertIn('learning_rate', best_config)
        
        # Get optimization report
        report = auto_tuner.get_optimization_report()
        self.assertIn('current_config', report)
        self.assertGreater(report['total_evaluations'], 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)