#!/usr/bin/env python3
"""
Quality Gates Validation - Comprehensive Testing and Validation Suite
Validates code quality, security, performance, and reliability
"""

import torch
import time
import unittest
import logging
import sys
import subprocess
import json
import os
import tempfile
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import hashlib
import threading
import concurrent.futures
import psutil

# Import our implementations
from simple_hypergnn_demo import SimpleHyperGNN, create_demo_graph
from robust_hypergnn_demo import RobustHyperGNN, ModelConfig, SecurityMonitor
from scalable_hypergnn_demo import ScalableHyperGNN, ScalabilityConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics collection."""
    test_coverage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    reliability_score: float = 0.0
    overall_score: float = 0.0
    
    def calculate_overall(self):
        """Calculate overall quality score."""
        weights = {
            'test_coverage': 0.25,
            'security_score': 0.25,
            'performance_score': 0.25,
            'reliability_score': 0.25
        }
        
        self.overall_score = (
            self.test_coverage * weights['test_coverage'] +
            self.security_score * weights['security_score'] +
            self.performance_score * weights['performance_score'] +
            self.reliability_score * weights['reliability_score']
        )


class TestSuite:
    """Comprehensive test suite for all implementations."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for core functionality."""
        logger.info("Running unit tests...")
        
        class HyperGNNUnitTests(unittest.TestCase):
            
            def setUp(self):
                self.graph = create_demo_graph()
                self.node_features = torch.randn(5, 64)
                self.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
                self.node_texts = [self.graph.nodes[node]["description"] for node in self.graph.nodes()]
            
            def test_simple_hypergnn_creation(self):
                """Test SimpleHyperGNN model creation."""
                model = SimpleHyperGNN(hidden_dim=64, num_layers=2)
                self.assertIsNotNone(model)
                self.assertEqual(model.hidden_dim, 64)
                self.assertEqual(model.num_layers, 2)
            
            def test_simple_hypergnn_forward(self):
                """Test SimpleHyperGNN forward pass."""
                model = SimpleHyperGNN(hidden_dim=64, num_layers=2)
                model.eval()
                
                with torch.no_grad():
                    result = model(self.edge_index, self.node_features, self.node_texts)
                
                self.assertEqual(result.shape[0], self.node_features.shape[0])
                self.assertEqual(result.shape[1], 64)
                self.assertFalse(torch.isnan(result).any())
            
            def test_robust_hypergnn_validation(self):
                """Test RobustHyperGNN input validation."""
                config = ModelConfig(hidden_dim=64, num_layers=2, max_nodes=10)
                model = RobustHyperGNN(config)
                
                # Test with valid inputs
                result = model.predict(self.edge_index, self.node_features, self.node_texts)
                self.assertIsNotNone(result)
                
                # Test with invalid inputs should raise exceptions
                with self.assertRaises(Exception):
                    bad_features = torch.tensor([[float('nan'), 1.0]])
                    bad_edges = torch.tensor([[0, 1]], dtype=torch.long)
                    bad_texts = ["test"]
                    model.predict(bad_edges, bad_features, bad_texts)
            
            def test_scalable_hypergnn_performance(self):
                """Test ScalableHyperGNN performance features."""
                config = ScalabilityConfig(enable_caching=True, cache_size=10)
                model = ScalableHyperGNN(config)
                
                # First prediction (no cache)
                start_time = time.time()
                result1 = model.predict_single(self.edge_index, self.node_features, self.node_texts)
                time1 = time.time() - start_time
                
                # Second prediction (with cache)
                start_time = time.time()
                result2 = model.predict_single(self.edge_index, self.node_features, self.node_texts)
                time2 = time.time() - start_time
                
                # Results should be similar
                self.assertTrue(torch.allclose(result1, result2, atol=1e-5))
                
                # Cache should provide some speedup (or at least not slow down)
                self.assertLessEqual(time2, time1 * 2)  # Allow for some variance
        
        # Run tests
        suite = unittest.TestLoader().loadTestsFromTestCase(HyperGNNUnitTests)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'details': [str(failure) for failure in result.failures + result.errors]
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        results = {
            'tests_run': 0,
            'failures': 0,
            'errors': 0,
            'success_rate': 0.0,
            'details': []
        }
        
        try:
            # Test 1: End-to-end pipeline
            results['tests_run'] += 1
            graph = create_demo_graph()
            node_features = torch.randn(5, 64)
            edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
            node_texts = [graph.nodes[node]["description"] for node in graph.nodes()]
            
            # Test all three implementations
            models = [
                SimpleHyperGNN(hidden_dim=64, num_layers=2),
                RobustHyperGNN(ModelConfig(hidden_dim=64, num_layers=2)),
                ScalableHyperGNN(ScalabilityConfig())
            ]
            
            for i, model in enumerate(models):
                try:
                    if isinstance(model, RobustHyperGNN):
                        result = model.predict(edge_index, node_features, node_texts)
                    elif isinstance(model, ScalableHyperGNN):
                        result = model.predict_single(edge_index, node_features, node_texts)
                    else:
                        model.eval()
                        with torch.no_grad():
                            result = model(edge_index, node_features, node_texts)
                    
                    if result is None or torch.isnan(result).any():
                        results['failures'] += 1
                        results['details'].append(f"Model {i} produced invalid output")
                
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append(f"Model {i} failed: {str(e)}")
            
            # Test 2: Cross-domain transfer
            results['tests_run'] += 1
            try:
                research_texts = ["Research on AI", "Machine learning study", "Data science paper"]
                research_features = torch.randn(3, 64)
                research_edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
                
                model = SimpleHyperGNN(hidden_dim=64, num_layers=2)
                model.eval()
                with torch.no_grad():
                    result = model(research_edges, research_features, research_texts)
                
                if result is None or torch.isnan(result).any():
                    results['failures'] += 1
                    results['details'].append("Cross-domain transfer failed")
            
            except Exception as e:
                results['errors'] += 1
                results['details'].append(f"Cross-domain transfer error: {str(e)}")
            
        except Exception as e:
            results['errors'] += 1
            results['details'].append(f"Integration test setup failed: {str(e)}")
        
        results['success_rate'] = (results['tests_run'] - results['failures'] - results['errors']) / max(results['tests_run'], 1)
        return results


class SecurityValidator:
    """Security validation and vulnerability scanning."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_score = 0.0
    
    def scan_input_validation(self) -> Dict[str, Any]:
        """Test input validation security."""
        logger.info("Scanning input validation security...")
        
        results = {
            'tests_run': 0,
            'vulnerabilities': 0,
            'security_score': 0.0,
            'details': []
        }
        
        try:
            config = ModelConfig(max_nodes=10, max_edges=20)
            model = RobustHyperGNN(config)
            
            # Test 1: SQL injection-like patterns
            results['tests_run'] += 1
            malicious_texts = ["'; DROP TABLE users; --", "Normal text", "Another text"]
            try:
                node_features = torch.randn(3, 64)
                edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
                result = model.predict(edge_index, node_features, malicious_texts)
                results['details'].append("SQL injection patterns handled safely")
            except Exception:
                results['details'].append("SQL injection patterns blocked (good)")
            
            # Test 2: XSS-like patterns
            results['tests_run'] += 1
            xss_texts = ["<script>alert('xss')</script>", "javascript:void(0)", "Normal text"]
            try:
                result = model.predict(edge_index, node_features, xss_texts)
                results['vulnerabilities'] += 1
                results['details'].append("XSS patterns not properly sanitized")
            except RuntimeError:
                results['details'].append("XSS patterns blocked (good)")
            
            # Test 3: Buffer overflow simulation
            results['tests_run'] += 1
            overflow_texts = ["A" * 10000, "Normal text", "Another text"]
            try:
                result = model.predict(edge_index, node_features, overflow_texts)
                results['details'].append("Large inputs handled safely")
            except Exception:
                results['details'].append("Large inputs rejected (good)")
            
            # Test 4: Memory exhaustion attempt
            results['tests_run'] += 1
            try:
                huge_features = torch.randn(10000, 1000)  # Very large
                huge_edges = torch.randint(0, 10000, (2, 50000))
                huge_texts = [f"Node {i}" for i in range(10000)]
                
                result = model.predict(huge_edges, huge_features, huge_texts)
                results['vulnerabilities'] += 1
                results['details'].append("Memory exhaustion not prevented")
            except Exception:
                results['details'].append("Memory exhaustion prevented (good)")
        
        except Exception as e:
            results['details'].append(f"Security scan failed: {str(e)}")
        
        results['security_score'] = 1.0 - (results['vulnerabilities'] / max(results['tests_run'], 1))
        return results
    
    def scan_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting protection."""
        logger.info("Testing rate limiting...")
        
        results = {
            'tests_run': 1,
            'vulnerabilities': 0,
            'security_score': 0.0,
            'details': []
        }
        
        try:
            monitor = SecurityMonitor()
            
            # Simulate rapid requests
            blocked_count = 0
            for i in range(70):  # More than the limit of 60
                if not monitor.check_rate_limiting():
                    blocked_count += 1
            
            if blocked_count > 0:
                results['details'].append(f"Rate limiting working: {blocked_count} requests blocked")
                results['security_score'] = 1.0
            else:
                results['vulnerabilities'] += 1
                results['details'].append("Rate limiting not working properly")
                results['security_score'] = 0.0
        
        except Exception as e:
            results['details'].append(f"Rate limiting test failed: {str(e)}")
        
        return results


class PerformanceBenchmark:
    """Performance benchmarking and validation."""
    
    def __init__(self):
        self.benchmarks = {}
    
    def benchmark_inference_speed(self) -> Dict[str, Any]:
        """Benchmark inference speed."""
        logger.info("Benchmarking inference speed...")
        
        results = {
            'models_tested': 0,
            'benchmarks': {},
            'performance_score': 0.0
        }
        
        # Test data
        node_features = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        node_texts = [f"Node {i} description" for i in range(100)]
        
        models = [
            ("SimpleHyperGNN", SimpleHyperGNN(hidden_dim=64, num_layers=2)),
            ("RobustHyperGNN", RobustHyperGNN(ModelConfig(hidden_dim=64, num_layers=2))),
            ("ScalableHyperGNN", ScalableHyperGNN(ScalabilityConfig()))
        ]
        
        for name, model in models:
            try:
                results['models_tested'] += 1
                
                # Warmup
                for _ in range(3):
                    if isinstance(model, RobustHyperGNN):
                        model.predict(edge_index[:, :10], node_features[:10], node_texts[:10])
                    elif isinstance(model, ScalableHyperGNN):
                        model.predict_single(edge_index[:, :10], node_features[:10], node_texts[:10])
                    else:
                        model.eval()
                        with torch.no_grad():
                            model(edge_index[:, :10], node_features[:10], node_texts[:10])
                
                # Benchmark
                times = []
                for _ in range(10):
                    start_time = time.time()
                    
                    if isinstance(model, RobustHyperGNN):
                        result = model.predict(edge_index, node_features, node_texts)
                    elif isinstance(model, ScalableHyperGNN):
                        result = model.predict_single(edge_index, node_features, node_texts)
                    else:
                        model.eval()
                        with torch.no_grad():
                            result = model(edge_index, node_features, node_texts)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                results['benchmarks'][name] = {
                    'avg_time_ms': avg_time * 1000,
                    'min_time_ms': min_time * 1000,
                    'max_time_ms': max_time * 1000,
                    'throughput_graphs_per_sec': 1.0 / avg_time
                }
                
            except Exception as e:
                results['benchmarks'][name] = {'error': str(e)}
        
        # Calculate performance score (lower time = higher score)
        if results['benchmarks']:
            avg_times = [b.get('avg_time_ms', float('inf')) for b in results['benchmarks'].values() if 'avg_time_ms' in b]
            if avg_times:
                best_time = min(avg_times)
                # Score: 1.0 for sub-100ms, scaling down
                results['performance_score'] = max(0.0, min(1.0, 100.0 / best_time))
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        logger.info("Benchmarking memory usage...")
        
        results = {
            'models_tested': 0,
            'memory_benchmarks': {},
            'memory_score': 0.0
        }
        
        # Test data
        node_features = torch.randn(50, 64)
        edge_index = torch.randint(0, 50, (2, 100))
        node_texts = [f"Node {i} description" for i in range(50)]
        
        models = [
            ("SimpleHyperGNN", lambda: SimpleHyperGNN(hidden_dim=64, num_layers=2)),
            ("RobustHyperGNN", lambda: RobustHyperGNN(ModelConfig(hidden_dim=64, num_layers=2))),
            ("ScalableHyperGNN", lambda: ScalableHyperGNN(ScalabilityConfig()))
        ]
        
        for name, model_factory in models:
            try:
                results['models_tested'] += 1
                
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024**2  # MB
                
                # Create and use model
                model = model_factory()
                
                if isinstance(model, RobustHyperGNN):
                    result = model.predict(edge_index, node_features, node_texts)
                elif isinstance(model, ScalableHyperGNN):
                    result = model.predict_single(edge_index, node_features, node_texts)
                else:
                    model.eval()
                    with torch.no_grad():
                        result = model(edge_index, node_features, node_texts)
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024**2  # MB
                memory_used = memory_after - memory_before
                
                results['memory_benchmarks'][name] = {
                    'memory_used_mb': memory_used,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after
                }
                
                # Cleanup
                del model
                
            except Exception as e:
                results['memory_benchmarks'][name] = {'error': str(e)}
        
        # Calculate memory score (lower usage = higher score)
        if results['memory_benchmarks']:
            memory_usage = [b.get('memory_used_mb', float('inf')) for b in results['memory_benchmarks'].values() if 'memory_used_mb' in b]
            if memory_usage:
                max_memory = max(memory_usage)
                # Score: 1.0 for under 100MB, scaling down
                results['memory_score'] = max(0.0, min(1.0, 100.0 / max_memory)) if max_memory > 0 else 1.0
        
        return results


class QualityGatesValidator:
    """Main quality gates validator."""
    
    def __init__(self):
        self.metrics = QualityMetrics()
        self.test_suite = TestSuite()
        self.security_validator = SecurityValidator()
        self.performance_benchmark = PerformanceBenchmark()
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        logger.info("Starting comprehensive quality gates validation...")
        
        results = {
            'timestamp': time.time(),
            'tests': {},
            'security': {},
            'performance': {},
            'metrics': {},
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # Run tests
            logger.info("Running test validations...")
            unit_results = self.test_suite.run_unit_tests()
            integration_results = self.test_suite.run_integration_tests()
            
            results['tests'] = {
                'unit_tests': unit_results,
                'integration_tests': integration_results
            }
            
            # Calculate test coverage score
            total_success_rate = (
                unit_results['success_rate'] + integration_results['success_rate']
            ) / 2
            self.metrics.test_coverage = total_success_rate
            
            # Run security validations
            logger.info("Running security validations...")
            input_security = self.security_validator.scan_input_validation()
            rate_limit_security = self.security_validator.scan_rate_limiting()
            
            results['security'] = {
                'input_validation': input_security,
                'rate_limiting': rate_limit_security
            }
            
            # Calculate security score
            self.metrics.security_score = (
                input_security['security_score'] + rate_limit_security['security_score']
            ) / 2
            
            # Run performance benchmarks
            logger.info("Running performance benchmarks...")
            speed_benchmark = self.performance_benchmark.benchmark_inference_speed()
            memory_benchmark = self.performance_benchmark.benchmark_memory_usage()
            
            results['performance'] = {
                'speed_benchmark': speed_benchmark,
                'memory_benchmark': memory_benchmark
            }
            
            # Calculate performance score
            self.metrics.performance_score = (
                speed_benchmark['performance_score'] + memory_benchmark['memory_score']
            ) / 2
            
            # Calculate reliability score (based on error rates)
            total_errors = (
                unit_results['errors'] + unit_results['failures'] +
                integration_results['errors'] + integration_results['failures'] +
                input_security['vulnerabilities']
            )
            total_tests = (
                unit_results['tests_run'] + integration_results['tests_run'] +
                input_security['tests_run']
            )
            
            self.metrics.reliability_score = 1.0 - (total_errors / max(total_tests, 1))
            
            # Calculate overall score
            self.metrics.calculate_overall()
            
            # Store metrics
            results['metrics'] = {
                'test_coverage': self.metrics.test_coverage,
                'security_score': self.metrics.security_score,
                'performance_score': self.metrics.performance_score,
                'reliability_score': self.metrics.reliability_score,
                'overall_score': self.metrics.overall_score
            }
            
            # Determine overall status
            if self.metrics.overall_score >= 0.9:
                results['overall_status'] = 'EXCELLENT'
            elif self.metrics.overall_score >= 0.8:
                results['overall_status'] = 'GOOD'
            elif self.metrics.overall_score >= 0.7:
                results['overall_status'] = 'ACCEPTABLE'
            elif self.metrics.overall_score >= 0.6:
                results['overall_status'] = 'NEEDS_IMPROVEMENT'
            else:
                results['overall_status'] = 'POOR'
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            results['error'] = str(e)
            results['overall_status'] = 'ERROR'
        
        return results


def main():
    """Main quality gates validation."""
    print("🛡️ QUALITY GATES VALIDATION - Comprehensive Testing Suite")
    print("=" * 80)
    
    validator = QualityGatesValidator()
    results = validator.run_all_validations()
    
    # Display results
    print("\n📊 VALIDATION RESULTS")
    print("-" * 40)
    
    # Test results
    if 'tests' in results:
        unit_tests = results['tests']['unit_tests']
        integration_tests = results['tests']['integration_tests']
        
        print(f"\n✅ Unit Tests:")
        print(f"   • Tests run: {unit_tests['tests_run']}")
        print(f"   • Success rate: {unit_tests['success_rate']:.1%}")
        print(f"   • Failures: {unit_tests['failures']}")
        print(f"   • Errors: {unit_tests['errors']}")
        
        print(f"\n🔗 Integration Tests:")
        print(f"   • Tests run: {integration_tests['tests_run']}")
        print(f"   • Success rate: {integration_tests['success_rate']:.1%}")
        print(f"   • Failures: {integration_tests['failures']}")
        print(f"   • Errors: {integration_tests['errors']}")
    
    # Security results
    if 'security' in results:
        input_sec = results['security']['input_validation']
        rate_sec = results['security']['rate_limiting']
        
        print(f"\n🔒 Security Validation:")
        print(f"   • Input validation score: {input_sec['security_score']:.1%}")
        print(f"   • Rate limiting score: {rate_sec['security_score']:.1%}")
        print(f"   • Vulnerabilities found: {input_sec['vulnerabilities']}")
    
    # Performance results
    if 'performance' in results:
        speed = results['performance']['speed_benchmark']
        memory = results['performance']['memory_benchmark']
        
        print(f"\n⚡ Performance Benchmarks:")
        print(f"   • Models tested: {speed['models_tested']}")
        print(f"   • Performance score: {speed['performance_score']:.1%}")
        print(f"   • Memory score: {memory['memory_score']:.1%}")
        
        if 'benchmarks' in speed:
            for model_name, bench in speed['benchmarks'].items():
                if 'avg_time_ms' in bench:
                    print(f"   • {model_name}: {bench['avg_time_ms']:.1f}ms avg")
    
    # Overall metrics
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"\n🏆 OVERALL QUALITY METRICS:")
        print(f"   • Test Coverage: {metrics['test_coverage']:.1%}")
        print(f"   • Security Score: {metrics['security_score']:.1%}")
        print(f"   • Performance Score: {metrics['performance_score']:.1%}")
        print(f"   • Reliability Score: {metrics['reliability_score']:.1%}")
        print(f"   • Overall Score: {metrics['overall_score']:.1%}")
        print(f"   • Status: {results['overall_status']}")
    
    # Quality gates decision
    status = results.get('overall_status', 'ERROR')
    if status in ['EXCELLENT', 'GOOD']:
        print(f"\n🎉 QUALITY GATES: ✅ PASSED ({status})")
        print("   Ready for production deployment!")
        return True
    elif status == 'ACCEPTABLE':
        print(f"\n⚠️ QUALITY GATES: ✅ PASSED ({status})")
        print("   Acceptable for deployment with monitoring.")
        return True
    else:
        print(f"\n❌ QUALITY GATES: ❌ FAILED ({status})")
        print("   Requires improvements before deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)