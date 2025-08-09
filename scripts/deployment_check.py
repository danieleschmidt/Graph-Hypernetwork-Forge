#!/usr/bin/env python3
"""Production deployment readiness checker."""

import sys
import os
import time
import torch
import subprocess
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import SyntheticDataGenerator


class DeploymentChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
    
    def add_check(self, name: str, passed: bool, details: str = "", critical: bool = True):
        """Add a check result."""
        check = {
            'name': name,
            'passed': passed,
            'details': details,
            'critical': critical
        }
        self.checks.append(check)
        
        if not passed:
            if critical:
                self.errors.append(f"❌ {name}: {details}")
            else:
                self.warnings.append(f"⚠️  {name}: {details}")
    
    def check_dependencies(self):
        """Check all required dependencies are available."""
        print("🔍 Checking Dependencies...")
        
        required_packages = [
            'torch', 'torch_geometric', 'transformers', 
            'sentence_transformers', 'numpy', 'networkx'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.add_check(f"Package {package}", True)
            except ImportError as e:
                self.add_check(f"Package {package}", False, str(e))
    
    def check_model_functionality(self):
        """Test core model functionality."""
        print("🧠 Testing Model Functionality...")
        
        try:
            # Test model creation
            model = HyperGNN(hidden_dim=64, num_layers=2, enable_caching=True)
            self.add_check("Model Creation", True)
            
            # Test data generation
            gen = SyntheticDataGenerator()
            graph = gen.generate_social_network(num_nodes=10)
            self.add_check("Data Generation", True)
            
            # Test forward pass
            node_features = torch.randn(graph.num_nodes, 64)
            model.eval()
            
            with torch.no_grad():
                predictions = model(graph.edge_index, node_features, graph.node_texts)
            
            # Validate output
            if predictions.shape == (10, 64):
                self.add_check("Forward Pass Shape", True)
            else:
                self.add_check("Forward Pass Shape", False, f"Got {predictions.shape}, expected (10, 64)")
            
            if not torch.isnan(predictions).any():
                self.add_check("Output Validity", True)
            else:
                self.add_check("Output Validity", False, "NaN values in output")
            
        except Exception as e:
            self.add_check("Model Functionality", False, str(e))
    
    def check_performance(self):
        """Test performance requirements."""
        print("⚡ Testing Performance...")
        
        try:
            model = HyperGNN(hidden_dim=64, num_layers=2)
            gen = SyntheticDataGenerator()
            graph = gen.generate_social_network(num_nodes=20)
            node_features = torch.randn(graph.num_nodes, 64)
            
            # Warmup
            model.eval()
            with torch.no_grad():
                _ = model(graph.edge_index, node_features, graph.node_texts)
            
            # Benchmark
            times = []
            for _ in range(3):
                start = time.time()
                with torch.no_grad():
                    _ = model(graph.edge_index, node_features, graph.node_texts)
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            throughput = graph.num_nodes / avg_time
            
            if avg_time < 2.0:
                self.add_check("Inference Speed", True, f"{avg_time:.3f}s for 20 nodes")
            else:
                self.add_check("Inference Speed", False, f"Too slow: {avg_time:.3f}s")
            
            if throughput > 10.0:
                self.add_check("Throughput", True, f"{throughput:.1f} nodes/sec")
            else:
                self.add_check("Throughput", False, f"Too low: {throughput:.1f} nodes/sec")
                
        except Exception as e:
            self.add_check("Performance Test", False, str(e))
    
    def check_memory_usage(self):
        """Check memory usage is reasonable."""
        print("💾 Testing Memory Usage...")
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and use model
            model = HyperGNN(hidden_dim=128, num_layers=3)
            gen = SyntheticDataGenerator()
            graph = gen.generate_social_network(num_nodes=50)
            node_features = torch.randn(graph.num_nodes, 128)
            
            model.eval()
            with torch.no_grad():
                _ = model(graph.edge_index, node_features, graph.node_texts)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            
            if memory_used < 200:
                self.add_check("Memory Usage", True, f"{memory_used:.1f} MB")
            else:
                self.add_check("Memory Usage", False, f"High usage: {memory_used:.1f} MB", critical=False)
            
        except ImportError:
            self.add_check("Memory Test", False, "psutil not available", critical=False)
        except Exception as e:
            self.add_check("Memory Test", False, str(e), critical=False)
    
    def check_serialization(self):
        """Test model serialization/deserialization."""
        print("💾 Testing Model Serialization...")
        
        try:
            import tempfile
            
            # Create and save model
            model1 = HyperGNN(hidden_dim=64, num_layers=2)
            
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(model1.state_dict(), f.name)
                
                # Load model
                model2 = HyperGNN(hidden_dim=64, num_layers=2)
                model2.load_state_dict(torch.load(f.name))
                
                # Test equivalence
                gen = SyntheticDataGenerator(seed=42)
                graph = gen.generate_social_network(num_nodes=5)
                node_features = torch.randn(graph.num_nodes, 64)
                
                model1.eval()
                model2.eval()
                with torch.no_grad():
                    pred1 = model1(graph.edge_index, node_features, graph.node_texts)
                    pred2 = model2(graph.edge_index, node_features, graph.node_texts)
                
                if torch.allclose(pred1, pred2, atol=1e-6):
                    self.add_check("Model Serialization", True)
                else:
                    self.add_check("Model Serialization", False, "Loaded model differs")
                
                # Cleanup
                os.unlink(f.name)
                
        except Exception as e:
            self.add_check("Model Serialization", False, str(e))
    
    def check_error_handling(self):
        """Test error handling and edge cases."""
        print("🛡️  Testing Error Handling...")
        
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Test empty graph
        try:
            model(torch.empty(2, 0), torch.empty(0, 32), [])
            self.add_check("Empty Input Handling", False, "Should have raised error")
        except Exception:
            self.add_check("Empty Input Handling", True)
        
        # Test dimension mismatch
        try:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            wrong_features = torch.randn(2, 16)  # Wrong dimension
            model(edge_index, wrong_features, ["text1", "text2"])
            self.add_check("Dimension Mismatch Handling", False, "Should have raised error")
        except Exception:
            self.add_check("Dimension Mismatch Handling", True)
    
    def check_docker_compatibility(self):
        """Check if Docker environment is ready."""
        print("🐳 Checking Docker Compatibility...")
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.add_check("Docker Available", True, result.stdout.strip(), critical=False)
            else:
                self.add_check("Docker Available", False, "Docker command failed", critical=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.add_check("Docker Available", False, "Docker not found", critical=False)
    
    def check_gpu_availability(self):
        """Check GPU availability and compatibility."""
        print("🎮 Checking GPU Availability...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            self.add_check("GPU Available", True, f"{gpu_count} GPUs ({gpu_name})", critical=False)
            
            # Test GPU functionality
            try:
                model = HyperGNN(hidden_dim=32, num_layers=2).cuda()
                gen = SyntheticDataGenerator()
                graph = gen.generate_social_network(num_nodes=5)
                node_features = torch.randn(graph.num_nodes, 32).cuda()
                
                model.eval()
                with torch.no_grad():
                    predictions = model(graph.edge_index.cuda(), node_features, graph.node_texts)
                
                self.add_check("GPU Functionality", True, critical=False)
            except Exception as e:
                self.add_check("GPU Functionality", False, str(e), critical=False)
        else:
            self.add_check("GPU Available", False, "No CUDA GPUs detected", critical=False)
    
    def check_system_resources(self):
        """Check system resources."""
        print("🖥️  Checking System Resources...")
        
        try:
            import psutil
            
            # CPU count
            cpu_count = psutil.cpu_count()
            if cpu_count >= 2:
                self.add_check("CPU Count", True, f"{cpu_count} CPUs")
            else:
                self.add_check("CPU Count", False, f"Only {cpu_count} CPU", critical=False)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 4:
                self.add_check("System Memory", True, f"{memory_gb:.1f} GB")
            else:
                self.add_check("System Memory", False, f"Only {memory_gb:.1f} GB", critical=False)
            
            # Disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            if disk_free_gb >= 1:
                self.add_check("Disk Space", True, f"{disk_free_gb:.1f} GB free")
            else:
                self.add_check("Disk Space", False, f"Only {disk_free_gb:.1f} GB free", critical=False)
                
        except ImportError:
            self.add_check("System Resources", False, "psutil not available", critical=False)
    
    def run_all_checks(self):
        """Run all deployment checks."""
        print("🚀 Production Deployment Readiness Check")
        print("=" * 60)
        
        self.check_dependencies()
        self.check_model_functionality()
        self.check_performance()
        self.check_memory_usage()
        self.check_serialization()
        self.check_error_handling()
        self.check_docker_compatibility()
        self.check_gpu_availability()
        self.check_system_resources()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate deployment readiness report."""
        print("\n" + "=" * 60)
        print("📋 DEPLOYMENT READINESS REPORT")
        print("=" * 60)
        
        passed_checks = sum(1 for check in self.checks if check['passed'])
        total_checks = len(self.checks)
        critical_failures = sum(1 for check in self.checks if not check['passed'] and check['critical'])
        
        print(f"\n✅ Checks Passed: {passed_checks}/{total_checks}")
        print(f"❌ Critical Failures: {critical_failures}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        # Show critical failures
        if self.errors:
            print(f"\n🚨 CRITICAL ISSUES:")
            for error in self.errors:
                print(f"   {error}")
        
        # Show warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   {warning}")
        
        # Deployment decision
        if critical_failures == 0:
            print(f"\n🎉 DEPLOYMENT READY!")
            print("✅ All critical checks passed")
            print("🚀 HyperGNN is ready for production deployment")
            deployment_ready = True
        else:
            print(f"\n❌ DEPLOYMENT BLOCKED!")
            print(f"🚨 {critical_failures} critical issues must be resolved")
            print("🔧 Please fix the issues above before deploying")
            deployment_ready = False
        
        # Performance summary
        performance_checks = [c for c in self.checks if 'performance' in c['name'].lower() or 'speed' in c['name'].lower() or 'throughput' in c['name'].lower()]
        if performance_checks:
            print(f"\n⚡ PERFORMANCE SUMMARY:")
            for check in performance_checks:
                if check['passed']:
                    print(f"   ✅ {check['name']}: {check['details']}")
        
        return {
            'deployment_ready': deployment_ready,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'critical_failures': critical_failures,
            'warnings': len(self.warnings),
            'checks': self.checks
        }


def main():
    """Run deployment readiness check."""
    checker = DeploymentChecker()
    
    try:
        report = checker.run_all_checks()
        return 0 if report['deployment_ready'] else 1
    except Exception as e:
        print(f"\n💥 DEPLOYMENT CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit(main())