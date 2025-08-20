#!/usr/bin/env python3
"""
Production Health Check - Graph Hypernetwork Forge
Comprehensive system validation and monitoring
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess

class HealthCheckResult:
    """Health check result container"""
    
    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }

class ProductionHealthChecker:
    """Comprehensive production health checker"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results: List[HealthCheckResult] = []
    
    def check_repository_structure(self) -> HealthCheckResult:
        """Validate repository structure"""
        try:
            required_files = [
                "README.md",
                "pyproject.toml", 
                "requirements.txt",
                "graph_hypernetwork_forge/__init__.py",
                "graph_hypernetwork_forge/models/hypergnn.py",
                "scripts/demo.py",
                "tests/"
            ]
            
            missing_files = []
            existing_files = []
            
            for file_path in required_files:
                full_path = self.repo_path / file_path
                if full_path.exists():
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            if missing_files:
                return HealthCheckResult(
                    "Repository Structure",
                    False,
                    f"Missing {len(missing_files)} required files",
                    {"missing": missing_files, "existing": existing_files}
                )
            
            return HealthCheckResult(
                "Repository Structure",
                True,
                f"All {len(required_files)} required files present",
                {"files_checked": len(required_files)}
            )
            
        except Exception as e:
            return HealthCheckResult(
                "Repository Structure", 
                False,
                f"Structure check failed: {e}",
                {"error": str(e)}
            )
    
    def check_python_syntax(self) -> HealthCheckResult:
        """Check Python syntax for all Python files"""
        try:
            python_files = list(self.repo_path.glob("**/*.py"))
            syntax_errors = []
            valid_files = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    compile(content, str(py_file), 'exec')
                    valid_files.append(str(py_file.relative_to(self.repo_path)))
                    
                except SyntaxError as e:
                    syntax_errors.append({
                        "file": str(py_file.relative_to(self.repo_path)),
                        "error": str(e),
                        "line": e.lineno
                    })
                except Exception as e:
                    syntax_errors.append({
                        "file": str(py_file.relative_to(self.repo_path)),
                        "error": str(e),
                        "line": None
                    })
            
            if syntax_errors:
                return HealthCheckResult(
                    "Python Syntax",
                    False,
                    f"Syntax errors in {len(syntax_errors)} files",
                    {"errors": syntax_errors, "valid_files": len(valid_files)}
                )
            
            return HealthCheckResult(
                "Python Syntax",
                True,
                f"All {len(valid_files)} Python files have valid syntax",
                {"files_checked": len(valid_files)}
            )
            
        except Exception as e:
            return HealthCheckResult(
                "Python Syntax",
                False,
                f"Syntax check failed: {e}",
                {"error": str(e)}
            )
    
    def check_imports(self) -> HealthCheckResult:
        """Check critical imports"""
        try:
            critical_imports = [
                ("json", "JSON processing"),
                ("os", "Operating system interface"),
                ("sys", "System-specific parameters"),
                ("pathlib", "Path handling"),
                ("typing", "Type hints")
            ]
            
            import_results = []
            failed_imports = []
            
            for module, description in critical_imports:
                try:
                    __import__(module)
                    import_results.append({"module": module, "status": "success", "description": description})
                except ImportError as e:
                    failed_imports.append({"module": module, "error": str(e), "description": description})
            
            # Test optional ML dependencies (graceful degradation)
            optional_imports = [
                ("torch", "PyTorch deep learning"),
                ("transformers", "Hugging Face transformers"),
                ("sentence_transformers", "Sentence embeddings"),
                ("torch_geometric", "Graph neural networks")
            ]
            
            optional_results = []
            for module, description in optional_imports:
                try:
                    __import__(module)
                    optional_results.append({"module": module, "status": "available", "description": description})
                except ImportError:
                    optional_results.append({"module": module, "status": "missing", "description": description})
            
            if failed_imports:
                return HealthCheckResult(
                    "Import Check",
                    False,
                    f"Failed to import {len(failed_imports)} critical modules",
                    {
                        "failed_critical": failed_imports,
                        "successful_critical": import_results,
                        "optional_status": optional_results
                    }
                )
            
            return HealthCheckResult(
                "Import Check",
                True,
                f"All {len(critical_imports)} critical imports successful",
                {
                    "critical_imports": import_results,
                    "optional_status": optional_results
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                "Import Check",
                False,
                f"Import check failed: {e}",
                {"error": str(e)}
            )
    
    def check_demo_functionality(self) -> HealthCheckResult:
        """Test demo script functionality"""
        try:
            demo_script = self.repo_path / "simple_production_demo.py"
            
            if not demo_script.exists():
                return HealthCheckResult(
                    "Demo Functionality",
                    False,
                    "Demo script not found",
                    {"expected_path": str(demo_script)}
                )
            
            # Test syntax first
            with open(demo_script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, str(demo_script), 'exec')
            
            # Try to run demo in a safe environment
            try:
                result = subprocess.run(
                    [sys.executable, str(demo_script)],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return HealthCheckResult(
                        "Demo Functionality",
                        True,
                        "Demo script executed successfully",
                        {
                            "exit_code": result.returncode,
                            "output_lines": len(result.stdout.split('\n')),
                            "contains_success": "Demo Completed Successfully" in result.stdout
                        }
                    )
                else:
                    return HealthCheckResult(
                        "Demo Functionality",
                        False,
                        f"Demo script failed with exit code {result.returncode}",
                        {
                            "exit_code": result.returncode,
                            "stderr": result.stderr[:500],
                            "stdout": result.stdout[:500]
                        }
                    )
                    
            except subprocess.TimeoutExpired:
                return HealthCheckResult(
                    "Demo Functionality",
                    False,
                    "Demo script timed out after 30 seconds",
                    {"timeout": 30}
                )
                
        except Exception as e:
            return HealthCheckResult(
                "Demo Functionality",
                False,
                f"Demo test failed: {e}",
                {"error": str(e)}
            )
    
    def check_configuration_files(self) -> HealthCheckResult:
        """Validate configuration files"""
        try:
            config_checks = []
            
            # Check pyproject.toml
            pyproject_path = self.repo_path / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    import tomllib
                    with open(pyproject_path, 'rb') as f:
                        config = tomllib.load(f)
                    
                    required_sections = ["project", "build-system"]
                    missing_sections = [s for s in required_sections if s not in config]
                    
                    if missing_sections:
                        config_checks.append({
                            "file": "pyproject.toml",
                            "status": "incomplete",
                            "missing_sections": missing_sections
                        })
                    else:
                        config_checks.append({
                            "file": "pyproject.toml",
                            "status": "valid",
                            "sections": list(config.keys())
                        })
                        
                except Exception as e:
                    config_checks.append({
                        "file": "pyproject.toml",
                        "status": "error",
                        "error": str(e)
                    })
            else:
                config_checks.append({
                    "file": "pyproject.toml",
                    "status": "missing"
                })
            
            # Check requirements.txt
            requirements_path = self.repo_path / "requirements.txt"
            if requirements_path.exists():
                try:
                    with open(requirements_path, 'r') as f:
                        requirements = f.read().strip().split('\n')
                    
                    valid_requirements = [r for r in requirements if r.strip() and not r.startswith('#')]
                    
                    config_checks.append({
                        "file": "requirements.txt",
                        "status": "valid",
                        "requirement_count": len(valid_requirements)
                    })
                    
                except Exception as e:
                    config_checks.append({
                        "file": "requirements.txt",
                        "status": "error",
                        "error": str(e)
                    })
            else:
                config_checks.append({
                    "file": "requirements.txt",
                    "status": "missing"
                })
            
            # Check for critical errors
            critical_errors = [c for c in config_checks if c["status"] in ["missing", "error"]]
            
            if critical_errors:
                return HealthCheckResult(
                    "Configuration Files",
                    False,
                    f"Issues with {len(critical_errors)} configuration files",
                    {"checks": config_checks, "critical_errors": len(critical_errors)}
                )
            
            return HealthCheckResult(
                "Configuration Files",
                True,
                "All configuration files are valid",
                {"checks": config_checks}
            )
            
        except Exception as e:
            return HealthCheckResult(
                "Configuration Files",
                False,
                f"Configuration check failed: {e}",
                {"error": str(e)}
            )
    
    def check_security_basics(self) -> HealthCheckResult:
        """Basic security checks"""
        try:
            security_issues = []
            security_good = []
            
            # Check for hardcoded secrets (basic patterns)
            secret_patterns = [
                ("password", r"password\s*=\s*['\"][^'\"]{3,}['\"]"),
                ("api_key", r"api[_-]?key\s*=\s*['\"][^'\"]{10,}['\"]"),
                ("secret", r"secret\s*=\s*['\"][^'\"]{8,}['\"]"),
                ("token", r"token\s*=\s*['\"][^'\"]{10,}['\"]")
            ]
            
            python_files = list(self.repo_path.glob("**/*.py"))
            
            for py_file in python_files[:10]:  # Limit to first 10 files for performance
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_name, pattern in secret_patterns:
                        import re
                        if re.search(pattern, content, re.IGNORECASE):
                            security_issues.append({
                                "file": str(py_file.relative_to(self.repo_path)),
                                "issue": f"Potential hardcoded {pattern_name}",
                                "severity": "medium"
                            })
                        
                except Exception:
                    continue
            
            # Check file permissions (if on Unix)
            if os.name == 'posix':
                try:
                    executable_files = []
                    for py_file in python_files[:5]:  # Check first 5
                        if os.access(py_file, os.X_OK):
                            executable_files.append(str(py_file.relative_to(self.repo_path)))
                    
                    if executable_files:
                        security_good.append({
                            "check": "executable_permissions",
                            "status": "good", 
                            "files": len(executable_files)
                        })
                        
                except Exception:
                    pass
            
            # Check for .gitignore
            gitignore_path = self.repo_path / ".gitignore"
            if gitignore_path.exists():
                security_good.append({
                    "check": "gitignore_present",
                    "status": "good"
                })
            else:
                security_issues.append({
                    "file": ".gitignore",
                    "issue": "Missing .gitignore file",
                    "severity": "low"
                })
            
            if security_issues:
                high_severity = [i for i in security_issues if i.get("severity") == "high"]
                if high_severity:
                    return HealthCheckResult(
                        "Security Basics",
                        False,
                        f"Found {len(high_severity)} high-severity security issues",
                        {"issues": security_issues, "good_practices": security_good}
                    )
            
            return HealthCheckResult(
                "Security Basics",
                True,
                f"Basic security check passed with {len(security_good)} good practices",
                {"issues": security_issues, "good_practices": security_good}
            )
            
        except Exception as e:
            return HealthCheckResult(
                "Security Basics",
                False,
                f"Security check failed: {e}",
                {"error": str(e)}
            )
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        print("🏥 Running Production Health Checks...")
        print("=" * 50)
        
        checks = [
            self.check_repository_structure,
            self.check_python_syntax,
            self.check_imports,
            self.check_configuration_files,
            self.check_demo_functionality,
            self.check_security_basics
        ]
        
        self.results = []
        
        for check_func in checks:
            print(f"\n🔍 Running {check_func.__name__.replace('check_', '').replace('_', ' ').title()}...")
            
            try:
                result = check_func()
                self.results.append(result)
                
                status_icon = "✅" if result.passed else "❌"
                print(f"{status_icon} {result.name}: {result.message}")
                
                if not result.passed and result.details:
                    print(f"   Details: {result.details}")
                    
            except Exception as e:
                error_result = HealthCheckResult(
                    check_func.__name__,
                    False,
                    f"Check failed with exception: {e}",
                    {"exception": str(e)}
                )
                self.results.append(error_result)
                print(f"❌ {check_func.__name__}: Exception - {e}")
        
        # Summary
        passed_checks = [r for r in self.results if r.passed]
        failed_checks = [r for r in self.results if not r.passed]
        
        print(f"\n📊 Health Check Summary")
        print(f"   ✅ Passed: {len(passed_checks)}/{len(self.results)}")
        print(f"   ❌ Failed: {len(failed_checks)}/{len(self.results)}")
        
        overall_health = len(failed_checks) == 0
        
        if overall_health:
            print(f"\n🎉 System is healthy and ready for production!")
        else:
            print(f"\n⚠️  System needs attention before production deployment")
            print(f"Failed checks: {[r.name for r in failed_checks]}")
        
        return {
            "overall_health": overall_health,
            "total_checks": len(self.results),
            "passed_checks": len(passed_checks),
            "failed_checks": len(failed_checks),
            "results": [r.to_dict() for r in self.results],
            "timestamp": time.time()
        }
    
    def save_report(self, output_path: str = "health_check_report.json"):
        """Save health check report"""
        report_data = {
            "metadata": {
                "timestamp": time.time(),
                "repo_path": str(self.repo_path),
                "python_version": sys.version,
                "platform": sys.platform
            },
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Health check report saved to: {output_path}")

def main():
    """Main health check execution"""
    try:
        checker = ProductionHealthChecker()
        results = checker.run_all_checks()
        checker.save_report()
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_health"] else 1)
        
    except Exception as e:
        print(f"❌ Health check system failed: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()