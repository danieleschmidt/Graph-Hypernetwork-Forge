#!/usr/bin/env python3
"""
Automated metrics collection script for Graph Hypernetwork Forge.
Collects and reports various project health metrics.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MetricsCollector:
    """Collects various project metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics_file = project_root / ".github" / "project-metrics.json"
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "development": self.collect_development_metrics(),
            "quality": self.collect_quality_metrics(),
            "performance": self.collect_performance_metrics(),
            "community": self.collect_community_metrics(),
            "ci_cd": self.collect_ci_cd_metrics(),
        }
        return metrics
    
    def collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development-related metrics."""
        metrics = {}
        
        # Lines of code
        try:
            result = subprocess.run(
                ["find", "graph_hypernetwork_forge", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                lines = sum(int(line.split()[0]) for line in result.stdout.strip().split('\n') if line.strip())
                metrics["lines_of_code"] = lines
        except Exception as e:
            print(f"Warning: Could not collect lines of code: {e}")
            
        # Test coverage
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=graph_hypernetwork_forge", "--cov-report=json", "tests/"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        metrics["test_coverage"] = round(coverage_data["totals"]["percent_covered"], 2)
        except Exception as e:
            print(f"Warning: Could not collect test coverage: {e}")
            
        # Technical debt (TODO/FIXME count)
        try:
            result = subprocess.run(
                ["grep", "-r", "-i", "TODO\\|FIXME\\|XXX\\|HACK", "graph_hypernetwork_forge/", "--include=*.py"],
                capture_output=True, text=True, cwd=self.project_root
            )
            todo_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics["technical_debt_items"] = todo_count
        except Exception as e:
            print(f"Warning: Could not collect technical debt: {e}")
            
        return metrics
    
    def collect_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Linting issues
        try:
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "graph_hypernetwork_forge/", "--format=json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.stdout.strip():
                linting_data = json.loads(result.stdout)
                metrics["linting_issues"] = len(linting_data)
            else:
                metrics["linting_issues"] = 0
        except Exception as e:
            print(f"Warning: Could not collect linting metrics: {e}")
            
        # Security issues
        try:
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", "graph_hypernetwork_forge/", "-f", "json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.stdout.strip():
                bandit_data = json.loads(result.stdout)
                metrics["security_issues"] = len(bandit_data.get("results", []))
            else:
                metrics["security_issues"] = 0
        except Exception as e:
            print(f"Warning: Could not collect security metrics: {e}")
            
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        # Run performance benchmarks
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-m", "performance", "--benchmark-json=benchmark.json"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            benchmark_file = self.project_root / "benchmark.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                    benchmarks = benchmark_data.get("benchmarks", [])
                    if benchmarks:
                        avg_time = sum(b["stats"]["mean"] for b in benchmarks) / len(benchmarks)
                        metrics["average_benchmark_time"] = round(avg_time * 1000, 2)  # Convert to ms
        except Exception as e:
            print(f"Warning: Could not collect performance metrics: {e}")
            
        return metrics
    
    def collect_community_metrics(self) -> Dict[str, Any]:
        """Collect community and GitHub metrics."""
        metrics = {}
        
        try:
            # GitHub stars, forks, issues, PRs
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "stargazerCount,forkCount,issues,pullRequests"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                github_data = json.loads(result.stdout)
                metrics["github_stars"] = github_data.get("stargazerCount", 0)
                metrics["github_forks"] = github_data.get("forkCount", 0)
                metrics["open_issues"] = len(github_data.get("issues", []))
                metrics["open_pull_requests"] = len(github_data.get("pullRequests", []))
        except Exception as e:
            print(f"Warning: Could not collect GitHub metrics: {e}")
            
        # Contributors
        try:
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                contributors = len(result.stdout.strip().split('\n'))
                metrics["contributors"] = contributors
        except Exception as e:
            print(f"Warning: Could not collect contributor metrics: {e}")
            
        return metrics
    
    def collect_ci_cd_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD metrics."""
        metrics = {}
        
        try:
            # Recent workflow runs
            result = subprocess.run(
                ["gh", "run", "list", "--limit", "50", "--json", "status,conclusion"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                runs_data = json.loads(result.stdout)
                completed_runs = [r for r in runs_data if r["status"] == "completed"]
                if completed_runs:
                    successful_runs = [r for r in completed_runs if r["conclusion"] == "success"]
                    success_rate = (len(successful_runs) / len(completed_runs)) * 100
                    metrics["build_success_rate"] = round(success_rate, 2)
        except Exception as e:
            print(f"Warning: Could not collect CI/CD metrics: {e}")
            
        return metrics
    
    def update_metrics_file(self, new_metrics: Dict[str, Any]) -> None:
        """Update the project metrics file."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
            
        # Update metrics while preserving structure
        if "metrics" not in existing_data:
            existing_data["metrics"] = {}
            
        for category, category_metrics in new_metrics.items():
            if category == "timestamp":
                existing_data["last_updated"] = category_metrics
            else:
                existing_data["metrics"][category] = category_metrics
                
        # Write updated metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Updated metrics file: {self.metrics_file}")
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report."""
        report = []
        report.append("# Project Metrics Report")
        report.append(f"Generated: {metrics['timestamp']}")
        report.append("")
        
        for category, category_metrics in metrics.items():
            if category == "timestamp":
                continue
                
            report.append(f"## {category.title()} Metrics")
            for metric, value in category_metrics.items():
                report.append(f"- {metric.replace('_', ' ').title()}: {value}")
            report.append("")
            
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--output", choices=["json", "report"], default="json",
                       help="Output format")
    parser.add_argument("--update-file", action="store_true",
                       help="Update the project metrics file")
    parser.add_argument("--category", choices=["development", "quality", "performance", "community", "ci_cd"],
                       help="Collect specific category only")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    collector = MetricsCollector(project_root)
    
    if args.category:
        method_name = f"collect_{args.category}_metrics"
        if hasattr(collector, method_name):
            metrics = {args.category: getattr(collector, method_name)()}
        else:
            print(f"Error: Unknown category: {args.category}")
            sys.exit(1)
    else:
        metrics = collector.collect_all_metrics()
    
    if args.output == "json":
        print(json.dumps(metrics, indent=2))
    else:
        print(collector.generate_report(metrics))
    
    if args.update_file:
        collector.update_metrics_file(metrics)


if __name__ == "__main__":
    main()