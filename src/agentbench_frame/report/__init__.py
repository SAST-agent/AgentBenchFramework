"""
Report generation for AgentBench.

Provides:
- ReportBuilder: Static site generator from experiment data
"""

from agentbench_frame.report.builder import ReportBuilder, build_report

__all__ = ["ReportBuilder", "build_report"]
