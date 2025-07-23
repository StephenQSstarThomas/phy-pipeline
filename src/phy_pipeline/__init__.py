"""
PHY-Pipeline: AI-powered physics problem processing pipeline.

This package provides tools for converting PDF physics problems to structured JSON format,
including question-answer matching, content processing, and deduplication.
"""

__version__ = "0.1.0"
__author__ = "PHYBench Team"
__email__ = "contact@phybench.com"

from .qa_matcher import QAMatcher, QAPair
from .simple_qa_processor import SimpleQAProcessor
from .mixed_qa_processor import MixedQAProcessor

__all__ = [
    "QAMatcher",
    "QAPair", 
    "SimpleQAProcessor",
    "MixedQAProcessor",
] 