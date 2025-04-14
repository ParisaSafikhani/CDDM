"""
ContextualAutoPyTorch - Domain-Aware Pretrained Model Selection for AutoML

This package implements the methodology described in:
"AutoML Meets Hugging Face: Domain-Aware Pretrained Model Selection for Text Classification"
"""

from .corpus_builder import HuggingFaceCorpusBuilder
from .domain_classifier import DomainClassifier
from .model_selector import ModelSelector
from .autopytorch_integration import ContextualAutoPyTorch

__version__ = "0.1.0"