#!/usr/bin/env python
"""
GLUE Benchmarks Example - Domain-Aware Model Selection for GLUE Datasets

This script demonstrates how to use CDDM to run benchmarks
on the GLUE datasets with domain-aware model selection.
"""

import os
import argparse
import logging
import sys
import glob
import pandas as pd

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autopytorch_integration import ContextualAutoPyTorch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("glue_benchmarks.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("glue_benchmarks")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run domain-aware AutoPyTorch on GLUE datasets')
    parser.add_argument('--glue_dir', type=str, default='/home/safikhani/main_repository/Auto-PyTorch_autoNLP/data/GLUE',
                        help='Directory containing GLUE datasets')
    parser.add_argument('--corpus_path', type=str, default='domain_classified_corpus.csv',
                        help='Path to the domain-classified corpus file')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process (e.g., "COLA")')
    parser.add_argument('--all', action='store_true', help='Process all GLUE datasets')
    parser.add_argument('--build_corpus', action='store_true', help='Build and classify a new corpus')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory to store cached models')
    parser.add_argument('--time_limit', type=int, default=1200,
                        help='Time limit for AutoPyTorch in seconds')
    return parser.parse_args()

def find_standardized_datasets(glue_dir):
    """Find all standardized GLUE datasets."""
    datasets = []
    
    # Check each subdirectory in GLUE directory
    for dataset_name in os.listdir(glue_dir):
        dataset_dir = os.path.join(glue_dir, dataset_name)
        if os.path.isdir(dataset_dir):
            # Look for standardized file
            std_file = os.path.join(dataset_dir, f"{dataset_name.lower()}_standardized.csv")
            if os.path.exists(std_file):
                datasets.append({
                    'name': dataset_name,
                    'path': std_file
                })
    
    logger.info(f"Found {len(datasets)} standardized GLUE datasets")
    return datasets

def build_and_classify_corpus(contextual_aml, args):
    """Build and classify a new corpus of models."""
    logger.info("Building corpus of text classification models")
    
    # Build corpus
    corpus_df = contextual_aml.build_corpus(output_file="text_classification_models.csv")
    
    # Clean corpus
    cleaned_df = contextual_aml.clean_corpus(
        corpus_df=corpus_df,
        output_file="cleaned_corpus.csv"
    )
    
    # Classify corpus
    classified_df = contextual_aml.classify_corpus(
        corpus_df=cleaned_df,
        output_file=args.corpus_path
    )
    
    logger.info(f"Corpus built and classified with {len(classified_df)} models")
    return classified_df

def main():
    args = parse_arguments()
    
    # Initialize ContextualAutoPyTorch
    if os.path.exists(args.corpus_path):
        logger.info(f"Using existing corpus: {args.corpus_path}")
        contextual_aml = ContextualAutoPyTorch(
            corpus_path=args.corpus_path,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir
        )
    else:
        logger.info("No corpus file found. Initializing without corpus.")
        contextual_aml = ContextualAutoPyTorch(
            cache_dir=args.cache_dir,
            results_dir=args.results_dir
        )
    
    # Build corpus if requested
    if args.build_corpus:
        build_and_classify_corpus(contextual_aml, args)
    
    # Find datasets
    datasets = find_standardized_datasets(args.glue_dir)
    
    # Filter by specific dataset if provided
    if args.dataset and not args.all:
        datasets = [d for d in datasets if d['name'].upper() == args.dataset.upper()]
        if not datasets:
            logger.error(f"Dataset {args.dataset} not found")
            return
    
    # Process datasets
    logger.info(f"Processing {len(datasets)} datasets")
    
    for dataset_info in datasets:
        try:
            logger.info(f"Processing dataset: {dataset_info['name']}")
            
            # Process dataset
            result = contextual_aml.process_dataset(
                dataset_path=dataset_info['path'],
                text_column='text',
                label_column='label'
            )
            
            logger.info(f"Completed {dataset_info['name']} with accuracy: {result['metrics']['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_info['name']}: {str(e)}")
    
    # Create final summary
    contextual_aml._create_benchmark_summary()
    logger.info("Benchmark completed")

if __name__ == "__main__":
    main() 