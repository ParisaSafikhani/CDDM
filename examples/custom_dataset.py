#!/usr/bin/env python
"""
CDDM - Corpus-Driven Domain Mapping

This package implements the methodology described in:
"AutoML Meets Hugging Face: Domain-Aware Pretrained Model Selection for Text Classification"
"""

import os
import argparse
import logging
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cddm import ContextualAutoPyTorch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("custom_dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("custom_dataset")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run domain-aware AutoPyTorch on a custom dataset')
    parser.add_argument('--dataset_path', type=str, help='Path to custom dataset CSV')
    parser.add_argument('--corpus_path', type=str, default='domain_classified_corpus.csv',
                        help='Path to the domain-classified corpus file')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of the text column in the dataset')
    parser.add_argument('--label_column', type=str, default='label',
                        help='Name of the label column in the dataset')
    parser.add_argument('--use_20newsgroups', action='store_true',
                       help='Use 20 Newsgroups dataset instead of custom dataset')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory to store cached models')
    parser.add_argument('--time_limit', type=int, default=1200,
                        help='Time limit for AutoPyTorch in seconds')
    return parser.parse_args()

def create_20newsgroups_dataset(output_path: str = "20newsgroups.csv"):
    """Create a CSV file from the 20 Newsgroups dataset."""
    logger.info("Creating 20 Newsgroups dataset")
    
    # Load 20 Newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': newsgroups.data,
        'label': newsgroups.target
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Created 20 Newsgroups dataset with {len(df)} samples and saved to {output_path}")
    
    return output_path

def main():
    args = parse_arguments()
    
    # Determine dataset path
    if args.use_20newsgroups:
        dataset_path = create_20newsgroups_dataset()
    elif args.dataset_path:
        dataset_path = args.dataset_path
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return
    else:
        logger.error("Please provide a dataset path or use --use_20newsgroups")
        return
    
    # Initialize ContextualAutoPyTorch
    if os.path.exists(args.corpus_path):
        logger.info(f"Using existing corpus: {args.corpus_path}")
        contextual_aml = ContextualAutoPyTorch(
            corpus_path=args.corpus_path,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir
        )
    else:
        logger.warning(f"Corpus file not found: {args.corpus_path}")
        logger.info("Building a new corpus...")
        
        # Initialize without corpus first
        contextual_aml = ContextualAutoPyTorch(
            cache_dir=args.cache_dir,
            results_dir=args.results_dir
        )
        
        # Build and classify corpus
        logger.info("Building corpus of text classification models")
        corpus_df = contextual_aml.build_corpus(output_file="text_classification_models.csv")
        cleaned_df = contextual_aml.clean_corpus(corpus_df=corpus_df, output_file="cleaned_corpus.csv")
        classified_df = contextual_aml.classify_corpus(corpus_df=cleaned_df, output_file=args.corpus_path)
        
        # Reinitialize with the new corpus
        contextual_aml = ContextualAutoPyTorch(
            corpus_path=args.corpus_path,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir
        )
    
    # Process dataset
    logger.info(f"Processing dataset: {dataset_path}")
    
    try:
        # Step 1: Classify dataset to identify domain
        dataset_domain = contextual_aml.classify_dataset(
            dataset_path=dataset_path, 
            text_column=args.text_column,
            label_column=args.label_column,
            use_zero_shot=True
        )
        
        logger.info(f"Dataset domain: {dataset_domain['primary_domain']} "
                  f"(score: {dataset_domain['primary_domain_score']:.4f})")
        
        # Step 2: Select appropriate model based on domain
        model_info = contextual_aml.select_model_for_dataset(
            dataset_path=dataset_path,
            text_column=args.text_column,
            label_column=args.label_column
        )
        
        logger.info(f"Selected model: {model_info['model_id']} "
                  f"(method: {model_info['selection_method']})")
        
        # Step 3: Process dataset end-to-end
        result = contextual_aml.process_dataset(
            dataset_path=dataset_path,
            text_column=args.text_column,
            label_column=args.label_column
        )
        
        # Log results
        logger.info(f"Completed processing with the following metrics:")
        for metric_name, metric_value in result['metrics'].items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
            
        logger.info(f"Results saved to {contextual_aml.run_dir}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 