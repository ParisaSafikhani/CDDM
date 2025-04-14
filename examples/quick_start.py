#!/usr/bin/env python
"""
CDDM Quick Start Guide

This script demonstrates how to use CDDM for domain-aware model selection.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cddm import ContextualAutoPyTorch

def create_sample_dataset(filename="sample_sentiment.csv"):
    """Create a sample sentiment analysis dataset."""
    print("Creating sample sentiment analysis dataset...")
    
    # Create a sample dataset
    data = {
        'text': [
            "I absolutely loved this movie! The acting was superb.",
            "The food at this restaurant was delicious and the service was excellent.",
            "This book was a complete waste of time. The plot made no sense.",
            "The customer service was terrible and the product arrived damaged.",
            "I highly recommend this hotel, the rooms were clean and comfortable.",
            "This concert was the best I've ever been to, amazing performance!",
            "The software is full of bugs and crashes constantly.",
            "I'm very disappointed with the quality of this product.",
            "The vacation was wonderful, beautiful beaches and friendly locals.",
            "This course was very informative and well-structured."
        ],
        'label': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  # 1 for positive, 0 for negative
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    return filename

def build_corpus(cddm, output_file="domain_classified_corpus.csv", max_models=100):
    """Build and classify a corpus of models."""
    print("\nBuilding a corpus of text classification models...")
    print("Note: This may take some time.")
    
    # Set max models to a lower number for demonstration
    cddm.corpus_builder.max_models = max_models
    
    # Build corpus
    corpus_df = cddm.build_corpus(output_file="text_classification_models.csv")
    
    # Clean corpus
    cleaned_df = cddm.clean_corpus(corpus_df=corpus_df, output_file="cleaned_corpus.csv")
    
    # Classify corpus
    classified_df = cddm.classify_corpus(corpus_df=cleaned_df, output_file=output_file)
    
    print(f"Corpus built and saved to {output_file}")
    return classified_df

def classify_and_select_model(cddm, dataset_path):
    """Classify dataset and select an appropriate model."""
    print(f"\nClassifying dataset: {dataset_path}")
    
    # Classify dataset
    dataset_domain = cddm.classify_dataset(
        dataset_path=dataset_path,
        text_column="text",
        label_column="label",
        use_zero_shot=True
    )
    
    print(f"Dataset Domain: {dataset_domain['primary_domain']}")
    print(f"Domain Score: {dataset_domain['primary_domain_score']:.4f}")
    
    # Select model
    print("\nSelecting appropriate model...")
    model_info = cddm.select_model_for_dataset(
        dataset_path=dataset_path,
        text_column="text",
        label_column="label"
    )
    
    print(f"Selected Model: {model_info['model_id']}")
    print(f"Selection Method: {model_info['selection_method']}")
    print(f"Domain: {model_info['domain']}")
    print(f"Domain Score: {model_info['domain_score']:.4f}")
    
    return model_info

def process_with_autopytorch(cddm, dataset_path):
    """Process dataset with AutoPyTorch integration."""
    print("\nProcessing dataset with AutoPyTorch integration...")
    
    try:
        # Process dataset end-to-end
        result = cddm.process_dataset(
            dataset_path=dataset_path,
            text_column="text",
            label_column="label"
        )
        
        print("\nPerformance Metrics:")
        for metric_name, metric_value in result['metrics'].items():
            print(f"{metric_name}: {metric_value:.4f}")
            
        return result
    except Exception as e:
        print(f"Error running end-to-end process: {str(e)}")
        print("This may be due to missing AutoPyTorch dependency.")
        return None

def process_with_sklearn(cddm, dataset_path, model_id):
    """Process dataset with scikit-learn."""
    print("\nProcessing dataset with scikit-learn...")
    
    # Generate embeddings
    embeddings, labels = cddm.generate_embeddings(
        dataset_path=dataset_path,
        text_column="text",
        label_column="label",
        model_id=model_id
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42
    )
    
    # Train a simple model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy

def main():
    """Main function demonstrating CDDM workflow."""
    print("CDDM Quick Start Guide")
    print("======================")
    
    # Initialize CDDM
    print("Initializing CDDM...")
    cddm = ContextualAutoPyTorch(
        cache_dir="cache",
        results_dir="results"
    )
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    # Check if corpus exists
    corpus_path = "domain_classified_corpus.csv"
    if not os.path.exists(corpus_path):
        build_corpus(cddm, corpus_path)
    else:
        print(f"\nUsing existing corpus: {corpus_path}")
    
    # Classify dataset and select model
    model_info = classify_and_select_model(cddm, dataset_path)
    
    # Try processing with AutoPyTorch
    autopytorch_result = process_with_autopytorch(cddm, dataset_path)
    
    # Process with scikit-learn
    if autopytorch_result is None:
        sklearn_accuracy = process_with_sklearn(cddm, dataset_path, model_info['model_id'])
    
    print("\nWorkflow Completed!")
    print("===================")

if __name__ == "__main__":
    main() 