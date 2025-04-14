"""
Corpus Builder - Extracts and processes models from Hugging Face Hub

This module implements the first phase of the methodology described in
Section 3.1 of the paper: Pre-trained Model Repository Integration.
"""

import os
import pandas as pd
import numpy as np
from huggingface_hub import HfApi
import time
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("corpus_builder")

class HuggingFaceCorpusBuilder:
    """
    Builds a corpus of pre-trained models from Hugging Face Hub with
    metadata extraction, dataset identification, and domain mapping.
    """
    
    def __init__(self, 
                 cache_dir: str = "cache", 
                 output_dir: str = "corpus",
                 max_models: int = 80000,
                 request_delay: float = 0.5):
        """
        Initialize the corpus builder.
        
        Args:
            cache_dir: Directory to store temporary files
            output_dir: Directory to save the corpus
            max_models: Maximum number of models to retrieve
            request_delay: Delay between API requests to avoid rate limiting
        """
        self.api = HfApi()
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.max_models = max_models
        self.request_delay = request_delay
        
        # Create directories if they don't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset keywords for identification
        self.dataset_keywords = ['dataset', 'corpus', 'data']
        
        logger.info(f"Initialized HuggingFaceCorpusBuilder with output to {output_dir}")
    
    def fetch_text_classification_models(self) -> List[Any]:
        """
        Retrieve text classification models from Hugging Face Hub.
        
        Returns:
            List of model objects from Hugging Face API
        """
        logger.info(f"Fetching up to {self.max_models} text classification models from Hugging Face Hub")
        models = self.api.list_models(filter="text-classification", limit=self.max_models)
        model_list = list(models)
        logger.info(f"Retrieved {len(model_list)} models")
        return model_list
    
    def extract_dataset_info_from_tags(self, tags: List[str]) -> List[str]:
        """
        Extract dataset names from model tags.
        
        Args:
            tags: List of tags associated with a model
            
        Returns:
            List of identified dataset names
        """
        datasets = []
        for tag in tags:
            tag_lower = tag.lower()
            if any(keyword in tag_lower for keyword in self.dataset_keywords):
                cleaned_tag = tag_lower
                for keyword in self.dataset_keywords:
                    cleaned_tag = cleaned_tag.replace(f"{keyword}:", "").replace(f"{keyword}-", "")
                    if cleaned_tag.endswith(keyword):
                        cleaned_tag = cleaned_tag[:len(cleaned_tag)-len(keyword)]
                cleaned_tag = cleaned_tag.strip()
                if cleaned_tag:
                    datasets.append(cleaned_tag)
        return datasets
    
    def get_dataset_description(self, dataset_name: str) -> str:
        """
        Retrieve dataset description from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Description string or error message
        """
        try:
            dataset_info = self.api.dataset_info(dataset_name)
            return dataset_info.description[:500] + '...' if dataset_info.description and len(dataset_info.description) > 500 else dataset_info.description or 'No description available'
        except Exception as e:
            return f"Error fetching info: {str(e)[:100]}"
    
    def extract_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Extract comprehensive information about a model.
        
        Args:
            model: Model object from Hugging Face API
            
        Returns:
            Dictionary containing model metadata
        """
        # Basic model info
        info = {
            'id': getattr(model, 'id', ''),
            'author': getattr(model, 'author', ''),
            'created_at': getattr(model, 'created_at', ''),
            'last_modified': getattr(model, 'last_modified', ''),
            'downloads': getattr(model, 'downloads', 0),
            'likes': getattr(model, 'likes', 0),
            'tags': ', '.join(model.tags) if getattr(model, 'tags', None) else '',
            'pipeline_tag': getattr(model, 'pipeline_tag', ''),
        }

        # Extract datasets from tags
        datasets = self.extract_dataset_info_from_tags(model.tags) if getattr(model, 'tags', None) else []
        info['datasets'] = ', '.join(datasets) if datasets else 'Unknown'
        
        # Get dataset descriptions
        dataset_descriptions = {}
        for dataset in datasets:
            if dataset not in dataset_descriptions:
                dataset_descriptions[dataset] = self.get_dataset_description(dataset)
                time.sleep(self.request_delay)  # Avoid rate limiting
        
        info['dataset_info'] = ' | '.join([f"{dataset}: {description}" 
                                          for dataset, description in dataset_descriptions.items()])
        
        # Model card data (if available)
        if hasattr(model, 'card_data'):
            card_fields = [
                'description', 'language', 'license', 'tags', 'metrics',
                'model_type', 'use_cases', 'limitations'
            ]
            for field in card_fields:
                info[f'card_{field}'] = getattr(model.card_data, field, '')
        
        return info
    
    def build_corpus(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Build the complete corpus by processing all models.
        
        Args:
            output_file: Optional filename for saving the corpus
            
        Returns:
            DataFrame containing the complete corpus
        """
        models = self.fetch_text_classification_models()
        
        # Process each model
        data = []
        for model in tqdm(models, desc="Processing models"):
            try:
                model_info = self.extract_model_info(model)
                data.append(model_info)
            except Exception as e:
                logger.error(f"Error processing model {getattr(model, 'id', 'Unknown')}: {str(e)}")
            time.sleep(self.request_delay)  # Avoid rate limiting
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            df.to_csv(output_path, index=False)
            logger.info(f"Corpus saved to {output_path} with {len(df)} models and {len(df.columns)} features")
        else:
            output_path = os.path.join(self.output_dir, "text_classification_models.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Corpus saved to {output_path} with {len(df)} models and {len(df.columns)} features")
        
        return df
    
    def clean_corpus(self, df: Optional[pd.DataFrame] = None, input_file: Optional[str] = None, 
                    output_file: str = "cleaned_corpus.csv") -> pd.DataFrame:
        """
        Clean and preprocess the corpus.
        
        Args:
            df: DataFrame containing the corpus (if None, load from input_file)
            input_file: Input file path (if df is None)
            output_file: Output file path for saving the cleaned corpus
            
        Returns:
            Cleaned DataFrame
        """
        # Load data if not provided
        if df is None and input_file:
            input_path = os.path.join(self.output_dir, input_file)
            df = pd.read_csv(input_path)
            logger.info(f"Loaded corpus from {input_path}")
        elif df is None:
            raise ValueError("Either df or input_file must be provided")
        
        # Cleaning steps
        # 1. Remove rows with no dataset information
        df_cleaned = df.dropna(subset=['datasets']).copy()
        
        # 2. Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates(subset=['id'])
        
        # 3. Filter out rows with minimal information
        df_cleaned = df_cleaned[df_cleaned['datasets'] != 'Unknown']
        
        # Save cleaned corpus
        output_path = os.path.join(self.output_dir, output_file)
        df_cleaned.to_csv(output_path, index=False)
        logger.info(f"Cleaned corpus saved to {output_path} with {len(df_cleaned)} models")
        
        return df_cleaned
    
    def analyze_corpus(self, df: Optional[pd.DataFrame] = None, input_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the corpus to extract statistics.
        
        Args:
            df: DataFrame containing the corpus (if None, load from input_file)
            input_file: Input file path (if df is None)
            
        Returns:
            Dictionary of corpus statistics
        """
        # Load data if not provided
        if df is None and input_file:
            input_path = os.path.join(self.output_dir, input_file)
            df = pd.read_csv(input_path)
            logger.info(f"Loaded corpus from {input_path}")
        elif df is None:
            raise ValueError("Either df or input_file must be provided")
        
        # Extract all dataset names
        all_datasets = set()
        for datasets_str in df['datasets'].dropna():
            for dataset in datasets_str.split(','):
                dataset = dataset.strip()
                if dataset and dataset != 'Unknown':
                    all_datasets.add(dataset)
        
        # Count model occurrences per dataset
        dataset_counts = {}
        for datasets_str in df['datasets'].dropna():
            for dataset in datasets_str.split(','):
                dataset = dataset.strip()
                if dataset and dataset != 'Unknown':
                    dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        # Generate statistics
        stats = {
            'total_models': len(df),
            'unique_datasets': len(all_datasets),
            'most_common_datasets': sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'avg_datasets_per_model': np.mean([len(datasets_str.split(',')) for datasets_str in df['datasets'].dropna()]),
            'models_with_description': df['card_description'].notna().sum(),
            'models_by_author': df.groupby('author').size().sort_values(ascending=False).head(10).to_dict()
        }
        
        logger.info(f"Corpus analysis completed with {stats['total_models']} models and {stats['unique_datasets']} unique datasets")
        return stats