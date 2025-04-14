"""
Model Selector - Finds the most appropriate pre-trained model for a dataset

This module implements the methodology described in Section 3.2 of the paper:
Selection of Domain-Specific Models using the domain-annotated corpus.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_selector")

class ModelSelector:
    """
    Selects appropriate pre-trained models for datasets based on domain alignment.
    """
    
    def __init__(self, 
                 corpus_path: str,
                 cache_dir: str = "cache",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 fallback_model: str = "all-MiniLM-L6-v2",
                 min_model_score: float = 0.5,
                 download_models: bool = True):
        """
        Initialize the model selector.
        
        Args:
            corpus_path: Path to the domain-classified corpus CSV file
            cache_dir: Directory to store downloaded models
            embedding_model: Model to use for embedding comparisons
            fallback_model: Default model to use when no good match is found
            min_model_score: Minimum score threshold for model selection
            download_models: Whether to download models automatically
        """
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.fallback_model = fallback_model
        self.min_model_score = min_model_score
        self.download_models = download_models
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load the model corpus
        logger.info(f"Loading model corpus from {corpus_path}")
        self.corpus_df = pd.read_csv(corpus_path)
        
        # Check if the corpus has the required columns
        required_columns = ['id', 'primary_domain', 'domain_score']
        missing_columns = [col for col in required_columns if col not in self.corpus_df.columns]
        if missing_columns:
            raise ValueError(f"Corpus is missing required columns: {missing_columns}")
        
        # Load embedding model for comparing dataset and model descriptions
        logger.info(f"Loading sentence transformer model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Pre-compute model embeddings
        if 'datasets' in self.corpus_df.columns and 'dataset_info' in self.corpus_df.columns:
            self.corpus_df['combined_text'] = self.corpus_df['datasets'] + " " + self.corpus_df['dataset_info'].fillna("")
            self._precompute_model_embeddings()
        else:
            logger.warning("Corpus doesn't have 'datasets' and 'dataset_info' columns. Some features will be limited.")
        
        logger.info(f"Initialized ModelSelector with {len(self.corpus_df)} models")
    
    def _precompute_model_embeddings(self):
        """
        Pre-compute embeddings for all models in the corpus.
        """
        logger.info("Pre-computing model embeddings...")
        texts = self.corpus_df['combined_text'].fillna("").astype(str).tolist()
        self.model_embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Pre-computed embeddings for {len(texts)} models")
    
    def find_models_by_domain(self, domain: str, min_score: Optional[float] = None) -> pd.DataFrame:
        """
        Find models that match a specific domain.
        
        Args:
            domain: Domain name to match
            min_score: Minimum domain score to consider (default: self.min_model_score)
            
        Returns:
            DataFrame with matching models, sorted by domain score
        """
        if min_score is None:
            min_score = self.min_model_score
            
        # Find matching models
        matching_models = self.corpus_df[
            (self.corpus_df['primary_domain'] == domain) & 
            (self.corpus_df['domain_score'] >= min_score)
        ].copy()
        
        # Sort by score
        if not matching_models.empty:
            matching_models.sort_values('domain_score', ascending=False, inplace=True)
            
        return matching_models
    
    def find_models_by_dataset_domain(self, dataset_domain: Dict[str, Any]) -> pd.DataFrame:
        """
        Find models that match the domain of a dataset.
        
        Args:
            dataset_domain: Dictionary with dataset domain classification results
            
        Returns:
            DataFrame with matching models, sorted by relevance
        """
        primary_domain = dataset_domain.get('primary_domain')
        if not primary_domain:
            logger.warning("No primary domain found in dataset_domain")
            return pd.DataFrame()
        
        # First try to find models matching the primary domain
        matching_models = self.find_models_by_domain(primary_domain)
        
        # If no models match the primary domain, try alternative domains
        if matching_models.empty and 'all_domains' in dataset_domain:
            for domain in dataset_domain['all_domains']:
                if domain != primary_domain:
                    domain_models = self.find_models_by_domain(domain)
                    if not domain_models.empty:
                        matching_models = domain_models
                        logger.info(f"Using alternative domain: {domain}")
                        break
        
        return matching_models
    
    def find_models_by_text_similarity(self, dataset_text: str, top_n: int = 5) -> pd.DataFrame:
        """
        Find models based on text similarity between dataset and model descriptions.
        
        Args:
            dataset_text: Text description of the dataset
            top_n: Number of top models to return
            
        Returns:
            DataFrame with matching models, sorted by similarity score
        """
        if not hasattr(self, 'model_embeddings'):
            logger.error("Model embeddings not pre-computed. Can't perform text similarity search.")
            return pd.DataFrame()
        
        # Encode the dataset text
        dataset_embedding = self.model.encode([dataset_text])[0]
        
        # Compute similarity with all model descriptions
        similarities = cosine_similarity([dataset_embedding], self.model_embeddings)[0]
        
        # Get top models
        self.corpus_df['similarity_score'] = similarities
        top_models = self.corpus_df.sort_values('similarity_score', ascending=False).head(top_n).copy()
        
        return top_models
    
    def select_model_for_dataset(self, 
                                dataset_domain: Dict[str, Any], 
                                dataset_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Select the best model for a dataset based on domain and text similarity.
        
        Args:
            dataset_domain: Dictionary with dataset domain classification results
            dataset_text: Optional text description of the dataset for similarity matching
            
        Returns:
            Dictionary with selected model information
        """
        selected_model = None
        selection_method = None
        
        # Step 1: Try to find models by domain
        domain_models = self.find_models_by_dataset_domain(dataset_domain)
        
        if not domain_models.empty:
            # Pick the model with the highest domain score
            selected_model = domain_models.iloc[0]
            selection_method = "domain"
            logger.info(f"Selected model by domain match: {selected_model['id']}")
        
        # Step 2: If no domain match or dataset_text is provided, try text similarity
        if (selected_model is None or selection_method != "domain") and dataset_text:
            similarity_models = self.find_models_by_text_similarity(dataset_text)
            
            if not similarity_models.empty and similarity_models.iloc[0]['similarity_score'] >= self.min_model_score:
                selected_model = similarity_models.iloc[0]
                selection_method = "similarity"
                logger.info(f"Selected model by text similarity: {selected_model['id']}")
        
        # Step 3: If no suitable model found, use fallback
        if selected_model is None:
            logger.info(f"No suitable model found. Using fallback model: {self.fallback_model}")
            selected_model = pd.Series({
                'id': self.fallback_model,
                'primary_domain': None,
                'domain_score': 0.0,
                'selection_method': 'fallback'
            })
            selection_method = "fallback"
        
        # Download the model if needed
        model_id = selected_model['id']
        if self.download_models:
            try:
                self._download_model(model_id)
            except Exception as e:
                logger.error(f"Error downloading model {model_id}: {str(e)}")
                # If download fails and not already using fallback, try fallback
                if selection_method != "fallback":
                    logger.info(f"Falling back to default model: {self.fallback_model}")
                    model_id = self.fallback_model
                    selection_method = "fallback"
                    self._download_model(model_id)
        
        # Format return value
        result = {
            'model_id': model_id,
            'selection_method': selection_method,
            'domain': selected_model.get('primary_domain'),
            'domain_score': selected_model.get('domain_score', 0.0),
            'similarity_score': selected_model.get('similarity_score', 0.0),
            'dataset_domain': dataset_domain.get('primary_domain'),
            'dataset_domain_score': dataset_domain.get('primary_domain_score', 0.0)
        }
        
        return result
    
    def _download_model(self, model_id: str) -> None:
        """
        Download a model from Hugging Face.
        
        Args:
            model_id: The ID of the model to download
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info(f"Downloading model {model_id}")
            # Download tokenizer and model to cache
            AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
            AutoModel.from_pretrained(model_id, cache_dir=self.cache_dir)
            logger.info(f"Model {model_id} downloaded successfully")
        except ImportError:
            logger.error("transformers package not installed. Cannot download models.")
            raise
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {str(e)}")
            raise
    
    def generate_embeddings(self, 
                           texts: List[str], 
                           model_id: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts using a specified model.
        
        Args:
            texts: List of text strings to encode
            model_id: Model ID to use for encoding (default: use the model from this class)
            
        Returns:
            Numpy array of embeddings
        """
        if model_id is None:
            # Use the default embedding model
            return self.model.encode(texts, show_progress_bar=True)
        else:
            try:
                from transformers import AutoTokenizer, AutoModel
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
                model = AutoModel.from_pretrained(model_id, cache_dir=self.cache_dir)
                
                # Process in batches
                batch_size = 32
                embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        model = model.to("cuda")
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Use mean of last hidden state as embedding
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(batch_embeddings.cpu().numpy())
                
                return np.vstack(embeddings)
            except ImportError:
                logger.error("transformers package not installed. Using fallback embedding model.")
                return self.model.encode(texts, show_progress_bar=True)
            except Exception as e:
                logger.error(f"Error generating embeddings with model {model_id}: {str(e)}")
                logger.info("Using fallback embedding model.")
                return self.model.encode(texts, show_progress_bar=True)