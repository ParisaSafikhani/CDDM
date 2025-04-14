"""
Domain Classifier - Automatically assigns domains to models and datasets

This module implements the methodology described in Section 3.2 of the paper:
Selection of Domain-Specific Models.
"""

import os
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("domain_classifier")

# Define standard domains with references, based on Table 1 from the paper
STANDARD_DOMAINS = {
    "Sentiment Analysis": "Detecting opinions, emotions, and sentiments in text.",
    "Topic Classification": "Assigning topics or categories to text documents.",
    "Emotion Recognition": "Identifying emotions such as joy, sadness, anger, and fear in text.",
    "Spam Detection": "Classifying emails or messages as spam or legitimate.",
    "Intent Classification": "Understanding the purpose or intent behind user queries.",
    "Hate Speech and Toxicity Detection": "Detecting hate speech, toxic, or abusive language in text.",
    "Named Entity Recognition (NER)": "Recognizing named entities such as people, organizations, and locations in text.",
    "Textual Entailment and Natural Language Inference": "Determining if one text logically follows from another.",
    "Emotion Cause Extraction": "Identifying the reasons or triggers for specific emotions in text.",
    "Language Identification": "Detecting the language of text, especially in multilingual settings.",
    "Document Classification": "Categorizing entire documents into predefined classes.",
    "Fake News Detection": "Detecting false or misleading news articles.",
    "Aspect-Based Sentiment Analysis": "Analyzing sentiment specific to different aspects of a product or service.",
    "Sarcasm Detection": "Identifying sarcasm or ironic statements in text.",
    "Propaganda Detection": "Detecting manipulative or biased content in text.",
    "Stance Detection": "Determining whether a text agrees or disagrees with a given statement.",
    "Humor Detection": "Recognizing jokes or humorous content in text.",
    "Relationship Extraction": "Extracting relationships between entities in text.",
    "Authorship Attribution": "Determining the author of a text based on writing style.",
    "Text Difficulty Classification": "Classifying text based on readability or difficulty level.",
    "Question Classification": "Categorizing questions based on their type or intent.",
    "Sexism and Misogyny Detection": "Detecting sexist or misogynistic content in text.",
    "Customer Feedback Classification": "Classifying customer feedback into predefined categories.",
    "Event Detection": "Identifying and extracting events from text.",
    "Gender Classification": "Classifying text or data based on gender.",
    "Code-Mixed Language Classification": "Classifying mixed-language content.",
    "Metaphor Detection": "Detecting metaphorical language in text.",
    "Irony Detection": "Identifying ironic statements in text.",
    "Argument Mining and Argument Classification": "Analyzing arguments and their structures in text.",
    "Deception Detection": "Detecting lies, fraud, or deceptive statements in text.",
}

class DomainClassifier:
    """
    Classifies models and datasets into domains using embedding-based
    similarity or keyword-based approaches.
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 output_dir: str = "domains",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 domains: Optional[Dict[str, str]] = None,
                 similarity_threshold: float = 0.3):
        """
        Initialize the domain classifier.
        
        Args:
            cache_dir: Directory to store temporary files
            output_dir: Directory to save domain classification results
            embedding_model: Sentence transformer model to use for embeddings
            domains: Dictionary of domain names and descriptions (default: STANDARD_DOMAINS)
            similarity_threshold: Minimum similarity score to assign a domain
        """
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        
        # Create directories if they don't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the sentence transformer model
        logger.info(f"Loading sentence transformer model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Set up domains
        self.domains = domains if domains is not None else STANDARD_DOMAINS
        self.domain_names = list(self.domains.keys())
        
        # Pre-compute embeddings for domain descriptions
        logger.info(f"Pre-computing embeddings for {len(self.domains)} domains")
        self.domain_embeddings = self.model.encode(list(self.domains.values()))
        
        # Domain keywords for the keyword-based approach
        self.domain_keywords = self._extract_keywords_from_domains()
        
        logger.info(f"Initialized DomainClassifier with {len(self.domains)} domains")
    
    def _extract_keywords_from_domains(self) -> Dict[str, List[str]]:
        """
        Extract keywords from domain descriptions.
        
        Returns:
            Dictionary mapping domain names to lists of keywords
        """
        domain_keywords = {}
        for domain, description in self.domains.items():
            # Extract keywords from domain name and description
            words = re.findall(r'\b\w+\b', domain.lower() + ' ' + description.lower())
            # Filter out common words and keep only meaningful keywords
            keywords = [w for w in words if len(w) > 3 and w not in [
                'and', 'the', 'text', 'from', 'such', 'with', 'based', 'into', 
                'their', 'whether', 'using', 'type', 'content'
            ]]
            domain_keywords[domain] = keywords
        return domain_keywords
    
    def _prepare_dataset_domains_df(self) -> pd.DataFrame:
        """
        Prepare a DataFrame with domain names and descriptions.
        
        Returns:
            DataFrame with domain names and descriptions
        """
        return pd.DataFrame(
            [(name, desc) for name, desc in self.domains.items()],
            columns=["domain", "description"]
        )
    
    def classify_text_embedding(self, text: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Classify a text into domains using embedding-based similarity.
        
        Args:
            text: Text to classify
            top_n: Number of top domains to return
            
        Returns:
            List of (domain_name, similarity_score) tuples
        """
        # Encode the text
        text_embedding = self.model.encode([text])[0]
        
        # Compute similarity with all domain descriptions
        similarities = cosine_similarity([text_embedding], self.domain_embeddings)[0]
        
        # Get top domains
        top_indices = similarities.argsort()[-top_n:][::-1]
        result = [(self.domain_names[i], similarities[i]) for i in top_indices 
                  if similarities[i] >= self.similarity_threshold]
        
        return result
    
    def classify_text_keywords(self, text: str, top_n: int = 3) -> List[Tuple[str, int]]:
        """
        Classify a text into domains using keyword matching.
        
        Args:
            text: Text to classify
            top_n: Number of top domains to return
            
        Returns:
            List of (domain_name, keyword_count) tuples
        """
        text_lower = text.lower()
        domain_matches = {}
        
        for domain, keywords in self.domain_keywords.items():
            matches = 0
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    matches += 1
            if matches > 0:
                domain_matches[domain] = matches
        
        # Return top matches
        top_domains = sorted(domain_matches.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return top_domains
    
    def classify_dataset(self, 
                         dataset_path: str, 
                         text_column: str = "text", 
                         sample_size: int = 100) -> Dict[str, Any]:
        """
        Classify a dataset into domains based on its text content.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text
            sample_size: Maximum number of samples to use for classification
            
        Returns:
            Dictionary with domain classification results
        """
        logger.info(f"Classifying dataset: {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Ensure text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        
        # Sample data if needed
        if len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df
        
        # Concatenate samples for overall analysis
        all_text = " ".join(df_sample[text_column].fillna("").astype(str).values)
        
        # Classify using both approaches
        embedding_results = self.classify_text_embedding(all_text)
        keyword_results = self.classify_text_keywords(all_text)
        
        # Individual sample classification for confidence measurement
        sample_domains = []
        for text in df_sample[text_column].fillna("").astype(str).values:
            if len(text.strip()) > 10:  # Skip very short texts
                sample_result = self.classify_text_embedding(text, top_n=1)
                if sample_result:
                    sample_domains.append(sample_result[0][0])
        
        # Calculate confidence from individual classifications
        domain_counts = Counter(sample_domains)
        total_samples = len(sample_domains)
        
        confidence_scores = {
            domain: count / total_samples 
            for domain, count in domain_counts.items()
        } if total_samples > 0 else {}
        
        # Determine final domain(s)
        final_domains = []
        
        # Prioritize domains with high confidence from individual samples
        high_confidence_domains = [
            domain for domain, score in confidence_scores.items() 
            if score >= 0.2  # At least 20% of samples agree
        ]
        
        if high_confidence_domains:
            final_domains.extend(high_confidence_domains)
        
        # Add top embedding results if not already included
        for domain, score in embedding_results:
            if domain not in final_domains and score >= 0.5:  # Higher threshold for corpus-level
                final_domains.append(domain)
        
        # Result dictionary
        result = {
            "dataset_path": dataset_path,
            "primary_domain": embedding_results[0][0] if embedding_results else None,
            "primary_domain_score": embedding_results[0][1] if embedding_results else 0,
            "all_domains": final_domains,
            "embedding_domains": embedding_results,
            "keyword_domains": keyword_results,
            "domain_confidence": confidence_scores,
            "sample_size": len(df_sample)
        }
        
        logger.info(f"Classified dataset {os.path.basename(dataset_path)} as: {result['primary_domain']}")
        return result
    
    def classify_models_corpus(self, 
                              corpus_path: str, 
                              text_column: str = "combined_text",
                              output_file: str = "domain_classified_models.csv") -> pd.DataFrame:
        """
        Classify all models in a corpus based on their dataset descriptions.
        
        Args:
            corpus_path: Path to the model corpus CSV file
            text_column: Column to use for classification (default: dataset name + description)
            output_file: Name of output file to save results
            
        Returns:
            DataFrame with domain classifications added
        """
        logger.info(f"Classifying models in corpus: {corpus_path}")
        
        # Load corpus
        df = pd.read_csv(corpus_path)
        
        # Create combined text column if it doesn't exist
        if text_column not in df.columns:
            if 'datasets' in df.columns and 'dataset_info' in df.columns:
                df[text_column] = df['datasets'] + " " + df['dataset_info'].fillna("")
            else:
                raise ValueError(f"Required columns not found in corpus and {text_column} doesn't exist")
        
        # Classify each model
        domains = []
        domain_scores = []
        all_domains_list = []
        
        for text in tqdm(df[text_column], desc="Classifying models"):
            result = self.classify_text_embedding(str(text))
            if result:
                domains.append(result[0][0])  # Primary domain
                domain_scores.append(result[0][1])  # Primary domain score
                all_domains_list.append(', '.join([d for d, s in result]))  # All identified domains
            else:
                domains.append(None)
                domain_scores.append(0)
                all_domains_list.append("")
        
        # Add results to DataFrame
        df['primary_domain'] = domains
        df['domain_score'] = domain_scores
        df['all_domains'] = all_domains_list
        
        # Save results
        output_path = os.path.join(self.output_dir, output_file)
        df.to_csv(output_path, index=False)
        logger.info(f"Domain classification complete. Results saved to {output_path}")
        
        # Count domain distribution
        domain_counts = Counter(domains)
        top_domains = domain_counts.most_common(10)
        logger.info(f"Top domains in corpus: {top_domains}")
        
        return df
    
    def classify_dataset_zero_shot(self, 
                                  dataset_path: str, 
                                  text_column: str = "text",
                                  label_column: str = "label",
                                  sample_size: int = 100,
                                  zero_shot_model_name: str = "cross-encoder/nli-deberta-v3-small") -> Dict[str, Any]:
        """
        Classify a dataset using zero-shot classification as described in Section 3.3.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels 
            sample_size: Maximum number of samples to use per class
            zero_shot_model_name: Model name for zero-shot classification
            
        Returns:
            Dictionary with zero-shot classification results
        """
        try:
            from transformers import pipeline
        except ImportError:
            logger.error("transformers package not installed. Cannot use zero_shot classification.")
            return {"error": "transformers package not installed"}
        
        logger.info(f"Classifying dataset with zero-shot: {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Ensure columns exist
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Get samples for each class
        class_samples = {}
        for label in df[label_column].unique():
            class_df = df[df[label_column] == label]
            if len(class_df) > sample_size:
                class_samples[label] = class_df.sample(sample_size, random_state=42)[text_column].tolist()
            else:
                class_samples[label] = class_df[text_column].tolist()
        
        # Initialize zero-shot classifier
        logger.info(f"Loading zero-shot classifier: {zero_shot_model_name}")
        classifier = pipeline("zero-shot-classification", model=zero_shot_model_name)
        
        # Classify samples for each class
        candidate_labels = list(self.domains.keys())
        
        class_domain_scores = {}
        for label, samples in class_samples.items():
            label_scores = {domain: 0 for domain in candidate_labels}
            
            # Sample up to 5 texts per class to keep processing manageable
            use_samples = samples[:5]
            
            for text in use_samples:
                if not isinstance(text, str) or len(text.strip()) < 10:
                    continue
                    
                try:
                    result = classifier(text, candidate_labels, multi_label=False)
                    for domain, score in zip(result['labels'], result['scores']):
                        label_scores[domain] += score
                except Exception as e:
                    logger.warning(f"Error in zero-shot classification: {str(e)}")
                    continue
            
            # Average scores for this class
            if len(use_samples) > 0:
                for domain in label_scores:
                    label_scores[domain] /= len(use_samples)
                    
            class_domain_scores[label] = label_scores
        
        # Compute average scores across all classes
        avg_scores = {domain: 0 for domain in candidate_labels}
        for label_scores in class_domain_scores.values():
            for domain, score in label_scores.items():
                avg_scores[domain] += score
                
        if class_domain_scores:
            for domain in avg_scores:
                avg_scores[domain] /= len(class_domain_scores)
        
        # Get top domains
        top_domains = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            "dataset_path": dataset_path,
            "primary_domain": top_domains[0][0] if top_domains else None,
            "primary_domain_score": top_domains[0][1] if top_domains else 0,
            "top_domains": top_domains[:5],
            "class_domain_scores": class_domain_scores,
            "method": "zero-shot",
            "model": zero_shot_model_name
        }
        
        logger.info(f"Zero-shot classification for {os.path.basename(dataset_path)}: {result['primary_domain']}")
        return result