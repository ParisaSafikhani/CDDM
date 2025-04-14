"""
AutoPyTorch Integration - Integrates domain-aware model selection with AutoPyTorch

This module implements the end-to-end methodology described in the paper, integrating
the domain-aware model selection with AutoPyTorch for text classification tasks.
"""

import os
import pandas as pd
import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("autopytorch_integration")

class ContextualAutoPyTorch:
    """
    Integrates domain-aware model selection with AutoPyTorch for text classification tasks.
    """
    
    def __init__(self, 
                 corpus_builder=None,
                 domain_classifier=None,
                 model_selector=None,
                 corpus_path: Optional[str] = None,
                 cache_dir: str = "cache",
                 results_dir: str = "results",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 fallback_model: str = "all-MiniLM-L6-v2",
                 min_model_score: float = 0.5):
        """
        Initialize the AutoPyTorch integration.
        
        Args:
            corpus_builder: Instance of HuggingFaceCorpusBuilder (if None, will be created if needed)
            domain_classifier: Instance of DomainClassifier (if None, will be created if needed)
            model_selector: Instance of ModelSelector (if None, will be created if needed)
            corpus_path: Path to the domain-classified corpus CSV file (required if model_selector is None)
            cache_dir: Directory to store downloaded models and cached data
            results_dir: Directory to save results
            embedding_model: Model to use for embedding calculations
            fallback_model: Default model to use when no good match is found
            min_model_score: Minimum score threshold for model selection
        """
        self.cache_dir = cache_dir
        self.results_dir = results_dir
        self.embedding_model = embedding_model
        self.fallback_model = fallback_model
        self.min_model_score = min_model_score
        
        # Create directories if they don't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up components
        self.corpus_builder = corpus_builder
        self.domain_classifier = domain_classifier
        self.model_selector = model_selector
        
        # Create timestamp for this run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize components if needed
        if self.model_selector is None and corpus_path:
            self._initialize_model_selector(corpus_path)
        elif self.model_selector is None:
            logger.warning("No model selector provided and no corpus path. Domain-aware model selection disabled.")
        
        logger.info(f"Initialized ContextualAutoPyTorch with results directory: {self.run_dir}")
    
    def _initialize_model_selector(self, corpus_path: str):
        """
        Initialize the model selector component.
        
        Args:
            corpus_path: Path to the domain-classified corpus CSV file
        """
        from .model_selector import ModelSelector
        
        logger.info(f"Initializing ModelSelector with corpus: {corpus_path}")
        self.model_selector = ModelSelector(
            corpus_path=corpus_path,
            cache_dir=self.cache_dir,
            embedding_model=self.embedding_model,
            fallback_model=self.fallback_model,
            min_model_score=self.min_model_score
        )
    
    def _initialize_domain_classifier(self):
        """
        Initialize the domain classifier component.
        """
        from .domain_classifier import DomainClassifier
        
        logger.info("Initializing DomainClassifier")
        self.domain_classifier = DomainClassifier(
            cache_dir=self.cache_dir,
            output_dir=self.run_dir,
            embedding_model=self.embedding_model
        )
    
    def _initialize_corpus_builder(self):
        """
        Initialize the corpus builder component.
        """
        from .corpus_builder import HuggingFaceCorpusBuilder
        
        logger.info("Initializing HuggingFaceCorpusBuilder")
        self.corpus_builder = HuggingFaceCorpusBuilder(
            cache_dir=self.cache_dir,
            output_dir=self.run_dir
        )
    
    def build_corpus(self, output_file: str = "text_classification_models.csv") -> pd.DataFrame:
        """
        Build a corpus of text classification models from Hugging Face.
        
        Args:
            output_file: File name for saving the corpus
            
        Returns:
            DataFrame containing the corpus
        """
        if self.corpus_builder is None:
            self._initialize_corpus_builder()
            
        logger.info("Building corpus of text classification models")
        return self.corpus_builder.build_corpus(output_file=output_file)
    
    def clean_corpus(self, corpus_df: Optional[pd.DataFrame] = None, 
                    input_file: Optional[str] = None,
                    output_file: str = "cleaned_corpus.csv") -> pd.DataFrame:
        """
        Clean and preprocess the corpus.
        
        Args:
            corpus_df: DataFrame containing the corpus (if None, load from input_file)
            input_file: Input file path (if corpus_df is None)
            output_file: Output file path for saving the cleaned corpus
            
        Returns:
            Cleaned DataFrame
        """
        if self.corpus_builder is None:
            self._initialize_corpus_builder()
            
        return self.corpus_builder.clean_corpus(
            df=corpus_df, 
            input_file=input_file, 
            output_file=output_file
        )
    
    def classify_corpus(self, 
                       corpus_df: Optional[pd.DataFrame] = None, 
                       input_file: Optional[str] = None,
                       output_file: str = "domain_classified_corpus.csv") -> pd.DataFrame:
        """
        Classify models in the corpus into domains.
        
        Args:
            corpus_df: DataFrame containing the corpus (if None, load from input_file)
            input_file: Input file path (if corpus_df is None)
            output_file: Output file path for saving the classified corpus
            
        Returns:
            DataFrame with domain classifications added
        """
        if self.domain_classifier is None:
            self._initialize_domain_classifier()
            
        # Load corpus if not provided
        if corpus_df is None and input_file:
            corpus_path = os.path.join(self.corpus_builder.output_dir if self.corpus_builder else "", input_file)
            corpus_df = pd.read_csv(corpus_path)
            logger.info(f"Loaded corpus from {corpus_path}")
        elif corpus_df is None:
            raise ValueError("Either corpus_df or input_file must be provided")
        
        # Create combined text column if needed
        if 'combined_text' not in corpus_df.columns and 'datasets' in corpus_df.columns and 'dataset_info' in corpus_df.columns:
            corpus_df['combined_text'] = corpus_df['datasets'] + " " + corpus_df['dataset_info'].fillna("")
        
        # Save corpus for classification
        temp_path = os.path.join(self.run_dir, "temp_corpus_for_classification.csv")
        corpus_df.to_csv(temp_path, index=False)
        
        # Classify corpus
        classified_df = self.domain_classifier.classify_models_corpus(
            corpus_path=temp_path,
            text_column='combined_text',
            output_file=output_file
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return classified_df
    
    def classify_dataset(self, 
                        dataset_path: str, 
                        text_column: str = "text",
                        label_column: str = "label",
                        use_zero_shot: bool = True) -> Dict[str, Any]:
        """
        Classify a dataset into domains.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            use_zero_shot: Whether to use zero-shot classification
            
        Returns:
            Dictionary with domain classification results
        """
        if self.domain_classifier is None:
            self._initialize_domain_classifier()
            
        if use_zero_shot:
            return self.domain_classifier.classify_dataset_zero_shot(
                dataset_path=dataset_path,
                text_column=text_column,
                label_column=label_column
            )
        else:
            return self.domain_classifier.classify_dataset(
                dataset_path=dataset_path,
                text_column=text_column
            )
    
    def select_model_for_dataset(self, 
                               dataset_path: str, 
                               text_column: str = "text",
                               label_column: str = "label",
                               use_zero_shot: bool = True) -> Dict[str, Any]:
        """
        Select the most appropriate pre-trained model for a dataset.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            use_zero_shot: Whether to use zero-shot classification for domain identification
            
        Returns:
            Dictionary with selected model information
        """
        if self.model_selector is None:
            raise ValueError("No model selector available. Initialize with corpus_path or provide model_selector.")
            
        # Classify dataset
        logger.info(f"Classifying dataset: {dataset_path}")
        dataset_domain = self.classify_dataset(
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
            use_zero_shot=use_zero_shot
        )
        
        # Prepare dataset text for similarity matching
        df = pd.read_csv(dataset_path)
        if text_column in df.columns:
            # Sample up to 100 examples for performance
            if len(df) > 100:
                df_sample = df.sample(100)
            else:
                df_sample = df
            dataset_text = " ".join(df_sample[text_column].fillna("").astype(str).tolist())
        else:
            dataset_text = None
        
        # Select model
        logger.info("Selecting appropriate pre-trained model")
        model_info = self.model_selector.select_model_for_dataset(
            dataset_domain=dataset_domain,
            dataset_text=dataset_text
        )
        
        # Add dataset info
        model_info['dataset_path'] = dataset_path
        model_info['text_column'] = text_column
        model_info['label_column'] = label_column
        
        # Save selection results
        os.makedirs(os.path.join(self.run_dir, "model_selections"), exist_ok=True)
        selection_file = os.path.join(self.run_dir, "model_selections", f"{os.path.basename(dataset_path).split('.')[0]}_selection.json")
        with open(selection_file, 'w') as f:
            json.dump({**model_info, **dataset_domain}, f, indent=2)
        
        logger.info(f"Selected model {model_info['model_id']} for dataset {os.path.basename(dataset_path)}")
        return model_info
    
    def generate_embeddings(self, 
                          dataset_path: str, 
                          text_column: str = "text",
                          label_column: str = "label",
                          model_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for a dataset using a specified or selected model.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            model_id: ID of the model to use (if None, will be automatically selected)
            
        Returns:
            Tuple of (embeddings, labels)
        """
        # Select model if not provided
        if model_id is None:
            model_info = self.select_model_for_dataset(
                dataset_path=dataset_path,
                text_column=text_column,
                label_column=label_column
            )
            model_id = model_info['model_id']
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Generate embeddings
        logger.info(f"Generating embeddings using model: {model_id}")
        texts = df[text_column].fillna("").astype(str).tolist()
        embeddings = self.model_selector.generate_embeddings(texts, model_id=model_id)
        
        # Get labels
        labels = df[label_column].values
        
        return embeddings, labels
    
    def run_autopytorch(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      total_time_limit: int = 1000,
                      per_run_time_limit: int = 300,
                      ensemble_size: int = 5,
                      resampling_strategy: str = "cv",
                      folds: int = 5) -> Dict[str, Any]:
        """
        Run AutoPyTorch on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            total_time_limit: Total time limit in seconds
            per_run_time_limit: Time limit per run in seconds
            ensemble_size: Size of the ensemble
            resampling_strategy: Resampling strategy ('cv' or 'holdout')
            folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with results and trained model
        """
        try:
            from autoPyTorch.api.tabular_classification import TabularClassificationTask
            from autoPyTorch.datasets.resampling_strategy import CrossValTypes
        except ImportError:
            logger.error("AutoPyTorch is not installed. Cannot run AutoPyTorch.")
            return {"error": "AutoPyTorch is not installed"}
        
        logger.info("Initializing AutoPyTorch")
        
        # Determine resampling strategy
        if resampling_strategy.lower() == "cv":
            resampling = CrossValTypes.k_fold_cross_validation
        else:
            resampling = CrossValTypes.holdout
        
        # Initialize AutoPyTorch
        api = TabularClassificationTask(
            resampling_strategy=resampling,
            resampling_strategy_args={"folds": folds} if resampling_strategy.lower() == "cv" else None,
            ensemble_size=ensemble_size
        )
        
        # Determine feat_types (all are numerical for embeddings)
        feat_types = ['numerical'] * X_train.shape[1]
        
        # Run AutoPyTorch
        logger.info(f"Running AutoPyTorch with time limit: {total_time_limit}s")
        start_time = time.time()
        
        api.search(
            optimize_metric='accuracy',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feat_types=feat_types,
            total_walltime_limit=total_time_limit,
            func_eval_time_limit_secs=per_run_time_limit,
            budget_type='epochs',
            min_budget=5,
            max_budget=50,
            enable_traditional_pipeline=True,
            all_supported_metrics=True,
            precision=32,
            load_models=True
        )
        
        # Calculate metrics
        y_pred = api.predict(X_test)
        y_pred_proba = api.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'runtime': time.time() - start_time
        }
        
        # Calculate AUPRC for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['auprc'] = average_precision_score(y_test, y_pred_proba[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            metrics['auprc'] = average_precision_score(y_test_bin, y_pred_proba, average='macro')
        
        logger.info(f"AutoPyTorch run completed in {metrics['runtime']:.2f}s")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'api': api,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def process_dataset(self, 
                       dataset_path: str, 
                       text_column: str = "text",
                       label_column: str = "label",
                       test_size: float = 0.2,
                       random_state: int = 42,
                       save_results: bool = True) -> Dict[str, Any]:
        """
        Process a dataset end-to-end using the domain-aware AutoPyTorch pipeline.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with complete results
        """
        start_time = time.time()
        
        # Step 1: Select appropriate model for the dataset
        logger.info(f"Processing dataset: {dataset_path}")
        model_info = self.select_model_for_dataset(
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column
        )
        
        # Step 2: Generate embeddings using the selected model
        logger.info(f"Generating embeddings with model: {model_info['model_id']}")
        embeddings, labels = self.generate_embeddings(
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
            model_id=model_info['model_id']
        )
        
        # Step 3: Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Step 4: Run AutoPyTorch on the embeddings
        logger.info("Running AutoPyTorch on the embeddings")
        autopytorch_results = self.run_autopytorch(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        # Combine results
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        
        results = {
            'dataset': dataset_name,
            'dataset_path': dataset_path,
            'model_id': model_info['model_id'],
            'selection_method': model_info['selection_method'],
            'domain': model_info['domain'],
            'metrics': autopytorch_results['metrics'],
            'runtime': time.time() - start_time
        }
        
        # Save results
        if save_results:
            # Save individual result
            os.makedirs(os.path.join(self.run_dir, "results"), exist_ok=True)
            result_file = os.path.join(self.run_dir, "results", f"{dataset_name}_result.json")
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = results.copy()
            serializable_results['metrics'] = {k: float(v) for k, v in serializable_results['metrics'].items()}
            
            with open(result_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Update summary file
            summary_file = os.path.join(self.run_dir, "results_summary.csv")
            summary_row = pd.DataFrame([{
                'dataset': dataset_name,
                'model_id': model_info['model_id'],
                'selection_method': model_info['selection_method'],
                'domain': model_info['domain'],
                'domain_score': model_info['domain_score'],
                **{f"metric_{k}": v for k, v in autopytorch_results['metrics'].items()}
            }])
            
            if os.path.exists(summary_file):
                summary_df = pd.read_csv(summary_file)
                summary_df = pd.concat([summary_df, summary_row], ignore_index=True)
            else:
                summary_df = summary_row
                
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"Results saved to {result_file} and summary updated")
        
        logger.info(f"Dataset {dataset_name} processed in {results['runtime']:.2f}s")
        return results
    
    def run_benchmark(self, 
                     datasets_dir: str, 
                     text_column: str = "text",
                     label_column: str = "label",
                     file_pattern: str = "*.csv") -> Dict[str, Any]:
        """
        Run benchmark on multiple datasets.
        
        Args:
            datasets_dir: Directory containing datasets
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            file_pattern: Pattern to match dataset files
            
        Returns:
            Dictionary with benchmark results
        """
        import glob
        
        # Find datasets
        dataset_paths = glob.glob(os.path.join(datasets_dir, file_pattern))
        logger.info(f"Found {len(dataset_paths)} datasets in {datasets_dir}")
        
        # Process each dataset
        results = {}
        for dataset_path in dataset_paths:
            try:
                dataset_name = os.path.basename(dataset_path).split('.')[0]
                logger.info(f"Processing dataset: {dataset_name}")
                
                result = self.process_dataset(
                    dataset_path=dataset_path,
                    text_column=text_column,
                    label_column=label_column,
                    save_results=True
                )
                
                results[dataset_name] = result
                logger.info(f"Completed {dataset_name} with accuracy: {result['metrics']['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_path}: {str(e)}")
                results[os.path.basename(dataset_path).split('.')[0]] = {"error": str(e)}
        
        # Generate final summary
        self._create_benchmark_summary()
        
        return results
    
    def _create_benchmark_summary(self) -> Dict[str, Any]:
        """
        Create a comprehensive summary of benchmark results.
        
        Returns:
            Dictionary with summary statistics
        """
        summary_file = os.path.join(self.run_dir, "results_summary.csv")
        if not os.path.exists(summary_file):
            logger.error("No summary file found to create benchmark summary")
            return {}
        
        summary_df = pd.read_csv(summary_file)
        
        # Generate summary metrics
        metric_columns = [col for col in summary_df.columns if col.startswith('metric_')]
        dataset_column = 'dataset'
        
        # Calculate metrics statistics
        metrics_stats = {}
        for col in metric_columns:
            if summary_df[col].dtype in [np.float64, np.int64]:
                metrics_stats[f'avg_{col}'] = summary_df[col].mean()
                metrics_stats[f'max_{col}'] = summary_df[col].max()
                metrics_stats[f'min_{col}'] = summary_df[col].min()
                
                if col.startswith('metric_') and 'time' not in col and 'runtime' not in col:
                    best_idx = summary_df[col].idxmax()
                    metrics_stats[f'best_dataset_{col}'] = summary_df.loc[best_idx, dataset_column]
        
        # Add overall stats
        metrics_stats['total_datasets'] = len(summary_df)
        metrics_stats['total_execution_time'] = summary_df['metric_runtime'].sum() if 'metric_runtime' in summary_df.columns else None
        
        # Save final summary
        final_summary_file = os.path.join(self.run_dir, "benchmark_final_summary.csv")
        pd.DataFrame([metrics_stats]).to_csv(final_summary_file, index=False)
        
        # Create detailed statistics for key metrics
        if metric_columns:
            metric_stats_df = summary_df[metric_columns].describe()
            metric_stats_file = os.path.join(self.run_dir, "benchmark_metric_statistics.csv")
            metric_stats_df.to_csv(metric_stats_file)
        
        logger.info(f"Benchmark summary created with {metrics_stats['total_datasets']} datasets")
        return metrics_stats