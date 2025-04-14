# CDDM: Corpus-Driven Domain Mapping

Implementation of the methodology described in the paper:
"AutoML Meets Hugging Face: Domain-Aware Pretrained Model Selection for Text Classification"

## Overview

CDDM is a framework that enhances AutoML by automatically selecting appropriate pre-trained models from the Hugging Face Hub based on domain matching. It addresses the challenge of finding the most suitable pre-trained model for a specific text classification task by analyzing the domain of the dataset and selecting models that were fine-tuned on similar domains.

The key innovation of CDDM is its domain-aware model selection approach, which:
1. Builds a corpus of text classification models from Hugging Face Hub
2. Classifies models and datasets into predefined domains
3. Selects models that match the domain of the target dataset
4. Integrates with AutoPyTorch for end-to-end AutoML

## Key Components

1. **Corpus Builder**: Extracts text classification models from Hugging Face Hub along with metadata and dataset information.
   - Processes model tags and descriptions to identify datasets
   - Extracts dataset information and descriptions
   - Cleans and filters the corpus for quality

2. **Domain Classifier**: Classifies models and datasets into predefined domains using embedding-based similarity.
   - Implements over 30 standard domains for text classification
   - Uses SentenceTransformer for embedding-based similarity matching
   - Supports both keyword-based and zero-shot classification approaches

3. **Model Selector**: Selects appropriate pre-trained models based on domain matching.
   - Ranks models by domain similarity score
   - Implements fallback mechanisms for domains with few models
   - Downloads and prepares selected models for fine-tuning

4. **AutoPyTorch Integration**: Integrates all components into a unified workflow for end-to-end benchmarking.
   - Streamlines dataset processing and embedding generation
   - Integrates with AutoPyTorch for hyperparameter optimization
   - Provides comprehensive performance metrics and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/ParisaSafikhani/CDDM.git
cd CDDM

# Install the package in development mode
pip install -e .
```

## Example Usage

```python
from cddm import ContextualAutoPyTorch

# Initialize the CDDM framework
cddm = ContextualAutoPyTorch(
    corpus_path="domain_classified_corpus.csv",  # Path to pre-built corpus
    cache_dir="cache",                           # Directory for cached models
    results_dir="results"                        # Directory for results
)

# Process a dataset end-to-end
result = cddm.process_dataset(
    dataset_path="my_dataset.csv",
    text_column="text",
    label_column="label"
)

# Access results
print(f"Selected model: {result['model_id']}")
print(f"Selection method: {result['selection_method']}")
print(f"Domain: {result['domain']}")
print(f"Accuracy: {result['metrics']['accuracy']}")
```

## Building a Corpus

If you need to build a corpus from scratch:

```python
from cddm import ContextualAutoPyTorch

# Initialize without a corpus
cddm = ContextualAutoPyTorch(
    cache_dir="cache",
    results_dir="results"
)

# Build and classify corpus
corpus_df = cddm.build_corpus(output_file="text_classification_models.csv")
cleaned_df = cddm.clean_corpus(corpus_df=corpus_df, output_file="cleaned_corpus.csv")
classified_df = cddm.classify_corpus(corpus_df=cleaned_df, output_file="domain_classified_corpus.csv")
```

```

## Custom Datasets

For custom datasets, use the provided example:

```bash
python examples/custom_dataset.py --dataset_path my_dataset.csv --corpus_path domain_classified_corpus.csv
```

## Implementation Details

The CDDM framework implements a modular approach to domain-aware model selection:

1. **Domain Classification**: Uses embedding-based similarity to classify datasets into predefined domains. 
2. **Corpus Matching**: Matches dataset domains with pre-trained models from the corpus.
3. **Model Selection**: Selects the most appropriate model based on domain match score.
4. **AutoML Integration**: Integrates with AutoPyTorch for automatic fine-tuning.

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{safikhani2023automl,
  title={AutoML Meets Hugging Face: Domain-Aware Pretrained Model Selection for Text Classification},
  author={Safikhani, Parisa and Broneske, David},
  booktitle={},
  year={2024}
}
```

## License

MIT License 
