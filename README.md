Collecting workspace information# Food Calorie Estimator

A machine learning system that estimates calorie content from natural language food descriptions using BERT-based models and the USDA food database.

## Features

- Natural language processing of food descriptions
- Calorie content estimation using fine-tuned BERT models 
- Fast similarity search with FAISS indexing
- Named Entity Recognition (NER) for ingredient extraction
- USDA FoodData Central integration

## Installation

1. Ensure Python >=3.13.2 is installed
2. Clone this repository
3. Install dependencies:

```bash
uv sync
```

## Configuration

1. Create a .env file in the project root:

```ini
USDA_API_KEY=your_api_key_here
SAMPLE_SIZE=1000
```

2. Obtain a USDA FoodData Central API key from [api.data.gov](https://api.data.gov)

## Available Commands

The following commands are available after installation:

### Data Pipeline Commands
- `fetch_data` - Fetches raw food data from USDA API
- `process_data` - Processes raw data into training format
- `fetch_and_process` - Combines fetch and process steps

### Model Training Commands
- `finetune_bert_maskedlm` - Fine-tunes BERT for masked language modeling
- `finetune_bert_regression` - Fine-tunes BERT for calorie prediction

### Evaluation Commands
- `evaluate_bert_regression` - Evaluates regression model performance

## Project Structure

```
├── data/               # Dataset storage
│   ├── raw/           # Raw USDA data
│   ├── train/         # Training splits
│   └── eval/          # Evaluation splits
├── models/            # Model implementations
├── src/               # Source code
│   ├── api/          # API implementation
│   ├── data_process/ # Data processing utilities
│   ├── evaluation/   # Model evaluation
│   ├── models/       # Model training
│   ├── nlp/          # NLP components
│   └── utils/        # Helper functions
└── tests/            # Unit tests
```

## Technical Details

- Uses BERT-based models fine-tuned on food descriptions
- Implements both masked language modeling and regression tasks
- FAISS indexing for efficient similarity search
- Data sampling with stratification to reduce bias
- Modular architecture for easy extension

## Development Status

- [x] USDA data fetching and processing
- [x] Basic BERT fine-tuning
- [x] Regression model implementation
- [ ] Improved embedding extraction (instead of creating an index from the embedding matrix of the finetuned model, create a new embedding space comprised of the sentence embeddings of all the raw descriptions (along with an ID) from the dataset, to allow reverse search.)
- [ ] Enhanced NER for ingredients
- [ ] Extended validation datasets
- [ ] API deployment

## Requirements

See pyproject.toml for complete dependencies:
- PyTorch
- Transformers
- FAISS-CPU
- Python-dotenv
- Accelerate