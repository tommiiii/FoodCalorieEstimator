# Food Calorie Estimator

- **Project Overview:**
  - Builds a Python API to estimate meal calorie content from a natural language description.
  - Utilizes advanced NLP with Named Entity Recognition (NER) and contextual embeddings.

- **Key Features:**
  - **Data Fetching:**  
    - Retrieves food entries from USDA FoodData Central.
    - Implements randomized and stratified sampling techniques to reduce bias.
  - **Data Processing:**  
    - Filters and processes raw food data for further analysis.
    - Stores filtered data under `data/processed/`.
  - **NLP & Embeddings:**  
    - Fine-tunes BERT to provide better domain specific embeddings.
    - Builds an embedding space using contextual representations.
    - Uses FAISS for fast semantic similarity search.

- **Implementation Highlights:**
  - NER and similarity search for handling ingredients not directly found in the dataset.
  - Data processing scripts ensure diverse data extraction using randomized page sampling.
  - Modular structure for ease of maintenance and future improvements.

- **Getting Started:**
  1. Set up your environment and install dependencies (e.g., PyTorch, Transformers, FAISS).
  2. Create a `.env` file with `SAMPLE_SIZE` and `USDA_API_KEY` required for data fetching.
  3. Run `fetch_data.py` to download raw USDA data.
  4. Execute `filter_data.py` to process and filter the food data.
  5. Fine-tune BERT using `finetune_bert.py`.
  6. Generate and index embeddings in `embeddings.py`.
  7. Launch the API (if integrated) via the modules under `src/api/`.

- **Next Steps:**
  - Enhance ingredient extraction with NER techniques.
  - Expand evaluation with separate validation datasets.
  - Implement additional sampling strategies to further reduce data bias.

- **Notes:**
  - The project is under active development; contributions and suggestions are welcome.
  - Ensure to update file paths and environment variables as needed.
