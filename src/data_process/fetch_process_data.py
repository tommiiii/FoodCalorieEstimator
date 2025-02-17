import os
import json
import math
import random
import requests
import dotenv

def init_config():
    dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    return {
        "api_key": os.getenv("USDA_API_KEY", ""),
        "base_url": "https://api.nal.usda.gov/fdc/v1/foods/search",
        "data_type": ["Foundation", "Survey (FNDDS)"],
        "sample_size": int(os.getenv("SAMPLE_SIZE", 1000)),
        "allowed_fields": ["fdcId", "description", "commonNames", "scientificName", "additionalDescriptions", "foodNutrients"]
    }

def fetch_data():
    config = init_config()
    
    params = {
        "api_key": config["api_key"],
        "query": "*",
        "dataType": config["data_type"],
        "pageSize": 1
    }
    response = requests.get(config["base_url"], params=params)
    data = response.json()
    total_count = data.get("totalHits", 0)
    print(f"Total available '{'+'.join(config['data_type'])}' entries:", total_count)

    page_size = 20
    total_pages = math.ceil(total_count / page_size)
    num_pages_to_fetch = config["sample_size"] // page_size
    max_page = min(total_pages, num_pages_to_fetch * 2)
    pages = list(range(1, max_page + 1))
    random.shuffle(pages)
    random_pages = pages[:min(num_pages_to_fetch, max_page)]

    filtered_entries = []
    print(f"Fetching {len(random_pages)} pages of {page_size} entries each")
    
    for i, page in enumerate(random_pages, start=1):
        params = {
            "api_key": config["api_key"],
            "dataType": config["data_type"],
            "pageSize": page_size,
            "pageNumber": page
        }
        response = requests.get(config["base_url"], params=params)
        if response.status_code == 200:
            page_data = response.json()
            foods = page_data.get("foods", [])
            for item in foods:
                filtered_item = {field: item[field] for field in config["allowed_fields"] if field in item}
                filtered_entries.append(filtered_item)
            print(f"{((i/len(random_pages))*100):0,.1f}% - {i}/{len(random_pages)}")
        else:
            print(f"Failed to fetch page {page}: Status code {response.status_code}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"usda_{len(filtered_entries)}.json")
    
    with open(output_file, "w") as f:
        json.dump(filtered_entries, f, indent=2)
    
    print(f"Raw data saved to {output_file}")
    return output_file

def process_data():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    raw_dir = os.path.join(output_dir, "raw")
    
    # Get the most recent raw data file
    latest_file = max([f for f in os.listdir(raw_dir) if f.startswith("usda_")],
                     key=lambda x: os.path.getmtime(os.path.join(raw_dir, x)))
    
    with open(os.path.join(raw_dir, latest_file)) as f:
        filtered_entries = json.load(f)

    filtered_entries.sort(key=lambda entry: entry.get("description", "").lower())
    model_data = [(x.get("description", ""), 
                   max([nutrient.get("value") for nutrient in x.get("foodNutrients", [])
                        if nutrient.get("unitName") == "KCAL" and nutrient.get("value") is not None] or [0]))
                  for x in filtered_entries]

    split_index = math.floor(len(model_data) * 0.8)
    train_data = model_data[:split_index]
    eval_data = model_data[split_index:]

    # Save training data
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    output_file_train = os.path.join(train_dir, f"usda_{len(train_data)}.json")
    with open(output_file_train, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"Training data saved to {output_file_train}")

    # Save evaluation data
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_file_eval = os.path.join(eval_dir, f"usda_{len(eval_data)}.json")
    with open(output_file_eval, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Evaluation data saved to {output_file_eval}")

def fetch_and_process():
    fetch_data()
    process_data()

if __name__ == "__main__":
    fetch_and_process()