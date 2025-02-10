import os
import json
import math
import random
import requests
import dotenv

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
API_KEY = os.getenv("USDA_API_KEY", "")
BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
DATA_TYPE = ["Foundation", "Survey (FNDDS)"]
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", 1000))
ALLOWED_FIELDS = ["fdcId", "description", "commonNames", "scientificName", "additionalDescriptions", "foodNutrients"]

params = {
    "api_key": API_KEY,
    "query": "*",
    "dataType": DATA_TYPE,
    "pageSize": 1
}
response = requests.get(BASE_URL, params=params)
data = response.json()
total_count = data.get("totalHits", 0) 
print(f"Total available '{" + ".join(DATA_TYPE)}' entries:", total_count)

page_size = 20 
total_pages = math.ceil(total_count / page_size)

num_pages_to_fetch = SAMPLE_SIZE // page_size
max_page = min(total_pages, num_pages_to_fetch * 2)
pages = list(range(1, max_page + 1))
random.shuffle(pages)
random_pages = pages[:min(num_pages_to_fetch, max_page)]

all_entries = []
filtered_entries = []

print(f"Fetching {len(random_pages)} pages of {page_size} entries each")
for i, page in enumerate(random_pages, start=1):
    params = {
        "api_key": API_KEY,
        "dataType": DATA_TYPE, 
        "pageSize": page_size,
        "pageNumber": page
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        page_data = response.json()
        foods = page_data.get("foods", [])
        for item in foods:
            filtered_item = {field: item[field] for field in ALLOWED_FIELDS if field in item}
            filtered_entries.append(filtered_item)
        print(f"{((i/len(random_pages))*100):0,.1f}% - {i}/{len(random_pages)}")
    else:
        print(f"Failed to fetch page {page}: Status code {response.status_code}")

print("Total fetched entries:", len(filtered_entries))

filtered_entries.sort(key=lambda entry: entry.get("description", "").lower())

model_data = [(x.get("description", ""), max([nutrient.get("value") for nutrient in x.get("foodNutrients", []) if nutrient.get("unitName") == "KCAL" and nutrient.get("value") is not None] or [0])) for x in filtered_entries]

split_index = math.floor(len(model_data) * 0.8)
train_data = model_data[:split_index]
eval_data = model_data[split_index:]

output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

output_file_train = os.path.join(os.path.join(output_dir, "train"), f"usda_{len(train_data)}.json")
with open(output_file_train, "w") as f:
    json.dump(train_data, f, indent=2)
    
print(f"Training data saved to {output_file_train}")
    
output_file_eval = os.path.join(os.path.join(output_dir, "eval"), f"usda_{len(eval_data)}.json")
with open(output_file_eval, "w") as f:
    json.dump(eval_data, f, indent=2)
    
print(f"Evaluation data saved to {output_file_eval}")

output_all_file = os.path.join(output_dir, "raw", f"usda_{len(filtered_entries)}.json")

with open(output_all_file, "w") as f:
    json.dump(filtered_entries, f, indent=2)