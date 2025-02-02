import os
import json
import math
import random
import requests
import dotenv

dotenv.load_dotenv()
API_KEY = os.getenv("USDA_API_KEY", "")
BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
DATA_TYPE = ["Foundation", "Survey (FNDDS)"]
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", 1000))

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
        all_entries.extend(foods)
        print(f"{((i/len(random_pages))*100):0,.1f}% - {i}/{len(random_pages)}")
    else:
        print(f"Failed to fetch page {page}: Status code {response.status_code}")

print("Total fetched entries:", len(all_entries))

all_entries.sort(key=lambda entry: entry.get("description", "").lower())

output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"usda_food_foundation_data_{SAMPLE_SIZE}_entries.json")
with open(output_file, "w") as f:
    json.dump(all_entries, f, indent=2)
print(f"Raw data saved to {output_file}")