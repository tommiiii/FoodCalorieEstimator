import json
import os
from dotenv import load_dotenv

load_dotenv()
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", 1000))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")
allowed_fields = ["fdcId", "description", "commonNames", "scientificName", "additionalDescriptions", "foodNutrients"]

filtered_data = []
with open(os.path.join(DATA_PATH, "raw", f"usda_food_foundation_data_{SAMPLE_SIZE}_entries.json"), "r") as f:
    data = json.load(f)
    for item in data:
        filtered_item = {field: item[field] for field in allowed_fields if field in item}
        filtered_data.append(filtered_item)

output_file = os.path.join(DATA_PATH, "processed", "usda_food_foundation_data_filtered.json")
with open(output_file, "w") as f:
    json.dump(filtered_data, f, indent=2)