"""
test_api.py
-----------
Milestone 3 – API Endpoint Tests
Tests all endpoints locally before Azure deployment
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_result(name, response):
    status = "✅" if response.status_code == 200 else "❌"
    print(f"{status} [{response.status_code}] {name}")
    if response.status_code == 200:
        data = response.json()
        print(f"   → {json.dumps(data, indent=2)[:300]}...")
    else:
        print(f"   → Error: {response.text[:200]}")
    print()

def run_tests():
    print("=" * 55)
    print("  Demand Forecasting API – Endpoint Tests")
    print("=" * 55)
    print()

    # 1. Root
    r = requests.get(f"{BASE_URL}/")
    print_result("GET /  (Root)", r)

    # 2. Health
    r = requests.get(f"{BASE_URL}/health")
    print_result("GET /health", r)

    # 3. Models info
    r = requests.get(f"{BASE_URL}/models/info")
    print_result("GET /models/info", r)

    # 4. Metrics
    r = requests.get(f"{BASE_URL}/metrics")
    print_result("GET /metrics", r)

    # 5. Single prediction
    payload = {
        "date":         "2024-06-01",
        "store_id":     1,
        "product_id":   5,
        "category":     "Electronics",
        "store_type":   "Urban",
        "price":        149.99,
        "stock_level":  30,
        "is_promotion": 1,
        "is_holiday":   0,
        "model":        "lightgbm",
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print_result("POST /predict (single)", r)

    # 6. Batch predictions
    batch_payload = {
        "predictions": [
            {**payload, "date": "2024-06-01", "product_id": 1},
            {**payload, "date": "2024-06-02", "product_id": 2, "category": "Food",     "price": 12.5},
            {**payload, "date": "2024-06-03", "product_id": 3, "category": "Clothing", "price": 89.0},
        ]
    }
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
    print_result("POST /predict/batch (3 items)", r)

    # 7. 30-day forecast
    r = requests.get(f"{BASE_URL}/forecast/30?store_id=1&product_id=5&category=Electronics&store_type=Urban&price=149.99")
    print_result("GET /forecast/30", r)

    # 8. Inventory recommendations
    r = requests.get(f"{BASE_URL}/inventory/recommend?store_id=1&store_type=Urban&safety_days=7")
    print_result("GET /inventory/recommend", r)

    print("=" * 55)
    print("  All tests complete!")
    print("=" * 55)

if __name__ == "__main__":
    run_tests()
