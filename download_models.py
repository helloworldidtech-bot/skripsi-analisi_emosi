import os
import requests

MODELS = {
    "best_model_70_30.pth": os.environ.get("MODEL70_URL", "PUT_URL_HERE"),
    "best_model_80_20.pth": os.environ.get("MODEL80_URL", "PUT_URL_HERE"),
}

for name, url in MODELS.items():
    if os.path.exists(name):
        print(f"{name} already exists, skipping")
        continue
    if url == "PUT_URL_HERE" or not url:
        print(f"Set a download URL for {name} in environment variables or edit this script.")
        continue
    print(f"Downloading {name}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Saved {name}")
