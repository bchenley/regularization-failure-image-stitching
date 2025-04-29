# Brandon Henley

import os

folders = [
    "report",
    "data/raw",
    "data/processed",
    "data/matches",
    "notebooks",
    "scripts",
    "src/data",
    "src/models",
    "src/homography",
    "src/evaluation",
    "outputs/logs",
    "outputs/results",
    "outputs/models",
    "outputs/figures"
]

def create_folders(base_path="."):
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

if __name__ == "__main__":
    print("Building project scaffold...")
    create_folders()
    print("Done!")