from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('USERNAME')
os.environ['KAGGLE_TOKEN'] = os.getenv('API_TOKEN')

def extract(dataset_dir: str, path: str):

    api = KaggleApi()
    api.authenticate()

    try:
        api.dataset_download_files(dataset_dir, path=path, unzip=True)
    except Exception as e:
        raise Exception(f"Unable to download dataset: {e}") 


if __name__ == "__main__":
    extract("bwandowando/philippine-spam-sms-messages", "./data/raw")