import requests
from dotenv import main
import os 
_ = main.load_dotenv(main.find_dotenv())

API_BASE_URL = os.getenv("API_BASE_URL")
def get_response(query: str = ""):
    url = API_BASE_URL
    payload = {"text": query}
    response = requests.post(url, json=payload)
    return response.json()


