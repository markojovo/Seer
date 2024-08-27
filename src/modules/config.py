import os
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ['FRED_API_KEY']
NEWS_API_KEY = os.environ['NEWS_API_KEY']