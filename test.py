from dotenv import load_dotenv
import os
load_dotenv(override=True)
api_key = os.getenv('PINECONE_KEY')
if api_key is None:
    print("Environment variable not found.")
else:
    print(f"API Key: {api_key}")
