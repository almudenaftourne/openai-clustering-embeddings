# imports
# imports
import pandas as pd
import tiktoken 
from dotenv import load_dotenv
import os
import openai
from openai.embeddings_utils import get_embedding
import time 
import apply_embedding

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base" # encoding for text-embedding-ada-002 
max_tokens = 2000 
dataset_path = "data/dataset.csv"

load_dotenv()
openai.api_key = os.environ["AZURE_AOAI_KEY"]
openai.api_type = os.environ["AZURE_AOAI_TYPE"]
openai.api_base = os.environ["AZURE_AOAI_ENDP"]
openai.api_version = os.environ["AZURE_AOAI_VER"]

# Load & inspect dataset
df = pd.read_csv(dataset_path, sep='delimiter', engine='python')
df = df.dropna()
df.head(2)

top_n = 1000
encoding = tiktoken.get_encoding(embedding_encoding)

# omit descrptions that are too long to embed
df["n_tokens"] = df.cambio.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
len(df)

def apply_embedding(x):
    time.sleep(1) # add 1 second delay
    return get_embedding(x, engine=embedding_model)

df["embedding"] = df.cambio.apply(lambda x: apply_embedding(x))
df.to_csv("data/daset_with_embeddings.csv")
