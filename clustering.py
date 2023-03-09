# imports
import numpy as np
import pandas as pd
import os
import openai
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from openai.embeddings_utils import get_embedding
import apply_embedding
from dotenv import load_dotenv

# load data
datafile_path = "data/daset_with_embeddings.csv"

# credentials 
load_dotenv()
openai.api_key = os.environ["AZURE_AOAI_KEY"]
openai.api_type = os.environ["AZURE_AOAI_TYPE"]
openai.api_base = os.environ["AZURE_AOAI_ENDP"]
openai.api_version = os.environ["AZURE_AOAI_VER"]

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.embedding.values)
matrix.shape

# find the clusters using k-means
n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
df["Cluster"] = labels

rev_per_cluster = 5

with open('data/categories.txt', 'w') as file:
    for i in range(n_clusters):
        print(f"Cluster {i} Theme:", end=" ", file=file)

        descriptions = "\n".join(
            df[df.Cluster == i]
            .cambio.str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .sample(rev_per_cluster, random_state=42)
            .values
        )
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f'What do the following descriptions have in common?\n\nDescriptions:\n"""\n{descriptions}\n"""\n\nTheme:',
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(response["choices"][0]["text"].replace("\n", ""), file=file)

        sample_cluster_rows = df[df.Cluster == i].sample(rev_per_cluster, random_state=42)
        for j in range(rev_per_cluster):
            print(sample_cluster_rows.cambio.str[:70].values[j], file=file)

        print("-" * 100, file=file)