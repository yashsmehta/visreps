from bonner.datasets.allen2021_natural_scenes._stimuli import load_captions
captions = load_captions()
print(captions[123])  # list[str] for NSD stimulus 123

from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?",
    config=types.EmbedContentConfig(output_dimensionality=768)
)

[embedding_obj] = result.embeddings
embedding_length = len(embedding_obj.values)

print(f"Length of embedding: {embedding_length}")