import numpy as np
from tqdm import tqdm
from bonner.datasets.allen2021_natural_scenes._stimuli import load_captions

from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
import logging

from visreps.dataloaders.neural import load_nsd_data

logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

output_path = "datasets/neural/nsd/gemini_representations.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

nsd_stimuli_cfg = {"neural_dataset": "nsd", "region": "ventral visual stream", "subject_idx": 0}

_, stimuli_dict = load_nsd_data(nsd_stimuli_cfg)
captions = load_captions()
print("Length of stimuli_dict: ", len(stimuli_dict))

gemini_representations = [] 
stimulus_ids = []
for stim_id in tqdm(stimuli_dict.keys(), desc="Extracting Gemini representations"):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=captions[int(stim_id)],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768)
    )
    embedding_np = np.array([np.array(result.embeddings[i].values) for i in range(len(captions[int(stim_id)]))])
    embedding_np = embedding_np.mean(axis=0)
    gemini_representations.append(embedding_np)
    stimulus_ids.append(stim_id)

np.savez_compressed(output_path, gemini_representations=gemini_representations, stimulus_ids=stimulus_ids)
print(f"Saved Gemini representations and stimulus ids to {output_path}")