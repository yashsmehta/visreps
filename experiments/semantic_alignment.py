import numpy as np

features_path = "datasets/neural/nsd/gemini_representations.npz"
data = np.load(features_path, allow_pickle=True)
gemini_representations = data['gemini_representations']
stimulus_ids = data['stimulus_ids']

print(gemini_representations.shape)
print(stimulus_ids.shape)

exit()