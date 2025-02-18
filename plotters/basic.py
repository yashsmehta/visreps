import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

# Define layer names and RSA scores for each experiment
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
rsa_exp1 = [0.0566, 0.0820, 0.0388, 0.0678, 0.1187, 0.1517, 0.1539, 0.1142]  # PCA 2 classes
rsa_exp2 = [0.0576, 0.0737, 0.0328, 0.0715, 0.0653, 0.1381, 0.1756, 0.1792]  # Default tiny imagine

# Create x-axis positions
x = range(len(layers))

# Create figure with higher resolution
plt.figure(figsize=(12, 7))

# Plot with enhanced styling
sns.lineplot(x=x, y=rsa_exp1, marker='o', linewidth=2, markersize=8, label='Exp1: PCA 2 classes')
sns.lineplot(x=x, y=rsa_exp2, marker='s', linewidth=2, markersize=8, label='Exp2: Default tiny imagine')

# Customize the plot
plt.xticks(x, layers, rotation=45)
plt.xlabel('Layer', fontsize=12, fontweight='bold')
plt.ylabel('RSA Score', fontsize=12, fontweight='bold')
plt.title('Representational Similarity Analysis Across Network Layers', fontsize=14, pad=20)

# Enhance legend
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high DPI
plt.savefig('rsa_analysis.png', dpi=300, bbox_inches='tight')
plt.close()