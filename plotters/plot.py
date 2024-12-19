import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('model_checkpoints/base/cfg1/results.csv')

# Filter the DataFrame for the specific layers
filtered_df = df[df['layer'].isin(['conv.5', 'fc.3'])]

# Plot the data
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_df, x='epoch', y='rsa_score', hue='layer', palette=['b', 'r'], marker='o')

# Set plot labels and title
plt.xlabel('Epoch')
plt.ylabel('RSA Score')
plt.title('RSA Score over training epochs')

# Save the plot
plt.savefig('plotters/imgs/rsa_score_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved as 'rsa_score_plot.png'")