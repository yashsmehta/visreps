import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('logs/brainscore_training.csv')

df_filtered = df[(df['region'] == 'OTC') & (df['metric'] == 'srpr') & (df['cv_split'] == 'train')]
df_filtered['score'] = pd.to_numeric(df_filtered['score'], errors='coerce')
df_filtered['model_layer_index'] = df_filtered['model_layer_index'] / max(df_filtered['model_layer_index'])

sns.set_theme(style="ticks", font_scale=1.25)  # Enhanced style and font scaling for better visual appeal
plt.figure(figsize=(12, 8))  # Increased figure size for better detail visibility

lineplot = sns.lineplot(data=df_filtered, x='model_layer_index', y='score', hue='epoch', 
                        palette='mako', linewidth=2.5)

# Setting more descriptive and larger title and labels
lineplot.set_title('Training Dynamics of ~AlexNet on Tiny ImageNet', fontsize=20)
lineplot.set_xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
lineplot.set_ylabel('Encoding Score (SRPR)', fontsize=16)

# Customizing the legend to be more informative and visually appealing
plt.legend(title='Training Epoch', title_fontsize='14', fontsize='13', loc='upper left', frameon=True, shadow=True)

# Making axes thicker
for axis in ['top','bottom','left','right']:
  lineplot.spines[axis].set_linewidth(3)

sns.despine(trim=True)  # Trimming the spines for a cleaner look

plt.savefig('plotters/test.png', dpi=600, bbox_inches='tight')
