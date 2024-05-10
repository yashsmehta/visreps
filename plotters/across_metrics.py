import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('logs/data_augmentation.csv', dtype={'conv_trainable': str, 'fc_trainable': str})
df = df[(df['region'] == 'OTC') & (df['cv_split'] == 'test') & (df['conv_trainable'] == '11111') & (df['fc_trainable'] == '111')]

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['model_layer_index'] = df['model_layer_index'] / max(df['model_layer_index'])
df['conv_trainable'] = df['conv_trainable'].astype(str)
df['fc_trainable'] = df['fc_trainable'].astype(str)

sns.set_theme(style="whitegrid", font_scale=1.5)  # Set a clean and professional style with larger fonts for readability
fig, axes = plt.subplots(1, 3, figsize=(21, 7))  # Creating 3 subplots for each metric with increased size for clarity

metrics = ['srpr', 'ersa', 'crsa']
base_colors = ['aquamarine', 'skyblue', 'orange']

for i, metric in enumerate(metrics):
    metric_df = df[df['metric'] == metric]
    unique_epochs = metric_df['epoch'].nunique()
    palette = sns.cubehelix_palette(n_colors=unique_epochs, start=i, rot=-0.5, dark=0.3, light=0.7, reverse=True)
    lineplot = sns.lineplot(ax=axes[i], data=metric_df, x='model_layer_index', y='score', hue='epoch', 
                            palette=palette, linewidth=3)

    lineplot.set_title(f'{metric.upper()}', fontsize=18)
    lineplot.set_xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
    lineplot.set_ylabel('Score', fontsize=16)
    lineplot.legend(title='Epoch', title_fontsize='14', fontsize='12', loc='upper left', frameon=True, shadow=True)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('plotters/across_metrics.png', dpi=600, bbox_inches='tight')
print('Saved fig!')
