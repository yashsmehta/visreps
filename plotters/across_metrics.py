import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('logs/partial_training.csv', dtype={'conv_trainable': str, 'fc_trainable': str})
df = df[(df['region'] == 'OTC') & (df['cv_split'] == 'test') & (df['conv_trainable'] == '11111') & (df['fc_trainable'] == '111')]

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['model_layer_index'] = df['model_layer_index'] / max(df['model_layer_index'])
df['conv_trainable'] = df['conv_trainable'].astype(str)
df['fc_trainable'] = df['fc_trainable'].astype(str)
print(df.shape)

sns.set_theme(style="ticks", font_scale=1.25)  # Enhanced style and font scaling for better visual appeal
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Creating 3 subplots for each metric

metrics = ['srpr', 'ersa', 'crsa']
for i, metric in enumerate(metrics):
    metric_df = df[df['metric'] == metric]
    lineplot = sns.lineplot(ax=axes[i], data=metric_df, x='model_layer_index', y='score', hue='epoch', 
                            palette='mako', linewidth=2.5)
    lineplot.set_title(f'Encoding Score ({metric.upper()})', fontsize=16)
    lineplot.set_xlabel('Model Depth (Normalized Layer Index)', fontsize=14)
    lineplot.set_ylabel('Score', fontsize=14)
    lineplot.legend(title='Epoch', title_fontsize='12', fontsize='11', loc='upper left', frameon=True, shadow=True)

for axis in ['top','bottom','left','right']:
    for ax in axes:
        ax.spines[axis].set_linewidth(2)

sns.despine(trim=True)  # Trimming the spines for a cleaner look

plt.tight_layout()
plt.savefig('plotters/test_metrics.png', dpi=600, bbox_inches='tight')
