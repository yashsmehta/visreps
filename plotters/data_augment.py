import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
  df = pd.read_csv('logs/data_augment.csv')

  df_filtered = df[(df['region'] == 'OTC') & (df['metric'] == 'srpr') & (df['cv_split'] == 'test') & (df['epoch'] == 50)].copy()
  df_filtered['score'] = pd.to_numeric(df_filtered['score'], errors='coerce')

  max_model_layer_index = df_filtered['model_layer_index'].max()
  df_filtered['model_layer_index'] = df_filtered['model_layer_index'].astype(float) / max_model_layer_index

  sns.set_theme(style="ticks", font_scale=1.25)
  plt.figure(figsize=(12, 8))

  lineplot = sns.lineplot(data=df_filtered, x='model_layer_index', y='score', hue='data_augment', 
                          style='data_augment', palette='viridis', linewidth=2.5)

  lineplot.set_title('Effect of Data Augmentation on Encoding Score', fontsize=20)
  lineplot.set_xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
  lineplot.set_ylabel('Encoding Score (SRPR)', fontsize=16)

  plt.legend(title='Data Augmentation', title_fontsize='14', fontsize='13', loc='upper left', frameon=True, shadow=True)

  for axis in ['top','bottom','left','right']:
    lineplot.spines[axis].set_linewidth(3)

  sns.despine(trim=True)

  plt.savefig('plotters/data_augment.png', dpi=600, bbox_inches='tight')
  print("saved fig!")
