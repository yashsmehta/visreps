import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
  df = pd.read_csv('logs/1conv1fc.csv', dtype={'conv_trainable': str, 'fc_trainable': str})
  df = df[(df['region'] == 'OTC') & (df['metric'] == 'srpr') & (df['cv_split'] == 'test') & (df['epoch'] == 50)]

  df['score'] = pd.to_numeric(df['score'], errors='coerce')
  df['model_layer_index'] = df['model_layer_index'] / max(df['model_layer_index'])

  sns.set_theme(style="ticks", font_scale=1.25)
  plt.figure(figsize=(10, 7))

  df_all_trainable = pd.read_csv('logs/full_train.csv')
  df_all_trainable = df_all_trainable[(df_all_trainable['region'] == 'OTC') & (df_all_trainable['metric'] == 'srpr') & (df_all_trainable['cv_split'] == 'test') & (df_all_trainable['epoch'] == 50)]
  df_all_trainable['score'] = pd.to_numeric(df_all_trainable['score'], errors='coerce')
  df_all_trainable['model_layer_index'] = df_all_trainable['model_layer_index'] / max(df_all_trainable['model_layer_index'])

  df_1stconv = df[df['conv_trainable'] == '10000'].sort_values(by='fc_trainable', ascending=False)
  df_lastconv = df[df['conv_trainable'] == '00001'].sort_values(by='fc_trainable', ascending=False)

  palette_1stconv = sns.color_palette("Blues", n_colors=3)
  palette_lastconv = sns.color_palette("Reds", n_colors=3)


  lineplot_1stconv = sns.lineplot(data=df_1stconv, x='model_layer_index', y='score', hue='fc_trainable', 
                                  palette=palette_1stconv, linewidth=2.5)
  lineplot_lastconv = sns.lineplot(data=df_lastconv, x='model_layer_index', y='score', hue='fc_trainable', 
                                  palette=palette_lastconv, linewidth=2.5)

  plt.title('1 Conv + 1 FC', fontsize=20)
  plt.xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
  plt.ylabel('Encoding Score (SRPR)', fontsize=16)

  handles, labels = plt.gca().get_legend_handles_labels()
  plt.legend(handles=handles, labels=labels, title='FC Trainable Layers', title_fontsize='14', fontsize='13', loc='upper left', frameon=True, shadow=True)

  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(3)

  sns.despine(trim=True)

  # Adding text annotation for line colors
  plt.text(0.95, 0.2, 'Last conv (Blue), First conv (Red)', verticalalignment='bottom', horizontalalignment='right',
          transform=plt.gca().transAxes, color='black', fontsize=12)

  plt.savefig('plotters/1conv1fc.png', dpi=600, bbox_inches='tight')
  print("saved fig!")

