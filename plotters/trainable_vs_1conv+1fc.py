import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('logs/partial_training.csv', dtype={'conv_trainable': str, 'fc_trainable': str})
df = df[(df['region'] == 'OTC') & (df['metric'] == 'srpr') & (df['cv_split'] == 'test') & (df['epoch'] == 40) & 
        ((df['conv_trainable'] == '00001') & (df['fc_trainable'] != '000') | 
         (df['conv_trainable'] == '11111') & (df['fc_trainable'] == '111'))]

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['model_layer_index'] = df['model_layer_index'] / max(df['model_layer_index'])
df['conv_trainable'] = df['conv_trainable'].astype(str)
df['fc_trainable'] = df['fc_trainable'].astype(str)
print(df.shape)

sns.set_theme(style="ticks", font_scale=1.25)  # Enhanced style and font scaling for better visual appeal
plt.figure(figsize=(12, 8))  # Increased figure size for better detail visibility

# Mapping fc_trainable values to more descriptive labels
df['trainable_description'] = df.apply(lambda x: 'all trainable' if x['conv_trainable'] == '11111' and x['fc_trainable'] == '111' else 'last conv+ x fc', axis=1)

lineplot = sns.lineplot(data=df, x='model_layer_index', y='score', hue='trainable_description', 
                        palette='mako', linewidth=2.5)

lineplot.set_title('Trainable Layers Impact on Encoding Score', fontsize=20)
lineplot.set_xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
lineplot.set_ylabel('Encoding Score (SRPR)', fontsize=16)

# Customizing the legend to be more informative and visually appealing
plt.legend(title='Trainable Layers', title_fontsize='14', fontsize='13', loc='upper left', frameon=True, shadow=True)

# Making axes thicker
for axis in ['top','bottom','left','right']:
  lineplot.spines[axis].set_linewidth(3)

sns.despine(trim=True)  # Trimming the spines for a cleaner look

plt.savefig('plotters/test.png', dpi=600, bbox_inches='tight')
