import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_filter_data(file_path, filters, dtype_dict):
    df = pd.read_csv(file_path, dtype=dtype_dict)
    query_string = " and ".join(filters)
    return df.query(query_string)

if __name__ == '__main__':
    dtype_dict = {
        'conv_trainable': 'string', 
        'fc_trainable': 'string', 
        'score': 'float64'
    }

    filters_single = [
        "region == 'OTC'",
        "metric == 'srpr'",
        "cv_split == 'test'",
        "epoch == 50",
        "(conv_trainable.str.count('1') + fc_trainable.str.count('1')) == 1"
    ]

    df = load_and_filter_data('logs/single_train.csv', filters_single, dtype_dict)
    df['model_layer_index'] = df['model_layer_index'] / df['model_layer_index'].max()

    df_conv_trainable = df.query("fc_trainable == '000' and conv_trainable.str.count('1') == 1")
    df_fc_trainable = df.query("conv_trainable == '00000' and fc_trainable.str.count('1') == 1")

    filters_full = [
        "data_augment == True",
        "region == 'OTC'",
        "metric == 'srpr'",
        "cv_split == 'test'",
        "epoch == 50"
    ]

    df_full_train = load_and_filter_data('logs/full_train.csv', filters_full, dtype_dict)
    df_full_train['model_layer_index'] = df_full_train['model_layer_index'] / df_full_train['model_layer_index'].max()

    sns.set_theme(style="whitegrid", font_scale=1.25)
    plt.figure(figsize=(12, 8))

    sns.set_theme(style="whitegrid", font_scale=1.25)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    df_conv_trainable_sorted = df_conv_trainable.sort_values(by='conv_trainable', ascending=False)
    sns.lineplot(ax=axes[0], data=df_conv_trainable_sorted, x='model_layer_index', y='score', 
                hue='conv_trainable', palette=sns.color_palette("Blues", n_colors=df_conv_trainable_sorted['conv_trainable'].nunique()), linewidth=2.5)

    sns.lineplot(ax=axes[0], data=df_full_train, x='model_layer_index', y='score', 
                color='orange', linewidth=2.5, alpha=0.7)
    axes[0].set_title('Single layer conv trainable', fontsize=20)
    axes[0].set_xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
    axes[0].set_ylabel('Encoding Score (SRPR)', fontsize=16)

    df_fc_trainable_sorted = df_fc_trainable.sort_values(by='fc_trainable', ascending=False)
    sns.lineplot(ax=axes[1], data=df_fc_trainable_sorted, x='model_layer_index', y='score', hue='fc_trainable', 
                palette=sns.color_palette("Greens", n_colors=df_fc_trainable_sorted['fc_trainable'].nunique()), linewidth=2.5)

    sns.lineplot(ax=axes[1], data=df_full_train, x='model_layer_index', y='score', 
                color='orange', linewidth=2.5, alpha=0.7)

    axes[1].set_title('Single layer fc trainable', fontsize=20)
    axes[1].set_xlabel('Model Depth (Normalized Layer Index)', fontsize=16)
    axes[1].set_ylabel('')

    y_min = min(df_full_train['score'].min(), df_fc_trainable['score'].min(), df_conv_trainable['score'].min())
    y_max = max(df_full_train['score'].max(), df_fc_trainable['score'].max(), df_conv_trainable['score'].max())
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    axes[0].legend(title='Conv Trainable', title_fontsize='14', fontsize='13', loc='upper left', frameon=True, shadow=True)
    axes[1].legend(title='FC Trainable', title_fontsize='14', fontsize='13', loc='upper left', frameon=True, shadow=True)

    for axis in ['top','bottom','left','right']:
        axes[0].spines[axis].set_linewidth(3)
        axes[1].spines[axis].set_linewidth(3)

    sns.despine(trim=True, offset=10)

    plt.savefig('plotters/single_train.png', dpi=600, bbox_inches='tight')
