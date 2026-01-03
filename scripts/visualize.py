import matplotlib.pyplot as plt
import pandas as pd

def tradeoff(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Color grouped by Model
    unique_models = df['Model'].unique()
    colors_model = plt.cm.tab10(range(len(unique_models)))
    color_map_model = dict(zip(unique_models, colors_model))
    
    for model in unique_models:
        model_data = df[df['Model'] == model]
        axes[0].scatter(model_data['Bias Score'], model_data['Cultural Accuracy'], 
                       color=color_map_model[model], label=model, s=100)
    
    axes[0].set_xlabel('Bias Score (closer to 0 is better)')
    axes[0].set_ylabel('Cultural Accuracy (higher is better)')
    axes[0].set_title('Grouped by Model')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True)
    
    # Subplot 2: Color grouped by Culture
    unique_cultures = df['Culture'].unique()
    colors_culture = plt.cm.tab10(range(len(unique_cultures)))
    color_map_culture = dict(zip(unique_cultures, colors_culture))
    
    for culture in unique_cultures:
        culture_data = df[df['Culture'] == culture]
        axes[1].scatter(culture_data['Bias Score'], culture_data['Cultural Accuracy'], 
                       color=color_map_culture[culture], label=culture, s=100)
    
    axes[1].set_xlabel('Bias Score (closer to 0 is better)')
    axes[1].set_ylabel('Cultural Accuracy (higher is better)')
    axes[1].set_title('Grouped by Culture')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True)
    
    # Subplot 3: Color grouped by Language
    unique_languages = df['Language'].unique()
    colors_language = plt.cm.tab10(range(len(unique_languages)))
    color_map_language = dict(zip(unique_languages, colors_language))
    
    for language in unique_languages:
        language_data = df[df['Language'] == language]
        axes[2].scatter(language_data['Bias Score'], language_data['Cultural Accuracy'], 
                       color=color_map_language[language], label=language, s=100)
    
    axes[2].set_xlabel('Bias Score (closer to 0 is better)')
    axes[2].set_ylabel('Cultural Accuracy (higher is better)')
    axes[2].set_title('Grouped by Language')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('tradeoff_plot.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('single_result_table.csv')
    tradeoff(df)