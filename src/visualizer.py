import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.makedirs('outputs', exist_ok=True)


def plot_eda(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Exploratory Data Analysis', fontsize=18)

    sns.histplot(df['Price'], kde=True, color='#3E8EDE', ax=axes[0, 0])
    axes[0, 0].set_title('Price Distribution')

    sns.boxplot(x='Location', y='Price', data=df, palette='Set2', ax=axes[0, 1])
    axes[0, 1].set_title('Price by Location')
    axes[0, 1].tick_params(axis='x', rotation=15)

    sns.scatterplot(x='Area_sqft', y='Price', hue='Location', data=df, ax=axes[0, 2], palette='tab10', alpha=0.7)
    axes[0, 2].set_title('Area vs Price')

    sns.boxplot(x='Bedrooms', y='Price', data=df, palette='Set3', ax=axes[1, 0])
    axes[1, 0].set_title('Bedrooms vs Price')

    sns.boxplot(x='Condition', y='Price', data=df, palette='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Condition vs Price')

    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Heatmap')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('outputs/eda_plots.png', dpi=150)
    plt.close(fig)


def plot_model_performance(test_results, y_test, y_pred, model, feature_columns):
    results_df = pd.DataFrame(test_results).T.reset_index().rename(columns={'index': 'Model'})

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.barplot(x='r2', y='Model', data=results_df.sort_values('r2', ascending=False), palette='magma', ax=axes[0])
    axes[0].set_title('Test R2 by Model')
    axes[0].set_xlim(0, 1)

    sns.scatterplot(x=y_test, y=y_pred, ax=axes[1], alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    axes[1].set_title('Actual vs Predicted')
    axes[1].set_xlabel('Actual Price')
    axes[1].set_ylabel('Predicted Price')

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_)
    else:
        importances = None

    if importances is not None:
        fi = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        fi = fi.sort_values('Importance', ascending=False).head(12)
        sns.barplot(x='Importance', y='Feature', data=fi, palette='viridis', ax=axes[2])
        axes[2].set_title('Top 12 Feature Importances')
    else:
        axes[2].text(0.5, 0.5, 'No feature importance available', ha='center', va='center')
        axes[2].set_title('Feature Importances')
        axes[2].set_axis_off()

    fig.tight_layout()
    fig.savefig('outputs/model_performance.png', dpi=150)
    plt.close(fig)
