# src/visuals.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(df, col):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[col], kde=True, ax=ax, color='teal')
    return fig

def plot_count(df, col):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=col, hue='class', data=df, ax=ax, palette='viridis')
    plt.xticks(rotation=45)
    return fig