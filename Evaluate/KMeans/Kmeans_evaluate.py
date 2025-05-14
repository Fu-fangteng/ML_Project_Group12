import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import fowlkes_mallows_score as FMI
from sklearn.metrics import calinski_harabasz_score

data = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")
X = data.iloc[:, 1:17].values
y_true = data.iloc[:, -1].values
n_clusters_range = range(2, 15)  
metrics = {
    'ARI': [],
    'NMI': [],
    'FMI': [],
    'Silhouette': [],
    'Davies-Bouldin': [],
    'Calinski-Harabasz': []
}

def evaluate_hierarchical(X, true_labels, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters,  random_state = 42)
    pred_labels = kmeans.fit_predict(X)
    results = {
        'ARI': ARI(true_labels, pred_labels),
        'NMI': NMI(true_labels, pred_labels),
        'FMI': FMI(true_labels, pred_labels),
        'Silhouette': silhouette_score(X, pred_labels),
        'Davies-Bouldin': davies_bouldin_score(X, pred_labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, pred_labels)
    }
    
    return results


for n in n_clusters_range:
    scores = evaluate_hierarchical(X, y_true, n)
    for metric in metrics:
        metrics[metric].append(scores[metric])

def plot_metrics(metrics, n_clusters_range):
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    metrics_info = {
        'ARI': {'name': 'Adjusted Rand Index', 'scale': 'linear', 'higher_better': True},
        'NMI': {'name': 'Normalized Mutual Info', 'scale': 'linear', 'higher_better': True},
        'FMI': {'name': 'Fowlkes-Mallows Index', 'scale': 'linear', 'higher_better': True},
        'Silhouette': {'name': 'Silhouette Score', 'scale': 'linear', 'higher_better': True},
        'Davies-Bouldin': {'name': 'Davies-Bouldin Index', 'scale': 'linear', 'higher_better': False},
        'Calinski-Harabasz': {'name': 'Calinski-Harabasz', 'scale': 'linear', 'higher_better': True}
    }

    for i, (metric, values) in enumerate(metrics.items()):
        if metrics_info[metric]['higher_better']:
            scaled_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            scaled_values = 1 - (values - np.min(values)) / (np.max(values) - np.min(values))
        
        plt.plot(n_clusters_range, scaled_values, 
                color=colors[i], linestyle=line_styles[i],
                label=f"{metrics_info[metric]['name']} ({metric})",
                linewidth=2)

    plt.title('KMeans', fontsize=14)
    plt.xlabel('number of clusters', fontsize=12)
    plt.ylabel('Evaluation', fontsize=12)
    plt.xticks(n_clusters_range)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/Evaluate/KMeans", bbox_inches='tight', dpi=300)
    plt.show()

def find_optimal_clusters(metrics, n_clusters_range):
    scores = np.zeros(len(n_clusters_range))
    for i, n in enumerate(n_clusters_range):
        score = metrics['ARI'][i] + metrics['NMI'][i] + metrics['FMI'][i]
        scores[i] = score
    optimal_n_kmeans = n_clusters_range[np.argmax(scores)]
    return optimal_n_kmeans, scores

optimal_n_kmeans, cluster_scores = find_optimal_clusters(metrics, n_clusters_range)

plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, cluster_scores, 'b-o', linewidth=2)
plt.axvline(x=optimal_n_kmeans, color='r', linestyle='--', label=f'best: {optimal_n_kmeans}')
plt.title('KMeans', fontsize=14)
plt.xlabel('number of clusters', fontsize=12)
plt.ylabel('score', fontsize=12)
plt.xticks(n_clusters_range)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/Evaluate/KMeans", bbox_inches='tight', dpi=300)
plt.show()
print(f"Best: {optimal_n_kmeans}")



def optimize_hierarchical_clusters(X, y_true, max_clusters=15):
    n_clusters_range = range(2, max_clusters+1)
    
    metrics = {k: [] for k in ['ARI', 'NMI', 'FMI', 'Silhouette', 'Davies-Bouldin','Calinski-Harabasz']}
    
    for n in n_clusters_range:
        scores = evaluate_hierarchical(X, y_true, n)
        for metric in metrics:
            metrics[metric].append(scores[metric])
    
    plot_metrics(metrics, n_clusters_range)
    optimal_n_kmeans, _ = find_optimal_clusters(metrics, n_clusters_range)
    return optimal_n_kmeans, metrics

optimal_n_kmeans, metrics = optimize_hierarchical_clusters(X, y_true, max_clusters=14)
