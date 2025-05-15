import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import fowlkes_mallows_score as FMI
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")
X = data.iloc[:, 1:17].values  
y_true = data.iloc[:, -1].values  

def find_eps_param(X, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f'K-Distance Graph (k={k})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.grid(True)
    plt.savefig("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/Evaluate/OPTICS/distance.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    gradients = np.gradient(k_distances)
    eps_estimate = k_distances[np.argmax(gradients)]
    print(f"Suggested eps parameter: {eps_estimate:.2f}")
    return eps_estimate

eps_estimate = find_eps_param(X, k=5)

min_samples_range = range(5, 20, 2)  
metrics = {
    'ARI': [],
    'NMI': [],
    'FMI': [],
    'Silhouette': [],
    'Davies-Bouldin': [],
    'Calinski-Harabasz': [],
    'n_clusters': [] 
}

def evaluate_optics(X, true_labels, min_samples, eps=None):
    optics = OPTICS(min_samples=min_samples, 
                   eps=eps,
                   metric='euclidean',
                   cluster_method='xi',  
                   n_jobs=-1)
    pred_labels = optics.fit_predict(X)
    n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    if n_clusters < 2:
        return {
            'ARI': 0,
            'NMI': 0,
            'FMI': 0,
            'Silhouette': -1,
            'Davies-Bouldin': np.inf,
            'Calinski-Harabasz': 0,
            'n_clusters': 0
        }
    
    results = {
        'ARI': ARI(true_labels, pred_labels),
        'NMI': NMI(true_labels, pred_labels),
        'FMI': FMI(true_labels, pred_labels),
        'Silhouette': silhouette_score(X, pred_labels),
        'Davies-Bouldin': davies_bouldin_score(X, pred_labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, pred_labels),
        'n_clusters': n_clusters
    }
    return results

for min_samples in min_samples_range:
    scores = evaluate_optics(X, y_true, min_samples, eps=eps_estimate)
    for metric in metrics:
        if metric in scores:
            metrics[metric].append(scores[metric])

def plot_optics_metrics(metrics, param_range, param_name='min_samples'):
    plt.figure(figsize=(14, 8))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(param_range, metrics['ARI'], 'b-', label='ARI')
    ax1.plot(param_range, metrics['NMI'], 'g-', label='NMI')
    ax1.plot(param_range, metrics['FMI'], 'r-', label='FMI')
    ax1.plot(param_range, metrics['Silhouette'], 'c-', label='Silhouette')
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('Metric Score', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax2.plot(param_range, metrics['Davies-Bouldin'], 'm--', label='Davies-Bouldin')
    ax2.plot(param_range, metrics['Calinski-Harabasz'], 'y--', label='Calinski-Harabasz')
    ax2.plot(param_range, metrics['n_clusters'], 'k:', label='Number of clusters')
    ax2.set_ylabel('DB/CH/Cluster Count', fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('OPTICS Clustering Evaluation Metrics', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(param_range)
    plt.savefig("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/Evaluate/OPTICS/evaluation.png", bbox_inches='tight', dpi=300)
    plt.show()

plot_optics_metrics(metrics, min_samples_range)

def find_optimal_params(metrics, param_range):
    scores = []
    for i in range(len(param_range)):
        if metrics['n_clusters'][i] < 2:
            scores.append(-np.inf)
            continue
        score = (metrics['ARI'][i] + 
                metrics['NMI'][i] + 
                metrics['FMI'][i] )
        scores.append(score)
    
    best_idx = np.argmax(scores)
    optimal_param = param_range[best_idx]
    return optimal_param, scores

optimal_min_samples, param_scores = find_optimal_params(metrics, min_samples_range)
