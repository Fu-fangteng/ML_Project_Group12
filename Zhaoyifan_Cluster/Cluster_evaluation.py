from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def Cluster_Evaluation(X,y_true,labels):
    # === 外部指标 ===
    print("\n--- External Evaluation Metrics ---")
    print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, labels):.4f}")
    print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(y_true, labels):.4f}")
    print(f"Fowlkes-Mallows Index (FMI): {fowlkes_mallows_score(y_true, labels):.4f}")

    # === 内部指标 ===
    print("\n--- Internal Evaluation Metrics ---")
    print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, labels):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.4f}")