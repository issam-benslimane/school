#1/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

#2/
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#3/
plt.scatter(X[:, 0], X[:, 1])
plt.title('Visualisation des données simulées')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print("Taille des données:", X.shape)

#4/
kmeans_random = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, random_state=42)
kmeans_random.fit(X)
kmeans_plus = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans_plus.fit(X)

#5/
score_random = silhouette_score(X, kmeans_random.labels_)
score_plus = silhouette_score(X, kmeans_plus.labels_)
print("Silhouette Score avec initialisation aléatoire:", score_random)
print("Silhouette Score avec K-means++:", score_plus)

#6/
#Les scores Silhouette élevés (environ 0.68 pour les deux méthodes d'initialisation) indiquent
#une bonne séparation et cohésion des clusters. Les visualisations montrent que les clusters
#correspondent aux groupes naturellement formés dans les données, confirmant l'efficacité de
#l'algorithme K-means avec les paramètres choisis.

#7/
#Les résultats sont identiques pour les deux méthodes d'initialisation. Cependant,
#K-means++ est généralement préféré puisqu'il plus robuste.

#8/
plt.scatter(X[:, 0], X[:, 1], c=kmeans_plus.labels_, cmap='viridis')
centers = kmeans_plus.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('Visualisation des données avec les centres de cluster')
plt.show()

#9/
#a)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#b)
print("Valeurs propres: ", pca.explained_variance_)
print("Vecteurs propres: ", pca.components_)

#c)
print("Inertie de chaque axe: ", pca.explained_variance_ratio_)

#d)
total_variance = sum(pca.explained_variance_ratio_)
print("Somme des inerties des axes:", total_variance) #il faut que cette valeur soit égale à 1, dans notre cas 1 apparait dans la console.

#e)
centers_pca = pca.transform(kmeans_plus.cluster_centers_)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_plus.labels_, cmap='viridis', marker='o', label='Data Points')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75, marker='x', label='Centers')
plt.title('Données et centres de cluster après PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


#d)
print("Inertie: ", pca.explained_variance_ratio_)

#f)
#L'application de PCA en combinaison avec le K-means a efficacement réduit
#la dimensionnalité des données, tout en conservant les distinctions claires
#entre les clusters. Cela a simplifié la visualisation et l'interprétation 
#des groupements principaux, démontrant l'efficacité de ces méthodes pour 
#identifier et comprendre les structures importantes dans les données.