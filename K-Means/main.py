import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 西瓜数据集
melon_data = np.array([
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.666, 0.091],
    [0.437, 0.211],
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042],
    [0.719, 0.103],
    [0.359, 0.188],
    [0.339, 0.241],
    [0.282, 0.257],
    [0.748, 0.232],
    [0.714, 0.346],
    [0.483, 0.312],
    [0.478, 0.437],
    [0.525, 0.369],
    [0.751, 0.489],
    [0.532, 0.472],
    [0.473, 0.376],
    [0.725, 0.445],
    [0.446, 0.459]
])

# 数据标准化
scaler = StandardScaler()
melon_data_scaled = scaler.fit_transform(melon_data)

# 聚类数范围
cluster_range = range(2, 21)

# 评估每个聚类数的轮廓系数
inertia_values = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(melon_data_scaled)
    inertia_values.append(kmeans.inertia_)

# 绘制肘部方法图
plt.plot(cluster_range, inertia_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# 选择最佳聚类数
best_k = np.argmin(np.diff(inertia_values)) + 2
print("Best number of clusters:", best_k)

# 使用最佳聚类数进行聚类
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
final_clusters = kmeans_final.fit_predict(melon_data_scaled)
clusters = kmeans_final.fit_predict(melon_data_scaled)

# 可视化最终聚类结果
plt.scatter(melon_data[:, 0], melon_data[:, 1], c=final_clusters, cmap='viridis', marker='o')
plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.title('KMeans Clustering on Watermelon Dataset')
plt.show()


plt.scatter(melon_data[clusters == 0, 0], melon_data[clusters == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(melon_data[clusters == 1, 0], melon_data[clusters == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(melon_data[clusters == 2, 0], melon_data[clusters == 2, 1], s=50, c='green', label='Cluster 3')

# 标记聚类中心
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=100, c='yellow', label='Centroids', marker='*')

plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.title('Watermelon Samples Clustering')
plt.legend()
plt.show()
