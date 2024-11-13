import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 指定保存路径
load_path = '.../p0.25q2_4h1node2vec_features.npy'


# 从 .npy 文件加载特征向量矩阵
X_loaded = np.load(load_path)

# 打印加载后的矩阵以确认
print(X_loaded)

# 将加载的矩阵赋值给变量 X
X = X_loaded
n_clusters =16


# 执行KMeans聚类并获取聚类标签
cluster_labels = KMeans(n_clusters, random_state=0).fit(X).labels_

# 计算 Silhouette Score
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# 计算 Calinski-Harabasz 指数
ch_score = calinski_harabasz_score(X, cluster_labels)
print(f"Calinski-Harabasz Score: {ch_score}")

# 计算 Davies-Bouldin 指数
db_score = davies_bouldin_score(X, cluster_labels)
print(f"Davies-Bouldin Score: {db_score}")

# 定义9个高级颜色
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF',
          '#33FFF5', '#FFC300', '#DAF7A6', '#C70039', '#581845',
          '#900C3F', '#FF5733', '#900C3F', '#C70039', '#FFC300',
          '#FF6347', '#4682B4', '#8A2BE2', '#FFD700', '#98FB98']

# 绘制三维聚类结果
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')  # 设置为3D坐标系

# 遍历每个聚类并绘制散点
for i in range(n_clusters):
    ax.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], X[cluster_labels == i, 2],  # X, Y, Z坐标
               color=colors[i], s=50, alpha=0.5, label=f'Cluster {i+1}')

ax.set_title('Individuals K-Means Clustering of 10000 in 2h-condition (3D View)')
ax.set_xlabel('Feature 1')  # 添加X轴说明
ax.set_ylabel('Feature 2')  # 添加Y轴说明
ax.set_zlabel('Feature 3')  # 添加Z轴说明

# 将图例放置在图外
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Clusters')

plt.show()
