import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal

# 예제 데이터 생성
np.random.seed(42)
mean = [2, 3]
cov = [[1, 0.5], [0.5, 2]]  # 공분산 행렬

# 100개의 2차원 데이터 포인트 생성
data = np.random.multivariate_normal(mean, cov, 100)

# 테스트 포인트
test_point = np.array([1, 1])

print("데이터 평균:", np.mean(data, axis=0))
print("데이터 공분산 행렬:")
data_cov = np.cov(data.T)
print(data_cov)
print("테스트 포인트:", test_point)

# 방법 1: scipy 사용
data_mean = np.mean(data, axis=0)
data_cov_inv = np.linalg.inv(data_cov)
mahal_dist_scipy = mahalanobis(test_point, data_mean, data_cov_inv)

# 방법 2: 직접 계산
def mahalanobis_distance(x, mean, cov):
    """마할라노비스 거리 직접 계산"""
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    distance = np.sqrt(diff.T @ cov_inv @ diff)
    return distance

mahal_dist_manual = mahalanobis_distance(test_point, data_mean, data_cov)

# 방법 3: 유클리드 거리와 비교
euclidean_dist = np.linalg.norm(test_point - data_mean)

print(f"\n마할라노비스 거리 (scipy): {mahal_dist_scipy:.4f}")
print(f"마할라노비스 거리 (manual): {mahal_dist_manual:.4f}")
print(f"유클리드 거리: {euclidean_dist:.4f}")

# 시각화
plt.figure(figsize=(12, 5))

# 서브플롯 1: 데이터와 테스트 포인트
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Data points')
plt.scatter(test_point[0], test_point[1], color='red', s=100, marker='x', label='Test point')
plt.scatter(data_mean[0], data_mean[1], color='green', s=100, marker='o', label='Mean')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 2: 마할라노비스 거리 등고선
plt.subplot(1, 2, 2)
x_range = np.linspace(-2, 6, 100)
y_range = np.linspace(-1, 7, 100)
X, Y = np.meshgrid(x_range, y_range)
pos = np.dstack((X, Y))

# 각 점에서의 마할라노비스 거리 계산
mahal_distances = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        mahal_distances[i, j] = mahalanobis_distance(point, data_mean, data_cov)

# 등고선 그리기
contours = plt.contour(X, Y, mahal_distances, levels=[1, 2, 3, 4], colors='blue', alpha=0.7)
plt.clabel(contours, inline=True, fontsize=8)

plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Data points')
plt.scatter(test_point[0], test_point[1], color='red', s=100, marker='x', label='Test point')
plt.scatter(data_mean[0], data_mean[1], color='green', s=100, marker='o', label='Mean')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Mahalanobis Distance Contours')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 데이터 포인트들의 마할라노비스 거리 계산
print("\n각 데이터 포인트의 마할라노비스 거리 (처음 10개):")
for i in range(10):
    dist = mahalanobis_distance(data[i], data_mean, data_cov)
    print(f"Point {i}: {data[i]} -> Distance: {dist:.4f}")

# 이상치 탐지 예제
threshold = 3.0  # 임계값
outliers = []
for i, point in enumerate(data):
    dist = mahalanobis_distance(point, data_mean, data_cov)
    if dist > threshold:
        outliers.append((i, point, dist))

print(f"\n임계값 {threshold}을 초과하는 이상치:")
for idx, point, dist in outliers:
    print(f"Index {idx}: {point} -> Distance: {dist:.4f}")