import math


# Helper to calculate euclidean distance.
def distance(p1, p2):
    sum_sq = 0.0
    for i in range(len(p1)):
        sum_sq += (p1[i] - p2[i]) ** 2
    return math.sqrt(sum_sq)


def run_kmeans(points, K):
    N = len(points)
    d = len(points[0])
    iter_num = 300
    eps = 0.0001

    # Init centroids array from the first K points
    centroids = [p[:] for p in points[:K]]

    for _ in range(iter_num):
        new_centroids = [[0.0] * d for _ in range(K)]
        counts = [0] * K

        for point in points:
            min_dist = float("inf")
            closest_cluster = 0

            # Find the points cluster
            for k, centroid in enumerate(centroids):
                dist = distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = k

            # Add point to its cluster
            counts[closest_cluster] += 1
            for j in range(d):
                new_centroids[closest_cluster][j] += point[j]

        converged = True

        # Calc and update each clusters centroid
        for k in range(K):
            if counts[k] > 0:
                for j in range(d):
                    new_centroids[k][j] /= counts[k]
            else:
                new_centroids[k] = centroids[k][:]

            delta = distance(centroids[k], new_centroids[k])
            if delta >= eps:
                converged = False

            centroids[k] = new_centroids[k]

        if converged:
            break

    # Assign each point to its final cluster to get the labels
    labels = []
    for point in points:
        min_dist = float("inf")
        closest_cluster = 0
        for k, centroid in enumerate(centroids):
            dist = distance(point, centroid)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = k
        labels.append(closest_cluster)

    return labels
