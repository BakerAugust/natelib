# Quick k-means with many hardcoded assumptions. K=3

points = {
    "a1": (2, 10),
    "a2": (2, 5),
    "a3": (8, 4),
    "b1": (5, 8),
    "b2": (7, 5),
    "b3": (6, 4),
    "c1": (1, 2),
    "c2": (4, 9),
}


def euc_dist(point_a, point_b):
    dist = ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** (1 / 2)
    return dist


def cluster_mean(points: list) -> tuple:
    new_x = sum([p[0] for p in points]) / len(points)
    new_y = sum([p[1] for p in points]) / len(points)
    return (new_x, new_y)


# initialize clusters with provided assignments
centroids = {"A": points["a1"], "B": points["b1"], "C": points["c1"]}

for i in range(5):  # Add stopping criteria
    # Assign points
    assignments = {"A": [], "B": [], "C": []}
    for k, v in points.items():
        center_dist = 100
        for label, centroid in centroids.items():
            d = euc_dist(centroid, v)
            if d < center_dist:
                assignment = label
                center_dist = d
        assignments[assignment].append(k)

    # Calculate new clusters
    for cluster_label, point_labels in assignments.items():
        centroids[cluster_label] = cluster_mean([points[l] for l in point_labels])
    print(assignments)
