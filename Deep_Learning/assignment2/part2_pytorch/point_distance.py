
import random
import numpy as np
import matplotlib.pyplot as plt


def main_original(n, delta, T):
    """
    n: is the number of uniformly at random generated points in the unit square
    delta: a maximal move of a point in one of four random directions: left, right, up, or down
    T: number of iterations
    return:
    lst_of_min_distances: of the minimum distances among all n points over times: t=0, 1, 2, \dots, T - 1,
    it is a list of reals of length T"""

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    points = [[random.uniform(0, 1), random.uniform(0, 1)] for _ in range(n)]

    lst_of_min_distances = []
    for t in range(T):
        # random motion
        for p in points:
            this_direction = random.choice(directions)
            this_delta = random.uniform(0, delta)
            p[0], p[1] = p[0] + this_delta * this_direction[0], p[1] + this_delta * this_direction[1]

        # compute min distance
        min_dist = float('inf')
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    p1, p2 = points[i], points[j]
                    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
        lst_of_min_distances.append(min_dist)

    return lst_of_min_distances

def main_optimized(n, delta, T):
    """
    n: is the number of uniformly at random generated points in the unit square
    delta: a maximal move of a point in one of four random directions: left, right, up, or down
    T: number of iterations
    return:
    lst_of_min_distances: of the minimum distances among all n points over times: t=0, 1, 2, \dots, T - 1,
    it is a list of reals of length T"""

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    points_init = np.random.uniform(0, 1, size=(n, 2, 1))

    # simu deltas, directions, moves
    lst_of_min_distances = []
    rng = np.random.default_rng(seed=10)
    simu_deltas = np.random.uniform(0, delta, n * T).reshape(n, 1, T)
    simu_directions = np.transpose(rng.choice(directions, (n, T)), (0, 2, 1))
    simu_moves = simu_deltas * simu_directions
    simu_cum_moves = np.cumsum(simu_moves, axis=2)

    # compute points location
    simu_points = points_init + simu_cum_moves
    simu_points_x = simu_points[:, 0:1, :]
    simu_points_y = simu_points[:, 1:2, :]

    # compute x/y axis distance among points in each t
    x_diff = simu_points_x - np.transpose(simu_points_x, (1, 0, 2))
    y_diff = simu_points_y - np.transpose(simu_points_y, (1, 0, 2))

    # set diagonal values to np.inf
    size = x_diff.shape[0]
    x_diff[range(size), range(size), :] = np.inf
    y_diff[range(size), range(size), :] = np.inf

    # get distance
    distances = np.sqrt(x_diff ** 2 + y_diff ** 2)

    # get min distance
    lst_of_min_distances = np.amin(distances, axis=(0, 1))

    return list(lst_of_min_distances)

random.seed(10)
np.random.seed(10)

n = 1500
delta = 1.0 / n
T = 20

result = main_optimized(n, delta, T)
# result = main_original(n, delta, T)
print("len:", len(result))
plt.plot(range(T), np.array(result) * np.sqrt(n))
plt.show()