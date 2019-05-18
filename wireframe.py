import numpy as np
import matplotlib.pyplot as plt

def find_3d_points(matches, P1, P2):
    X1 = matches[:, :2]
    X2 = matches[:, 2:]

    points_3d = []
    err = 0

    for (x1, x2) in zip(X1, X2):
        M = np.stack([
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
            x2[0] * P2[2] - P2[0],
            x2[1] * P2[2] - P2[1]], axis=0)
        U, s, V_T = np.linalg.svd(M)
        point_3d = V_T[3] / V_T[3, 3]
        points_3d.append(point_3d)

        x1_proj = P1 @ np.array(point_3d)
        x1_proj = x1_proj[:2] / x1_proj[2]
        err += np.linalg.norm(x1_proj - x1)**2

        x2_proj = P2 @ np.array(point_3d)
        x2_proj = x2_proj[:2] / x2_proj[2]
        err += np.linalg.norm(x2_proj - x2)**2

    err /= 2*X1.shape[0]

    return np.array(points_3d)[:, :3], err

def rotM_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

f = 35
K = np.diag([f, f, 1])

theta = np.pi / 4.

R1 = rotM_y(theta)
C1 = np.expand_dims(np.array([-1, 0, 1]), 1)
t1 = -R1 @ C1
P1 = K @ np.concatenate([R1, t1], 1)

R2 = rotM_y(-theta)
C2 = np.expand_dims(np.array([1, 0, 1]), 1)
t2 = -R2 @ C2
P2 = K @ np.concatenate([R2, t2], 1)

from cat_smile import *
# from angel_demon import *
matches = wireframe(show=True)
matches[:, 0] = -matches[:, 0]

point_cloud, _ = find_3d_points(matches, P1, P2)
pairwise(point_cloud)

def plot_3d(points_3d, matches, P1, P2):
    X1 = matches[:, :2]
    X2 = matches[:, 2:]
    homog_points_3d = np.concatenate([points_3d, np.ones([points_3d.shape[0], 1])], axis=1)
    X1_proj = homog_points_3d @ P1.T
    X2_proj = homog_points_3d @ P2.T
    X1_proj = X1_proj[:, :2] / X1_proj[:, 2:]
    X2_proj = X2_proj[:, :2] / X2_proj[:, 2:]
    plt.plot(X1[:, 0], X1[:, 1], '+r')
    plt.plot(X1_proj[:, 0], X1_proj[:, 1], '+b')
    plt.show()
    plt.plot(X2[:, 0], X2[:, 1], '+r')
    plt.plot(X2_proj[:, 0], X2_proj[:, 1], '+b')
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D

    # inliers = (points_3d[:, 0] > np.percentile(points_3d[:, 0], 2)) \
    #     * (points_3d[:, 0] < np.percentile(points_3d[:, 0], 98))
    # points_3d = points_3d[inliers]
    # inliers = (points_3d[:, 1] > np.percentile(points_3d[:, 1], 2)) \
    #     * (points_3d[:, 1] < np.percentile(points_3d[:, 1], 98))
    # points_3d = points_3d[inliers]
    # inliers = (points_3d[:, 2] > np.percentile(points_3d[:, 2], 2)) \
    #     * (points_3d[:, 2] < np.percentile(points_3d[:, 2], 98))
    # points_3d = points_3d[inliers]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker='.')

    X, Y, Z = points_3d.T
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    # rotate the axes and update
    ax.view_init(45, -180)
    plt.pause(10)
    for angle in range(-180, -361, -5):
        ax.view_init(45, angle)
        plt.draw()
        plt.pause(0.000001)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

plot_3d(point_cloud, matches, P1, P2)
