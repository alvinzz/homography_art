import numpy as np
import matplotlib.pyplot as plt

def wireframe(show=False):
    # create wireframe data
    matches = np.zeros([0, 4])
    y = np.linspace(-1, 1, 100)

    plt.figure('1')
    p1 = np.stack([np.sqrt(1 - y**2), y], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*np.sqrt(1 - (1/(y[21]-y[0]) * (y[:14] - y[21]))**2), y[:14]], 1)
    p2 = np.concatenate([p2, 
        np.stack([0.7*np.sqrt(1 - (1/(y[27]-y[7]) * (y[14:27] - y[7]))**2), y[14:27]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([0.7*0.2*(1 - (1.5*y[27:73])**2) + 0.1, y[27:73]], 1)], 0) #0.2*(1-(2*x)^2)
    p2 = np.concatenate([p2, 
        np.stack([0.7*np.sqrt(1 - (1/(y[92]-y[72]) * (y[73:86] - y[92]))**2), y[73:86]], 1)], 0)
    p2 = np.concatenate([p2, 
        np.stack([0.7*np.sqrt(1 - (1/(y[99]-y[78]) * (y[86:100] - y[78]))**2), y[86:100]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-np.sqrt(1 - y**2), y], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*-2*np.sqrt(1 - y**2), y], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([0.5*(y[39]-y[20]) * np.sqrt(1 - (2/(y[39]-y[20]) * (y[20:40] - (y[20]+y[39])/2))**2) + 0.1, y[20:40]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*(0.6*np.sqrt(1 - (2/(y[39]-y[20]) * (y[20:40] - (y[20]+y[39])/2))**2) - 0.8), y[20:40]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-0.5*(y[39]-y[20]) * np.sqrt(1 - (2/(y[39]-y[20]) * (y[20:40] - (y[20]+y[39])/2))**2) + 0.1, y[20:40]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*-0.8*np.ones_like(y[20:40]), y[20:40]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([0.5*(y[79]-y[60]) * np.sqrt(1 - (2/(y[79]-y[60]) * (y[60:80] - (y[60]+y[79])/2))**2) + 0.1, y[60:80]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*(0.6*np.sqrt(1 - (2/(y[79]-y[60]) * (y[60:80] - (y[60]+y[79])/2))**2) - 0.8), y[60:80]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-0.5*(y[79]-y[60]) * np.sqrt(1 - (2/(y[79]-y[60]) * (y[60:80] - (y[60]+y[79])/2))**2) + 0.1, y[60:80]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*-0.8*np.ones_like(y[60:80]), y[60:80]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([1.5*y[33:67]**2 - 0.6, y[33:67]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*(5*(y[33:50] + 0.2)**2 - 1.5), y[33:50]], 1)
    p2 = np.concatenate([p2,
        np.stack([0.7*(5*(y[50:67] - 0.2)**2 - 1.5), y[50:67]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-4*y[42:58]**2 - 0.2, y[42:58]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*-1.15*np.ones_like(y[42:58]), y[42:58]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-0.3*np.ones_like(y[42:58]), y[42:58]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.7*(-1.15*np.ones_like(y[42:50]) - 0.2/(y[49]-y[42])*(y[42:50]-y[42])), y[42:50]], 1)
    p2 = np.concatenate([p2,
        np.stack([0.7*(-1.35*np.ones_like(y[50:58]) + 0.2/(y[57]-y[50])*(y[50:58]-y[50])), y[50:58]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([12*y[40:60]**2 - 1.0, y[40:60]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.5*(20*y[40:60]**2 - 2.8), y[40:60]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    if show:
        plt.figure('1')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.figure('2')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    else:
        plt.close('1')
        plt.close('2')

    return matches

def pairwise(point_cloud):
    with open('cat_smile_data.scad', 'w') as f:
        f.write('data = [\n')

        f.write('    [')
        for ind in range(100):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(100, 200)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[0][0], point_cloud[0][1], point_cloud[0][2]))
        f.write('],\n')


        f.write('    [')
        for ind in range(200, 220):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(220, 240)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[200][0], point_cloud[200][1], point_cloud[200][2]))
        f.write('],\n')
        f.write('    [')
        for ind in range(240, 260):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(260, 280)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[240][0], point_cloud[240][1], point_cloud[240][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[200][0], point_cloud[200][1], point_cloud[200][2]))
        f.write('[{},{},{}],'.format(point_cloud[1][0], point_cloud[1][1], point_cloud[1][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[219][0], point_cloud[219][1], point_cloud[219][2]))
        f.write('[{},{},{}],'.format(point_cloud[240][0], point_cloud[240][1], point_cloud[240][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[259][0], point_cloud[259][1], point_cloud[259][2]))
        f.write('[{},{},{}],'.format(point_cloud[98][0], point_cloud[98][1], point_cloud[98][2]))
        f.write('],\n')

        f.write('    [')
        for ind in range(280, 314):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('],\n')

        f.write('    [')
        for ind in range(314, 330):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(330, 346)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[314][0], point_cloud[314][1], point_cloud[314][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(
            (point_cloud[322][0]+point_cloud[321][0])/2,
            (point_cloud[322][1]+point_cloud[321][1])/2,
            (point_cloud[322][2]+point_cloud[321][2])/2))
        f.write('[{},{},{}],'.format(
            (point_cloud[219][0]+point_cloud[240][0])/2,
            (point_cloud[219][1]+point_cloud[240][1])/2,
            (point_cloud[219][2]+point_cloud[240][2])/2))
        f.write('],\n')

        f.write('    [')
        for ind in range(346, 366):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('],\n')

        f.write('];\n')