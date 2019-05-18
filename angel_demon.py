import numpy as np
import matplotlib.pyplot as plt

def wireframe(show=False):
    # create wireframe data
    matches = np.zeros([0, 4])
    y = np.linspace(-1.5, 1.5, 300)

    plt.figure('1')
    p1 = np.stack([np.sqrt(((y[199]-y[100])/2)**2 - y[100:200]**2), y[100:200]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([np.sqrt(((y[199]-y[100])/2)**2 - y[100:200]**2), y[100:200]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-np.sqrt(((y[199]-y[100])/2)**2 - y[100:200]**2), y[100:200]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-np.sqrt(((y[199]-y[100])/2)**2 - y[100:200]**2), y[100:200]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([(((y[124]-y[0])/2)**2 - (y[:114] - (y[0]+y[124])/2)**2)**(1/2), y[:114]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.2*(y[:75] + 2.5)**2 - 0.1, y[:75]], 1)
    p2 = np.concatenate([p2,
        np.stack([0.4*y[75:114]**2 + 0.3, y[75:114]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-10*(((y[40]-y[0])/2)**2 - (y[:40] - (y[0]+y[40])/2)**2)**(1/2), y[:40]], 1)
    p1 = np.concatenate([p1,
        np.stack([-8*(((y[78]-y[40])/2)**2 - (y[40:78] - (y[40]+y[77])/2)**2)**(1/2) - 0.2, y[40:78]], 1)], 0)
    p1 = np.concatenate([p1,
        np.stack([-6*(((y[114]-y[78])/2)**2 - (y[78:114] - (y[78]+y[113])/2)**2)**(1/2) - 0.1, y[78:114]], 1)], 0)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-7.5*y[:30] - 11.15, y[:30]], 1)
    p2 = np.concatenate([p2,
        np.stack([4*y[30:60] + 2.7, y[30:60]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([-3*y[60:90] - 3.6, y[60:90]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([6*y[90:114] + 1.85, y[90:114]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([(((y[124]-y[0])/2)**2 - (y[-114:] - (y[-1]+y[-125])/2)**2)**(1/2), y[-114:]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([0.4*y[-114:-75]**2 + 0.3, y[-114:-75]], 1)
    p2 = np.concatenate([p2,
        np.stack([0.2*(y[-75:] - 2.5)**2 - 0.1, y[-75:]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-6*(((y[-115]-y[-79])/2)**2 - (y[-114:-78] - (y[-79]+y[-114])/2)**2)**(1/2) - 0.1, y[-114:-78]], 1)
    p1 = np.concatenate([p1,
        np.stack([-8*(((y[-79]-y[-41])/2)**2 - (y[-78:-40] - (y[-41]+y[-78])/2)**2)**(1/2) - 0.2, y[-78:-40]], 1)], 0)
    p1 = np.concatenate([p1,
        np.stack([-10*(((y[-41]-y[-1])/2)**2 - (y[-40:] - (y[-1]+y[-41])/2)**2)**(1/2), y[-40:]], 1)], 0)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-6*y[-114:-90] + 1.85, y[-114:-90]], 1)
    p2 = np.concatenate([p2,
        np.stack([3*y[-90:-60] - 3.6, y[-90:-60]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([-4*y[-60:-30] + 2.7, y[-60:-30]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([7.5*y[-30:] - 11.15, y[-30:]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([0.4*np.sqrt(((y[199]-y[100])/2)**2 - y[100:200]**2) + 0.8, y[100:200]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-10*(y[100:120] + 0.5)**2 + 0.75, y[100:120]], 1)
    p2 = np.concatenate([p2,
        np.stack([np.sqrt(((y[199]-y[100])/2)**2 - y[120:180]**2), y[120:180]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([-10*(y[180:200] - 0.5)**2 + 0.75, y[180:200]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-0.4*np.sqrt(((y[199]-y[100])/2)**2 - y[100:200]**2) + 0.8, y[100:200]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-1.8*(y[100:140] + 0.5)**2 + 0.75, y[100:140]], 1)
    p2 = np.concatenate([p2,
        np.stack([np.sqrt(((y[199]-y[100])/2)**2 - y[140:160]**2), y[140:160]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([-1.8*(y[160:200] - 0.5)**2 + 0.75, y[160:200]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([np.sqrt(((y[174]-y[125])/2)**2 - (y[125:175] - (y[125]+y[174])/2)**2) - 0.8, y[125:175]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-1.8*np.sqrt(((y[174]-y[125])/2)**2 - (y[125:148] - (y[125]+y[174])/2)**2) - 0.55, y[125:148]], 1)
    p2 =  np.concatenate([p2,
        np.stack([-0.45 + 20*y[148:150], y[148:150]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([-0.45 - 20*y[150:152], y[150:152]], 1)], 0)
    p2 = np.concatenate([p2,
        np.stack([-1.8*np.sqrt(((y[174]-y[125])/2)**2 - (y[152:175] - (y[125]+y[174])/2)**2) - 0.55, y[152:175]], 1)], 0)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-np.sqrt(((y[174]-y[125])/2)**2 - (y[125:175] - (y[125]+y[174])/2)**2) - 0.8, y[125:175]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-2*np.sqrt(((y[174]-y[125])/2)**2 - (y[125:175] - (y[125]+y[174])/2)**2) - 0.55, y[125:175]], 1)
    plt.plot(p2[:, 1], p2[:, 0], color='b')
    matches = np.concatenate([matches, np.concatenate([p1, p2], 1)], 0)

    plt.figure('1')
    p1 = np.stack([-20*(((y[154]-y[145])/2)**6 - (y[145:155] - (y[145]+y[154])/2)**6)**(1/6) - 1.05, y[145:155]], 1)
    plt.plot(p1[:, 1], p1[:, 0], color='b')
    plt.figure('2')
    p2 = np.stack([-20*(((y[154]-y[145])/2)**6 - (y[145:155] - (y[145]+y[154])/2)**6)**(1/6) - 1.05, y[145:155]], 1)
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
    with open('angel_demon_data.scad', 'w') as f:
        f.write('data = [\n')

        f.write('    [')
        for ind in range(100):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(100, 200)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[0][0], point_cloud[0][1], point_cloud[0][2]))
        f.write('],\n')

        f.write('    [')
        for ind in reversed(range(200, 314)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in range(314, 428):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('],\n')

        f.write('    [')
        for ind in range(428, 542):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(542, 656)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('],\n')

        f.write('    [')
        for ind in range(656, 756):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(756, 856)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[656][0], point_cloud[656][1], point_cloud[656][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[656][0], point_cloud[656][1], point_cloud[656][2]))
        f.write('[{},{},{}],'.format(point_cloud[250][0], point_cloud[250][1], point_cloud[250][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[755][0], point_cloud[755][1], point_cloud[755][2]))
        f.write('[{},{},{}],'.format(point_cloud[492][0], point_cloud[492][1], point_cloud[492][2]))
        f.write('],\n')

        f.write('    [')
        for ind in range(856, 906):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        for ind in reversed(range(906, 956)):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('[{},{},{}],'.format(point_cloud[856][0], point_cloud[856][1], point_cloud[856][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[856][0], point_cloud[856][1], point_cloud[856][2]))
        f.write('[{},{},{}],'.format(point_cloud[424][0], point_cloud[424][1], point_cloud[424][2]))
        f.write('],\n')
        f.write('    [')
        f.write('[{},{},{}],'.format(point_cloud[905][0], point_cloud[905][1], point_cloud[905][2]))
        f.write('[{},{},{}],'.format(point_cloud[545][0], point_cloud[545][1], point_cloud[545][2]))
        f.write('],\n')

        f.write('    [')
        for ind in range(956, 966):
            f.write('[{},{},{}],'.format(point_cloud[ind][0], point_cloud[ind][1], point_cloud[ind][2]))
        f.write('],\n')

        f.write('];\n')