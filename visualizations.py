import sys
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np


def draw_frame(**kwargs):
    # draws a frame of the simulation.
    if 'x' in kwargs:
        px = list(kwargs['x'])
        px.append([-100, -100])
        px.append([-100, -100])
        px = np.array(px)
    else:
        print 'position not supplied, exiting'
        sys.exit()

    if 'v' in kwargs:
        pv = list(kwargs['v'])
        pv.append([1, 0])
        pv.append([-1, 0])
        pv = np.array(pv)
    else:
        print 'speeds not given. exiting'

    rr = 1.2 * kwargs['box_size'] / 40.

    arrow_arr = [[px[i, 0], px[i, 1], rr * pv[i, 0], rr * pv[i, 1]] for i in xrange(len(px))]

    if 'fig' in kwargs:
        fig = kwargs['fig']
        if not 'ax' in kwargs:
            ax = plt.axes([0, 0, 1, 1])
        else:
            ax = kwargs['ax']

    else:
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes([0, 0, 1, 1])

    x, y, u, v = zip(*arrow_arr)

    colors = (np.arctan2(pv[:, 1], pv[:, 0])) % (2 * np.pi)
    colors[-1] = 0
    colors[-2] = 2 * np.pi

    quiv = ax.quiver(x, y, u, v, colors, angles='xy', scale_units='xy', scale=1, headaxislength=9,
                     cmap='isolum_rainbow')
    if 'rad' in kwargs:
        patch = []
        for x in px:
            circ = Circle((x[0], x[1]), radius=kwargs['rad'])
            patch.append(circ)

        p = PatchCollection(patch, color='w', edgecolor='k', lw=0.2, alpha=0.1)
        p.set_zorder(0)
        ax.add_collection(p)
    ax.set_aspect(1)

    if 'box_size' in kwargs:
        plt.ylim(-0, kwargs['box_size'])
        plt.xlim(-0, kwargs['box_size'])

    plt.xticks([])
    plt.yticks([])

    if 'save_name' in kwargs:
        plt.savefig(kwargs['save_name'])
        plt.close()
        return -1, [quiv, p]
    else:
        return fig, [quiv, p]


def initialize_movie_writer(**kwargs):
    ''' Initializes movie writer for simulation animation.
        Parameters
        ----------
        There are only optional keyword inputs here.
        'metadata': metadata for movie.  'title', 'artist', 'comment' are metadata inputs.
        'qm' : sets frame rate.  Framerate is 40/qm.

        Returns
        ---------
        FFMpegWriter : manimation object
            writer for animation'''
    FFMpegWriter = manimation.writers['ffmpeg']

    if 'metadata' in kwargs:
        metadata = kwargs['metadata']
    else:
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
    if 'qm' in kwargs:
        qm = kwargs['qm']
        if 1 <= qm <= 4:
            qm = qm
        else:
            qm = 1
    else:
        qm = 1

    return FFMpegWriter(fps=int(40. / qm), bitrate=3000, metadata=metadata)


def write_movie(data, writer, qm=1, **kwargs):
    ''' Writes the movie to an mp4 file.
        Parameters
        ----------
        data: list of arrays
            Data list has formate of [X, V, t, box_size, max_dist]. box size and max_dist are floats.
            X is an array of positions from simulation. V is array of orientations from simulation.
        writer : manimation movie writer

        Returns
        ---------
        None'''
    X = data[0]
    V = data[1]
    t = data[2]
    box_size = data[3]
    max_dist = data[4]

    if 'movie_path_and_name' not in kwargs:
        name = 'default.mp4'

    else:
        name = kwargs['movie_path_and_name']
        if name.split('.')[-1] != 'mp4':
            name += '.mp4'

    fig = plt.figure(figsize=(3, 3))
    ax = plt.axes([0, 0, 1, 1])

    with writer.saving(fig, name, 100):
        for i in xrange(len(t) - 1):
            if i % qm == 0:
                fig, ca = draw_frame(fig=fig, ax=ax, x=X[i], v=V[i],
                                     box_size=box_size, rad=max_dist)
                writer.grab_frame()
                for cobject in ca:
                    cobject.remove()
