import sys

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import isolum_rainbow

import visualizations as v

def draw_frame(**kwargs):
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
        pv.append([1,0])
        pv.append([-1, 0])
        pv = np.array(pv)
    else:
        print 'speeds not given. exiting'


    rr = 1.2 * kwargs['box_size']/40.

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

    X, Y, U, V = zip(*arrow_arr)



    colors = (np.arctan2(pv[:,1], pv[:,0]))%(2*np.pi)
    colors[-1]=0
    colors[-2]=2*np.pi



    #plt.scatter(px[:, 0], px[:, 1], alpha=.5, c = 'k')
    #plt.scatter(px[0, 0], px[0, 1], alpha=.5, c='r')
    plt.ion()
    quiv = quiver_plot=ax.quiver(X, Y, U, V, colors, angles='xy', scale_units='xy', scale=1, headaxislength=3, cmap='isolum_rainbow')

    plt.gca().set_aspect(1)

    if 'box_size' in kwargs:
        plt.ylim(-0, kwargs['box_size'])
        plt.xlim(-0, kwargs['box_size'])

    plt.xticks([])
    plt.yticks([])

    if 'save_name' in kwargs:
        plt.savefig(kwargs['save_name'])
        plt.close()
    else:
        return fig, [quiv]

def initialize_movie_writer(**kwargs):
    FFMpegWriter = manimation.writers['ffmpeg']

    if 'metadata' in kwargs:
        metadata = kwargs['metadata']
    else:
        metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    return FFMpegWriter(fps=40, bitrate=3000, metadata=metadata)


def write_movie(data, writer, **kwargs):
    X = data[0]
    V = data[1]
    t = data[2]
    box_size = data[3]

    if 'movie_path_and_name' not in kwargs:
        name = 'default.mp4'

    else:
        name = kwargs['movie_path_and_name']

    fig = plt.figure(figsize=(3, 3))
    ax = plt.axes([0, 0, 1, 1])

    with writer.saving(fig, name, 200):
        for i in xrange(len(t) - 1):
            fig, ca = v.draw_frame(fig=fig, ax=ax, x=X[i], v=V[i],
                                   box_size=box_size)
            writer.grab_frame()
            for cobject in ca:
                cobject.remove()