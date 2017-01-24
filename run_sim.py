import argparse as ap
import time

import numpy as np

import SimulationObject as SO
import visualizations as v
from input import parameter_dict

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Simulate flocking with rotation.')

    parser.add_argument('-t', dest='sim_time', type=int, help='time to run simulation', default=15)
    parser.add_argument('-o', dest='output_file', type=str, help='output movie name', default='output.mp4')
    args = parser.parse_args()

    sim = SO.SimulationObject(parameter_dict)

    delta_t = 0.02
    total_time = args.sim_time

    steps = total_time / delta_t
    t = np.linspace(0, total_time, steps)

    st = time.time()
    X = []
    V = []

    print('...running simulation...')
    for i in xrange(len(t) - 1):
        ti = t[i]
        ti1 = t[i + 1]
        sim.run_step_alt(ti, ti1)
        X.append(sim.current_points)
        V.append(sim.current_v)

        if i == len(t) - 2:
            et = time.time()

    print('time per step', (et - st) / len(t))

    print('...creating movie...')
    writer = v.initialize_movie_writer()
    data = np.array([X, V, t, sim.box_size, sim._max_dist])
    v.write_movie(data, writer, movie_path_and_name=args.output_file)
    print('...done!')
