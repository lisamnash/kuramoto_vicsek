import sys
import logging

import numpy as np

import sdeint as sdint
import scipy.spatial as spatial

import matplotlib.pyplot as plt


class SimulationObject:
    def __init__(self, parameter_dict):

        # parameters
        self.num_points = 200
        self.box_size = 30
        self.nu = 3
        self.C = 1.5
        self.speed = 2
        self.wamp = 1
        self.ww = 0
        self._max_dist = 2
        self._max_num = 10
        # I will store current step

        self.current_points = 0
        self.current_v = 0
        self.F = -1
        self.G = -1
        self.set_parameters(parameter_dict)
        self._initialize_step(parameter_dict)
        self.vshape = np.shape(self.current_v)


        self.vbar, self.NiNk = self.calc_vbar()
        #self.F_fun, self.G_fun = self.time_0()

    def set_parameters(self, parameter_dict):
        # type: (object) -> object
        """
        :type parameter_dict: object

        :rtype: None

        """
        if 'num_points' in parameter_dict:
            self.num_points = parameter_dict['num_points']
        else:
            logging.info('Number of points not given.  Assigning to default of 200')

        if 'box_size' in parameter_dict:
            self.box_size = parameter_dict['box_size']
        else:
            logging.info('Box size not given.  Assigning to default of 30')

        if 'nu' in parameter_dict:
            self.nu = parameter_dict['nu']
        else:
            logging.info('Nu not given.  Assigning to default of 3')

        if 'C' in parameter_dict:
            self.C = parameter_dict['C']
        else:
            logging.info('C not given.  Assigning to default of 1')

        if 'speed' in parameter_dict:
            self.speed = parameter_dict['speed']
        else:
            logging.info('speed not given.  Assigning to default of 2')

        if 'w0' in parameter_dict:
            self.w0 = 1
        else:
            logging.info('w0 not given.  Assigning to defulat of 1')

        if 'max_dist' in parameter_dict:
            self._max_dist = parameter_dict['max_dist']
        else:
            logging.info('Max dist not given. Assigning to default of 2')

        if 'max_num' in parameter_dict:
            self._max_num = parameter_dict['max_num']
        else:
            logging.info('Max num not set.  Assigning to default of 10')
        return None

    def set_ww(self, parameter_dict):
        if 'w_amp' in parameter_dict:
            w_amp = parameter_dict['w_amp']
        else:
            w_amp = 1
        self.w_amp = w_amp
        self.ww = w_amp*(1/10.*np.random.random(self.num_points)+1)

    def _initialize_step(self, parameter_dict):
        if 'initial_flocks' in parameter_dict:
            x_points, y_points = self.place_flocks(parameter_dict)
            x_points = np.array([item for sublist in x_points for item in sublist])
            y_points = np.array([item for sublist in y_points for item in sublist])
            self.num_points = len(x_points)
            #TODO: Add statement saying num_points in dict has been overridden
        else:
            bs = self.box_size/2
            x_points = (2*bs) * (np.random.random(self.num_points)) + bs
            y_points = (2*bs) * (np.random.random(self.num_points)) + bs
        self.current_points = np.array(zip(x_points, y_points))

        orientation = 2 * np.pi * np.random.random(self.num_points)
        x_v = np.cos(orientation)
        y_v = np.sin(orientation)

        self.current_v = np.array(zip(x_v, y_v))
        self.set_ww(parameter_dict)

        self.F = self.set_F()

    def place_flocks(self, parameter_dict):

        initial_flocks = parameter_dict['initial_flocks']

        x_points = []
        y_points = []
        for flock in initial_flocks:
            center = flock[0]
            length = flock[1]
            num_in_flock = flock[2]
            x = center[0] + length/2. * (np.random.random(num_in_flock))
            y = center[1] + length/2. * (np.random.random(num_in_flock))
            x_points.append(list(x))
            y_points.append(list(y))

        return x_points, y_points



    def set_max_dist(self, max_dist):
        if max_dist <= 0:
            logging.log('max distance cannot be set less than or equal to zero.  Setting to 2.')
            self._max_dist = 2
        else:
            self._max_dist = max_dist

    def calc_vbar(self):
        mytree = spatial.cKDTree(self.current_points)
        dist, indexes = mytree.query(self.current_points, k=self._max_num)

        ri = np.zeros((len(self.current_points), self._max_num), dtype=int)
        rk = np.zeros_like(ri, dtype=int)

        good_inds = ((dist < self._max_dist) & (dist > 0))
        ri[good_inds] = indexes[good_inds]
        rk[good_inds] = 1

        # I should get the angle and average
        ori = np.arctan2(self.current_v[:, 1], self.current_v[:, 0])

        mean_n = []
        for i in range(len(ri)):
            nei = ri[i]
            mm = (np.arctan2(np.sum(np.sin(ori[nei])), np.sum(np.cos(ori[nei])))) % (2 * np.pi)
            mean_n.append(mm)

        vbar = np.array(zip(np.cos(mean_n), np.sin(mean_n)))

        return vbar, [np.array(ri), np.array(rk)]

    def calc_force(self):
        X = np.array(self.current_points)
        ri = self.NiNk
        Ni = ri[0].astype(int)

        NP = len(X)

        Xni = X[Ni]

        vecs = np.array([(X[i] - Xni[i]) for i in xrange(len(Xni))])
        mags = np.sum(abs(vecs) ** 2, axis=-1) ** (1 / 2.)

        mags[np.where(mags == 0)] = 1
        mags = np.reshape(mags, [NP, self._max_num, 1])
        vec_hat = vecs / mags

        sig = 1
        ep = 0.25
        force = np.array(4*sig*(((12*ep**12)/mags**13) - (6*ep**6)/mags**7))*vec_hat

        force_v = np.sum(force, axis=1)

        return force_v

    def f_def(self, v, t):
        v = np.reshape(v, self.vshape)
        v_rot2 = (v[:, 0] + 1j * v[:, 1]) * np.exp(1j * np.pi / 2)
        v_perp2 = np.array([np.real(v_rot2), np.imag(v_rot2)]).T

        v_perp2 = np.reshape(v_perp2, (len(self.ww), 2))

        return np.array([self.ww[i] * v_perp2[i] for i in xrange(len(self.ww))]).flatten()

    def g_def(self, v, t, vbar):
        v = np.reshape(v, self.vshape)
        id_m = np.array([[1, 0], [0, 1]])
        ret = []
        for k in xrange(len(v)):
            t_p = np.tensordot(v[k], v[k], axes=0)
            p_vk = id_m - t_p
            ret_s = self.nu * np.dot(p_vk, vbar[k]) + self.C * np.dot(p_vk, np.random.random(2))

            ret_s = np.reshape(ret_s, [2, 1])
            ret.append(ret_s)

        ret = np.array(ret)

        return np.reshape(ret, (2 * len(ret), 1))

    def set_G(self):
        vbar, rd = self.calc_vbar()

        return lambda v, t: self.g_def(v, t, vbar)

    def set_F(self):

        return lambda v, t: self.f_def(v, t)

    def run_step(self, ti, ti1):
        delta_t = ti1 - ti
        if delta_t <=0:
            sys.exit()
            #add error message
        self.G = self.set_G()
        vv = sdint.stratint(self.F, self.G, np.array(self.current_v).flatten(), [ti, ti1])
        vv = vv[-1]
        vv = np.reshape(vv, self.vshape)
        vv = np.array(vv)

        vv_mags = np.sqrt(vv[:, 0] ** 2 + vv[:, 1] ** 2)

        vv = np.array([vv[i] / vv_mags[i] for i in xrange(len(vv))])

        xx = (self.current_points + delta_t * self.speed * vv) % self.box_size


        self.current_v = vv
        self.current_points = xx

        mxm = max(xx.flatten())
        if mxm > 100:
            sys.exit()








