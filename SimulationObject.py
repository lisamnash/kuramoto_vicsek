import sys

import numpy as np
import scipy.spatial as spatial

import helper_functions as hf


class SimulationObject:
    def __init__(self, parameter_dict):

        '''Initializes the class
                Parameters
                ----------
                parameter_dict : dictionary of initial values for simulation.
                See class member explanation below for details.

                Class members
                ----------
                num_points : int
                   Number of birds in our simulation.
                box_size : float or int
                    Side-length of simulation box.  The simulation has periodic boundary conditions
                nu : float or int
                    Co-efficient of alignment.  Characterizes how good the birds are at aligning with each other
                C : float or int
                    Co-efficient of  error.  Characterizes the maximum magnitude of the error in alignment for a single
                    bird
                speed: float or int
                    The speed (in simulation units/simulation time) that a bird moves.  All birds move at constant speed
                w_amp : float or int
                    Amplitude of randomness on circle radius
                w0 : float or int
                    base circle size
                ww : float or int
                    array of circle sizes (essentially)
                _max_dist : float or int
                    Maximum distance each bird looks to calculate its correction to alignment.
                current_points: [num_points x 2] array of floats
                    The current positions of all the birds.
                current_v: [num_points x 2] array of floats
                    Orientations of birds.  These are unit vectors.
                F, G : lambda functions of v, and t
                    These functions are used to calculate dv for the numerical integraion
                vshape: 2x1 array
                    The initial shape of the v array
                vbar: [num_points x 2] array of floats
                    The average orientation of each point as calculated from its neighbors.
                   '''

        # parameters
        self.num_points = 200
        self.box_size = 30
        self.nu = 3
        self.C = 1.5
        self.speed = 2
        self.w_amp = 1
        self.ww = 0
        self.w0 = 0
        self.w_amp = 0
        self._max_dist = 2
        self._max_num = 20
        self.current_points = 0
        self.current_v = 0
        self.F = -1
        self.G = -1
        self.set_parameters(parameter_dict)
        self._initialize_step(parameter_dict)
        self.vshape = np.shape(self.current_v)
        self.vbar, self.NiNk = self.calc_vbar()

    def set_parameters(self, parameter_dict):
        '''Sets the parameters for the simulation based on the values in parameter_dict.  Does nothing if value not assigned in parameter_dict
            Parameters
            ----------
            parameter_dict: dictionary
                simulation input

            Returns
            ----------
            None
        '''
        if 'num_points' in parameter_dict:
            if hf.check_type(parameter_dict['num_points'], [int, float]):
                self.num_points = parameter_dict['num_points']
            else:
                print('num_points given is improper type.  Assigning to default of 200.')
        else:
            print('Number of points not given.  Assigning to default of 200')

        if 'box_size' in parameter_dict:
            if 'box_size' in parameter_dict:
                if hf.check_type(parameter_dict['box_size'], [int, float]):
                    self.box_size = parameter_dict['box_size']
                else:
                    print('box_size given is improper type.  Assigning to default of 30.')
        else:
            print('box_size not given.  Assigning to default of 30')

        if 'nu' in parameter_dict:
            if hf.check_type(parameter_dict['nu'], [int, float]):
                self.nu = parameter_dict['nu']
            else:
                print('Type Error: Type for nu should be int or float.  Assigning to default of 3. ')

        else:
            print('Nu not given.  Assigning to default of 3.')

        if 'C' in parameter_dict:
            if hf.check_type(parameter_dict['C'], [int, float]):
                self.C = parameter_dict['C']
            else:
                print('Type Error: Type for C should be int or float.  Exiting program. Assigning to default of 1')

        else:
            print('C not given.  Assigning to default of 1')

        if 'speed' in parameter_dict:
            if hf.check_type(parameter_dict['speed'], [int, float]):
                self.speed = parameter_dict['speed']
            else:
                print('Type Error: Type for nu should be int or float.  Assigning to default of 2.')

        else:
            print('speed not given.  Assigning to default of 2.')

        if 'w0' in parameter_dict:
            if hf.check_type(parameter_dict['w0'], [int, float]):
                self.w0 = parameter_dict['w0']
            else:
                print('Type Error: Type for w0 should be int or float.  Assigning to default of 1. ')
        else:
            print('w0 not given.  Assigning to default of 1')

        if 'max_dist' in parameter_dict:
            if hf.check_type(parameter_dict['max_dist'], [int, float]):
                self._max_dist = parameter_dict['max_dist']
            else:
                print('Type Error: Type for max_dist should be int or float.  Assigning to default of 2. ')
        else:
            print('Max dist not given. Assigning to default of 2')

        if 'max_num' in parameter_dict:
            if hf.check_type(parameter_dict['max_num'], [int, float]):
                self._max_num = parameter_dict['max_num']
            else:
                print('Type Error: Type for max_num should be int or float.  Assigning to default of 20. ')
        else:
            print('Max num not set.  Assigning to default of 20')
        return None

    def set_ww(self, parameter_dict):
        '''Sets the array of circle radii once flocks have been placed.  This is necessary because placing flocks
        changes the number of points.
            Parameters
            ----------
            parameter_dict: dictionary
                simulation input

            Returns
            ----------
            None'''
        if 'w_amp' in parameter_dict:
            if hf.check_type(parameter_dict['w_amp'], [int, float]):
                self.w_amp = parameter_dict['w_amp']
            else:
                print('Type Error: Type for w_amp should be int or float.  Assigning to default of 0.1 ')
                self.w_amp = 0.1
        else:
            self.w_amp = 0.1
            print('w_amp set to default value of 0.1')
        self.ww = self.w_amp * np.random.random(self.num_points) + self.w0

    def _initialize_step(self, parameter_dict):
        '''Function to iniitalize the simulation.
        Parameters
        ----------
        parameter_dict: dictionary
            simulation input

        Returns
        ----------
        None'''
        if 'initial_flocks' in parameter_dict:
            x_points, y_points = self.place_flocks(parameter_dict)
            x_points = np.array([item for sublist in x_points for item in sublist])
            y_points = np.array([item for sublist in y_points for item in sublist])
            self.num_points = len(x_points)
            print 'placing flocks...This overides value of num_points in input dictionary...'
        else:
            bs = self.box_size / 2
            x_points = (2 * bs) * (np.random.random(self.num_points)) + bs
            y_points = (2 * bs) * (np.random.random(self.num_points)) + bs
        self.current_points = np.array(zip(x_points, y_points))

        orientation = 2 * np.pi * np.random.random(self.num_points)
        x_v = np.cos(orientation)
        y_v = np.sin(orientation)

        self.current_v = np.array(zip(x_v, y_v))
        self.set_ww(parameter_dict)

        self.F = self.set_F()

    def place_flocks(self, parameter_dict):
        ''' Places birds
            Parameters
            ----------
            parameter_dict: dictionary
                simulation input

            Returns
            ----------
            x_points : [num_points x 1] array of floats
                x positions of birds
            y_points : [num_points x 1] array of floats
                y positions of birds'''
        initial_flocks = parameter_dict['initial_flocks']

        x_points = []
        y_points = []
        for flock in initial_flocks:
            center = flock[0]
            length = flock[1]
            num_in_flock = flock[2]
            x = center[0] + length / 2. * (np.random.random(num_in_flock))
            y = center[1] + length / 2. * (np.random.random(num_in_flock))
            x_points.append(list(x))
            y_points.append(list(y))

        return x_points, y_points

    def set_max_dist(self, max_dist):
        '''Sets the maximum distance that a bird will look for its neighbors
            Parameters
            -----------
            max_dist : float or int
                Numer of units a bird will check for neighbors'''
        if max_dist <= 0:
            print('max distance cannot be set less than or equal to zero.  Setting to 2.')
            self._max_dist = 2
        else:
            self._max_dist = max_dist

    def calc_vbar(self):
        # Calculates the average velocities from neighboring birds depending on max_dist and max_num.
        my_tree = spatial.cKDTree(self.current_points)
        dist, indexes = my_tree.query(self.current_points, k=self._max_num)

        ri = np.zeros((len(self.current_points), self._max_num), dtype=int)
        rk = np.zeros_like(ri, dtype=int)

        good_inds = (dist < self._max_dist)
        ri[good_inds] = indexes[good_inds]
        rk[good_inds] = 1

        # I should get the angle and average
        ori = np.arctan2(self.current_v[:, 1], self.current_v[:, 0])

        mean_n = []
        for i in range(len(ri)):
            nei = ri[i][np.where(rk[i] == 1)[0]]
            mm = (np.arctan2(np.sum(np.sin(ori[nei])), np.sum(np.cos(ori[nei])))) % (2 * np.pi)
            mean_n.append(mm)

        vbar = np.array(zip(np.cos(mean_n), np.sin(mean_n)))
        return vbar, [np.array(ri), np.array(rk)]

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
            r_ori = 2 * np.pi * np.random.random()
            ret_s = self.nu * np.dot(p_vk, vbar[k]) + self.C * np.dot(p_vk, [np.sin(r_ori), np.cos(r_ori)])

            ret_s = np.reshape(ret_s, [2, 1])
            ret.append(ret_s)

        ret = np.array(ret)

        return np.reshape(ret, 2 * len(ret))

    def set_G(self):
        vbar, rd = self.calc_vbar()

        return lambda v, t: self.g_def(v, t, vbar)

    def set_F(self):

        return lambda v, t: self.f_def(v, t)

    def run_step_alt(self, ti, ti1):
        ''' Propagates the simulation one step forward in time using simple Euler method.
            Parameters
            ----------
            ti : float
                time of start of this step
            ti1 : float
                time start of the next tep

            Returns
            ----------
            x_points : [num_points x 1] array of floats
                x positions of birds
            y_points : [num_points x 1] array of floats
                y positions of birds'''
        delta_t = ti1 - ti
        if delta_t <= 0:
            print 'time delta for step must be greater than 0'
            sys.exit()

        self.G = self.set_G()
        dv = self.F(self.current_v, ti) + self.G(self.current_v, ti)
        dv = np.reshape(dv, self.vshape)
        vv = self.current_v + delta_t * dv
        vv_mags = np.sqrt(vv[:, 0] ** 2 + vv[:, 1] ** 2)
        vv = np.array([vv[i] / vv_mags[i] for i in xrange(len(vv))])

        xx = (self.current_points + delta_t * self.speed * vv) % self.box_size

        self.current_v = vv
        self.current_points = xx
