import taichi as ti
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from sksparse.cholmod import cholesky
import time
import ipdb

@ti.data_oriented
class rollers_lubrications:
    '''
    Define the short-ranged lubrication interactions between rollers
    '''
    def __init__(self, system):
        # spinners suspension
        self.system = system

        self.N = system.N
        self.dim = system.dim

        self.radius = system.radius
        self.diameter = 2.0 * self.radius
        self.viscosity = system.viscosity

        self.kT = system.kT
        self.deltaT = system.deltaT

        # particle properties
        self.pos = system.pos
        self.ort = system.ort

        self.vel_det = system.vel_det
        self.ang_det = system.ang_det

        self.vel_stoch = system.vel_stoch
        self.ang_stoch = system.ang_stoch

        self.force = system.force
        self.torque = system.torque

        self.vel_lub = None
        self.ang_lub = None

        self.force_lub = None
        self.torque_lub = None

        # tabulated values config
        self.tabulated_data_dir = './lubrication_resistance_functions/'

        #self.tabulated_dist_name = 'res_dists_interpolated.npy'
        self.tabulated_dist_name = 'res_dists_interpolated_full_subs_RPY.npy'
        self.tabulated_dist_num = 1024
        self.tabulated_dist = None

        #self.tabulated_funcs_name = 'res_scalars_interpolated.npy'
        self.tabulated_funcs_name = 'res_scalars_interpolated_full_subs_RPY.npy'
        self.tabulated_funcs_num = 10
        self.tabulated_funcs = None

        # scaled by particle radius
        self.sep_max = 2.0
        self.sep_min = 0.0001

        self.dist_max = self.sep_max + 2.0
        self.dist_min = self.sep_min + 2.0

        self.sep_max_log = np.log10(self.sep_max)
        self.sep_min_log = np.log10(self.sep_min)

        self.dr_log = (self.sep_max_log - self.sep_min_log) / (self.tabulated_dist_num - 1)

        # sparse matrix info
        self.elements_num_max = self.N * (self.dim * 4)**2 * 2
        self.elements_num = None
        self.rows = None
        self.cols = None
        self.vals = None
        self.neighbors_num = None

        # interaction range
        self.lub_cutoff = system.lub_cutoff

        # normalization factors
        self.radius_inv = 1.0 / self.radius
        self.R_FU_factor = 6.0 * np.pi * self.viscosity * self.radius
        self.R_FW_factor = 6.0 * np.pi * self.viscosity * self.radius**2
        self.R_TU_factor = 6.0 * np.pi * self.viscosity * self.radius**2
        self.R_TW_factor = 6.0 * np.pi * self.viscosity * self.radius**3

        self.noise_level = np.sqrt(0.0)
        self.brownian_factor = np.sqrt(2.0 * self.kT / self.deltaT)

        # wall corrections
        # Not Implemented
        self.wall_correction_on = system.lub_wall_correction_on

        # specify data layout
        self.layout()

    def layout(self):
        '''
        specify data layout
        '''
        # global variables
        self.elements_num = ti.field(ti.i32, shape=())

        # field quantities
        self.vel_lub = ti.Vector.field(self.dim, dtype=ti.f64)
        self.ang_lub = ti.Vector.field(self.dim, dtype=ti.f64)

        self.force_lub = ti.Vector.field(self.dim, dtype=ti.f64)
        self.torque_lub = ti.Vector.field(self.dim, dtype=ti.f64)

        self.tabulated_dist = ti.field(dtype=ti.f64)
        self.tabulated_funcs = ti.field(dtype=ti.f64)

        self.neighbors_num = ti.field(dtype=ti.i32)
        self.rows = ti.field(dtype=ti.i32)
        self.cols = ti.field(dtype=ti.i32)
        self.vals = ti.field(dtype=ti.f64)

        # place field quantities
        ti.root.dense(ti.i, self.N).place(self.vel_lub, self.ang_lub)
        ti.root.dense(ti.i, self.N).place(self.force_lub, self.torque_lub)


        ti.root.dense(ti.i, self.tabulated_dist_num).place(self.tabulated_dist)
        ti.root.dense(ti.ij, (self.tabulated_dist_num, self.tabulated_funcs_num)).place(self.tabulated_funcs)

        # not the best choice but easier to implement
        ti.root.dense(ti.i, self.N).place(self.neighbors_num)
        ti.root.dense(ti.i, self.elements_num_max).place(self.rows)
        ti.root.dense(ti.i, self.elements_num_max).place(self.cols)
        ti.root.dense(ti.i, self.elements_num_max).place(self.vals)

        # dynamic node
        #ti.root.dynamic(ti.i, self.elements_num_max).place(self.rows)
        #ti.root.dynamic(ti.i, self.elements_num_max).place(self.cols)
        #ti.root.dynamic(ti.i, self.elements_num_max).place(self.vals)

    def init(self):
        '''
        initializations
        '''
        self.load_tabulated_values()

    def load_tabulated_values(self):
        '''
        load pre-computed tabulated lubrication resistance functions
        '''
        tabulated_dist_np = np.load(self.tabulated_data_dir+self.tabulated_dist_name)
        self.tabulated_dist.from_numpy(tabulated_dist_np.flatten())

        tabulated_funcs_np = np.load(self.tabulated_data_dir+self.tabulated_funcs_name)
        self.tabulated_funcs.from_numpy(tabulated_funcs_np)
        print('----------lubrication info---------------')
        print('tabulated lubrication resistance functions loaded: '+self.tabulated_data_dir+self.tabulated_dist_name)
        print('tabulated range: {dist_min}a to {dist_max}a'.format(dist_min=self.dist_min, dist_max=self.dist_max))
        print('data points num: {num} spacing in log space: {dr_log}'.format(num=self.tabulated_dist_num, dr_log=self.dr_log))
        print('------------------------------------------')

    def compute(self):
        '''
        compute lubrication interactions
        '''
        self.clear()

        #self.compute_force_from_trans()

        #self.compute_force_from_rot()

        #self.compute_torque_from_trans()

        #self.compute_torque_from_rot()

        # direct solvers - cholesky
        #self.compute_lub_free()
        #self.compute_lub_constrained()

    def apply_resistance_tensors_bulk(self):
        '''
        compute [F_b; T_b] = R_{bb} * [U_b; W_b]
        '''
        self.compute_force_from_trans()
        self.compute_force_from_rot()
        self.compute_torque_from_trans()
        self.compute_torque_from_rot()

    def compute_lub_free(self):
        '''
        compute lubrication interactions for the free kinetics case
        '''
        # build resistance matrix
        self.build_resistance_matrix()

        self.check_isolated_particles()

        # get resistance matrix from gpu
        elements_num = self.elements_num.to_numpy()
        rows = self.rows.to_numpy()[:elements_num]
        cols = self.cols.to_numpy()[:elements_num]
        vals = self.vals.to_numpy()[:elements_num]

        # construct sparse matrix
        R_lub = coo_matrix( (vals, (rows,cols)), shape=(self.N*self.dim*2, self.N*self.dim*2) ).tocsc()

        # generate random noise
        W = np.random.randn(self.N*self.dim*2)

        # cholesky decomposition
        cholesky_factor = cholesky(R_lub)
        # correlated noise R^{1/2} * W
        sqrt_R = cholesky_factor.L()
        force_torque_brownian = self.noise_level * self.brownian_factor * cholesky_factor.apply_Pt(sqrt_R.dot(W))

        # get forces and torques
        force = self.force.to_numpy()
        torque = self.torque.to_numpy()
        b = np.concatenate( (force.flatten(), torque.flatten()) )
        # brownian forces
        b += force_torque_brownian

        # solve R * [U; W] = [F; T]
        x = cholesky_factor(b)

        # extract translational and angular velocities
        vel = x[:self.N*self.dim].reshape((self.N, self.dim))
        ang = x[self.N*self.dim:].reshape((self.N, self.dim))

        # update lubrication translational and angular velocity
        self.vel_lub.from_numpy(vel)
        self.ang_lub.from_numpy(ang)
        self.update_lub_velocities()

    def compute_lub_constrained(self):
        '''
        compute lubrication interactions for the constrained kinetics case
        '''
        # build resistance matrix
        self.build_resistance_matrix()

        self.check_isolated_particles()

        # get resistance matrix from gpu
        elements_num = self.elements_num.to_numpy()
        rows = self.rows.to_numpy()[:elements_num]
        cols = self.cols.to_numpy()[:elements_num]
        vals = self.vals.to_numpy()[:elements_num]

        # construct sparse matrix
        R_lub = coo_matrix( (vals, (rows,cols)), shape=(self.N*self.dim*2, self.N*self.dim*2) ).tocsc()

        # get corresponding sub-matrix
        offset = self.N * self.dim
        R_FU_lub = R_lub[:offset, :offset]
        R_FW_lub = R_lub[:offset,offset:]
        R_TU_lub = R_lub[offset:, :offset]
        R_TW_lub = R_lub[offset:, offset:]

        # get forces
        force = self.force.to_numpy()

        # get prescribed angular velocity
        ang_prescribed = np.zeros_like(force)
        ang_prescribed[:,2] = 2.0 * np.pi * self.field_freq
        ang = ang_prescribed.flatten()

        # solve for translational velocity first
        # R_{FU} * U = F - R_{FW} * W
        b = force.flatten() - R_FW_lub.dot(ang)
        cholesky_factor = cholesky(R_FU_lub)
        vel = cholesky_factor(b)

        # then solve for torques
        # T = R_{TU} * U + R_{TW} * W
        torque = R_TU_lub.dot(vel) + R_TW_lub.dot(ang)

        # update
        ang = ang.reshape((self.N, self.dim))
        vel = vel.reshape((self.N, self.dim))
        torque = torque.reshape((self.N, self.dim))

        self.vel_lub.from_numpy(vel)
        self.ang_lub.from_numpy(ang)
        self.update_lub_velocities()

        # torque unused
        #self.torque.from_numpy(torque)

    def clear(self):
        '''
        clear related arrays
        '''
        self.clear_neighbors()

        self.clear_velocities()

        self.clear_forces_torques()

    @ti.kernel
    def clear_neighbors(self):
        '''
        clear neighbors
        '''
        for i in ti.ndrange(self.N):
            self.neighbors_num[i] = 0

        self.elements_num[None] = 0

    @ti.kernel
    def clear_velocities(self):
        '''
        clear translational velocities
        '''
        for i in ti.ndrange(self.N):
            for j in ti.static(range(self.dim)):
                self.vel_lub[i][j] = 0.0
                self.ang_lub[i][j] = 0.0

    @ti.kernel
    def clear_forces_torques(self):
        '''
        clear translational velocities
        '''
        for i in ti.ndrange(self.N):
            for j in ti.static(range(self.dim)):
                self.force_lub[i][j] = 0.0
                self.torque_lub[i][j] = 0.0

    @ti.kernel
    def set_angular_velocities(self, omega_x: ti.f64, omega_y: ti.f64, omega_z: ti.f64):
        '''
        set angular velocities for each particles
        '''
        for i in ti.ndrange(self.N):
            self.ang_lub[i][0] = omega_x
            self.ang_lub[i][1] = omega_y
            self.ang_lub[i][2] = omega_z

    @ti.kernel
    def update_lub_velocities(self):
        '''
        accumulate lubrication velocities into deterministic velocities
        '''
        for i in ti.ndrange(self.N):
            for j in ti.static(range(self.dim)):
                self.vel_det[i][j] += self.vel_lub[i][j]
                self.ang_det[i][j] += self.ang_lub[i][j]

    @ti.kernel
    def compute_force_from_trans(self):
        '''
        F = R_{FU} * U
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # loop each pair of particles
            if (i < j):
                # variables
                ri = ti.Vector([0.]*self.dim)
                rj = ti.Vector([0.]*self.dim)
                rij = ti.Vector([0.]*self.dim)

                veli = ti.Vector([0.]*self.dim)
                velj = ti.Vector([0.]*self.dim)

                forcei = ti.Vector([0.]*self.dim)
                forcej = ti.Vector([0.]*self.dim)

                R_FU_ii = ti.Matrix([[0.]*self.dim]*self.dim)
                R_FU_ij = ti.Matrix([[0.]*self.dim]*self.dim)
                R_FU_ji = ti.Matrix([[0.]*self.dim]*self.dim)
                R_FU_jj = ti.Matrix([[0.]*self.dim]*self.dim)

                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    rj[k] = self.pos[j][k]
                    veli[k] = self.vel_lub[i][k]
                    velj[k] = self.vel_lub[j][k]

                # rij
                for k in ti.static(range(self.dim)):
                    rij[k] = ri[k] - rj[k]

                r = rij.norm()

                # within cutoff
                if (r < self.lub_cutoff):

                    # resistance tensor - bulk
                    self.resistance_FU_lub(i, j, rij, R_FU_ii, R_FU_ij, R_FU_ji, R_FU_jj)

                    # resistance tensor - wall correction
                    if ti.static(self.wall_correction_on == True):
                        print('lubrication with wall correction NOT implemented!')

                    # F_i = R_{FU}^{ij} * U_j

                    # i-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcei[k] += R_FU_ii[k,l] * veli[l]

                    # i-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcei[k] += R_FU_ij[k,l] * velj[l]

                    # j-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcej[k] += R_FU_ji[k,l] * veli[l]

                    # j-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcej[k] += R_FU_jj[k,l] * velj[l]

                    # update global arrays
                    # i-th particle
                    for k in ti.static(range(self.dim)):
                        self.force_lub[i][k] += forcei[k] * self.R_FU_factor

                    # j-th particle
                    for k in ti.static(range(self.dim)):
                        self.force_lub[j][k] += forcej[k] * self.R_FU_factor

    @ti.func
    def resistance_FU_lub(self, i, j, r_vec, R_FU_ii: ti.template(), R_FU_ij: ti.template(), R_FU_ji: ti.template(), R_FU_jj: ti.template()):
        '''
        compute R_FU 3x3 lubrication resistance tensor for a pair of particles
        '''
        # rescale rij with particle radius
        r_vec *= self.radius_inv

        # particle distance
        r = r_vec.norm()
        epsilon = r - 2.0

        # unit displacement vector
        r_hat = -r_vec / r # !Negative sign to match formula convention

        # construct displacement tensors
        L1 = ti.Matrix([[0.]*self.dim]*self.dim)
        L2 = ti.Matrix([[0.]*self.dim]*self.dim)

        # L1 - squeezing motion: r_i * r_j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                L1[k,l] = r_hat[k] * r_hat[l]
                L2[k,l] = -1.0 * L1[k,l]

        # L2 - shearing motion: delta_ij - r_i * r_j
        for k in ti.static(range(self.dim)):
            L2[k,k] += 1.0

        # cast sepration distance into tabulated indexs
        epsilon_log = ti.log(epsilon) / ti.log(10)
        dist_index = ti.cast(ti.floor( (epsilon_log - self.sep_min_log) / self.dr_log ), ti.i32)

        # get lubrication resitance functions
        X11A, Y11A = 0.0, 0.0
        X12A, Y12A = 0.0, 0.0
        if epsilon < self.sep_min:
            # touching particles - using minimum separation distance
            print('overlapping occurs: ', i, j, r)
            X11A = self.tabulated_funcs[0, 0]
            X12A = self.tabulated_funcs[0, 1]
            Y11A = self.tabulated_funcs[0, 2]
            Y12A = self.tabulated_funcs[0, 3]
        else:
            # non-touching particles - linear interpolation
            X11A = self.lub_funcs(epsilon, dist_index, 0)
            X12A = self.lub_funcs(epsilon, dist_index, 1)
            Y11A = self.lub_funcs(epsilon, dist_index, 2)
            Y12A = self.lub_funcs(epsilon, dist_index, 3)

        # pair i-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FU_ii[k,l] = X11A * L1[k,l] + Y11A * L2[k,l]

        # pair i-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FU_ij[k,l] = X12A * L1[k,l] + Y12A * L2[k,l]

        # pair j-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FU_ji[k,l] = R_FU_ij[k,l]

        # pair j-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FU_jj[k,l] = R_FU_ii[k,l]

    @ti.kernel
    def compute_force_from_rot(self):
        '''
        F = R_{FW} * W
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # loop each pair of particles
            if (i < j):
                # variables
                ri = ti.Vector([0.]*self.dim)
                rj = ti.Vector([0.]*self.dim)
                rij = ti.Vector([0.]*self.dim)

                angi = ti.Vector([0.]*self.dim)
                angj = ti.Vector([0.]*self.dim)

                forcei = ti.Vector([0.]*self.dim)
                forcej = ti.Vector([0.]*self.dim)

                R_FW_ii = ti.Matrix([[0.]*self.dim]*self.dim)
                R_FW_ij = ti.Matrix([[0.]*self.dim]*self.dim)
                R_FW_ji = ti.Matrix([[0.]*self.dim]*self.dim)
                R_FW_jj = ti.Matrix([[0.]*self.dim]*self.dim)

                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    rj[k] = self.pos[j][k]
                    angi[k] = self.ang_lub[i][k]
                    angj[k] = self.ang_lub[j][k]

                # rij
                for k in ti.static(range(self.dim)):
                    rij[k] = ri[k] - rj[k]

                r = rij.norm()

                # within cutoff
                if (r < self.lub_cutoff):

                    # resistance tensor - bulk
                    self.resistance_FW_lub(i, j, rij, R_FW_ii, R_FW_ij, R_FW_ji, R_FW_jj)

                    # resistance tensor - wall correction
                    if ti.static(self.wall_correction_on == True):
                        print('lubrication with wall correction NOT implemented!')

                    # F_i = R_{FW}^{ij} * W_j

                    # i-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcei[k] += R_FW_ii[k,l] * angi[l]

                    # i-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcei[k] += R_FW_ij[k,l] * angj[l]

                    # j-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcej[k] += R_FW_ji[k,l] * angi[l]

                    # j-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            forcej[k] += R_FW_jj[k,l] * angj[l]

                    # update global arrays
                    # i-th particle
                    for k in ti.static(range(self.dim)):
                        self.force_lub[i][k] += forcei[k] * self.R_FW_factor

                    # j-th particle
                    for k in ti.static(range(self.dim)):
                        self.force_lub[j][k] += forcej[k] * self.R_FW_factor

    @ti.func
    def resistance_FW_lub(self, i, j, r_vec, R_FW_ii: ti.template(), R_FW_ij: ti.template(), R_FW_ji: ti.template(), R_FW_jj: ti.template()):
        '''
        compute R_FW 3x3 lubrication resistance tensor for a pair of particles
        '''
        # rescale rij with particle radius
        r_vec *= self.radius_inv

        # particle distance
        r = r_vec.norm()
        epsilon = r - 2.0

        # unit displacement vector
        r_hat = -r_vec / r # !Negative sign to match formula convention

        # construct displacement tensors
        L3 = ti.Matrix([[0.]*self.dim]*self.dim)

        # L3 - vortex motion: epsilon_ijk * r_k (left-handed!)
        L3[0,0] = 0.0
        L3[0,1] =-r_hat[2]
        L3[0,2] = r_hat[1]

        L3[1,0] = r_hat[2]
        L3[1,1] = 0.0
        L3[1,2] =-r_hat[0]

        L3[2,0] =-r_hat[1]
        L3[2,1] = r_hat[0]
        L3[2,2] = 0.0

        # cast sepration distance into tabulated indexs
        epsilon_log = ti.log(epsilon) / ti.log(10)
        dist_index = ti.cast(ti.floor( (epsilon_log - self.sep_min_log) / self.dr_log ), ti.i32)

        # get lubrication resitance functions
        Y11B = 0.0
        Y12B = 0.0
        if epsilon < self.sep_min:
            # touching particles - using minimum separation distance
            print('overlapping occurs: ', i, j, r)
            Y11B = self.tabulated_funcs[0, 4]
            Y12B = self.tabulated_funcs[0, 5]
        else:
            # non-touching particles - linear interpolation
            Y11B = self.lub_funcs(epsilon, dist_index, 4)
            Y12B = self.lub_funcs(epsilon, dist_index, 5)

        # pair i-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FW_ii[k,l] =-Y11B * L3[k,l]

        # pair i-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FW_ij[k,l] = Y12B * L3[k,l]

        # pair j-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FW_ji[k,l] =-R_FW_ij[k,l]

        # pair j-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_FW_jj[k,l] =-R_FW_ii[k,l]

    @ti.kernel
    def compute_torque_from_trans(self):
        '''
        T = R_{TU} * U
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # loop each pair of particles
            if (i < j):
                # variables
                ri = ti.Vector([0.]*self.dim)
                rj = ti.Vector([0.]*self.dim)
                rij = ti.Vector([0.]*self.dim)

                veli = ti.Vector([0.]*self.dim)
                velj = ti.Vector([0.]*self.dim)

                torquei = ti.Vector([0.]*self.dim)
                torquej = ti.Vector([0.]*self.dim)

                R_TU_ii = ti.Matrix([[0.]*self.dim]*self.dim)
                R_TU_ij = ti.Matrix([[0.]*self.dim]*self.dim)
                R_TU_ji = ti.Matrix([[0.]*self.dim]*self.dim)
                R_TU_jj = ti.Matrix([[0.]*self.dim]*self.dim)

                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    rj[k] = self.pos[j][k]
                    veli[k] = self.vel_lub[i][k]
                    velj[k] = self.vel_lub[j][k]

                # rij
                for k in ti.static(range(self.dim)):
                    rij[k] = ri[k] - rj[k]

                r = rij.norm()

                # within cutoff
                if (r < self.lub_cutoff):

                    # resistance tensor - bulk
                    self.resistance_TU_lub(i, j, rij, R_TU_ii, R_TU_ij, R_TU_ji, R_TU_jj)

                    # resistance tensor - wall correction
                    if ti.static(self.wall_correction_on == True):
                        print('lubrication with wall correction NOT implemented!')

                    # T_i = R_{TU}^{ij} * U_j

                    # i-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquei[k] += R_TU_ii[k,l] * veli[l]

                    # i-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquei[k] += R_TU_ij[k,l] * velj[l]

                    # j-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquej[k] += R_TU_ji[k,l] * veli[l]

                    # j-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquej[k] += R_TU_jj[k,l] * velj[l]

                    # update global arrays
                    # i-th particle
                    for k in ti.static(range(self.dim)):
                        self.torque_lub[i][k] += torquei[k] * self.R_TU_factor
                    # j-th particle
                    for k in ti.static(range(self.dim)):
                        self.torque_lub[j][k] += torquej[k] * self.R_TU_factor

    @ti.func
    def resistance_TU_lub(self, i, j, r_vec, R_TU_ii: ti.template(), R_TU_ij: ti.template(), R_TU_ji: ti.template(), R_TU_jj: ti.template()):
        '''
        compute R_TU 3x3 lubrication resistance tensor for a pair of particles
        '''
        # rescale rij with particle radius
        r_vec *= self.radius_inv

        # particle distance
        r = r_vec.norm()
        epsilon = r - 2.0

        # unit displacement vector
        r_hat = -r_vec / r # !Negative sign to match formula convention

        # construct displacement tensors
        L3 = ti.Matrix([[0.]*self.dim]*self.dim)

        # L3 - vortex motion: epsilon_ijk * r_k (left-handed!)
        L3[0,0] = 0.0
        L3[0,1] =-r_hat[2]
        L3[0,2] = r_hat[1]

        L3[1,0] = r_hat[2]
        L3[1,1] = 0.0
        L3[1,2] =-r_hat[0]

        L3[2,0] =-r_hat[1]
        L3[2,1] = r_hat[0]
        L3[2,2] = 0.0

        # cast sepration distance into tabulated indexs
        epsilon_log = ti.log(epsilon) / ti.log(10)
        dist_index = ti.cast(ti.floor( (epsilon_log - self.sep_min_log) / self.dr_log ), ti.i32)

        # get lubrication resitance functions
        Y11B = 0.0
        Y12B = 0.0
        if epsilon < self.sep_min:
            # touching particles - using minimum separation distance
            print('overlapping occurs: ', i, j, r)
            Y11B = self.tabulated_funcs[0, 4]
            Y12B = self.tabulated_funcs[0, 5]
        else:
            # non-touching particles - linear interpolation
            Y11B = self.lub_funcs(epsilon, dist_index, 4)
            Y12B = self.lub_funcs(epsilon, dist_index, 5)

        # pair i-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TU_ii[k,l] = Y11B * L3[k,l]

        # pair i-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TU_ij[k,l] = Y12B * L3[k,l]

        # pair j-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TU_ji[k,l] =-R_TU_ij[k,l]

        # pair j-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TU_jj[k,l] =-R_TU_ii[k,l]

    @ti.kernel
    def compute_torque_from_rot(self):
        '''
        T = R_{TW} * W
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # loop each pair of particles
            if (i < j):
                # variables
                ri = ti.Vector([0.]*self.dim)
                rj = ti.Vector([0.]*self.dim)
                rij = ti.Vector([0.]*self.dim)

                angi = ti.Vector([0.]*self.dim)
                angj = ti.Vector([0.]*self.dim)

                torquei = ti.Vector([0.]*self.dim)
                torquej = ti.Vector([0.]*self.dim)

                R_TW_ii = ti.Matrix([[0.]*self.dim]*self.dim)
                R_TW_ij = ti.Matrix([[0.]*self.dim]*self.dim)
                R_TW_ji = ti.Matrix([[0.]*self.dim]*self.dim)
                R_TW_jj = ti.Matrix([[0.]*self.dim]*self.dim)

                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    rj[k] = self.pos[j][k]
                    angi[k] = self.ang_lub[i][k]
                    angj[k] = self.ang_lub[j][k]

                # rij
                for k in ti.static(range(self.dim)):
                    rij[k] = ri[k] - rj[k]

                r = rij.norm()

                # within cutoff
                if (r < self.lub_cutoff):

                    # resistance tensor - bulk
                    self.resistance_TW_lub(i, j, rij, R_TW_ii, R_TW_ij, R_TW_ji, R_TW_jj)

                    # resistance tensor - wall correction
                    if ti.static(self.wall_correction_on == True):
                        print('lubrication with wall correction NOT implemented!')

                    # T_i = R_{TW}^{ij} * W_j

                    # i-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquei[k] += R_TW_ii[k,l] * angi[l]

                    # i-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquei[k] += R_TW_ij[k,l] * angj[l]

                    # j-i pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquej[k] += R_TW_ji[k,l] * angi[l]

                    # j-j pair
                    for k in ti.static(range(self.dim)):
                        for l in ti.static(range(self.dim)):
                            torquej[k] += R_TW_jj[k,l] * angj[l]

                    # update global arrays
                    # i-th particle
                    for k in ti.static(range(self.dim)):
                        self.torque_lub[i][k] += torquei[k] * self.R_TW_factor
                    # j-th particle
                    for k in ti.static(range(self.dim)):
                        self.torque_lub[j][k] += torquej[k] * self.R_TW_factor

    @ti.func
    def resistance_TW_lub(self, i, j, r_vec, R_TW_ii: ti.template(), R_TW_ij: ti.template(), R_TW_ji: ti.template(), R_TW_jj: ti.template()):
        '''
        compute R_TW 3x3 lubrication resistance tensor for a pair of particles
        '''
        # rescale rij with particle radius
        r_vec *= self.radius_inv

        # particle distance
        r = r_vec.norm()
        epsilon = r - 2.0

        # unit displacement vector
        r_hat = -r_vec / r # !Negative sign to match formula convention

        # construct displacement tensors
        L1 = ti.Matrix([[0.]*self.dim]*self.dim)
        L2 = ti.Matrix([[0.]*self.dim]*self.dim)

        # L1 - squeezing motion: r_i * r_j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                L1[k,l] = r_hat[k] * r_hat[l]
                L2[k,l] = -1.0 * L1[k,l]

        # L2 - shearing motion: delta_ij - r_i * r_j
        for k in ti.static(range(self.dim)):
            L2[k,k] += 1.0

        # cast sepration distance into tabulated indexs
        epsilon_log = ti.log(epsilon) / ti.log(10)
        dist_index = ti.cast(ti.floor( (epsilon_log - self.sep_min_log) / self.dr_log ), ti.i32)

        # get lubrication resitance functions
        X11C, Y11C = 0.0, 0.0
        X12C, Y12C = 0.0, 0.0
        if epsilon < self.sep_min:
            # touching particles - using minimum separation distance
            print('overlapping occurs: ', i, j, r)
            X11C = self.tabulated_funcs[0, 6]
            X12C = self.tabulated_funcs[0, 7]
            Y11C = self.tabulated_funcs[0, 8]
            Y12C = self.tabulated_funcs[0, 9]
        else:
            # non-touching particles - linear interpolation
            X11C = self.lub_funcs(epsilon, dist_index, 6)
            X12C = self.lub_funcs(epsilon, dist_index, 7)
            Y11C = self.lub_funcs(epsilon, dist_index, 8)
            Y12C = self.lub_funcs(epsilon, dist_index, 9)

        # pair i-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TW_ii[k,l] = X11C * L1[k,l] + Y11C * L2[k,l]

        # pair i-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TW_ij[k,l] = X12C * L1[k,l] + Y12C * L2[k,l]

        # pair j-i
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TW_ji[k,l] = R_TW_ij[k,l]

        # pair j-j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                R_TW_jj[k,l] = R_TW_ii[k,l]

    @ti.kernel
    def build_resistance_matrix(self):
        '''
        build resistance matrix by assuming pairwise additivity
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # loop each pair of particles
            if (i < j):
                # variables
                ri = ti.Vector([0.]*self.dim)
                rj = ti.Vector([0.]*self.dim)
                rij = ti.Vector([0.]*self.dim)

                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    rj[k] = self.pos[j][k]

                # rij
                for k in ti.static(range(self.dim)):
                    rij[k] = ri[k] - rj[k]

                r = rij.norm()

                # within cutoff
                if (r < self.lub_cutoff):

                    # resistance tensor - R_{FU} R_{TU} R_{FW} R_{TW}
                    self.resistance_matrix_lub(i, j, rij)

                    # resistance tensor - wall correction
                    if ti.static(self.wall_correction_on == True):
                        print('lubrication with wall correction NOT implemented!')

                    # add up neighbors number
                    self.neighbors_num[i] += 1
                    self.neighbors_num[j] += 1

    @ti.func
    def update_resistance_matrix(self, i, j, ind_row, ind_col, R_pair_ii: ti.template(), R_pair_ij: ti.template(), R_pair_ji: ti.template(), R_pair_jj: ti.template()):
        '''
        update entrys of resistance matrix into global arrays
        '''
        rows_offset = ind_row * self.N * self.dim 
        cols_offset = ind_col * self.N * self.dim
        # i-i block
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                ind = ti.atomic_add(self.elements_num[None], 1)
                if self.elements_num[None] > self.elements_num_max:
                    print("current element number ", self.elements_num[None], " exceed max length ", self.elements_num_max)

                self.vals[ind] = R_pair_ii[k,l]
                self.rows[ind] = k + i*self.dim + rows_offset
                self.cols[ind] = l + i*self.dim + cols_offset

                # dynamic nodes
                #ti.append(self.vals.parent(), self.elements_num, R_FU_pair[k,l])
                #ti.append(self.rows.parent(), self.elements_num, k+i*self.dim)
                #ti.append(self.cols.parent(), self.elements_num, l+i*self.dim)

        # i-j block
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                ind = ti.atomic_add(self.elements_num[None], 1)
                if self.elements_num[None] > self.elements_num_max:
                    print("current element number ", self.elements_num[None], " exceed max length ", self.elements_num_max)

                self.vals[ind] = R_pair_ij[k,l]
                self.rows[ind] = k + i*self.dim + rows_offset
                self.cols[ind] = l + j*self.dim + cols_offset

        # j-i block
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                ind = ti.atomic_add(self.elements_num[None], 1)
                if self.elements_num[None] > self.elements_num_max:
                    print("current element number ", self.elements_num[None], " exceed max length ", self.elements_num_max)

                self.vals[ind] = R_pair_ji[k,l]
                self.rows[ind] = k + j*self.dim + rows_offset
                self.cols[ind] = l + i*self.dim + cols_offset

        # j-j block
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                ind = ti.atomic_add(self.elements_num[None], 1)
                if self.elements_num[None] > self.elements_num_max:
                    print("current element number ", self.elements_num[None], " exceed max length ", self.elements_num_max)

                self.vals[ind] = R_pair_jj[k,l]
                self.rows[ind] = k + j*self.dim + rows_offset
                self.cols[ind] = l + j*self.dim + cols_offset

    @ti.func
    def resistance_matrix_lub(self, i, j, r_vec):
        '''
        compute each 6x6 block resistance tensor
        | F | = |R_{FU} R_{FW}| | U |
        | T |   |R_{TU} R_{TW}| | W |

        The data arrangement of each resistance tensor is:
        |R_ii R_ij|
        |R_ji R_jj|
        '''
        # rescale rij with particle radius
        r_vec *= self.radius_inv

        # particle distance
        r = r_vec.norm()
        epsilon = r - 2.0

        # unit displacement vector
        r_hat = -r_vec / r # !Negative sign to match formula convention

        # construct displacement tensors
        L1 = ti.Matrix([[0.]*self.dim]*self.dim)
        L2 = ti.Matrix([[0.]*self.dim]*self.dim)
        L3 = ti.Matrix([[0.]*self.dim]*self.dim)

        # L1 - squeezing motion: r_i * r_j
        for k in ti.static(range(self.dim)):
            for l in ti.static(range(self.dim)):
                L1[k,l] = r_hat[k] * r_hat[l]
                L2[k,l] = -1.0 * L1[k,l]

        # L2 - shearing motion: delta_ij - r_i * r_j
        for k in ti.static(range(self.dim)):
            L2[k,k] += 1.0

        # L3 - vortex motion: epsilon_ijk * r_k (left-handed!)
        L3[0,0] = 0.0
        L3[0,1] =-r_hat[2]
        L3[0,2] = r_hat[1]

        L3[1,0] = r_hat[2]
        L3[1,1] = 0.0
        L3[1,2] =-r_hat[0]

        L3[2,0] =-r_hat[1]
        L3[2,1] = r_hat[0]
        L3[2,2] = 0.0

        # cast separation distance into tabulated indexs
        epsilon_log = ti.log(epsilon) / ti.log(10)
        dist_index = ti.cast(ti.floor( (epsilon_log - self.sep_min_log) / self.dr_log ), ti.i32)

        # get lubrication resitance functions
        X11A, Y11A, Y11B, X11C, Y11C = 0.0, 0.0, 0.0, 0.0, 0.0
        X12A, Y12A, Y12B, X12C, Y12C = 0.0, 0.0, 0.0, 0.0, 0.0
        if epsilon < self.sep_min:
            # touching particles - using minimum separation distance
            print('WARNING: overlapping occurs between ', i, ' and ', j,' particles with distance ', r)
            X11A = self.tabulated_funcs[0, 0]
            X12A = self.tabulated_funcs[0, 1]
            Y11A = self.tabulated_funcs[0, 2]
            Y12A = self.tabulated_funcs[0, 3]
            Y11B = self.tabulated_funcs[0, 4]
            Y12B = self.tabulated_funcs[0, 5]
            X11C = self.tabulated_funcs[0, 6]
            X12C = self.tabulated_funcs[0, 7]
            Y11C = self.tabulated_funcs[0, 8]
            Y12C = self.tabulated_funcs[0, 9]
        else:
            # non-touching particles - linear interpolation
            X11A = self.lub_funcs(epsilon, dist_index, 0)
            X12A = self.lub_funcs(epsilon, dist_index, 1)
            Y11A = self.lub_funcs(epsilon, dist_index, 2)
            Y12A = self.lub_funcs(epsilon, dist_index, 3)
            Y11B = self.lub_funcs(epsilon, dist_index, 4)
            Y12B = self.lub_funcs(epsilon, dist_index, 5)
            X11C = self.lub_funcs(epsilon, dist_index, 6)
            X12C = self.lub_funcs(epsilon, dist_index, 7)
            Y11C = self.lub_funcs(epsilon, dist_index, 8)
            Y12C = self.lub_funcs(epsilon, dist_index, 9)

        ## pair i-i
        #R_FU_pair_ii = self.R_FU_factor * (X11A * L1 + Y11A * L2)
        #R_FW_pair_ii =-self.R_FW_factor * (Y11B * L3)
        #R_TU_pair_ii = self.R_TU_factor * (Y11B * L3)
        #R_TW_pair_ii = self.R_TW_factor * (X11C * L1 + Y11C * L2)

        ## pair i-j
        #R_FU_pair_ij = self.R_FU_factor * (X12A * L1 + Y12A * L2)
        #R_FW_pair_ij = self.R_FW_factor * (Y12B * L3)
        #R_TU_pair_ij = self.R_TU_factor * (Y12B * L3)
        #R_TW_pair_ij = self.R_TW_factor * (X12C * L1 + Y12C * L2)

        ## pair j-i
        #R_FU_pair_ji = self.R_FU_factor * (X12A * L1 + Y12A * L2)
        #R_FW_pair_ji =-self.R_FW_factor * (Y12B * L3)
        #R_TU_pair_ji =-self.R_TU_factor * (Y12B * L3)
        #R_TW_pair_ji = self.R_TW_factor * (X12C * L1 + Y12C * L2)

        ## pair j-j
        #R_FU_pair_jj = self.R_FU_factor * (X11A * L1 + Y11A * L2)
        #R_FW_pair_jj = self.R_FW_factor * (Y11B * L3)
        #R_TU_pair_jj =-self.R_TU_factor * (Y11B * L3)
        #R_TW_pair_jj = self.R_TW_factor * (X11C * L1 + Y11C * L2)

        #########################################################
        # symmetry used
        #########################################################
        # pair i-i
        R_FU_pair_ii = self.R_FU_factor * (X11A * L1 + Y11A * L2)
        R_FW_pair_ii =-self.R_FW_factor * (Y11B * L3)
        R_TU_pair_ii = self.R_TU_factor * (Y11B * L3)
        R_TW_pair_ii = self.R_TW_factor * (X11C * L1 + Y11C * L2)

        # pair i-j
        R_FU_pair_ij = self.R_FU_factor * (X12A * L1 + Y12A * L2)
        R_FW_pair_ij = self.R_FW_factor * (Y12B * L3)
        R_TU_pair_ij = self.R_TU_factor * (Y12B * L3)
        R_TW_pair_ij = self.R_TW_factor * (X12C * L1 + Y12C * L2)

        # pair j-i
        R_FU_pair_ji = 1.0 * R_FU_pair_ij
        R_FW_pair_ji =-1.0 * R_FW_pair_ij
        R_TU_pair_ji =-1.0 * R_TU_pair_ij
        R_TW_pair_ji = 1.0 * R_TW_pair_ij

        # pair j-j
        R_FU_pair_jj = 1.0 * R_FU_pair_ii
        R_FW_pair_jj =-1.0 * R_FW_pair_ii
        R_TU_pair_jj =-1.0 * R_TU_pair_ii
        R_TW_pair_jj = 1.0 * R_TW_pair_ii

        # push those elements into global arrays

        # R_{FU}
        self.update_resistance_matrix(i, j, 0, 0, R_FU_pair_ii, R_FU_pair_ij, R_FU_pair_ji, R_FU_pair_jj)
        # R_{FW}
        self.update_resistance_matrix(i, j, 0, 1, R_FW_pair_ii, R_FW_pair_ij, R_FW_pair_ji, R_FW_pair_jj)
        # R_{TU}
        self.update_resistance_matrix(i, j, 1, 0, R_TU_pair_ii, R_TU_pair_ij, R_TU_pair_ji, R_TU_pair_jj)
        # R_{TW}
        self.update_resistance_matrix(i, j, 1, 1, R_TW_pair_ii, R_TW_pair_ij, R_TW_pair_ji, R_TW_pair_jj)

    @ti.func
    def lub_funcs(self, epsilon, dist_index, func_index):
        '''
        get lubrication resistance function values from linear interpolation of
        tabulated values

        index 0    1    2    3    4    5    6    7    8    9
        funcs X11A X12A Y11A Y12A Y11B Y12B X11C X12C Y11C Y12C

        tabulated values are evenly distributed in the log scale.

        '''

        # cast sepration distance into tabulated indexs
        indl = dist_index
        indr = indl + 1

        # do linear interpolation
        xl = self.tabulated_dist[indl]
        yl = self.tabulated_funcs[indl,func_index]

        xr = self.tabulated_dist[indr]
        yr = self.tabulated_funcs[indr,func_index]

        x = epsilon + 2.0

        val = self.linear_interp(x, xl, yl, xr, yr)
        return val

    @ti.func
    def linear_interp(self, x, xl, yl, xr, yr):
        '''
        A simple linear interpolation
        '''
        dydx = (yr - yl) / (xr - xl)
        return yl + dydx * (x - xl)

    @ti.kernel
    def check_isolated_particles(self):
        '''
        add resistance tensors for isolated particles
        '''
        for i in ti.ndrange(self.N):
            if self.neighbors_num[i] == 0:
                # using bulk resistance tensor for isolated particles
                # R_FU = 6*pi*eta*a * delta_ij
                for k in ti.static(range(self.dim)):
                    ind = ti.atomic_add(self.elements_num[None], 1)
                    if self.elements_num[None] > self.elements_num_max:
                        print("current element number ", self.elements_num[None], " exceed max length ", self.elements_num_max)

                    self.vals[ind] = self.R_FU_factor
                    self.rows[ind] = k + i*self.dim
                    self.cols[ind] = k + i*self.dim

                # R_FW=R_TU=0
                # pass

                # R_TW = 8*pi*eta*a^3 * delta_ij
                rows_offset = self.N * self.dim
                cols_offset = self.N * self.dim
                for k in ti.static(range(self.dim)):
                    ind = ti.atomic_add(self.elements_num[None], 1)
                    if self.elements_num[None] > self.elements_num_max:
                        print("current element number ", self.elements_num[None], " exceed max length ", self.elements_num_max)

                    self.vals[ind] = self.R_TW_factor / 3.0 * 4.0
                    self.rows[ind] = k + i*self.dim + rows_offset
                    self.cols[ind] = k + i*self.dim + cols_offset

