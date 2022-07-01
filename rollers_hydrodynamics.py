import taichi as ti
import numpy as np
from rollers_lubrications import rollers_lubrications
from rollers_fluctuations import rollers_fluctuations
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix, csc_matrix, diags
from sksparse.cholmod import cholesky
from functools import partial
from gmres_solver import gmres_solver
from gmres_solver import gmres_solver_cpu, gmres_counter
import time
import ipdb

@ti.data_oriented
class rollers_hydrodynamics:
    '''
    Define the hydrodynamic interactions between Quincke rollers
    '''
    def __init__(self, system):
        # rollers suspension
        self.system = system

        self.N = system.N
        self.dim = system.dim
        self.L=system.L

        self.radius = system.radius
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

        self.force_bulk = None
        self.torque_bulk = None

        self.sparsity_mask = system.sparsity_mask

        # wall particle properties
        self.N_wall = system.N_wall
        self.wall_shape = system.wall_shape
        self.wall_type = system.wall_type

        self.pos_wall = system.pos_wall

        self.vel_wall = system.vel_wall
        self.ang_wall = system.ang_wall

        self.vel_det_wall = system.vel_det_wall
        self.ang_det_wall = system.ang_det_wall

        self.force_wall = system.force_wall
        self.torque_wall = system.torque_wall

        self.sparsity_mask_wall = system.sparsity_mask_wall

        # hydrodynamic interaction cutoff range
        self.hydro_cutoff = max(system.Lx, system.Ly, system.Lz)
        #self.hydro_cutoff = 5.0 * self.radius

        self.UT_self_factor = 1.0
        self.WT_self_factor = 0.5

        # normalization factors
        self.radius_inv = 1.0 / self.radius
        self.M_UF_factor = 1.0 / (8.0 * np.pi * self.viscosity * self.radius)
        self.M_UT_factor = 1.0 / (8.0 * np.pi * self.viscosity * self.radius**2)
        self.M_WF_factor = 1.0 / (8.0 * np.pi * self.viscosity * self.radius**2)
        self.M_WT_factor = 1.0 / (8.0 * np.pi * self.viscosity * self.radius**3)

        # wall corrections
        self.wall_correction_on = system.RPY_wall_correction_on

        # only valid for small particle-wall seperation
        self.wall_lubrication_on = False

        # lubrications corrections
        self.lubrications_on = system.lubrications_on
        if ti.static(self.lubrications_on == True):
            self.lubrications = rollers_lubrications(system)
            self.deltaR = None # resistance matrix with RPY substracted

        # using uncorrelated hydrodynamics - self-mobility only
        self.hydro_uncorrelated = system.hydro_uncorrelated

        # thermal flucutations
        self.fluctuations_on = system.fluctuations_on
        if ti.static(self.fluctuations_on == True):
            self.fluctuations = rollers_fluctuations(system)

        # GMRES related
        self.gmres_tol = system.gmres_tol
        self.gmres_maxiter = system.gmres_maxiter
        self.gmres_iter_tot = 0
        self.gmres_residual_on = system.gmres_residual_on
        self.gmres_save_prev_result = True
        self.gmres_x0 = None
        self.gmres_pc_on = system.gmres_pc_on
        self.gmres_pc_side = system.gmres_pc_side

        if ti.static(self.N_wall > 0):
            self.gmres_solver = gmres_solver(self)

        # specify data layout
        self.layout()

    def layout(self):
        '''
        specify data layout
        '''
        # field quantities
        self.force_bulk = ti.Vector.field(self.dim, dtype=ti.f64)
        self.torque_bulk = ti.Vector.field(self.dim, dtype=ti.f64)

        # place field quantities
        ti.root.dense(ti.i, self.N).place(self.force_bulk, self.torque_bulk)

    def init(self):
        '''
        initialization
        '''
        if ti.static(self.lubrications_on == True):
            self.lubrications.init()

    def compute(self):
        '''
        compute hydrodynamic interactions
        '''
        # update forces and torques
        self.update_forces_torques_bulk()

        # clear arrays for accumulation
        self.clear()

        if ti.static(self.lubrications_on == True):
            self.solve_kinetics_with_lubrications()
        else:
            self.solve_kinetics_without_lubrications()

        #self.solve_kinetics_lubrications_only()

        # generate thermal noise
        if ti.static(self.fluctuations_on == True):
            self.fluctuations.generate_thermal_noise()

    def apply_mobility_tensors_bb(self):
        '''
        compute [U_b; W_b] = M_{bb} * [F_b; T_b]
        '''
        if ti.static(self.hydro_uncorrelated == False):
            self.compute_bulk_trans_from_bulk_force()

            self.compute_bulk_trans_from_bulk_torque()

            self.compute_bulk_rot_from_bulk_force()

            self.compute_bulk_rot_from_bulk_torque()
        else:
            self.compute_trans_from_force_uncorrelated()

            self.compute_trans_from_torque_uncorrelated()

            self.compute_rot_from_force_uncorrelated()

            self.compute_rot_from_torque_uncorrelated()

    def solve_kinetics_with_lubrications(self):
        '''
        compute [I + M * R ] * U = M * F
        '''
        # prepare right hand side = M * F
        self.apply_mobility_tensors_bb()

        RHS_vel = self.vel_det.to_numpy()
        RHS_ang = self.ang_det.to_numpy()
        b = np.concatenate((RHS_vel.flatten(), RHS_ang.flatten()))

        # build lubrication resistance tensor
        self.lubrications.clear_neighbors()
        self.lubrications.build_resistance_matrix()

        # get resistance matrix from gpu
        elements_num = self.lubrications.elements_num.to_numpy()
        rows = self.lubrications.rows.to_numpy()[:elements_num]
        cols = self.lubrications.cols.to_numpy()[:elements_num]
        vals = self.lubrications.vals.to_numpy()[:elements_num]

        # construct sparse matrix
        self.deltaR = coo_matrix( (vals, (rows,cols)), shape=(self.N*self.dim*2, self.N*self.dim*2) ).tocsc()

        # [I + M * R] * U
        def lubrication_correction(vel_ang, deltaR = None):
            # compute: R * U
            force_torque = deltaR.dot(vel_ang)

            # unpack data
            force = force_torque[0:self.dim*self.N].reshape((self.N, self.dim))
            torque = force_torque[self.dim*self.N:].reshape((self.N, self.dim))

            # set forces and torques
            self.force_bulk.from_numpy(force)
            self.torque_bulk.from_numpy(torque)

            # clear velocities
            self.clear_velocities()

            # compute M * (R * U)
            self.apply_mobility_tensors_bb()

            # retrive results
            LHS_vel = self.vel_det.to_numpy()
            LHS_ang = self.ang_det.to_numpy()

            LHS_vel_ang = np.concatenate((LHS_vel.flatten(), LHS_ang.flatten()))

            # U + M * (R * U)
            return vel_ang + LHS_vel_ang

        # define linear operator
        linear_operator_lub = partial(lubrication_correction, deltaR = self.deltaR)

        system_size = 2 * self.dim * self.N
        A = spla.LinearOperator((system_size,system_size), matvec=linear_operator_lub, dtype='float64')

        # build preconditioner
        if ti.static(self.gmres_pc_on == True):
            # construct M_bulk
            M_UF = np.ones(self.dim*self.N) * self.M_UF_factor * (4.0/3.0)
            M_WT = np.ones(self.dim*self.N) * self.M_WT_factor
            M_diags = np.concatenate((M_UF, M_WT))
            # I + M_bulk * R
            IMdR = diags(np.ones(system_size), format='csc') + diags(M_diags, format='csc') * self.deltaR
            IMdR_inv = cholesky(IMdR)

            # M = P^{-1} = (I + M_bulk * R)^{-1}
            def preconditioner(x, IMdR_inv = None):
                return IMdR_inv(x)
            PC_partial = partial(preconditioner, IMdR_inv = IMdR_inv)
            PC = spla.LinearOperator((system_size, system_size), matvec = PC_partial, dtype='float64')
        else:
            PC = None

        # GMRES callback function
        gmres_callback = gmres_counter(residual_on=self.gmres_residual_on)

        # normalize right hand side
        b_norm = np.linalg.norm(b)
        if b_norm > 0:
            b /= b_norm

        # call the GMRES solver
        # A * x = b
        #start_time = time.time()

        (x, info) = gmres_solver_cpu(A, b, x0=self.gmres_x0, tol=self.gmres_tol, maxiter=self.gmres_maxiter, M=PC, callback=gmres_callback, PC_side=self.gmres_pc_side)

        if (info < 0):
            print('illegal input or breakdown.')
            exit()
        if (info > 0):
            print('maximum number of iterations reached.')
            exit()

        #end_time = time.time()
        #print('gmres time: '+str(end_time-start_time))

        self.gmres_iter_tot += gmres_callback.niter
        self.gmres_x0 = x

        # scale the solution back
        if b_norm > 0:
            x *= b_norm

        # unpack results
        vel = x[0:self.dim*self.N].reshape((self.N, self.dim))
        ang = x[self.dim*self.N:].reshape((self.N, self.dim))

        # update gpu arrays
        self.vel_det.from_numpy(vel)
        self.ang_det.from_numpy(ang)

    def solve_kinetics_without_lubrications(self):
        '''
        compute U = M * F with far-field RPY only
        '''
        self.apply_mobility_tensors_bb()

    def solve_kinetics_lubrications_only(self):
        '''
        compute R * U = F with near-field lubrication only
        '''
        # compute lubrication resistance tensor
        self.lubrications.clear_neighbors()
        self.lubrications.build_resistance_matrix()

        # get resistance matrix from gpu
        elements_num = self.lubrications.elements_num.to_numpy()
        rows = self.lubrications.rows.to_numpy()[:elements_num]
        cols = self.lubrications.cols.to_numpy()[:elements_num]
        vals = self.lubrications.vals.to_numpy()[:elements_num]

        # construct sparse matrix
        R_lub = coo_matrix( (vals, (rows,cols)), shape=(self.N*self.dim*2, self.N*self.dim*2) ).tocsc()

        # cholesky decomposition
        cholesky_factor = cholesky(R_lub)

        # get forces and torques
        force = self.force.to_numpy()
        torque = self.torque.to_numpy()
        b = np.concatenate( (force.flatten(), torque.flatten()) )

        # solve R * [U; W] = [F; T]
        x = cholesky_factor(b)

        # extract translational and angular velocities
        vel = x[:self.N*self.dim].reshape((self.N, self.dim))
        ang = x[self.N*self.dim:].reshape((self.N, self.dim))

        # update lubrication translational and angular velocity
        self.vel_det.from_numpy(vel)
        self.ang_det.from_numpy(ang)

    @ti.kernel
    def clear(self):
        '''
        clear related arrays
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.vel_det[i][k] = 0.0
                self.ang_det[i][k] = 0.0

                self.vel_stoch[i][k] = 0.0
                self.ang_stoch[i][k] = 0.0

        if ti.static(self.N_wall > 0):
            for i in ti.ndrange(self.N_wall):
                for k in ti.static(range(self.dim)):
                    self.vel_det_wall[i][k] = 0.0
                    self.ang_det_wall[i][k] = 0.0

                    self.force_wall[i][k] = 0.0
                    self.torque_wall[i][k] = 0.0

    @ti.kernel
    def clear_velocities(self):
        '''
        clear velocity arrays
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.vel_det[i][k] = 0.0
                self.ang_det[i][k] = 0.0

    @ti.kernel
    def update_forces_torques_bulk(self):
        '''
        update forces and torques
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.force_bulk[i][k] = self.force[i][k]
                self.torque_bulk[i][k] = self.torque[i][k]

    @ti.kernel
    def copy_forces_torques_lub(self):
        '''
        copy lubrication forces and torques
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.force_bulk[i][k] = self.lubrications.force_lub[i][k]
                self.torque_bulk[i][k] = self.lubrications.torque_lub[i][k]

    @ti.kernel
    def clear_auxiliary(self):
        '''
        clear auxiliary arrays
        '''
        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.vel_det_wall[i][k] = 0.0
                self.ang_det_wall[i][k] = 0.0

    @ti.kernel
    def set_wall_velocities(self):
        '''
        set velocities of wall particles
        only fixed wall implemented
        '''
        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.vel_wall[i][k] = 0.0
                self.ang_wall[i][k] = 0.0

    def solve_force_torque_wall_gpu(self):
        '''
        solve forces and torques acting on wall particles
        [U_w; W_w] = M_{wb} * [F_b; T_b] + M_{ww} * [F_w; T_w]
        i.e.
        M_{ww} * [F_w; T_w] = [U_w; W_w] - M_{wb} * [F_b; T_b]
        '''
        # reset GMRES solver
        self.gmres_solver.reset()

        # compute M_{wb} * [F_b; T_b]
        self.compute_wall_trans_from_bulk_force()
        self.compute_wall_trans_from_bulk_torque()
        self.compute_wall_rot_from_bulk_force()
        self.compute_wall_rot_from_bulk_torque()
        # prepare right hand side
        # [U_w; W_w] - M_{wb} * [F_b; T_b]
        self.update_b()

        # M_{ww} * [F_w; T_w]
        def grand_mobility_tensor_wall():
            # unpack data from x
            self.unpack_data()

            # clear auxiliary gpu arrays
            self.clear_auxiliary()
            # launch computaion
            self.compute_wall_trans_from_wall_force()
            self.compute_wall_trans_from_wall_torque()
            self.compute_wall_rot_from_wall_force()
            self.compute_wall_rot_from_wall_torque()

            # pack results to Ax
            self.pack_data()

        # define the linear operator
        self.gmres_solver.matvec=grand_mobility_tensor_wall

        # call the GMRES solver
        # A * x = b
        info = self.gmres_solver.solve()
        if (info < 0):
            print('illegal input or breakdown.')
            exit()
        if (info > 0):
            print('maximum number of iterations reached.')
            exit()

        self.gmres_iter_tot += self.gmres_solver.niter

        # update force and torque on wall particles with gmres solution
        self.update_force_torque_wall()
        #if self.system.timestep == 0:
        #    np.save(self.system.output_dir+'force_wall_gpu.npy', self.force_wall.to_numpy())
        #    np.save(self.system.output_dir+'torque_wall_gpu.npy', self.torque_wall.to_numpy())
        #    ipdb.set_trace()
        #ipdb.set_trace()

    def solve_force_torque_wall_scipy(self):
        '''
        solve forces and torques acting on wall particles
        [U_w; W_w] = M_{wb} * [F_b; T_b] + M_{ww} * [F_w; T_w]
        i.e.
        M_{ww} * [F_w; T_w] = [U_w; W_w] - M_{wb} * [F_b; T_b]
        '''
        # compute M_{wb} * [F_b; T_b]
        self.compute_wall_trans_from_bulk_force()
        self.compute_wall_trans_from_bulk_torque()
        self.compute_wall_rot_from_bulk_force()
        self.compute_wall_rot_from_bulk_torque()

        # prepare right hand side
        # [U_w; W_w] - M_{wb} * [F_b; T_b]
        RHS_vel = self.vel_wall.to_numpy() - self.vel_det_wall.to_numpy()
        RHS_ang = self.ang_wall.to_numpy() - self.ang_det_wall.to_numpy()
        b = np.concatenate((RHS_vel.flatten(),RHS_ang.flatten()))

        # M_{ww} * [F_w; T_w]
        def grand_mobility_tensor_wall(force_torque):
            # unpack data
            force = force_torque[0:self.dim*self.N_wall].reshape((self.N_wall, self.dim))
            torque = force_torque[self.dim*self.N_wall:].reshape((self.N_wall, self.dim))
            # transfer force and torque data to gpu
            self.force_wall.from_numpy(force)
            self.torque_wall.from_numpy(torque)

            # clear auxiliary gpu arrays
            self.clear_auxiliary()
            # launch computaion
            self.compute_wall_trans_from_wall_force()
            self.compute_wall_trans_from_wall_torque()
            self.compute_wall_rot_from_wall_force()
            self.compute_wall_rot_from_wall_torque()

            # retrive results from gpu
            vel = self.vel_det_wall.to_numpy()
            ang = self.ang_det_wall.to_numpy()
            return np.concatenate((vel.flatten(),ang.flatten()))

        # define the linear operator
        system_size = 2 * self.N_wall * self.dim
        A = spla.LinearOperator((system_size,system_size), matvec=grand_mobility_tensor_wall, dtype='float64')

        # GMRES callback function
        gmres_callback = gmres_counter(residual_on=self.gmres_residual_on)

        # normalize right hand side
        b_norm = np.linalg.norm(b)
        if b_norm > 0:
            b /= b_norm

        # call the GMRES solver
        # A * x = b
        #start_time = time.time()
        (x, info) = spla.gmres(A, b, x0=None, tol=self.gmres_tol, maxiter=self.gmres_maxiter, M=None, callback=gmres_callback)
        if (info < 0):
            print('illegal input or breakdown.')
            exit()
        if (info > 0):
            print('maximum number of iterations reached.')
            exit()
        #end_time = time.time()
        #print('gmres time: '+str(end_time-start_time))

        self.gmres_iter_tot += gmres_callback.niter
        self.gmres_x0 = x

        # scale the solution back
        if b_norm > 0:
            x *= b_norm

        # unpack results
        force = x[0:self.dim*self.N_wall].reshape((self.N_wall, self.dim))
        torque = x[self.dim*self.N_wall:].reshape((self.N_wall, self.dim))

        # update gpu arrays
        self.force_wall.from_numpy(force)
        self.torque_wall.from_numpy(torque)

        #if self.system.timestep == 0:
        #    np.save(self.system.output_dir+'force_wall_cpu.npy', self.force_wall.to_numpy())
        #    np.save(self.system.output_dir+'torque_wall_cpu.npy', self.torque_wall.to_numpy())
        #    ipdb.set_trace()
        #ipdb.set_trace()

    @ti.kernel
    def update_b(self):
        '''
        update the right side vector b of the GMRES solver
        b = [U_w; W_w] - M_{wb} * [F_b; T_b]
        b = 6N x 1
        U_w = N x 3
        W_w = N x 3
        '''
        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.gmres_solver.b[i*self.dim+k] = self.vel_wall[i][k] - self.vel_det_wall[i][k]

        offset = self.N_wall * self.dim

        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.gmres_solver.b[offset+i*self.dim+k] = self.ang_wall[i][k] - self.ang_det_wall[i][k]

    @ti.kernel
    def update_force_torque_wall(self):
        '''
        update the force and torque vector from the GMRES solver
        '''
        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.force_wall[i][k] = self.gmres_solver.x[i*self.dim+k]

        offset = self.N_wall * self.dim

        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.torque_wall[i][k] = self.gmres_solver.x[offset+i*self.dim+k]

    @ti.kernel
    def unpack_data(self):
        '''
        unpack a 6Nx1 vector into two Nx3 vectors
        '''
        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.force_wall[i][k] = self.gmres_solver.x[i*self.dim+k]

        offset = self.N_wall * self.dim

        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.torque_wall[i][k] = self.gmres_solver.x[offset+i*self.dim+k]

    @ti.kernel
    def pack_data(self):
        '''
        pack two Nx3 vectors into a 6Nx1 vector
        '''
        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.gmres_solver.Ax[i*self.dim+k] = self.vel_det_wall[i][k]

        offset = self.N_wall * self.dim

        for i in ti.ndrange(self.N_wall):
            for k in ti.static(range(self.dim)):
                self.gmres_solver.Ax[offset+i*self.dim+k] = self.ang_det_wall[i][k]

    @ti.kernel
    def compute_bulk_trans_from_bulk_force(self):
        '''
        U_b = M_{UF} * F_b
        '''
        for imx,i,j in ti.ndrange(3,self.N, self.N):
            # variables
            imx=imx-1
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_UF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                
                ri[k] = self.pos[i][k]
                rj[k] = self.pos[j][k]+imx*self.L[0]
                force[k] = self.force_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]
            #make sure image is not self-mobility
            if imx!=0:
                j=-1
            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UF_RPY(i, j, rij, M_UF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_UF_SingleWall(i, j, rij, rj[2], M_UF)

                # U_i = M_{UF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det[i][k] += vel_trans[k] * self.M_UF_factor

    @ti.kernel
    def compute_wall_trans_from_bulk_force(self):
        '''
        U_w = M_{UF} * F_b
        '''
        for i,j in ti.ndrange(self.N_wall, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_UF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos[j][k]
                force[k] = self.force_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UF_RPY(i, j, rij, M_UF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_UF_SingleWall(i, j, rij, rj[2], M_UF)

                # U_i = M_{UF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det_wall[i][k] += vel_trans[k] * self.M_UF_factor

    @ti.kernel
    def compute_bulk_trans_from_wall_force(self):
        '''
        U_b = M_{UF} * F_w
        '''
        for i,j in ti.ndrange(self.N, self.N_wall):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_UF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos_wall[j][k]
                force[k] = self.force_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UF_RPY(i, j, rij, M_UF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_UF_SingleWall(i, j, rij, rj[2], M_UF)

                # U_i = M_{UF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det[i][k] += vel_trans[k] * self.M_UF_factor

    @ti.kernel
    def compute_wall_trans_from_wall_force(self):
        '''
        U_w = M_{UF} * F_w
        '''
        for i,j in self.sparsity_mask_wall:
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_UF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos_wall[j][k]
                force[k] = self.force_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UF_RPY(i, j, rij, M_UF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_UF_SingleWall(i, j, rij, rj[2], M_UF)

                # U_i = M_{UF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det_wall[i][k] += vel_trans[k] * self.M_UF_factor

    @ti.func
    def mobility_UF_RPY(self, i, j, r_vec, M_UF: ti.template()):
        '''
        compute M_UF 3x3 RPY mobility tensor
        '''
        if (i==j):
            # self-mobility
            M_UF[0,0] = 4.0 / 3.0
            M_UF[0,1] = 0.0
            M_UF[0,2] = 0.0
            M_UF[1,1] = 4.0 / 3.0
            M_UF[1,2] = 0.0
            M_UF[2,2] = 4.0 / 3.0
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r2 = r_vec.dot(r_vec)
            r = ti.sqrt(r2)

            r_inv = 1.0 / r
            r2_inv = r_inv * r_inv

            if (r>=2.0):
                # far field
                c1 = 1.0 + 2.0 / 3.0 * r2_inv
                c2 = (1.0 - 2.0 * r2_inv) * r2_inv

                M_UF[0,0] = (c1 + c2*r_vec[0]*r_vec[0])*r_inv
                M_UF[0,1] = (     c2*r_vec[0]*r_vec[1])*r_inv
                M_UF[0,2] = (     c2*r_vec[0]*r_vec[2])*r_inv
                M_UF[1,1] = (c1 + c2*r_vec[1]*r_vec[1])*r_inv
                M_UF[1,2] = (     c2*r_vec[1]*r_vec[2])*r_inv
                M_UF[2,2] = (c1 + c2*r_vec[2]*r_vec[2])*r_inv
            else:
                # near field
                c1 = (4.0/3.0)*(1.0 - 9.0/32.0*r)
                c2 = (4.0/3.0)*(3.0/32.0*r_inv)

                M_UF[0,0] = c1 + c2*r_vec[0]*r_vec[0]
                M_UF[0,1] =      c2*r_vec[0]*r_vec[1]
                M_UF[0,2] =      c2*r_vec[0]*r_vec[2]
                M_UF[1,1] = c1 + c2*r_vec[1]*r_vec[1]
                M_UF[1,2] =      c2*r_vec[1]*r_vec[2]
                M_UF[2,2] = c1 + c2*r_vec[2]*r_vec[2]

        # mobility tensor symmetry
        M_UF[1,0] = M_UF[0,1]
        M_UF[2,0] = M_UF[0,2]
        M_UF[2,1] = M_UF[1,2]

    @ti.func
    def mobility_UF_SingleWall(self, i, j, r_vec, h, M_UF: ti.template()):
        '''
        compute single wall correction of M_UF
        '''
        if (i==j):
            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 8.0/15.0*ti.log(gap_inv) - 0.9588
                R2 = gap_inv

                M1 = 1.0 / R1
                M2 = 1.0 / R2

                M_UF[0,0] *= M1
                M_UF[1,1] *= M1
                M_UF[2,2] *= M2
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h2_inv = h_inv * h_inv
                h3_inv = h2_inv * h_inv
                h5_inv = h3_inv * h2_inv

                M_UF[0,0] += -(9.0*h_inv - 2.0*h3_inv + h5_inv) / 12.0
                M_UF[1,1] += -(9.0*h_inv - 2.0*h3_inv + h5_inv) / 12.0
                M_UF[2,2] += -(9.0*h_inv - 4.0*h3_inv + h5_inv) / 6.0
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r = r_vec.norm()

            if (r>= 2.0):
                # non-overlapping case
                h *= self.radius_inv # normalize height with particle radius
                R_vec = r_vec
                R_vec[2] += 2*h # image j -> i
                h_hat = h / R_vec[2]

                R = R_vec.norm()
                e_vec = R_vec / R

                R_inv = 1.0 / R
                R3_inv = R_inv * R_inv * R_inv
                R5_inv = R3_inv * R_inv * R_inv

                c1 = -(3.0*(1.0+2.0*h_hat*(1.0-h_hat)*e_vec[2]*e_vec[2])*R_inv + 2.0*(1.0-3.0*e_vec[2]*e_vec[2])*R3_inv - 2.0*(1.0-5.0*e_vec[2]*e_vec[2])*R5_inv) / 3.0
                c2 = -(3.0*(1.0-6.0*h_hat*(1.0-h_hat)*e_vec[2]*e_vec[2])*R_inv - 6.0*(1.0-5.0*e_vec[2]*e_vec[2])*R3_inv +10.0*(1.0-7.0*e_vec[2]*e_vec[2])*R5_inv) / 3.0
                c3 = e_vec[2] * (3.0*h_hat*(1.0-6.0*(1.0-h_hat)*e_vec[2]*e_vec[2])*R_inv - 6.0*(1.0-5.0*e_vec[2]*e_vec[2])*R3_inv + 10.0*(2.0-7.0*e_vec[2]*e_vec[2])*R5_inv) * 2.0 / 3.0
                c4 = e_vec[2] * (3.0*h_hat*R_inv - 10.0*R5_inv) * 2.0 / 3.0
                c5 = -(3.0*h_hat*h_hat*e_vec[2]*e_vec[2]*R_inv + 3.0*e_vec[2]*e_vec[2]*R3_inv + (2.0 - 15.0*e_vec[2]*e_vec[2])*R5_inv) * 4.0 / 3.0

                # c1*delta_ij + c2*e_i*e_j + c3*e_i*delta_j3 + c4*delta_i3*e_j + c5*delta_i3*delta_j3
                M_UF[0,0] += c1 + c2*e_vec[0]*e_vec[0]
                M_UF[0,1] +=      c2*e_vec[0]*e_vec[1]
                M_UF[0,2] +=      c2*e_vec[0]*e_vec[2] + c3*e_vec[0]

                M_UF[1,0] +=      c2*e_vec[1]*e_vec[0]
                M_UF[1,1] += c1 + c2*e_vec[1]*e_vec[1]
                M_UF[1,2] +=      c2*e_vec[1]*e_vec[2] + c3*e_vec[1]

                M_UF[2,0] +=      c2*e_vec[2]*e_vec[0]               + c4*e_vec[0]
                M_UF[2,1] +=      c2*e_vec[2]*e_vec[1]               + c4*e_vec[1]
                M_UF[2,2] += c1 + c2*e_vec[2]*e_vec[2] + c3*e_vec[2] + c4*e_vec[2] + c5

    @ti.kernel
    def compute_trans_from_force_uncorrelated(self):
        '''
        U = M_{UF} * F - self-mobility only
        '''
        for i in ti.ndrange(self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_UF = ti.Matrix.rows([ri,ri,ri])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                force[k] = self.force_bulk[i][k]

            # mobility tensor - bulk

            # self-mobility
            M_UF[0,0] = 4.0 / 3.0
            M_UF[0,1] = 0.0
            M_UF[0,2] = 0.0
            M_UF[1,1] = 4.0 / 3.0
            M_UF[1,2] = 0.0
            M_UF[2,2] = 4.0 / 3.0
            # mobility tensor symmetry
            M_UF[1,0] = M_UF[0,1]
            M_UF[2,0] = M_UF[0,2]
            M_UF[2,1] = M_UF[1,2]

            # mobility tensor - wall correction
            h = ri[2]

            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 8.0/15.0*ti.log(gap_inv) - 0.9588
                R2 = gap_inv

                M1 = 1.0 / R1
                M2 = 1.0 / R2

                M_UF[0,0] *= M1
                M_UF[1,1] *= M1
                M_UF[2,2] *= M2
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h2_inv = h_inv * h_inv
                h3_inv = h2_inv * h_inv
                h5_inv = h3_inv * h2_inv

                M_UF[0,0] += -(9.0*h_inv - 2.0*h3_inv + h5_inv) / 12.0
                M_UF[1,1] += -(9.0*h_inv - 2.0*h3_inv + h5_inv) / 12.0
                M_UF[2,2] += -(9.0*h_inv - 4.0*h3_inv + h5_inv) / 6.0

            # U_i = M_{UF}^{ij} * F_j
            for k in ti.static(range(self.dim)):
                for l in ti.static(range(self.dim)):
                    vel_trans[k] += M_UF[k,l]*force[l]

            # update global arrays
            for k in ti.static(range(self.dim)):
                self.vel_det[i][k] += vel_trans[k] * self.M_UF_factor

    @ti.kernel
    def compute_bulk_trans_from_bulk_torque(self):
        '''
        U_b = M_{UT} * T_b
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_UT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos[j][k]
                torque[k] = self.torque_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UT_RPY(i, j, rij, M_UT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    # symmetry M_{UT,ij}^{alpha,beta} = M_{WF,ji}^{beta,alpha}
                    self.mobility_UT_SingleWall(j, i, -rij, ri[2], M_UT)

                # U_i = M_{UT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det[i][k] += vel_trans[k] * self.M_UT_factor

    @ti.kernel
    def compute_wall_trans_from_bulk_torque(self):
        '''
        U_w = M_{UT} * T_b
        '''
        for i,j in ti.ndrange(self.N_wall, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_UT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos[j][k]
                torque[k] = self.torque_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UT_RPY(i, j, rij, M_UT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    # symmetry M_{UT,ij}^{alpha,beta} = M_{WF,ji}^{beta,alpha}
                    self.mobility_UT_SingleWall(j, i, -rij, ri[2], M_UT)

                # U_i = M_{UT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det_wall[i][k] += vel_trans[k] * self.M_UT_factor

    @ti.kernel
    def compute_bulk_trans_from_wall_torque(self):
        '''
        U_b = M_{UT} * T_w
        '''
        for i,j in ti.ndrange(self.N, self.N_wall):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_UT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos_wall[j][k]
                torque[k] = self.torque_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UT_RPY(i, j, rij, M_UT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    # symmetry M_{UT,ij}^{alpha,beta} = M_{WF,ji}^{beta,alpha}
                    self.mobility_UT_SingleWall(j, i, -rij, ri[2], M_UT)

                # U_i = M_{UT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det[i][k] += vel_trans[k] * self.M_UT_factor

    @ti.kernel
    def compute_wall_trans_from_wall_torque(self):
        '''
        U_w = M_{UT} * T_w
        '''
        for i,j in self.sparsity_mask_wall:
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_UT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos_wall[j][k]
                torque[k] = self.torque_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_UT_RPY(i, j, rij, M_UT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    # symmetry M_{UT,ij}^{alpha,beta} = M_{WF,ji}^{beta,alpha}
                    self.mobility_UT_SingleWall(j, i, -rij, ri[2], M_UT)

                # U_i = M_{UT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_trans[k] += M_UT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.vel_det_wall[i][k] += vel_trans[k] * self.M_UT_factor

    @ti.func
    def mobility_UT_RPY(self, i, j, r_vec, M_UT: ti.template()):
        '''
        compute M_UT 3x3 RPY mobility tensor
        '''
        if (i==j):
            # self-mobility
            M_UT[0,0] = 0.0
            M_UT[0,1] = 0.0
            M_UT[0,2] = 0.0
            M_UT[1,1] = 0.0
            M_UT[1,2] = 0.0
            M_UT[2,2] = 0.0
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r2 = r_vec.dot(r_vec)
            r = ti.sqrt(r2)
            r3 = r2*r

            r3_inv = 1.0/r3

            if (r>=2.0):
                # far field
                M_UT[0,0] =  0.0
                M_UT[0,1] =  r_vec[2]*r3_inv
                M_UT[0,2] = -r_vec[1]*r3_inv
                M_UT[1,1] =  0.0
                M_UT[1,2] =  r_vec[0]*r3_inv
                M_UT[2,2] =  0.0
            else:
                # near field
                c1 = 0.5*(1.0-3.0/8.0*r)

                M_UT[0,0] =  0.0
                M_UT[0,1] =  c1*r_vec[2]
                M_UT[0,2] = -c1*r_vec[1]
                M_UT[1,1] =  0.0
                M_UT[1,2] =  c1*r_vec[0]
                M_UT[2,2] =  0.0

        # mobility tensor symmetry
        M_UT[1,0] = -M_UT[0,1]
        M_UT[2,0] = -M_UT[0,2]
        M_UT[2,1] = -M_UT[1,2]

    @ti.func
    def mobility_UT_SingleWall(self, i, j, r_vec, h, M_UT: ti.template()):
        '''
        compute single wall correction of M_UT
        '''
        if (i==j):
            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 1.0/10.0*ti.log(gap_inv) + 0.1895

                M1 = 1.0/R1

                M_UT[0,1] += M1
                M_UT[1,0] += -M1
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h2_inv = h_inv * h_inv
                h4_inv = h2_inv * h2_inv

                M_UT[0,1] +=  h4_inv / 8.0
                M_UT[1,0] += -h4_inv / 8.0

                # apply factor
                M_UT[0,1] *= self.UT_self_factor
                M_UT[1,0] *= self.UT_self_factor
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r = r_vec.norm()

            if (r>=2.0):
                # non-overlapping case
                h *= self.radius_inv # normalize height with particle radius
                R_vec = r_vec
                R_vec[2] += 2*h # image j -> i
                h_hat = h / R_vec[2]

                R = R_vec.norm()
                e_vec = R_vec / R

                R_inv = 1.0 / R
                R2_inv = R_inv * R_inv
                R4_inv = R2_inv * R2_inv

                c1 = R2_inv 
                c2 = (6.0*h_hat*e_vec[2]*e_vec[2]*R2_inv + (1.0-10.0*e_vec[2]*e_vec[2])*R4_inv) * 2.0
                c3 = -e_vec[2] * (3.0*h_hat*R2_inv - 5.0*R4_inv) * 2.0
                c4 = -e_vec[2] * (h_hat*R2_inv - R4_inv) * 2.0

                # c1*epsilon_ijk*e_k + c2*epsilon_3ki*e_k*delta_j3 + c3*epsilon_3ki*e_k*ej + c4*epsilon_3ij
                M_UT[0,0] -=                             - c3*e_vec[1]*e_vec[0] 
                M_UT[0,1] -= - c1*e_vec[2]               + c3*e_vec[0]*e_vec[0] - c4
                M_UT[0,2] -=   c1*e_vec[1]

                M_UT[1,0] -=   c1*e_vec[2]               - c3*e_vec[1]*e_vec[1] + c4
                M_UT[1,1] -=                               c3*e_vec[0]*e_vec[1]
                M_UT[1,2] -= - c1*e_vec[0]

                M_UT[2,0] -= - c1*e_vec[1] - c2*e_vec[1] - c3*e_vec[1]*e_vec[2]
                M_UT[2,1] -=   c1*e_vec[0] + c2*e_vec[0] + c3*e_vec[0]*e_vec[2]

    @ti.kernel
    def compute_trans_from_torque_uncorrelated(self):
        '''
        U = M_{UT} * T - self-mobility only
        '''
        for i in ti.ndrange(self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            vel_trans = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_UT = ti.Matrix.rows([ri,ri,ri])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                torque[k] = self.torque_bulk[i][k]

            # mobility tensor - bulk
            # zeros

            # mobility tensor - wall correction
            h = ri[2]

            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 1.0/10.0*ti.log(gap_inv) + 0.1895

                M1 = 1.0/R1

                M_UT[0,1] += M1
                M_UT[1,0] += -M1
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h2_inv = h_inv * h_inv
                h4_inv = h2_inv * h2_inv

                M_UT[0,1] +=  h4_inv / 8.0
                M_UT[1,0] += -h4_inv / 8.0

                # apply factor
                M_UT[0,1] *= self.UT_self_factor
                M_UT[1,0] *= self.UT_self_factor

            # U_i = M_{UT}^{ij} * T_j
            for k in ti.static(range(self.dim)):
                for l in ti.static(range(self.dim)):
                    vel_trans[k] += M_UT[k,l]*torque[l]

            # update global arrays
            for k in ti.static(range(self.dim)):
                self.vel_det[i][k] += vel_trans[k] * self.M_UT_factor

    @ti.kernel
    def compute_bulk_rot_from_bulk_force(self):
        '''
        W_b = M_{WF} * F_b
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_WF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos[j][k]
                force[k] = self.force_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WF_RPY(i, j, rij, M_WF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WF_SingleWall(i, j, rij, rj[2], M_WF)

                # W_i = M_{WF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det[i][k] += vel_angular[k] * self.M_WF_factor

    @ti.kernel
    def compute_wall_rot_from_bulk_force(self):
        '''
        W_w = M_{WF} * F_b
        '''
        for i,j in ti.ndrange(self.N_wall, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_WF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos[j][k]
                force[k] = self.force_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WF_RPY(i, j, rij, M_WF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WF_SingleWall(i, j, rij, rj[2], M_WF)

                # W_i = M_{WF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det_wall[i][k] += vel_angular[k] * self.M_WF_factor

    @ti.kernel
    def compute_bulk_rot_from_wall_force(self):
        '''
        W_b = M_{WF} * F_w
        '''
        for i,j in ti.ndrange(self.N, self.N_wall):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_WF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos_wall[j][k]
                force[k] = self.force_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WF_RPY(i, j, rij, M_WF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WF_SingleWall(i, j, rij, rj[2], M_WF)

                # W_i = M_{WF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det[i][k] += vel_angular[k] * self.M_WF_factor

    @ti.kernel
    def compute_wall_rot_from_wall_force(self):
        '''
        W_w = M_{WF} * F_w
        '''
        for i,j in self.sparsity_mask_wall:
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_WF = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos_wall[j][k]
                force[k] = self.force_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WF_RPY(i, j, rij, M_WF)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WF_SingleWall(i, j, rij, rj[2], M_WF)

                # W_i = M_{WF}^{ij} * F_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WF[k,l]*force[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det_wall[i][k] += vel_angular[k] * self.M_WF_factor

    @ti.func
    def mobility_WF_RPY(self, i, j, r_vec, M_WF: ti.template()):
        '''
        compute M_WF 3x3 RPY mobility tensor
        '''
        if (i==j):
            # self-mobility
            M_WF[0,0] = 0.0
            M_WF[0,1] = 0.0
            M_WF[0,2] = 0.0
            M_WF[1,1] = 0.0
            M_WF[1,2] = 0.0
            M_WF[2,2] = 0.0
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r2 = r_vec.dot(r_vec)
            r = ti.sqrt(r2)
            r3 = r2*r

            r3_inv = 1.0/r3

            if (r>=2.0):
                # far field
                M_WF[0,0] =  0.0
                M_WF[0,1] =  r_vec[2]*r3_inv
                M_WF[0,2] = -r_vec[1]*r3_inv
                M_WF[1,1] =  0.0
                M_WF[1,2] =  r_vec[0]*r3_inv
                M_WF[2,2] =  0.0
            else:
                # near field
                c1 = 0.5*(1.0-3.0/8.0*r)

                M_WF[0,0] =  0.0
                M_WF[0,1] =  c1*r_vec[2]
                M_WF[0,2] = -c1*r_vec[1]
                M_WF[1,1] =  0.0
                M_WF[1,2] =  c1*r_vec[0]
                M_WF[2,2] =  0.0

        # mobility tensor symmetry
        M_WF[1,0] = -M_WF[0,1]
        M_WF[2,0] = -M_WF[0,2]
        M_WF[2,1] = -M_WF[1,2]

    @ti.func
    def mobility_WF_SingleWall(self, i, j, r_vec, h, M_WF: ti.template()):
        '''
        compute single wall correction of M_WF
        '''
        if (i==j):
            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 1.0/10.0*ti.log(gap_inv) + 0.1895

                M1 = 1.0/R1

                M_WF[0,1] += -M1
                M_WF[1,0] += M1
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h2_inv = h_inv * h_inv
                h4_inv = h2_inv * h2_inv

                M_WF[0,1] += -h4_inv / 8.0
                M_WF[1,0] +=  h4_inv / 8.0
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r = r_vec.norm()

            if (r>=2.0):
                # non-overlapping case
                h *= self.radius_inv # normalize height with particle radius
                R_vec = r_vec
                R_vec[2] += 2*h # image j -> i
                h_hat = h / R_vec[2]

                R = R_vec.norm()
                e_vec = R_vec / R

                R_inv = 1.0 / R
                R2_inv = R_inv * R_inv
                R4_inv = R2_inv * R2_inv

                c1 = R2_inv 
                c2 = (6.0*h_hat*e_vec[2]*e_vec[2]*R2_inv + (1.0-10.0*e_vec[2]*e_vec[2])*R4_inv) * 2.0
                c3 = -e_vec[2] * (3.0*h_hat*R2_inv - 5.0*R4_inv) * 2.0
                c4 = -e_vec[2] * (h_hat*R2_inv - R4_inv) * 2.0

                # c1*epsilon_ijk*e_k + c2*epsilon_3ki*e_k*delta_j3 + c3*epsilon_3ki*e_k*ej + c4*epsilon_3ij
                M_WF[0,0] -=                             - c3*e_vec[1]*e_vec[0] 
                M_WF[0,1] -=   c1*e_vec[2]               - c3*e_vec[1]*e_vec[1] + c4
                M_WF[0,2] -= - c1*e_vec[1] - c2*e_vec[1] - c3*e_vec[1]*e_vec[2]

                M_WF[1,0] -= - c1*e_vec[2]               + c3*e_vec[0]*e_vec[0] - c4
                M_WF[1,1] -=                               c3*e_vec[0]*e_vec[1]
                M_WF[1,2] -=   c1*e_vec[0] + c2*e_vec[0] + c3*e_vec[0]*e_vec[2]

                M_WF[2,0] -=   c1*e_vec[1]
                M_WF[2,1] -= - c1*e_vec[0]

    @ti.kernel
    def compute_rot_from_force_uncorrelated(self):
        '''
        W = M_{WF} * F - self-mobility only
        '''
        for i in ti.ndrange(self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            M_WF = ti.Matrix.rows([ri,ri,ri])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                force[k] = self.force_bulk[i][k]

            # mobility tensor - bulk
            # zeros

            # mobility tensor - wall correction
            h = ri[2]

            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 1.0/10.0*ti.log(gap_inv) + 0.1895

                M1 = 1.0/R1

                M_WF[0,1] += -M1
                M_WF[1,0] += M1
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h2_inv = h_inv * h_inv
                h4_inv = h2_inv * h2_inv

                M_WF[0,1] += -h4_inv / 8.0
                M_WF[1,0] +=  h4_inv / 8.0

            # W_i = M_{WF}^{ij} * F_j
            for k in ti.static(range(self.dim)):
                for l in ti.static(range(self.dim)):
                    vel_angular[k] += M_WF[k,l]*force[l]

            # update global arrays
            for k in ti.static(range(self.dim)):
                self.ang_det[i][k] += vel_angular[k] * self.M_WF_factor

    @ti.kernel
    def compute_bulk_rot_from_bulk_torque(self):
        '''
        W_b = M_{WT} * T_b
        '''
        for i,j in ti.ndrange(self.N, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_WT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos[j][k]
                torque[k] = self.torque_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WT_RPY(i, j, rij, M_WT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WT_SingleWall(i, j, rij, rj[2], M_WT)

                # W_i = M_{WT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det[i][k] += vel_angular[k] * self.M_WT_factor

    @ti.kernel
    def compute_wall_rot_from_bulk_torque(self):
        '''
        W_w = M_{WT} * T_b
        '''
        for i,j in ti.ndrange(self.N_wall, self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_WT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos[j][k]
                torque[k] = self.torque_bulk[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WT_RPY(i, j, rij, M_WT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WT_SingleWall(i, j, rij, rj[2], M_WT)

                # W_i = M_{WT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det_wall[i][k] += vel_angular[k] * self.M_WT_factor

    @ti.kernel
    def compute_bulk_rot_from_wall_torque(self):
        '''
        W_b = M_{WT} * T_w
        '''
        for i,j in ti.ndrange(self.N, self.N_wall):
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_WT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                rj[k] = self.pos_wall[j][k]
                torque[k] = self.torque_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WT_RPY(i, j, rij, M_WT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WT_SingleWall(i, j, rij, rj[2], M_WT)

                # W_i = M_{WT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det[i][k] += vel_angular[k] * self.M_WT_factor

    @ti.kernel
    def compute_wall_rot_from_wall_torque(self):
        '''
        W_w = M_{WT} * T_w
        '''
        for i,j in self.sparsity_mask_wall:
            # variables
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_WT = ti.Matrix.rows([rij,rij,rij])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos_wall[j][k]
                torque[k] = self.torque_wall[j][k]

            # rij
            for k in ti.static(range(self.dim)):
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            # within cutoff
            if (r <= self.hydro_cutoff):

                # mobility tensor - bulk
                self.mobility_WT_RPY(i, j, rij, M_WT)

                # mobility tensor - wall correction
                if ti.static(self.wall_correction_on == True):
                    self.mobility_WT_SingleWall(i, j, rij, rj[2], M_WT)

                # W_i = M_{WT}^{ij} * T_j
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        vel_angular[k] += M_WT[k,l]*torque[l]

                # update global arrays
                for k in ti.static(range(self.dim)):
                    self.ang_det_wall[i][k] += vel_angular[k] * self.M_WT_factor

    @ti.func
    def mobility_WT_RPY(self, i, j, r_vec, M_WT: ti.template()):
        '''
        compute M_WT 3x3 RPY mobility tensor
        '''
        if (i==j):
            # self-mobility
            M_WT[0,0] = 1.0
            M_WT[0,1] = 0.0
            M_WT[0,2] = 0.0
            M_WT[1,1] = 1.0
            M_WT[1,2] = 0.0
            M_WT[2,2] = 1.0
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r2 = r_vec.dot(r_vec)
            r = ti.sqrt(r2)
            r3 = r2*r

            r_inv = 1.0/r
            r2_inv = r_inv*r_inv
            r3_inv = r2_inv*r_inv

            if (r>=2.0):
                # far field
                c1 = -0.5
                c2 =  1.5 * r2_inv

                M_WT[0,0] = (c1 + c2*r_vec[0]*r_vec[0]) * r3_inv
                M_WT[0,1] = (     c2*r_vec[0]*r_vec[1]) * r3_inv
                M_WT[0,2] = (     c2*r_vec[0]*r_vec[2]) * r3_inv
                M_WT[1,1] = (c1 + c2*r_vec[1]*r_vec[1]) * r3_inv
                M_WT[1,2] = (     c2*r_vec[1]*r_vec[2]) * r3_inv
                M_WT[2,2] = (c1 + c2*r_vec[2]*r_vec[2]) * r3_inv
            else:
                # near field
                c1 = 1.0 - 27.0/32.0*r + 5.0/64.0*r3
                c2 = 9.0/32.0*r_inv - 3.0/64.0*r

                M_WT[0,0] = c1 + c2*r_vec[0]*r_vec[0] 
                M_WT[0,1] =      c2*r_vec[0]*r_vec[1] 
                M_WT[0,2] =      c2*r_vec[0]*r_vec[2] 
                M_WT[1,1] = c1 + c2*r_vec[1]*r_vec[1] 
                M_WT[1,2] =      c2*r_vec[1]*r_vec[2] 
                M_WT[2,2] = c1 + c2*r_vec[2]*r_vec[2] 

        # mobility tensor symmetry
        M_WT[1,0] = M_WT[0,1]
        M_WT[2,0] = M_WT[0,2]
        M_WT[2,1] = M_WT[1,2]

    @ti.func
    def mobility_WT_SingleWall(self, i, j, r_vec, h, M_WT: ti.template()):
        '''
        compute single wall correction of M_WT
        '''
        if (i==j):
            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 2.0/5.0*ti.log(gap_inv)-0.3817
                R2 = 1.2020569031595942 - 1.9348022005446792*gap

                M1 = 1.0 / R1
                M2 = 1.0 / R2

                M_WT[0,0] *= M1
                M_WT[1,1] *= M1
                M_WT[2,2] *= M2
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h3_inv = h_inv * h_inv * h_inv

                M_WT[0,0] += - h3_inv * 5.0 / 16.0
                M_WT[1,1] += - h3_inv * 5.0 / 16.0
                M_WT[2,2] += - h3_inv / 8.0

                # apply factor
                M_WT[0,0] *= self.WT_self_factor
                M_WT[1,1] *= self.WT_self_factor
                M_WT[2,2] *= self.WT_self_factor
        else:
            # pair-mobility
            r_vec *= self.radius_inv # normalize with particle radius
            r = r_vec.norm()

            if (r>= 2.0):
                # non-overlapping case
                h *= self.radius_inv # normalize height with particle radius
                R_vec = r_vec
                R_vec[2] += 2*h # image j -> i
                h_hat = h / R_vec[2]

                R = R_vec.norm()
                e_vec = R_vec / R

                R_inv = 1.0 / R
                R3_inv = R_inv * R_inv * R_inv

                c1 = (1.0-6.0*e_vec[2]*e_vec[2]) * R3_inv / 2.0
                c2 = -R3_inv * 3.0 / 2.0
                c3 =  R3_inv * e_vec[2] * 3.0
                c4 =  R3_inv * 3.0

                # c1*delta_ij + c2*e_i*e_j + c3*delta_i3*e_j + c4*epsilon_3ki*epsilon_3lj*e_k*e_l
                M_WT[0,0] += c1 + c2*e_vec[0]*e_vec[0] + c4*e_vec[1]*e_vec[1]
                M_WT[0,1] +=      c2*e_vec[0]*e_vec[1] - c4*e_vec[1]*e_vec[0]
                M_WT[0,2] +=      c2*e_vec[0]*e_vec[2]

                M_WT[1,0] +=      c2*e_vec[1]*e_vec[0] - c4*e_vec[0]*e_vec[1]
                M_WT[1,1] += c1 + c2*e_vec[1]*e_vec[1] + c4*e_vec[0]*e_vec[0]
                M_WT[1,2] +=      c2*e_vec[1]*e_vec[2]

                M_WT[2,0] +=      c2*e_vec[2]*e_vec[0] + c3*e_vec[0]
                M_WT[2,1] +=      c2*e_vec[2]*e_vec[1] + c3*e_vec[1]
                M_WT[2,2] += c1 + c2*e_vec[2]*e_vec[2] + c3*e_vec[2]

    @ti.kernel
    def compute_rot_from_torque_uncorrelated(self):
        '''
        W = M_{WT} * T - self-mobility only
        '''
        for i in ti.ndrange(self.N):
            # variables
            ri = ti.Vector([0.]*self.dim)
            vel_angular = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)
            M_WT = ti.Matrix.rows([ri,ri,ri])

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos[i][k]
                torque[k] = self.torque_bulk[i][k]

            # mobility tensor - bulk

            # self-mobility
            M_WT[0,0] = 1.0
            M_WT[0,1] = 0.0
            M_WT[0,2] = 0.0
            M_WT[1,1] = 1.0
            M_WT[1,2] = 0.0
            M_WT[2,2] = 1.0

            # mobility tensor symmetry
            M_WT[1,0] = M_WT[0,1]
            M_WT[2,0] = M_WT[0,2]
            M_WT[2,1] = M_WT[1,2]

            # mobility tensor - wall correction
            h = ri[2]

            # self-mobility
            if ti.static(self.wall_lubrication_on == True):
                # lubrication
                gap = h*self.radius_inv - 1.0
                gap_inv = 1.0 / gap

                R1 = 2.0/5.0*ti.log(gap_inv)-0.3817
                R2 = 1.2020569031595942 - 1.9348022005446792*gap

                M1 = 1.0 / R1
                M2 = 1.0 / R2

                M_WT[0,0] *= M1
                M_WT[1,1] *= M1
                M_WT[2,2] *= M2
            else:
                # method of reflection
                h *= self.radius_inv # normalize height with particle radius

                h_inv = 1.0 / h
                h3_inv = h_inv * h_inv * h_inv

                M_WT[0,0] += - h3_inv * 5.0 / 16.0
                M_WT[1,1] += - h3_inv * 5.0 / 16.0
                M_WT[2,2] += - h3_inv / 8.0

                # apply factor
                M_WT[0,0] *= self.WT_self_factor
                M_WT[1,1] *= self.WT_self_factor
                M_WT[2,2] *= self.WT_self_factor

            # W_i = M_{WT}^{ij} * T_j
            for k in ti.static(range(self.dim)):
                for l in ti.static(range(self.dim)):
                    vel_angular[k] += M_WT[k,l]*torque[l]

            # update global arrays
            for k in ti.static(range(self.dim)):
                self.ang_det[i][k] += vel_angular[k] * self.M_WT_factor

