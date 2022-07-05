# Quincke rollers suspension
# Author: Hang (johannyuan@gmail.com)

import taichi as ti
import numpy as np
from scipy.spatial import cKDTree
from rollers_interactions import rollers_interactions
from rollers_hydrodynamics import rollers_hydrodynamics
import configparser
import sys
import matplotlib.pyplot as plt
import time
from datetime import datetime
import ipdb

@ti.data_oriented
class rollers_suspension:
    '''
    Quincke rollers suspension simulation based on Stokesian Dynamics.
    unit system: length - micrometer | mass - milligram | time - second
    '''
    def __init__(self, configfile):
        # setup parameters
        self.config = None # system configuration
        self.parse_parameters(configfile)

        # variables
        self.L = None # simulation box size
        self.F0 = None # external forces
        self.T0 = None # external torques
        self.E0 = None # external fields

        # particle properties
        self.pos = None # position
        self.ort = None # orientation - Quaternion

        self.vel = None # velocity
        self.vel_det = None # deterministic velocity
        self.vel_stoch = None # stochastic velocity
        self.ang = None # angular velocity
        self.ang_det = None # deterministic angular velocity
        self.ang_stoch = None # stochastic angular velocity

        if ti.static(self.integrator == 'AB'):
            self.vel_det_prev = None # deterministic linear velocity at previous step
            self.ang_det_prev = None # deterministic angular velocity at previous step

        self.force = None # force
        self.torque = None # torque

        self.efield = None # electric field
        self.dipole = None # electric dipole moment

        self.sparsity_mask = None # mask

        # wall particle properties
        self.pos_wall = None # position - wall particles
        self.vel_wall = None # velocity - wall particles
        self.vel_det_wall = None # deterministic velocity - wall particles
        self.ang_wall = None # angular velocity - wall particles
        self.ang_det_wall = None # deterministic angular velocity - wall particles
        self.force_wall = None # force - wall particles
        self.torque_wall = None # torque - wall particles
        self.sparsity_mask_wall = None # mask - wall particles

        ######################
        # pending changes
        ######################
        self.wall_spacing = 2.1 * self.radius
        self.wall_shape_sets = {'Circle', 'Torus'}
        #if wall_shape not in self.wall_shape_sets:
        #    raise NotImplementedError
        self.wall_shape = 'Circle'
        self.wall_type_sets = {'No-slip'}
        #if wall_type not in self.wall_type_sets:
        #    raise NotImplementedError
        self.wall_type = 'No-slip'
        ######################

        self.timestep = 0

        # output related
        self.file_trajectory = None
        self.file_mechanics = None
        self.file_kinetics = None
        self.file_log = None

        # GUI related
        self.resolution = None
        self.gui = None
        self.video = None

        # colors and scales
        self.bg_color = ti.rgb_to_hex((0.0,0.0,0.0))
        self.wall_color = 0x112f41
        self.particle_color = 0x068587
        self.particle_scale = self.show_scale
        self.wall_particle_color = 0x068587
        self.wall_particle_scale = self.show_scale
        self.boundary_color = 0xebaca2
        self.arrow_color_vel = 0xff0000
        self.arrow_scale_vel = 0.005 * self.show_scale
        self.arrow_color_dipole = 0xffff00
        self.arrow_scale_dipole = 0.005 * self.show_scale
        self.arrow_color_efield = 0x008000
        self.arrow_scale_efield = 0.005 * self.show_scale
        self.arrow_color_force = 0x00FFFF
        self.arrow_scale_force = 0.005 * self.show_scale

        # specify data layout according to system parameters
        self.layout()

        # particle interactions
        self.interactions = rollers_interactions(self)

        # hydrodynamic interactions
        self.hydrodynamics = rollers_hydrodynamics(self)

        # ensure all bitmasked arrays are deactivated
        ti.root.deactivate_all()

    def parse_parameters(self, configfile):
        '''
        initialize from a config file
        '''
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        #########################
        # Simulation Parameters #
        #########################
        SimulaitonParameters = self.config['Simulation Parameters']
        self.dim = SimulaitonParameters.getint('dim', fallback=3)
        self.N = SimulaitonParameters.getint('N')
        self.steps = SimulaitonParameters.getint('steps', fallback=1)

        self.Lx = SimulaitonParameters.getfloat('Lx')
        self.Ly = SimulaitonParameters.getfloat('Ly')
        self.Lz = SimulaitonParameters.getfloat('Lz')

        self.N_wall = SimulaitonParameters.getint('N_wall', fallback=0)
        self.enforce2D = SimulaitonParameters.getboolean('enforce2D', fallback=True)
        self.gap = SimulaitonParameters.getfloat('gap', fallback=0.0)

        print("----------Simulation Parameters----------")
        print("Number of particles = {}".format(self.N))
        print("Number of steps = {}".format(self.steps))
        print("Simulation box = {Lx} x {Ly} x {Lz}".format(Lx=self.Lx, Ly=self.Ly, Lz=self.Lz))

        print("Number of wall particles = {}".format(self.N_wall))
        print("enforce 2D = {}".format(self.enforce2D))
        print("gap distance = {}".format(self.gap))
        print("#########################################\n")

        #########################
        # Integrator Settings   #
        #########################
        IntegratorSettings = self.config['Integrator Settings']
        self.integrator = IntegratorSettings.get('integrator', fallback='AB')
        self.integrator_sets= {'Euler','AB'}
        if self.integrator not in self.integrator_sets:
            raise NotImplementedError

        self.deltaT = IntegratorSettings.getfloat('deltaT', fallback=0.001)
        self.disp_limit_on = IntegratorSettings.getboolean('disp_limit_on', fallback=False)
        self.disp_limit = IntegratorSettings.getfloat('disp_limit', fallback=1.0)

        self.integrator_dipole = IntegratorSettings.get('integrator_dipole', fallback='RK2')
        self.integrator_dipole_sets = {'RK2'}
        if self.integrator_dipole not in self.integrator_dipole_sets:
            raise NotImplementedError

        self.periods_dipole = IntegratorSettings.getint('periods_dipole', fallback=10)
        self.deltaT_dipole = self.deltaT / self.periods_dipole
        IntegratorSettings['deltaT_dipole'] = str(self.deltaT_dipole)

        print("----------Integrator Settings----------")
        print("integrator = {}".format(self.integrator))
        print("time step = {}".format(self.deltaT))
        print("disp_limit_on = {} disp_limit = {}".format(self.disp_limit_on, self.disp_limit))

        print("integrator_dipole = {}".format(self.integrator_dipole))
        print("time step = {} periods_dipole = {}".format(self.deltaT_dipole, self.periods_dipole))
        print("#########################################\n")

        #######################
        # Material Properties #
        #######################
        MaterialProperties = self.config['Material Properties']
        self.kT = MaterialProperties.getfloat('kT', fallback=0.004141947)
        self.radius = MaterialProperties.getfloat('radius', fallback=1.0)
        self.viscosity = MaterialProperties.getfloat('viscosity', fallback=0.001)
        self.epsilon_p = MaterialProperties.getfloat('epsilon_p', fallback=1.0)
        self.epsilon_l = MaterialProperties.getfloat('epsilon_l', fallback=1.0)
        self.sigma_p = MaterialProperties.getfloat('sigma_p', fallback=0)
        self.sigma_l = MaterialProperties.getfloat('sigma_l', fallback=0)
        
        # derived quantities
        self.epsilon0 = 8.8541878128e-12 # vacuum permittivity
        self.epsilon_pl = (self.epsilon_p - self.epsilon_l) / (self.epsilon_p + 2*self.epsilon_l)
        self.sigma_pl = (self.sigma_p - self.sigma_l) / (self.sigma_p + 2*self.sigma_l)

        self.chi_inf = self.radius**3*self.epsilon_pl # high-frequency polarizability
        self.chi_0 = self.radius**3*self.sigma_pl # low-frequency polarizability
        self.tau_MW = self.epsilon0*(self.epsilon_p + 2*self.epsilon_l)/(self.sigma_p + 2*self.sigma_l) # Maxwell-Wagner relaxation time
        self.tau_MW_inv = 1.0 / self.tau_MW

        # archive derived quantities
        MaterialProperties['chi_inf'] = str(self.chi_inf)
        MaterialProperties['chi_0'] = str(self.chi_0)
        MaterialProperties['tau_MW'] = str(self.tau_MW)

        print("----------Material Properties----------")
        print("kT = {} radius = {} viscosity = {}".format(self.kT, self.radius, self.viscosity))
        print("epsilon_p = {} sigma_p  = {}".format(self.epsilon_p, self.sigma_p))
        print("epsilon_l = {} sigma_l  = {}".format(self.epsilon_l, self.sigma_l))
        print("chi_inf = {} chi_0 = {}".format(self.chi_inf, self.chi_0))
        print("tau_MW = {}".format(self.tau_MW))
        print("#########################################\n")

        ###################
        # External Fields #
        ###################
        ExternalFields = self.config['External Fields']

        self.efield_type = ExternalFields.get('efield_type', fallback='constant')
        self.efield_type_sets = {'constant','pulse'}
        if self.efield_type not in self.efield_type_sets:
            raise NotImplementedError

        self.Ex = ExternalFields.getfloat('Ex', fallback=0.0)
        self.Ey = ExternalFields.getfloat('Ey', fallback=0.0)
        self.Ez = ExternalFields.getfloat('Ez', fallback=0.0)

        self.efield_T1 = ExternalFields.getint('efield_T1', fallback=1)
        self.efield_T2 = ExternalFields.getint('efield_T2', fallback=1)
        self.efield_period = self.efield_T1 + self.efield_T2

        print("----------External Fields----------")
        print("Field Type = " + self.efield_type)
        print("Field strength = {} {} {}".format(self.Ex, self.Ey, self.Ez))
        print("Field on-time = {T1} Field off-time = {T2}, Field Period = {T}".format(T1=self.efield_T1, T2=self.efield_T2, T=self.efield_period))
        print("#########################################\n")

        ##########################
        # Interaction Settings   #
        ##########################
        InteractionSettings = self.config['Interaction Settings']

        # constant forces
        self.onebody_Fconst_on = InteractionSettings.getboolean('onebody_Fconst_on', fallback=False)
        self.Fx = InteractionSettings.getfloat('Fx', fallback=0.0)
        self.Fy = InteractionSettings.getfloat('Fy', fallback=0.0)
        self.Fz = InteractionSettings.getfloat('Fz', fallback=0.0)

        # constant torques
        self.onebody_Tconst_on = InteractionSettings.getboolean('onebody_Tconst_on', fallback=False)
        self.Tx = InteractionSettings.getfloat('Tx', fallback=0.0)
        self.Ty = InteractionSettings.getfloat('Ty', fallback=0.0)
        self.Tz = InteractionSettings.getfloat('Tz', fallback=0.0)

        # external electric torque
        self.onebody_Eext_on = InteractionSettings.getboolean('onebody_Eext_on', fallback=False)

        # dipole-dipole interactions
        self.pair_dipole_on = InteractionSettings.getboolean('pair_dipole_on', fallback=False)
        self.dipole_cutoff = InteractionSettings.getfloat('dipole_cutoff', fallback=5.0*self.radius)
        self.dipole_A = InteractionSettings.getfloat('dipole_A', fallback=1.0)
        self.dipole_kappa = InteractionSettings.getfloat('dipole_kappa', fallback=0.0)

        self.image_dipole_on = InteractionSettings.getboolean('image_dipole_on', fallback=False)
        self.image_dipole_A = InteractionSettings.getfloat('image_dipole_A', fallback=1.0)
        self.image_dipole_kappa = InteractionSettings.getfloat('image_dipole_kappa', fallback=0.0)

        # yukawa-type interaction
        self.pair_yukawa_on = InteractionSettings.getboolean('pair_yukawa_on', fallback=False)
        self.yukawa_cutoff = InteractionSettings.getfloat('yukawa_cutoff', fallback=5.0*self.radius)
        self.yukawa_A = InteractionSettings.getfloat('yukawa_A', fallback=1.0)
        self.yukawa_kappa = InteractionSettings.getfloat('yukawa_kappa', fallback=0.0)

        print("----------Interaction Settings----------")
        print("include constant forces = {}".format(self.onebody_Fconst_on))
        print("external forces = {} {} {}".format(self.Fx, self.Fy, self.Fz))

        print("include constant torques = {}".format(self.onebody_Tconst_on))
        print("external torques = {} {} {}".format(self.Tx, self.Ty, self.Tz))

        print("include external electric torques = {}".format(self.onebody_Eext_on))

        print("#########################################\n")

        ##########################
        # Hydrodynamics Settings #
        ##########################

        HydrodynamicsSettings= self.config['Hydrodynamics Settings']
        self.lubrications_on = HydrodynamicsSettings.getboolean('lubrications_on', fallback=False)
        self.lub_cutoff = HydrodynamicsSettings.getfloat('lub_cutoff', fallback=4.0*self.radius)
        self.hydro_uncorrelated = HydrodynamicsSettings.getboolean('hydro_uncorrelated', fallback=False)
        self.RPY_wall_correction_on = HydrodynamicsSettings.getboolean('RPY_wall_correction_on', fallback=False)
        self.lub_wall_correction_on = HydrodynamicsSettings.getboolean('lub_wall_correction_on', fallback=False)
        self.fluctuations_on = HydrodynamicsSettings.getboolean('fluctuations_on', fallback=False)
        self.thermal_noise_type = HydrodynamicsSettings.get('thermal_noise_type', fallback='uncorrelated')
        self.noise_types_set = {'uncorrelated'}
        if self.thermal_noise_type not in self.noise_types_set:
            raise NotImplementedError

        self.thermal_noise_level = HydrodynamicsSettings.getfloat('thermal_noise_level', fallback=0.0)

        print("----------Hydrodynamics Settings----------")
        print("include lubrication corrections = {}".format(self.lubrications_on))
        print("lubrication cutoff = {}".format(self.lub_cutoff))
        print("using uncorrelated hydrodynamics = {}".format(self.hydro_uncorrelated))
        print("include far-field wall corrections = {}".format(self.RPY_wall_correction_on))
        print("include near-field wall corrections = {}".format(self.lub_wall_correction_on))
        print("include thermal fluctuations = {}".format(self.fluctuations_on))
        print("thermal noise type = {}".format(self.thermal_noise_type))
        print("thermal noise level = {}".format(self.thermal_noise_level))

        print("#########################################\n")

        ##########################
        # GMRES Settings         #
        ##########################

        GMRESSettings = self.config['GMRES Settings']
        self.gmres_tol = GMRESSettings.getfloat('gmres_tol', fallback=1e-5)
        self.gmres_maxiter = GMRESSettings.getint('gmres_maxiter', fallback=31)
        self.gmres_residual_on = GMRESSettings.getboolean('gmres_residual_on', fallback=False)
        self.gmres_pc_on = GMRESSettings.getboolean('gmres_pc_on', fallback=False)
        self.gmres_pc_side = GMRESSettings.get('gmres_pc_side', fallback='right')

        print("----------GMRES Settings----------")
        print("tolerance = {}".format(self.gmres_tol))
        print("maxiter = {}".format(self.gmres_maxiter))
        print("show residuals = {}".format(self.gmres_residual_on))
        print("using preconditioner = {}".format(self.gmres_pc_on))
        print("preconditioner side = {}".format(self.gmres_pc_side))

        print("#########################################\n")

        ##########################
        # Boundary Settings      #
        ##########################
        BoundarySettings = self.config['Boundary Settings']
        self.boundary_circle_on = BoundarySettings.getboolean('boundary_circle_on')

        ###################
        # Input Settings  #
        ###################
        InputSettings = self.config['Input Settings']
        self.input_dir = InputSettings.get('input_dir', fallback='./data/')
        self.input_filename = InputSettings.get('input_filename', fallback='input.npy')
        self.input_lable = InputSettings.getint('input_label', fallback=0)

        print("----------Input Settings----------")
        print("input_folder = {}".format(self.input_dir))
        print("input_filename = {}".format(self.input_filename))
        print("input_label = {}".format(self.input_lable))
        print("#########################################\n")

        ###################
        # Output Settings #
        ###################
        OutputSettings = self.config['Output Settings']
        self.output_freq = OutputSettings.getint('output_freq', fallback=1)
        self.output_dir = OutputSettings.get('output_dir', fallback='./data/')
        self.output_filename = OutputSettings.get('output_filename', fallback='rollers')

        self.output_trajectory_on = OutputSettings.getboolean('output_trajectory_on', fallback=False)
        self.output_mechanics_on = OutputSettings.getboolean('output_mechanics_on', fallback=False)
        self.output_kinetics_on = OutputSettings.getboolean('output_kinetics_on', fallback=False)
        self.output_log_on = OutputSettings.getboolean('output_log_on', fallback=False)

        print("----------Output Settings----------")
        print("output_folder = {}".format(self.output_dir))
        print("output_filename = {}".format(self.output_filename))
        print("output_freq = {}".format(self.output_freq))

        print("output_trajectory_on = {}".format(self.output_trajectory_on))
        print("output_mechanics_on = {}".format(self.output_mechanics_on))
        print("output_kinetics_on = {}".format(self.output_kinetics_on))
        print("output_log_on = {}".format(self.output_log_on))
        print("#########################################\n")

        ###################
        # GUI Settings    #
        ###################
        GUISettings = self.config['GUI Settings']
        self.show_gui = GUISettings.getboolean('show_gui', fallback=True)
        self.gui_window_on = GUISettings.getboolean('gui_window_on', fallback=True)
        self.record_video = GUISettings.getboolean('record_video', fallback=False)
        self.show_scale = GUISettings.getfloat('show_scale', fallback=1.0)

        print("----------GUI Settings----------")
        print("show_gui = {}".format(self.show_gui))
        print("record_video = {}".format(self.record_video))
        print("show_scale = {}".format(self.show_scale))
        print("#########################################\n")

    def layout(self):
        '''
        specify data layout
        '''
        # variables
        self.L = ti.field(ti.f64, shape=self.dim)
        self.F0 = ti.field(ti.f64, shape=self.dim)
        self.T0 = ti.field(ti.f64, shape=self.dim)
        self.E0 = ti.field(ti.f64, shape=self.dim)

        # particle properties
        self.pos = ti.Vector.field(self.dim, dtype=ti.f64)
        self.ort = ti.Vector.field(4, dtype=ti.f64)

        # kinetic quantities
        self.vel = ti.Vector.field(self.dim, dtype=ti.f64)
        self.vel_det = ti.Vector.field(self.dim, dtype=ti.f64)
        self.vel_stoch = ti.Vector.field(self.dim, dtype=ti.f64)
        self.ang = ti.Vector.field(self.dim, dtype=ti.f64)
        self.ang_det = ti.Vector.field(self.dim, dtype=ti.f64)
        self.ang_stoch = ti.Vector.field(self.dim, dtype=ti.f64)
        if ti.static(self.integrator == 'AB'):
            self.vel_det_prev = ti.Vector.field(self.dim, dtype=ti.f64)
            self.ang_det_prev = ti.Vector.field(self.dim, dtype=ti.f64)

        # mechanical quantities
        self.force = ti.Vector.field(self.dim, dtype=ti.f64)
        self.torque = ti.Vector.field(self.dim, dtype=ti.f64)

        self.efield = ti.Vector.field(self.dim, dtype=ti.f64)
        self.dipole = ti.Vector.field(self.dim, dtype=ti.f64)

        self.sparsity_mask = ti.field(dtype=ti.i32)

        if ti.static(self.N_wall > 0):
            self.pos_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.vel_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.vel_det_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.ang_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.ang_det_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.force_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.torque_wall = ti.Vector.field(self.dim, dtype=ti.f64)
            self.sparsity_mask_wall = ti.field(dtype=ti.i32)

        # place field quantities
        ti.root.dense(ti.i, self.N).place(self.pos, self.ort)
        ti.root.dense(ti.i, self.N).place(self.vel, self.vel_det, self.vel_stoch)
        ti.root.dense(ti.i, self.N).place(self.ang, self.ang_det, self.ang_stoch)
        if ti.static(self.integrator == 'AB'):
            ti.root.dense(ti.i, self.N).place(self.vel_det_prev, self.ang_det_prev)
        ti.root.dense(ti.i, self.N).place(self.force, self.torque)
        ti.root.dense(ti.i, self.N).place(self.efield, self.dipole)
        ti.root.bitmasked(ti.ij, (self.N, self.N)).place(self.sparsity_mask)

        if ti.static(self.N_wall > 0):
            ti.root.dense(ti.i, self.N_wall).place(self.pos_wall)
            ti.root.dense(ti.i, self.N_wall).place(self.vel_wall, self.vel_det_wall)
            ti.root.dense(ti.i, self.N_wall).place(self.ang_wall, self.ang_det_wall)
            ti.root.dense(ti.i, self.N_wall).place(self.force_wall, self.torque_wall)
            ti.root.bitmasked(ti.ij, (self.N_wall, self.N_wall)).place(self.sparsity_mask_wall)

    def init(self):
        '''
        initialization
        '''
        self.init_variables()

        self.init_particles()

        #self.place_single_roller()
        #self.place_two_rollers()
        #self.place_three_rollers()
        self.load_initial_config()

        #self.load_wall_particles()

        #self.activate_wall_particles()

        # initialize interactions
        self.interactions.init()

        self.hydrodynamics.init()

        self.init_output_file()

        self.save_config()

        print('system initialized.')

    @ti.kernel
    def init_variables(self):
        '''
        initialize global variables
        '''
        self.L[0] = self.Lx
        self.L[1] = self.Ly
        self.L[2] = self.Lz

        self.F0[0] = self.Fx
        self.F0[1] = self.Fy
        self.F0[2] = self.Fz

        self.T0[0] = self.Tx
        self.T0[1] = self.Ty
        self.T0[2] = self.Tz

        self.E0[0] = self.Ex
        self.E0[1] = self.Ey
        self.E0[2] = self.Ez

    @ti.kernel
    def init_particles(self):
        '''
        initialize particles' properties 
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.pos[i][k] = 0.0

                self.vel[i][k] = 0.0
                self.vel_det[i][k] = 0.0
                self.vel_stoch[i][k] = 0.0

                self.ang[i][k] = 0.0
                self.ang_det[i][k] = 0.0
                self.ang_stoch[i][k] = 0.0

                self.force[i][k] = 0.0
                self.torque[i][k] = 0.0

                self.efield[i][k] = 0.0
                self.dipole[i][k] = 0.0

        # Quaternion - unused
        for i in ti.ndrange(self.N):
            for k in ti.static(range(4)):
                self.ort[i][k] = 1.0

    def load_wall_particles(self):
        '''
        place wall particles
        '''
        if self.N_wall <= 0:
            return

        if self.wall_shape == 'Circle':
            wall_particles_pos = self.place_wall_particles_circle()
        elif self.wall_shape == 'Torus':
            raise NotImplementedError
        # update wall particles position
        self.pos_wall.from_numpy(wall_particles_pos)

    def place_wall_particles_circle(self):
        '''
        place wall particles as a circular wall
        '''
        wall_particles_pos = np.zeros( (self.N_wall, self.dim) )

        dtheta = 2 * np.pi / self.N_wall
        wall_radius = min(self.Lx, self.Ly, self.Lz) / 2.0

        for i in range(self.N_wall):
            theta = i * dtheta
            wall_particles_pos[i][0] = wall_radius * np.cos(theta)
            wall_particles_pos[i][1] = wall_radius * np.sin(theta)
            wall_particles_pos[i][2] = self.radius + self.gap

        return wall_particles_pos

    @ti.kernel
    def activate_wall_particles(self):
        '''
        specify sparsity mask for wall particles
        '''
        if ti.static(self.N_wall <= 0):
            return

        hydro_cutoff_wall = 1.5 * self.wall_spacing
        #hydro_cutoff_wall = max(self.Lx, self.Ly, self.Lz)
        for i, j in ti.ndrange(self.N_wall, self.N_wall):
            # only activate particle pairs within the cutoff
            ri = ti.Vector([0.]*self.dim)
            rj = ti.Vector([0.]*self.dim)
            rij = ti.Vector([0.]*self.dim)

            for k in ti.static(range(self.dim)):
                ri[k] = self.pos_wall[i][k]
                rj[k] = self.pos_wall[j][k]
                rij[k] = ri[k] - rj[k]

            r = rij.norm()
            if (r <= hydro_cutoff_wall):
                self.sparsity_mask_wall[i,j] = 1

    def load_initial_config(self):
        '''
        load user-defined initial particles configuration
        '''
        # lattice
        particles_pos = self.place_particles_lattice()
        # torus
        #particles_pos = self.place_particles_torus()
        # torus - linear gradient
        #particles_pos,particles_dipole = self.place_particles_torus_gradient()
        # from file
        #particles_pos = self.place_particles_fromfile()
        # update particles position
        self.pos.from_numpy(particles_pos)
        #self.dipole.from_numpy(particles_dipole)

    def place_particles_fromfile(self):
        '''
        load particles position from file
        '''
        input_file = self.input_dir + self.input_filename
        label = self.input_lable
        init_pos = np.load(input_file)
        return init_pos[label,:,:]

    def place_single_roller(self):
        '''
        only for single roller test
        '''
        self.pos[0][2] = self.radius + self.gap
        self.ang[0][0] = 2000

    def place_two_rollers(self):
        '''
        only for two rollers test
        '''
        self.pos[0][2] = self.radius + self.gap
        self.pos[1][2] = self.radius + self.gap

        self.pos[0][0] =  self.radius - 0.01 * self.radius
        self.pos[1][0] = -self.radius

        #self.dipole[0][2] = self.chi_inf * self.E0[2]
        #self.dipole[1][2] = self.chi_inf * self.E0[2]

        #self.ang[0][1] = -2000
        #self.ang[1][1] =  2000

    def place_three_rollers(self):
        '''
        only for three rollers test
        '''
        self.pos[0][2] = self.radius + self.gap
        self.pos[1][2] = self.radius + self.gap
        self.pos[2][2] = self.radius + self.gap

        self.pos[0][0] =  0.0
        self.pos[1][0] =  2.0 * self.radius + 0.01 * self.radius
        self.pos[2][0] = -2.0 * self.radius - 0.01 * self.radius

        #self.dipole[0][2] = self.chi_inf * self.E0[2]
        #self.dipole[1][2] = self.chi_inf * self.E0[2]

        #self.ang[0][1] = -2000
        #self.ang[1][1] =  2000

    def place_particles_lattice(self):
        '''
        generate particles on lattices
        '''
        particles_pos = np.zeros( (self.N, self.dim) )

        Nx = 32
        dx = 2.0 * self.radius + 2.1 * self.radius
        dy = 2.0 * self.radius + 2.1 * self.radius
        offset_x = -Nx*dx/2.0
        offset_y = -(self.N//Nx)*dy/2.0
        for i in range(self.N):
            row = i // Nx
            col = i % Nx
            particles_pos[i][0] = offset_x + row * dx
            particles_pos[i][1] = offset_y + col * dy
            particles_pos[i][2] = self.radius + self.gap

        return particles_pos

    def place_particles_torus(self):
        '''
        generate particles on torus
        '''
        particles_pos = np.zeros( (self.N, self.dim) )

        Nt = 128
        R0 = self.Lx / 2.0 - 2.0 * self.radius
        dr = 2.0 * self.radius + 1.1 * self.radius
        dtheta = 2.0 * np.pi / Nt

        for i in range(self.N):
            nr = i // Nt
            ntheta = i % Nt
            r = R0 - nr*dr
            theta = dtheta*ntheta 
            particles_pos[i][0] = r * np.cos(theta)
            particles_pos[i][1] = r * np.sin(theta)
            particles_pos[i][2] = self.radius + self.gap

        return particles_pos

    def place_particles_torus_gradient(self):
        '''
        generate particles on torus
        '''
        particles_pos = np.zeros((self.N, self.dim))
        particles_dipole = np.zeros((self.N, self.dim))
        dipole_init = 10.0
        # min seperation along theta direction
        dh_min = 2.0 * self.radius + 2.1 * self.radius
        # seperation along radial direction
        dr = 2.0 * self.radius + 3.1 * self.radius

        # max radius
        R = self.Lx / 2.0 - 4.0 * self.radius

        dh_factor = dh_min * R

        pi = ti.asin(1.0)*2.0

        level = 0
        # radius of current level
        r = R - level * dr
        # seperation of current level
        dh = dh_factor / r
        # number of particles at current level
        pnum = (2.0 * pi * r) // dh
        # theta seperation of current level
        dtheta = (2.0 * pi) / pnum

        offset = 0
        cnt = 0

        for i in range(self.N):
            if cnt >= pnum:
                # next level
                level += 1
                offset += pnum
                cnt = 0
                # radius of current level
                r = R - level * dr
                # seperation of current level
                dh = dh_factor / r
                # number of particles at current level
                N = (2.0 * pi * r) // dh
                # theta seperation of current level
                dtheta = 2.0 * pi / pnum

            # theta position of current particle
            ntheta = (i - offset) % pnum
            theta = dtheta*ntheta

            particles_pos[i][0] = r * np.cos(theta)
            particles_pos[i][1] = r * np.sin(theta)
            particles_pos[i][2] = self.radius + self.gap

            particles_dipole[i][0] = - np.sin(theta)
            particles_dipole[i][1] =   np.cos(theta)
            particles_dipole[i][2] = 0.0

            cnt += 1

        return particles_pos, particles_dipole

    @ti.func
    def dPdt_func(self, dipole, ang_vel, efield):
        '''
        i-th particle dipole change rate:
        dP/dt = omega x (P - chi_inf*E) - 1/tau_MW * (P - chi_0*E)
        '''
        P_epsilon = dipole - self.chi_inf*efield
        P_sigma = dipole - self.chi_0*efield
        return ang_vel.cross(P_epsilon) - self.tau_MW_inv*P_sigma

    @ti.func
    def Runge_Kutta_2nd(self, i):
        '''
        a second order Runge-Kutta method
        '''
        # initial conditions
        dipole_init = ti.Vector([0.]*self.dim)
        ang_vel_init = ti.Vector([0.]*self.dim)
        efield_init = ti.Vector([0.]*self.dim)

        # dipole moments at current time step t
        dipole = ti.Vector([0.]*self.dim)

        for k in ti.static(range(self.dim)):
            dipole_init[k] = self.dipole[i][k]
            ang_vel_init[k] = self.ang[i][k]
            efield_init[k] = self.efield[i][k]
            dipole[k] = self.dipole[i][k]

        # propagate forward t -> t+dt
        for step in ti.static(range(self.periods_dipole)):
            # assumed angular velocity and electric field are unchanged during
            # dipole moment evolution

            k1 = self.dPdt_func(dipole, ang_vel_init, efield_init)

            dipole_k2 = dipole + k1*self.deltaT_dipole

            k2 = self.dPdt_func(dipole_k2, ang_vel_init, efield_init)

            dipole += 0.5*self.deltaT_dipole*(k1+k2)

        # update dipole moments t+dt
        for k in ti.static(range(self.dim)):
            self.dipole[i][k] = dipole[k]

    @ti.kernel
    def update_dipole_moments(self):
        '''
        evolve dipole moments according to the surface charge conservation
        equation.
        '''
        # update dipole moments of each particles
        for i in ti.ndrange(self.N):
            if ti.static(self.integrator_dipole == 'RK2'):
                self.Runge_Kutta_2nd(i)

    @ti.func
    def Euler_forward(self, i):
        '''
        Euler forward method:
        r(t+\delta) = r(t) + \delta t*v(t)
        '''
        for k in ti.static(range(self.dim)):
            self.vel[i][k] = self.vel_det[i][k] + self.vel_stoch[i][k]
            self.ang[i][k] = self.ang_det[i][k] + self.ang_stoch[i][k]
        
        if ti.static(self.integrator == 'AB'):
            for k in ti.static((range(self.dim))):
                self.vel_det_prev[i][k] = self.vel_det[i][k]
                self.ang_det_prev[i][k] = self.ang_det[i][k]

    @ti.func
    def Adams_Bashforth_2nd(self, i):
        '''
        a two steps Adams-Bashforth method
        r(t+\delta t) = r(t) + 1.5*\delta t*v(t) - 0.5*v(t-\delta t)
        '''
        for k in ti.static(range(self.dim)):
            self.vel[i][k] = 1.5 * self.vel_det[i][k] - 0.5 * self.vel_det_prev[i][k] + self.vel_stoch[i][k]
            self.ang[i][k] = 1.5 * self.ang_det[i][k] - 0.5 * self.ang_det_prev[i][k] + self.ang_stoch[i][k]

        for k in ti.static(range(self.dim)):
            self.vel_det_prev[i][k] = self.vel_det[i][k]
            self.ang_det_prev[i][k] = self.ang_det[i][k]

    @ti.kernel
    def update_configurations(self):
        '''
        update particles' configuration: position and orientation
        '''
        for i in ti.ndrange(self.N):
            if ti.static(self.integrator == 'Euler'):
                # Euler forward method
                self.Euler_forward(i)
            elif ti.static(self.integrator == 'AB'):
                # Adams-Bashforth method
                if self.timestep == 0:
                    self.Euler_forward(i)
                else:
                    self.Adams_Bashforth_2nd(i)

            # update positions
            if ti.static(self.enforce2D == True):
                # 2D simulation
                dx = self.vel[i][0] * self.deltaT
                dy = self.vel[i][1] * self.deltaT
                if ti.static(self.disp_limit_on == True):
                    dr = ti.sqrt(dx**2 + dy**2)
                    if (dr < self.disp_limit):
                        self.pos[i][0] += dx
                        self.pos[i][1] += dy
                    else:
                        # avoid large steps caused by overlapping particles
                        print('warning: max disp limit triggered ',dr)
                        scale_factor = self.disp_limit / dr
                        dx *= scale_factor
                        dy *= scale_factor
                        self.pos[i][0] += dx
                        self.pos[i][1] += dy

                        # rescale velocities
                        self.vel[i][0] *= scale_factor
                        self.vel[i][1] *= scale_factor
                else:
                    self.pos[i][0] += dx
                    self.pos[i][1] += dy

            else:
                # 3D simulation
                dx = self.vel[i][0] * self.deltaT
                dy = self.vel[i][1] * self.deltaT
                dz = self.vel[i][2] * self.deltaT
                if ti.static(self.disp_limit_on == True):
                    dr = ti.sqrt(dx**2 + dy**2 + dz**2)
                    if (dr < self.disp_limit):
                        self.pos[i][0] += dx
                        self.pos[i][1] += dy
                        self.pos[i][2] += dz
                    else:
                        # avoid large steps
                        print('warning: max disp limit triggered ',dr)
                        scale_factor = self.disp_limit / dr
                        dx *= scale_factor
                        dy *= scale_factor
                        dz *= scale_factor
                        self.pos[i][0] += dx
                        self.pos[i][1] += dy
                        self.pos[i][2] += dz

                        # rescale velocities
                        self.vel[i][0] *= scale_factor
                        self.vel[i][1] *= scale_factor
                        self.vel[i][2] *= scale_factor
                else:
                    self.pos[i][0] += dx
                    self.pos[i][1] += dy
                    self.pos[i][2] += dz

            # update orientations
            # NOT implemented
            # update self.ort with self.ang

    @ti.kernel
    def update_external_fields(self, timestep: ti.i32):
        '''
        apply external fields
        '''
        if ti.static(self.efield_type == 'constant'):
            # constant
            self.E0[0] = self.Ex
            self.E0[1] = self.Ey
            self.E0[2] = self.Ez
        elif ti.static(self.efield_type == 'pulse'):
            if timestep % self.efield_period <= self.efield_T1:
                # 0 < t <= T1 - constant
                self.E0[0] = self.Ex
                self.E0[1] = self.Ey
                self.E0[2] = self.Ez
            else:
                # T1 < t <= T2 - off
                self.E0[0] = 0.0
                self.E0[1] = 0.0
                self.E0[2] = 0.0

    def advance(self):
        '''
        advance the simulation one time step forward
        '''
        # apply external fields
        self.update_external_fields(self.timestep)
        #print(self.timestep, self.E0[2], self.interactions.E0[2])

        # update dipole moments
        self.update_dipole_moments()

        # particle interactions computation
        # update efield, force and torque
        self.interactions.compute()

        # hydrodynamic computation
        # update translational and angular velocity
        self.hydrodynamics.compute()

        # update particle positions/orientations
        self.update_configurations()

        # timestep
        self.timestep += 1

    def solve(self):
        '''
        solve system for specificed number of timesteps
        '''

        if ti.static(self.show_gui == True):
            width = int(self.show_scale * (self.Lx + 2.0 * self.radius))
            if width % 2 == 1:
                width += 1
            height = int(self.show_scale * (self.Ly + 2.0 * self.radius))
            if height % 2 == 1:
                height += 1
            self.resolution = (width, height)
            self.gui = ti.GUI(name='Quincke Rollers Suspension', res=self.resolution, show_gui=self.gui_window_on)

        if ti.static(self.record_video == True):
            self.video = ti.VideoManager(self.output_dir, framerate=25, automatic_build=False)

        timer_start = time.time()
        timer_prev = timer_start

        while self.timestep < self.steps:

            self.advance()

            if self.timestep % self.output_freq == 0:
                # output data
                if ti.static(self.output_trajectory_on == True):
                    self.output_trajectory()
                if ti.static(self.output_mechanics_on == True):
                    self.output_mechanics()
                if ti.static(self.output_kinetics_on == True):
                    self.output_kinetics()
                if ti.static(self.output_log_on == True):
                    self.output_logs()

                # print
                timer_now = time.time()
                dt = timer_now - timer_prev
                print('timestep: {timestep} walltime: {walltime} tps: {tps}'.format(timestep=self.timestep, walltime=timer_now-timer_start, tps=self.output_freq/dt))
                timer_prev = timer_now

                # gui rendering
                if ti.static(self.show_gui == True):
                    self.render()

        # close output files
        if ti.static(self.output_trajectory_on == True):
            self.file_trajectory.close()
        if ti.static(self.output_mechanics_on == True):
            self.file_mechanics.close()
        if ti.static(self.output_kinetics_on == True):
            self.file_kinetics.close()
        if ti.static(self.output_log_on == True):
            self.file_log.close()

        # output GMRES info
        print('total number of GMRES iterations = {}'.format(self.hydrodynamics.gmres_iter_tot))
        print('average number of iterations per step = {}'.format(self.hydrodynamics.gmres_iter_tot / self.steps))

        # generate video
        if ti.static(self.record_video == True):
            self.video.make_video(gif=False, mp4=True)

    def render(self):
        '''
        render current configuration of particles
        '''
        self.gui.clear(self.bg_color)
        # positions
        pos = self.pos.to_numpy()
        pos_xy = pos[:,0:2]
        pos_xy *= self.show_scale
        pos_xy[:,0] += self.resolution[0] / 2.0
        pos_xy[:,1] += self.resolution[1] / 2.0
        # relative coordinates
        pos_xy[:,0] /= self.resolution[0]
        pos_xy[:,1] /= self.resolution[1]

        # wall positions
        if ti.static(self.N_wall > 0):
            pos_wall = self.pos_wall.to_numpy()
            pos_xy_wall = pos_wall[:,0:2]
            pos_xy_wall *= self.show_scale
            pos_xy_wall[:,0] += self.resolution[0] / 2.0
            pos_xy_wall[:,1] += self.resolution[1] / 2.0
            # relative coordinates
            pos_xy_wall[:,0] /= self.resolution[0]
            pos_xy_wall[:,1] /= self.resolution[1]

        # dipoles
        dipole = self.dipole.to_numpy()
        dipole_xy = dipole[:,0:2]
        dipole_norm = np.linalg.norm(dipole_xy, ord=2, axis=1).reshape((self.N,1))
        dipole_xy /= dipole_norm
        #dipole_xy *= self.arrow_scale_dipole
        # treat small values as zero
        mask = dipole_norm < 1e-9
        scales = np.ones_like(dipole_norm) * self.arrow_scale_dipole
        scales[mask] = 0.0
        dipole_xy *= scales

        # velocity
        vel = self.vel.to_numpy()
        vel_xy = vel[:,0:2]
        vel_norm = np.linalg.norm(vel_xy, ord=2, axis=1)
        vel_xy /= vel_norm.reshape((self.N,1))
        vel_xy *= self.arrow_scale_vel

        # electric fields
        #efield = self.efield.to_numpy()
        #efield_xy = efield[:,0:2]
        #efield_norm = np.linalg.norm(efield_xy, ord=2, axis=1)
        #efield_xy /= efield_norm.reshape((self.N,1))
        #efield_xy *= self.arrow_scale_efield

        # forces
        #force = self.force.to_numpy()
        #force_xy = force[:,0:2]
        #force_norm = np.linalg.norm(force_xy, ord=2, axis=1).reshape((self.N,1))
        #force_xy /= force_norm
        #force_xy *= self.arrow_scale_force
        #force_ref = 5.0
        #force_max = 2*force_ref
        #force_norm[force_norm > force_max] = force_max
        #force_xy *= force_norm / force_ref

        # circular wall
        self.gui.circle([0.5,0.5], color=self.wall_color,
                radius=self.Lx/2.0*self.show_scale)
        # particles
        self.gui.circles(pos_xy, color=self.particle_color,
                radius=self.radius*self.particle_scale)
        # wall particles
        if ti.static(self.N_wall > 0):
            self.gui.circles(pos_xy_wall, color=self.wall_particle_color,
                    radius=self.radius*self.wall_particle_scale)
        # dipoles
        #self.gui.arrows(pos_xy, dir=dipole_xy, radius=2.0,
        #        color=self.arrow_color_dipole,
        #        tip_scale=self.arrow_scale_dipole)
        # velocity
        #self.gui.arrows(pos_xy, dir=vel_xy, radius=2.0,
        #        color=self.arrow_color_vel,
        #        tip_scale=self.arrow_scale_vel)
        # electric fields
        #self.gui.arrows(pos_xy, dir=efield_xy, radius=2.0,
        #        color=self.arrow_color_efield,
        #        tip_scale=self.arrow_scale_efield)
        # forces
        #self.gui.arrows(pos_xy, dir=force_xy, radius=2.0,
        #        color=self.arrow_color_force,
        #        tip_scale=self.arrow_scale_force)
        # simulation box
        self.gui.rect((0,0), (1,1), color=self.boundary_color, radius=2.0)

        if ti.static(self.record_video == True):
            img = self.gui.get_image()
            self.video.write_frame(img)

        self.gui.show()

    def init_output_file(self):
        '''
        intialize output files
        '''
        output_file = self.output_dir+self.output_filename

        # trajectory
        if ti.static(self.output_trajectory_on == True):
            filename = output_file+'.trajectory'
            self.file_trajectory = open(filename, 'w')
            self.file_trajectory.write('# id x y z px py pz Ex Ey Ez\n')
            self.output_trajectory()

        # mechanics
        if ti.static(self.output_mechanics_on == True):
            filename = output_file+'.mechanics'
            self.file_mechanics= open(filename, 'w')
            self.file_mechanics.write('# id fx fy fz tx ty tz\n')
            self.output_mechanics()

        # kinetics
        if ti.static(self.output_kinetics_on == True):
            filename = output_file+'.kinetics'
            self.file_kinetics = open(filename, 'w')
            self.file_kinetics.write('# id vx vy vz wx wy wz\n')
            self.output_kinetics()

        # log quantities
        if ti.static(self.output_log_on == True):
            filename = output_file+'.log'
            self.file_log= open(filename, 'w')
            self.file_log.write('# timestep realtime Lx Ly Lz An At V Vn Vt Vtb\n')
            self.output_logs()

    def output_trajectory(self):
        '''
        output trajectory related quantities
        '''
        pos = self.pos.to_numpy()
        dipole = self.dipole.to_numpy()
        efield = self.efield.to_numpy()
        self.file_trajectory.write('{timestep}\n'.format(timestep=self.timestep))
        for i in range(self.N):
            self.file_trajectory.write('{uid} {x} {y} {z} {px} {py} {pz} {Ex} {Ey} {Ez}\n'.format(uid=i, x=pos[i,0], y=pos[i,1], z=pos[i,2], px=dipole[i,0], py=dipole[i,1], pz=dipole[i,2], Ex=efield[i,0], Ey=efield[i,1], Ez=efield[i,2]))

    def output_mechanics(self):
        '''
        output mechanics related quantities
        '''
        force = self.force.to_numpy()
        torque = self.torque.to_numpy()
        self.file_mechanics.write('{timestep}\n'.format(timestep=self.timestep))
        for i in range(self.N):
            self.file_mechanics.write('{uid} {fx} {fy} {fz} {tx} {ty} {tz}\n'.format(uid=i, fx=force[i,0], fy=force[i,1], fz=force[i,2], tx=torque[i,0], ty=torque[i,1], tz=torque[i,2]))

    def output_kinetics(self):
        '''
        output kinetics related quantities
        '''
        vel = self.vel.to_numpy()
        ang = self.ang.to_numpy()
        self.file_kinetics.write('{timestep}\n'.format(timestep=self.timestep))
        for i in range(self.N):
            self.file_kinetics.write('{uid} {vx} {vy} {vz} {wx} {wy} {wz}\n'.format(uid=i, vx=vel[i,0], vy=vel[i,1], vz=vel[i,2], wx=ang[i,0], wy=ang[i,1], wz=ang[i,2]))

    def output_logs(self):
        '''
        output logged quantities
        '''
        # time
        realtime = self.timestep * self.deltaT

        # total angular momentum
        pos = self.pos.to_numpy()
        vel = self.vel.to_numpy()
        if ti.static(self.enforce2D == True):
            pos[:,2] = 0.0
            vel[:,2] = 0.0
        angular_momentum = np.mean(np.cross(pos, vel), axis=0)
        #scale = 100000
        #angular_momentum /= scale

        # asymmetry parameters
        neighbors_num = 2
        dist_cutoff = self.interactions.dipole_cutoff
        eps_val = 1e-15

        pos_xy = pos[:,0:2]
        tree = cKDTree(pos_xy) # build KDtree for neighbors lookup
        dist, idx = tree.query(pos_xy, k=neighbors_num)

        # virtual isotropic force functional form
        def virtual_force(r_norm, uvr):
            return 1.0 / r_norm * uvr

        # add forces from each neighbors
        force_neighbors = np.zeros_like(pos_xy)
        for k in range(1,neighbors_num):
            ri = pos_xy[idx[:,0]]
            rj = pos_xy[idx[:,k]]
            rij = ri - rj
            r_norm = np.linalg.norm(rij, ord=2, axis=1).reshape((self.N,1))
            uvr = rij / r_norm
            mask = (r_norm <= dist_cutoff).flatten()
            force_virtual = virtual_force(r_norm, uvr)
            force_neighbors[mask,:] += force_virtual[mask,:]

        # normalize total forces
        force_neighbors_norm = np.linalg.norm(force_neighbors, ord=2, axis=1).reshape((self.N,1))
        mask = (force_neighbors_norm < eps_val).flatten()
        force_neighbors_norm[mask] = eps_val
        force_neighbors_unit = force_neighbors / force_neighbors_norm
        force_neighbors_unit[mask,:] = 0.0 # remove zero vectors

        # decompose into normal and tangential components
        pos_norm = np.linalg.norm(pos_xy, ord=2, axis=1).reshape((self.N,1))
        normal_vec = pos_xy / pos_norm
        tangent_vec = np.zeros_like(normal_vec)
        tangent_vec[:,0] = -normal_vec[:,1]
        tangent_vec[:,1] =  normal_vec[:,0]

        force_normal = np.sum(force_neighbors_unit * normal_vec, axis=1)
        An = np.mean(force_normal)
        force_tangent = np.sum(force_neighbors_unit * tangent_vec, axis=1)
        At = np.mean(force_tangent)

        # average velocities
        vel_xy = vel[:,0:2]
        vel_amp = np.linalg.norm(vel_xy, ord=2, axis=1).reshape((self.N,1))
        V = np.mean(vel_amp)
        # decompose into normal and tangential components
        vel_normal = np.sum(vel_xy * normal_vec, axis=1)
        Vn = np.mean(vel_normal)
        vel_tangent = np.sum(vel_xy * tangent_vec, axis=1)
        Vt = np.mean(vel_tangent)

        # boundary layer velocity
        bc_radius = self.interactions.bc_circ_radius
        r_max = bc_radius - 2*self.radius
        r_min = bc_radius - 8*self.radius
        mask = (pos_norm < r_max) & (pos_norm > r_min)
        Vtb = np.mean(vel_tangent[mask.flatten()])

        self.file_log.write('{timestep} {realtime} {Lx} {Ly} {Lz} {An} {At} {V} {Vn} {Vt} {Vtb}\n'.format(timestep=self.timestep, realtime=realtime, Lx=angular_momentum[0], Ly=angular_momentum[1], Lz=angular_momentum[2], An=An, At=At, V=V, Vn=Vn, Vt=Vt, Vtb=Vtb))

    def save_config(self):
        '''
        save configuration file
        '''
        with open(self.output_dir+'rollers.config', 'w') as configfile:
            self.config.write(configfile)

if __name__ == '__main__':
    # run Quincke rollers suspension
    if len(sys.argv) < 2:
        print('Please specify system config file.')
        exit()

    configfile = sys.argv[1]

    #ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, print_preprocessed=True)
    #ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, kernel_profiler=True)
    ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)
    #ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32)
    #ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, random_seed=0)
    #ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, random_seed=int(time.time()*1000))

    suspension = rollers_suspension(configfile)

    suspension.init()
    #ipdb.set_trace()
    suspension.solve()

