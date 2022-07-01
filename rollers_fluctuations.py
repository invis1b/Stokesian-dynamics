import taichi as ti
import numpy as np

@ti.data_oriented
class rollers_fluctuations:
    '''
    Generate thermal fluctuations of Quincke rollers
    '''
    def __init__(self, system):
        # rollers suspension
        self.system = system

        self.N = system.N
        self.dim = system.dim

        self.radius = system.radius
        self.viscosity = system.viscosity

        self.kT = system.kT
        self.deltaT = system.deltaT

        # particle properties
        self.vel_stoch = system.vel_stoch
        self.ang_stoch = system.ang_stoch

        self.noise_type = system.thermal_noise_type

        self.noise_level = np.sqrt(system.thermal_noise_level)
        self.brownian_factor = np.sqrt(2.0 * self.kT / self.deltaT)
        self.brownian_M_UF = self.noise_level * self.brownian_factor * np.sqrt(1.0 / (6.0 * np.pi * self.viscosity * self.radius))
        self.brownian_M_WT = self.noise_level * self.brownian_factor * np.sqrt(1.0 / (8.0 * np.pi * self.viscosity * self.radius**3))

    def generate_thermal_noise(self):
        '''
        generate thermal noise
        '''
        if ti.static(self.noise_type == 'uncorrelated'):
            self.generate_uncorrelated_brownian_noise()
        else:
            raise NotImplementedError

    @ti.kernel
    def generate_uncorrelated_brownian_noise(self):
        '''
        generate uncorrelated brownian noise (bulk mobility tensor)
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.vel_stoch[i][k] = self.brownian_M_UF * ti.randn()
                self.ang_stoch[i][k] = self.brownian_M_WT * ti.randn()

    @ti.kernel
    def generate_correlated_brownian_noise(self):
        '''
        generate correlated brownian noise
        '''
        pass

    @ti.kernel
    def generate_thermal_drift(self):
        '''
        generate thermal drift term
        '''
        pass

