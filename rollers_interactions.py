import taichi as ti
import numpy as np
import ipdb

@ti.data_oriented
class rollers_interactions:
    '''
    Define the interactions between Quincke rollers
    '''
    def __init__(self, system):
        # rollers suspension
        self.system = system

        self.N = system.N
        self.dim = system.dim

        # particle properties
        self.particle_radius = system.radius

        self.pos = system.pos
        self.ort = system.ort

        self.vel = system.vel
        self.ang = system.ang

        self.force = system.force
        self.torque = system.torque

        self.efield = system.efield
        self.dipole = system.dipole
        
        # interactions

        # constant forces
        self.onebody_Fconst_on = system.onebody_Fconst_on
        self.F0 = system.F0 # external forces

        # constant torques
        self.onebody_Tconst_on = system.onebody_Tconst_on
        self.T0 = system.T0 # external torques

        # external fields
        self.onebody_Eext_on = system.onebody_Eext_on
        self.E0 = system.E0 # external fields

        # dipole-dipole interactions
        self.pair_dipole_on = True
        #self.dipole_type = 'soft'
        self.dipole_type = 'hard'
        self.dipole_cutoff = system.dipole_cutoff
        #self.dipole_cutoff = 5.0 * self.particle_radius

        self.dipole_r_min = 2.0 * self.particle_radius
        self.dipole_A = system.dipole_A
        self.dipole_kappa = system.dipole_kappa

        # image dipoles
        self.image_dipole_on = system.image_dipole_on
        self.image_dipole_A = system.image_dipole_A
        self.image_dipole_kappa = system.image_dipole_kappa

        # yukawa potential
        self.pair_yukawa_on = system.pair_yukawa_on
        #self.yukawa_type = 'soft'
        self.yukawa_type = 'hard'
        self.yukawa_cutoff = system.yukawa_cutoff
        #self.yukawa_cutoff = (2.0 + 3.0) * self.particle_radius
        self.yukawa_r_min = (2.0 + 0.1) * self.particle_radius
        self.yukawa_A = system.yukawa_A
        #self.yukawa_A = 1000.0
        self.yukawa_kappa = system.yukawa_kappa
        #self.yukawa_kappa = 1.0 / (0.20 * self.particle_radius)

        # circular confinement
        self.boundary_circle_on = system.boundary_circle_on
        self.bc_circ_reflect_dipole_on = True
        self.bc_circ_origin = ti.Vector([0]*self.dim)
        self.bc_circ_radius = min(system.Lx, system.Ly, system.Lz) / 2.0 - 1.0 * self.particle_radius
        self.bc_circ_strength = 1000 * 1
        self.bc_circ_depth = system.radius * 2.0

        self.enforce2D = system.enforce2D

        # create data layout
        self.layout()

    def layout(self):
        '''
        specify data layout
        '''
        pass

    def init(self):
        '''
        initialization
        '''
        pass

    def compute(self):
        '''
        compute all interactions
        '''

        # clear arrays before computation
        self.clear()

        # boundary interactions
        self.compute_boundary_interactions()

        # onebody interactions
        self.compute_onebody_interactions()

        # pairwise interactions
        self.compute_pairwise_interactions()

        # remove z-component forces
        if ti.static(self.enforce2D == True):
            self.zeros_forces_z()

    @ti.func
    def pair_dipole(self, rij, pi, pj, efieldi: ti.template(), efieldj: ti.template(), forcei: ti.template(), forcej: ti.template(), torquei: ti.template(), torquej: ti.template()):
        '''
        dipole-dipole interactions between i and j particles

        U(r) = A * exp(-kappa*r) * [1/r^3 * (pi * pj) - 3/r^5 * (pi * r) * (pj * r)]

        '''
        # auxiliary variables
        r = rij.norm()

        # modify interactions for nearly-touching particles to avoid divergence
        if ti.static(self.dipole_type == 'soft'):
            if r < self.dipole_r_min:
                rij = rij / r * self.dipole_r_min
                r = self.dipole_r_min

        r_inv = 1.0 / r
        r_inv2 = r_inv * r_inv
        r_inv3 = r_inv2 * r_inv
        r_inv5 = r_inv3 * r_inv2
        r_inv7 = r_inv5 * r_inv2

        pidotpj = pi.dot(pj)
        pidotr = pi.dot(rij)
        pjdotr = pj.dot(rij)

        picrosspj = pi.cross(pj)
        picrossr = pi.cross(rij)
        pjcrossr = pj.cross(rij)

        screen_factor = ti.exp(-self.dipole_kappa*r)
        energy_scale = self.dipole_A

        factor1 = 3.0*r_inv5*pidotpj - 15.0*r_inv7*pidotr*pjdotr
        factor2 = 3.0*r_inv5*pjdotr
        factor3 = 3.0*r_inv5*pidotr
        factor4 = (r_inv3*pidotpj - 3.0*r_inv5*pidotr*pjdotr)*self.dipole_kappa*r_inv

        # electric field
        efieldi_local = energy_scale * screen_factor * (-r_inv3*pj + 3*r_inv5*pjdotr*rij)
        efieldj_local = energy_scale * screen_factor * (-r_inv3*pi + 3*r_inv5*pidotr*rij)

        # force
        forcei_local = energy_scale * screen_factor * (factor1*rij + factor2*pi + factor3*pj + factor4*rij)
        forcej_local = -1.0*forcei_local

        # torque
        torquei_local = pi.cross(efieldi_local)
        torquej_local = pj.cross(efieldj_local)

        # update results
        efieldi += efieldi_local
        efieldj += efieldj_local

        forcei += forcei_local
        forcej += forcej_local

        torquei += torquei_local
        torquej += torquej_local

    @ti.func
    def pair_dipole_oneway(self, rij, pi, pj, efieldi: ti.template(), forcei: ti.template(), torquei: ti.template()):
        '''
        dipole-dipole interactions between i and j particles (only add up for i-th particle)
        '''
        # auxiliary variables
        r = rij.norm()

        # modify interactions for nearly-touching particles to avoid divergence
        if ti.static(self.dipole_type == 'soft'):
            if r < self.dipole_r_min:
                rij = rij / r * self.dipole_r_min
                r = self.dipole_r_min

        r_inv = 1.0 / r
        r_inv2 = r_inv * r_inv
        r_inv3 = r_inv2 * r_inv
        r_inv5 = r_inv3 * r_inv2
        r_inv7 = r_inv5 * r_inv2

        pidotpj = pi.dot(pj)
        pidotr = pi.dot(rij)
        pjdotr = pj.dot(rij)

        picrosspj = pi.cross(pj)
        picrossr = pi.cross(rij)
        pjcrossr = pj.cross(rij)

        screen_factor = ti.exp(-self.image_dipole_kappa*r)
        energy_scale = self.image_dipole_A

        factor1 = 3.0*r_inv5*pidotpj - 15.0*r_inv7*pidotr*pjdotr
        factor2 = 3.0*r_inv5*pjdotr
        factor3 = 3.0*r_inv5*pidotr
        factor4 = (r_inv3*pidotpj - 3.0*r_inv5*pidotr*pjdotr)*self.image_dipole_kappa*r_inv

        # electric field
        efieldi_local = energy_scale * screen_factor * (-r_inv3*pj + 3*r_inv5*pjdotr*rij)

        # force
        forcei_local = energy_scale * screen_factor * (factor1*rij + factor2*pi + factor3*pj + factor4*rij)

        # torque
        torquei_local = pi.cross(efieldi_local)

        # update results
        efieldi += efieldi_local
        forcei += forcei_local
        torquei += torquei_local

    @ti.func
    def pair_yukawa(self, rij, forcei: ti.template(), forcej: ti.template()):
        '''
        yukawa interactions between i and j particles

        E(r) = A * exp(-\kappa * r) / r

        '''
        r = rij.norm()
        eij = rij / r

        # modify for nearly-touching particles to avoid divergence
        if ti.static(self.yukawa_type == 'soft'):
            if r < self.yukawa_r_min:
                r = self.yukawa_r_min

        r -= 2.0 * self.particle_radius
        r_inv = 1.0 / r

        screen_factor = ti.exp(-self.yukawa_kappa*r)
        energy_scale = self.yukawa_A

        force_factor = energy_scale * screen_factor * (self.yukawa_kappa + r_inv) * r_inv

        forcei += force_factor * eij
        forcej -= force_factor * eij

    @ti.func
    def onebody_Eext(self, dipole, efield: ti.template(), torque: ti.template()):
        '''
        compute interactions with external electric fields
        '''
        # external fields
        E0 = ti.Vector([self.E0[0], self.E0[1], self.E0[2]])
        # torque pxE0
        torque0 = dipole.cross(E0)

        # accumulate
        for k in ti.static(range(self.dim)):
            # electric fields
            efield[k] += E0[k]

            # electric forces
            # zero for uniform fields

            # electric torques
            torque[k] += torque0[k]

    @ti.func
    def onebody_Fconst(self, force: ti.template()):
        '''
        apply a constant force to each particles
        '''
        # add up constant forces
        for k in ti.static(range(self.dim)):
            force[k] += self.F0[k]

    @ti.func
    def onebody_Tconst(self, torque: ti.template()):
        '''
        apply a constant force to each particles
        '''
        # add up constant torques
        for k in ti.static(range(self.dim)):
            torque[k] += self.T0[k]

    @ti.kernel
    def compute_onebody_interactions(self):
        '''
        per particle interactions
        '''
        for i in ti.ndrange(self.N):
            # auxiliary variables
            efield = ti.Vector([0.]*self.dim)
            force = ti.Vector([0.]*self.dim)
            torque = ti.Vector([0.]*self.dim)

            # external electric fields
            if ti.static(self.onebody_Eext_on == True):
                dipole = ti.Vector([0.]*self.dim)
                for k in ti.static(range(self.dim)):
                    dipole[k] = self.dipole[i][k]
                self.onebody_Eext(dipole, efield, torque)

            # image dipole - self image
            if ti.static(self.image_dipole_on == True):
                ri = ti.Vector([0.]*self.dim)
                pi = ti.Vector([0.]*self.dim)
                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    pi[k] = self.dipole[i][k]

                ri_image = ri
                ri_image[2] = -1.0 * ri_image[2]
                rii_image = ri - ri_image

                pi_image = pi
                pi_image[0] = -1.0 * pi_image[0]
                pi_image[1] = -1.0 * pi_image[1]

                # i <- i_image
                self.pair_dipole_oneway(rii_image, pi, pi_image, efield, force, torque)

            # constant forces
            if ti.static(self.onebody_Fconst_on == True):
                self.onebody_Fconst(force)

            # constant torques
            if ti.static(self.onebody_Tconst_on == True):
                self.onebody_Tconst(torque)

            # update i-th particle
            for k in ti.static(range(self.dim)):
                self.efield[i][k] += efield[k]
                self.force[i][k] += force[k]
                self.torque[i][k] += torque[k]

    @ti.kernel
    def compute_pairwise_interactions(self):
        '''
        compute pairwise interactions
        '''
        # brutal force for now, should use nerighbors list instead
        for i,j in ti.ndrange(self.N, self.N):
            if i<j:
                # auxiliary variables
                ri = ti.Vector([0.]*self.dim)
                rj = ti.Vector([0.]*self.dim)
                efieldi = ti.Vector([0.]*self.dim)
                efieldj = ti.Vector([0.]*self.dim)
                forcei= ti.Vector([0.]*self.dim)
                forcej= ti.Vector([0.]*self.dim)
                torquei= ti.Vector([0.]*self.dim)
                torquej= ti.Vector([0.]*self.dim)

                for k in ti.static(range(self.dim)):
                    ri[k] = self.pos[i][k]
                    rj[k] = self.pos[j][k]

                rij = ri - rj
                r = rij.norm()

                # dipole-dipole interaction
                if ti.static(self.pair_dipole_on == True):
                    if r < self.dipole_cutoff:
                        pi = ti.Vector([0.]*self.dim)
                        pj = ti.Vector([0.]*self.dim)
                        # dipole moments
                        for k in ti.static(range(self.dim)):
                            pi[k] = self.dipole[i][k]
                            pj[k] = self.dipole[j][k]

                        self.pair_dipole(rij, pi, pj, efieldi, efieldj, forcei, forcej, torquei, torquej)

                        # image dipoles
                        if ti.static(self.image_dipole_on == True):
                            # assume wall at z = 0

                            # image positions
                            ri_image = ri
                            ri_image[2] = -1.0 * ri_image[2]

                            rj_image = rj
                            rj_image[2] = -1.0 * rj_image[2]

                            rij_image = ri - rj_image
                            rji_image = rj - ri_image

                            # image dipoles
                            pi_image = pi
                            pi_image[0] = -1.0 * pi_image[0]
                            pi_image[1] = -1.0 * pi_image[1]

                            pj_image = pj
                            pj_image[0] = -1.0 * pj_image[0]
                            pj_image[1] = -1.0 * pj_image[1]

                            # i <- j_image
                            self.pair_dipole_oneway(rij_image, pi, pj_image, efieldi, forcei, torquei)

                            # j <- i_image
                            self.pair_dipole_oneway(rji_image, pj, pi_image, efieldj, forcej, torquej)

                # yukawa potential
                if ti.static(self.pair_yukawa_on == True):
                    if r < self.yukawa_cutoff:
                        self.pair_yukawa(rij, forcei, forcej)

                # update (i,j) pair
                for k in ti.static(range(self.dim)):
                    # electric field
                    self.efield[i][k] += efieldi[k]
                    self.efield[j][k] += efieldj[k]
                    # force
                    self.force[i][k] += forcei[k]
                    self.force[j][k] += forcej[k]
                    # torque
                    self.torque[i][k] += torquei[k]
                    self.torque[j][k] += torquej[k]

    @ti.func
    def boundary_circ_normal(self, r_vec, force: ti.template()):
        '''
        interactions with a circular wall along the normal direction
        '''
        r = r_vec.norm()
        n = r_vec / r

        if r <= self.bc_circ_radius - self.particle_radius + self.bc_circ_depth:
            # linear
            for k in ti.static(range(self.dim)):
                force[k] += -(r - self.bc_circ_radius + self.particle_radius) / self.bc_circ_depth * self.bc_circ_strength * n[k]
        else:
            # constant
            for k in ti.static(range(self.dim)):
                force[k] += -self.bc_circ_strength * n[k]

    @ti.func
    def boundary_reflect_dipole(self, i, r_vec):
        '''
        reflect in-plane dipole moments
        '''
        r = r_vec.norm()
        n = r_vec / r # assumed 2d normal vector [nx,ny,0]
        t = n
        t[0] = -n[1]
        t[1] =  n[0]

        dipole = self.dipole[i]

        # reflect normal component and keep tangential component
        pn = dipole.dot(n)
        pt = dipole.dot(t)

        dipole_reflected = - pn * n + pt * t

        # update reflected dipole moments
        for k in ti.static(range(self.dim)):
            self.dipole[i][k] = dipole_reflected[k]

    @ti.kernel
    def compute_boundary_interactions(self):
        '''
        compute boundary interactions
        '''
        for i in ti.ndrange(self.N):
            # auxiliary variables
            force = ti.Vector([0.]*self.dim)

            # circular boundary
            if ti.static(self.boundary_circle_on == True):
                r_vec = ti.Vector([0.]*self.dim)
                if ti.static(self.enforce2D == True):
                    r_vec[0] = self.pos[i][0] - self.bc_circ_origin[0]
                    r_vec[1] = self.pos[i][1] - self.bc_circ_origin[1]
                else:
                    for k in ti.static(range(self.dim)):
                        r_vec[k] = self.pos[i][k] - self.bc_circ_origin[k]

                dist = r_vec.norm()
                if dist > self.bc_circ_radius - self.particle_radius:
                    self.boundary_circ_normal(r_vec, force)
                    if ti.static(self.bc_circ_reflect_dipole_on == True):
                        self.boundary_reflect_dipole(i, r_vec)

            # update i-th particle
            for k in ti.static(range(self.dim)):
                self.force[i][k] += force[k]

    @ti.kernel
    def clear(self):
        '''
        clear relevant arrays
        '''
        for i in ti.ndrange(self.N):
            for k in ti.static(range(self.dim)):
                self.efield[i][k] = 0.0
                self.force[i][k] = 0.0
                self.torque[i][k] = 0.0

    @ti.kernel
    def zeros_forces_z(self):
        '''
        zeros z-component forces
        '''
        for i in ti.ndrange(self.N):
            self.force[i][2] = 0.0
