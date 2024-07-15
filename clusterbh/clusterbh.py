import numpy as np
from ssptools import evolve_mf
from scipy.special import hyp2f1
from scipy.integrate import solve_ivp


class clusterBH:
    '''

    Parameters
    ----------
    N : int
        Initial number of stars

    rhoh : float
        Initial half-mass density

    m0 : float, optional
        For Kroupa (2001) IMF 0.06-150 Msun Defaults to 0.606.

    fc : float, optional
        equation (50) Defaults to 1.

    rg : float, optional
        [kpc] Defaults to 8.

    Z : float, optional
        Metalicity. Defaults to 0.0002.

    f : float, optional
        Prefactor that relates relaxation of stars with the half mass
        relaxation. Defaults to 1.

    Mval : float, optional
        Contribution of stellar evolution in half mass radius after core
        collapse. Defaults to 1.5.

    zeta : float, optional
        Energy loss per half mass relaxation. Defaults to 0.1.

    a0 : float, optional
        Fix zeroth order in ψ. Defaults to 1.

    a2 : float, optional
        Ignore 2nd order in ψ for now. Defaults to 0.

    kick : bool, optional
        We include natal kicks. Defaults to True.

    tsev : float, optional
        In Myrs. Time instance when stars start evolving. Defaults to 2.

    n : float, optional
        Exponent in the power-law for tides. Defaults to 1.5.

    r : float, optional
        Ratio of the half mass radius over the Virial radius initially.
        Defaults to 1.0.

    alpha_c : float, optional
        0.65% Evaporation of stars initially, approximating tides.
        Defaults to 0.0065.

    gamma : float, optional
        Parameter of Coulomb logarithm as used in the cmc models.
        Defaults to 0.02.

    kin : float, optional
        Kinetic term of escaping stars. Defaults to 0.9.

    alpha : float, optional
        Defaults to -0.5.

    mlo : float, optional
        Defaults to 5.

    mup : float, optional
        Defaults to 50.

    sigmans : float, optional
        Velocity dispersion in km/s. Defaults to 265.

    mns : float, optional
        Mass of Neutron stars in Msun. Defaults to 1.4.

    tend : float, optional
        Final time instance where we integrate to. Defaults to 13.8e3.

    dtout : float, optional
        Myr. Time step for integration. Defaults to 2.

    Mst_min : float, optional
        [Msun] Stop criterion. Defaults to 100.

    integration_method : str, optional
        Integration method. Defaults to "RK45".

    tidal : bool, optional
        We activate tides. Defaults to True.

    escapers : bool, optional
        Condition for escapers to carry negative energy as they escape the
        cluster due to a tidal field. Defaults to False.

    Rht : float, optional
        Ratio of rh/rt to give correct Mdot. It is not necessarily
        the final value of rh / rt. Defaults to 0.125.

    Vc : float, optional
        [km/s] circular velocity of singular isothermal galaxy Defaults to 220.

    a_slopes : 3-tuple of float
        3-component IMF slopes. Defaults to (-1.3, -2.3, -2.3).

    m_breaks : 3-tuple of float
        Break masses. Defaults to (0.08, 0.5, 1., 150.).

    nbins : 3-tuple of int
        Number of bins. Defaults to (5, 5, 20).

    Attributes
    ----------
    ntrh : float
        Number of initial relaxations in order to compute core collapse
        instance. By default, has been fit to N-body models to give a value
        of 3.21.

    beta : float
        Ejection rate of black holes from the core per relaxation. By default,
        has been fit to N-body models to give a value of 0.0028.

    nu : float
        Mass loss rate of stars. By default, has been fit to N-body models to
        give a value of 0.0823.

    a1 : float
        Linear order in ψ. By default, has been fit to N-body models to give a
        value of 1.47.

    FeH : float
        Metallicity, computed as log10(Z / Zsolar) where Zsolar=0.014.

    M0 : float
        Total mass of stars (cluster) initially.

    rh0 : float
        Half mass radius initially.

    vesc0 : float
        Initial escape velocity as a function of mass and density.
        Based on M0 an rho0, multiplied by fc.

    Mbh0 : float
        Expected initial mass of black holes due to kicks.

    M0 : float
        remove BH st.ev. losses

    mbh : float
        average BH mass (initially)

    f0 : float
        Initial fraction of black holes. It should give us something close to
        0.05 for poor metal clusters.

    Nbh0 : int
        Initial number of black holes.

    Mst_lost : float
        Mass of stars lost in order to form black holes.

    t_bhcreation : float
        Time in Myrs that is required in order to form these black holes.

    trh0 : float
        Initial relaxation in Myrs. Because we start with black holes, they are
        included here so that trh is continuous. It is a small error which is
        practically absorbed in ntrh and corects itself after a few Myrs.

    tcc : float
        Core collapse in Myrs.
    '''

    def __init__(self, N, rhoh, *, m0=0.606, fc=1, rg=8, Z=0.0002, mbh=20, f=1,
                 Mval=1.5, zeta=0.1, a0=1, a2=0, kick=True, tsev=2, n=1.5, r=1.,
                 alpha_c=0.0065, gamma=0.02, kin=0.9,
                 alpha=-0.5, mlo=5, mup=50, sigmans=265, mns=1.4,
                 tend=13.8e3, dtout=2, Mst_min=100, integration_method="RK45",
                 tidal=True, escapers=False, Rht=0.125, Vc=220.,
                 a_slopes=[-1.3, -2.3, -2.3], m_breaks=[0.08, 0.5, 1., 150.],
                 nbins=[5, 5, 20],
                 ntrh=3.21, beta=0.0028, nu=0.0823, a1=1.47):

        self.G = 0.004499  # pc^3 /Msun /Myr^2

        # Cluster ICs
        self.N = N
        self.m0 = m0
        self.fc = fc
        self.rg = rg
        self.Z = Z
        self.f = f
        self.Mval = Mval

        # Model parameters
        self.zeta = zeta
        self.a0 = a0
        self.a2 = a2
        self.kick = kick
        self.tsev = tsev
        self.n = n
        self.r = r
        self.alpha_c = alpha_c
        self.gamma = gamma
        self.kin = kin

        # Parameters that were fit to N-body.
        self.ntrh = ntrh  # 3.21
        self.beta = beta  # 0.0028
        self.nu = nu  # 0.0823
        self.a1 = a1  # 1.47

        # BHMF

        self.alpha = alpha
        self.mlo = mlo
        self.mup = mup

        self.sigmans = sigmans
        self.mns = mns

        # Some integration params.
        self.tend = tend
        self.dtout = dtout
        self.Mst_min = Mst_min
        self.integration_method = integration_method

        self.output = False  # Just a string in order to save the results.
        self.outfile = "cluster.txt"  # File to save the results if needed.

        # Mass loss mechanism
        self.tidal = tidal
        self.escapers = escapers
        self.Rht = Rht
        self.Vc = Vc

        self.a_slopes = a_slopes
        self.m_breaks = m_breaks
        self.nbins = nbins

        self.FeH = np.log10(self.Z / 0.014)

        self.M0 = self.m0 * N
        self.rh0 = (3 * self.M0 / (8 * np.pi * rhoh))**(1. / 3)

        self.vesc0 = 50 * (self.M0 / 1e5)**(1. / 3) * (rhoh / 1e5)**(1. / 6)

        self.vesc0 *= self.fc  # Augment the value for different King models.

        # Implement kicks for this IMF with such metallicity.
        self.ibh = evolve_mf.InitialBHPopulation.from_IMF(
            self.m_breaks, self.a_slopes, self.nbins,
            self.FeH, N0=self.N, vesc=self.vesc0, natal_kicks=self.kick
        )

        self.Mbh0 = self.ibh.Mtot
        self.mbh = np.sum(self.ibh.M) / np.sum(self.ibh.N)

        self.M0 = self.M0 + self.Mbh0 - self.ibh.Ms_lost
        self.f0 = self.Mbh0 / self.M0

        # Check if we have kicks so that we can fix mb and compute the upper
        # mass of black holes at each time instance.
        if (self.kick):
            self.mb = ((9 * np.pi / 2)**(1. / 6)
                       * self.sigmans * self.mns / self.vesc0)

        self.Nbh0 = np.round(self.ibh.Ntot)
        self.Mst_lost = self.ibh.Ms_lost
        self.t_bhcreation = self.ibh.age

        self.trh0 = self._trh(self.M0 + self.Mbh0, self.rh0,
                              self.Mbh0 / (self.Mbh0 + self.M0), 0)
        self.tcc = self.ntrh * self.trh0

        self.evolve(N, rhoh)

    def _rt(self, M):
        '''Tidal radius.'''
        O2 = (self.Vc * 1.023 / (self.rg * 1e3)) ** 2  # Angular vel. squared
        return (self.G * M / (2 * O2)) ** (1. / 3)

    def _mst(self, t):
        '''Average mass of stars, due to stellar evolution.
        It takes as an input either a value or an array for simplicity.
        This is really sloppy.
        Does not take into consideration the effect of tides.
        '''
        if t > self.tsev:
            return self.m0 * (t / self.tsev)**(-self.nu)
        else:
            return self.m0

    def _psi(self, fbh):
        '''Friction term ψ in the relaxation.
        It is characteristic of mass spectrum, here due to black holes.
        We neglect O(2) corrections.
        '''
        psi = self.a0 + self.a1 * abs(fbh) / 0.01
        return psi

    def _N(self, M, fbh, t):
        '''
        Number of particles, approximately stars. We write it as a function of
        the total mass and the black hole fraction. The number of black holes
        is a minor correction since typically 1 star in 1e3 becomes a black
        hole, so clusters with 1e6 have 1e3 black holes roughly speaking, a
        small correction. If mbh is needed, one can add a term M * fbh / mbh.
        '''
        # Varying average mass for stars. Mass of BHs is constant for now.
        return np.round(M * ((1 - fbh) / self._mst(t))).astype(int)

    def _trh(self, M, rh, fbh, t):
        '''
        Relaxation as defined by Spitzer. Here we consider the effect of mass
        spectrum due to black holes. When they vanish, we get unity, which is
        not entirely true.
        '''
        Np = self._N(M, fbh, t)
        if M > 0 and rh > 0:
            return (0.138 * np.sqrt(M * rh ** 3 / self.G)
                    / (self._mst(t) * self._psi(fbh) * np.log(self.gamma * Np)))
        else:
            return 1e-99

    def find_mmax(self, Mbh):
        '''
        We find the maximum value of the black hole mass at each time instance.
        We assume that heavy black holes are the only ones ejected from the
        core, a decent approximation.
        '''
        a2 = self.alpha + 2

        if (self.kick):
            def integr(mm, qmb, qlb):
                a2 = self.alpha + 2
                b = a2 / 3
                h1 = hyp2f1(1, b, b + 1, -qmb ** 3)
                h2 = hyp2f1(1, b, b + 1, -qlb ** 3)

                return mm ** a2 * (1 - h1) - self.mlo ** a2 * (1 - h2)

            # invert eq. 52 from AG20
            Np = 1000
            mmax_ = np.linspace(self.mlo, self.mup, Np)
            # qml = mmax_ / self.mlo
            qmb, qlb = mmax_ / self.mb, self.mlo / self.mb

            A = self.Mbh0 / integr(self.mup, self.mup / self.mb, qlb)

            Mbh_ = A * integr(mmax_, qmb, qlb)
            mmax = np.interp(Mbh, Mbh_, mmax_)

        else:
            # eq 51 in AG20
            mmax = (
                Mbh / self.Mbh0 * (self.mup**a2 - self.mlo**a2) + self.mlo**a2
            )**(1. / a2)

        # TBD: Set to 0 when MBH = 0
        return mmax

    def _logcheck(self, t, y):
        return 0 if (y[0] > self.Mst_min) else -1

    def odes(self, t, y):
        '''The differential equations to be solved.'''

        Mst = y[0]  # Mass of stars.
        Mbh = y[1]  # Mass of black holes.
        rh = y[2]  # Half mass radius.

        # Total mass of the cluster.
        # It overestimates a bit initially because we assume Mbh>0 from start.
        M = Mst + Mbh

        # Fraction of black holes.
        # For the same reason it whould be considered after core collapse.
        fbh = Mbh / M

        rt = self._rt(M)
        trh = self._trh(M, rh, fbh, t)  # Relaxation
        tcc = self.tcc   # Core collapse.
        tsev = self.tsev  # Stellar evolution.
        # Np = self._N(M, fbh, t)  # Number of particles
        psi = self._psi(fbh)  # Friction term.
        tbh = self.t_bhcreation  # Time instance when black holes are created.
        trhstar = trh * self.f * psi  # If not needed, change to trh.

        # At first the derivatives are set equal to zero, then we build them up.
        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0

        # This would be the term for mass segregation.
        # Here it is kept constant and equal to 3.
        M_val = 3  # if t < tcc else self.Mval

        Mst_dotsev = 0

        # Stellar mass loss.

        if t >= tsev:

            # Stars lose mass due to stellar evolution.
            Mst_dotsev -= self.nu * Mst / t

            # The cluster expands for this reason. It is because we assume a
            # uniform distribution initially. With mass segregation up until
            # core collapse, we should get a better description.
            rh_dot -= (M_val - 2) * Mst_dotsev / M * rh

        if tcc < t < tbh:

            # Reduce Mbh because heavy stars are ejected. This must happen if
            # core collapse occurs due to stars.
            Mbh_dot -= self.alpha_c * M / trh

        # Add tidal mass loss.

        if (self.tidal):  # Check if we have tides.

            # Power-law for generality. The user may choose any form they wish.
            xi = 0.6 * self.zeta * (rh / rt / self.Rht) ** self.n

            # Total stellar mass loss rate is the combination of ejections and
            # tides. alpha_c is treated as ejections, not a constant
            # evaporation rate, and thus it is a property of the cluster and
            # not the tidal field.
            xi_total = xi + self.alpha_c if t > tcc else xi

            # Mass loss rate of stars due to tides. We add this to the already
            # known expression for mass-loss rate due to stellar evolution.
            Mst_dot -= xi * M / trhstar + (xi_total - xi) * M / trh

        # Note that we now subtract 2ν from the +ν term already in Mst_dot.
        # This is an approximation since heavy stars residing in a shell
        # around the core should contribute. This is captured better if we
        # use a parameter that describes mass segregation.
        rh_dot += 2 * Mst_dot / M * rh

        # Correct the total stellar mass loss rate.
        Mst_dot += Mst_dotsev

        # Keep only evaporation, not ejections here so use xi and not xi_total.
        # If escapers carry negative energy as they leave, the half mass radius
        # is expected to increase since it is similar to emitting positive
        # energy. The effect is proportional to tides.
        if self.escapers:
            rh_dot += 6 * xi / self.r / trhstar * (1 - self.kin) * rh ** 2 / rt

        # BH escape.
        # It occurs after core collapse, in the balanced phase of the cluster.

        # Expansion due to energy loss. It is due to the fact that the cluster
        # has a negative heat capacity due to its gravitational nature.
        if t >= tcc:
            rh_dot += self.zeta * rh / trh

        # Check if we have black holes so that we can evolve them as well.
        if Mbh > 0 and t >= tcc:
            # Ejection each relaxation.
            Mbh_dot -= self.beta * M / trh
            # Contraction since black holes are removed.
            rh_dot += 2 * Mbh_dot / M * rh

        derivs = [Mst_dot]
        derivs.append(Mbh_dot)
        derivs.append(rh_dot)

        # Return the derivatives in an array.
        return np.array(derivs, dtype=object)

    # Extract the solution using the above differential equations.
    def evolve(self, N, rhoh):
        Mst = [self.M0]  # Stellar mass initially.
        Mbh = [self.Mbh0]  # Initial black hole mass.
        rh = [self.rh0]  # Initial half mass radius.

        y = [Mst[0], Mbh[0], rh[0]]  # Combine them in a multivariable.

        def Mst_min_event(t, y):  # Event in order to stop when stars are lost.
            return y[0] - self.Mst_min

        # Find solutions as long as the event holds.
        Mst_min_event.terminal = True
        # Stop when we find a t such that the event is no longer True.
        Mst_min_event.direction = -1

        if self.dtout is not None:
            t_eval = np.arange(0, self.tend, self.dtout)
        else:
            t_eval = None

        # Solution.
        sol = solve_ivp(self.odes, [0, self.tend], y,
                        method=self.integration_method, t_eval=t_eval)

        self.t = [x / 1e3 for x in sol.t]  # Time in Gyrs.
        self.t = np.array(self.t)
        self.Mst = sol.y[0]  # Stellar mass.
        self.Mbh = sol.y[1]  # Black hole mass.
        self.rh = sol.y[2]  # Half mass radius

        cbh = (self.Mbh > 0)  # Where the mass of black holes is present.

        # Upper mass of black holes in the mass function at each instance.
        self.mmax = np.zeros_like(self.Mbh)
        self.mmax[cbh] = self.find_mmax(self.Mbh[cbh])

        # Some derived quantities.

        self.M = self.Mst + self.Mbh  # Total mass of the cluster.
        self.rt = self._rt(self.M)  # Tidal radius.
        self.fbh = self.Mbh / self.M  # Black hole fraction.
        self.psi = self._psi(self.fbh)  # Friction term ψ.

        # List of stellar average mass over time.
        self.mst_av = [self._mst(x) for x in sol.t]

        # Number of components.
        self.Np = np.array([self._N(x, y, z) for (x, y, z)
                            in zip(self.M, self.fbh, sol.t)])

        # Average mass of cluster over time, includes black holes.
        # No significant change is expected given that Nbh <= O(1e3).
        self.mav = self.M / self.Np

        # Energy of the cluster at each time instance.
        self.E = -self.r * self.G * self.M ** 2 / (4 * self.rh)

        # Virial radius as a function of time. It is of course a constant
        # fraction of rh since r does not vary in this approach.
        self.rv = -self. G * self. M ** 2 / (4 * self.E)

        # Relaxation.
        self.trh = np.array([self._trh(x, y, z, u) for (x, y, z, u)
                             in zip(self.M, self.rh, self.fbh, sol.t)])

        # output
        if (self.output):
            f = open(self.outfile, "w")
            for i in range(len(self.t)):
                f.write(f"{self.t[i]:12.5e} {self.Mbh[i]:12.5e} "
                        f"{self.M[i]:12.5e} {self.rh[i]:12.5e}\n")
            f.close()
