import numpy as np
import logging
import ssptools
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

    Mval0 : float, optional
        Initial value for mass segregation, taken for a homologous distribution
        of stars.

    Mval_cc : float, optional
        Contribution of stellar evolution in half-mass radius after
        core collapse.

    Mvalf : float, optional
        Final parameter for mass segregation.

    zeta : float, optional
        Energy loss per half mass relaxation. Defaults to 0.1.

    a0 : float, optional
        Fix zeroth order in ψ. Defaults to 1.

    a11 : float, optional
        First prefactor in ψ.

    a12 : float, optional
        Second prefactor in ψ.

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

    x : float, optional
        Exponent used for finite escape time from Lagrange points.

    tcross_index : float, optional
        Prefactor for half mass crossing time defined as index/sqrt(G rhoh).

    alpha_c : float, optional
        0.65% Evaporation of stars initially, approximating tides.
        Defaults to 0.0065.

    gamma : float, optional
        Parameter of Coulomb logarithm as used in the cmc models.
        Defaults to 0.02.

    gamma0 : float, optional
        An initial value for gamma exponent in psi.

    kin : float, optional
        Kinetic term of escaping stars. Defaults to 0.9.

    bmax : float, optional
        Exponent for parameter psi. The choices are between 2 and 2.5,
        the former indicates the same velocity dispersion between components
        while the latter complete equipartition.

    b1 : float, optional
        Correction to exponent of mbh / m in parameter psi. It appears
        because the properties within the half-mass radius differ.

    b2 : float, optional
        Exponent of the BH fraction for the BH ejection rate. Taken from
        Pina (2023).

    b3 : float, optional
        Exponent of the BH fraction for the BH ejection rate. Taken from
        Pina (2023).

    b4 : float, optional
        Exponent of the mass ratio for the BH ejection rate.

    fbh_crit : float, optional
        Critical value of the BH fraction to use in exponent gamma.
        It is a pair with gamma0.

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
        instance.

    beta : float
        Ejection rate of black holes from the core per relaxation.

    nu : float
        Mass loss rate of stars.

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

    def __init__(self, N0, rhoh, *, fc=1, rg=8, Z=0.0002, f=1,
                 Mval0=3.0, Mval_cc=3, Mvalf=4, zeta=0.070272,
                 a0=1, a11=2, a12=0.15, a2=0, kick=True,
                 tsev=1, n=2.974729, r=0.8, x=0.75, tcross_index=1,
                 alpha_c=0.0, gamma=0.02, gamma0=2.12, kin=0.9,
                 bmax=2.215548, b1=1, b2=1.5, b3=0.22, b4=0.4, fbh_crit=0.1,
                 tend=13.8e3, dtout=2, Mst_min=100, integration_method="RK45",
                 tidal=True, escapers=False, Rht=0.106094, Vc=220., Zsolar=0.02,
                 a_slopes=[-1.3, -2.3, -2.3], m_breaks=[0.08, 0.5, 1., 150.],
                 nbins=[5, 5, 20], ibh_kwargs=None,
                 ntrh=0.036606, beta=0.055909, nu=0.072):

        self.G = 0.004499  # pc^3 /Msun /Myr^2
        self.Zsolar = Zsolar

        # Cluster ICs
        self.N0 = N0
        self.fc = fc
        self.rg = rg
        self.Z = Z

        # Model parameters
        self.a0 = a0
        self.a11 = a11
        self.a12 = a12
        self.a2 = a2
        self.tsev = tsev
        self.x = x
        self.r = r
        self.f = f
        self.tcross_index = tcross_index
        self.alpha_c = alpha_c
        self.gamma = gamma
        self.gamma0 = gamma0
        self.kin = kin
        self.Mval0 = Mval0
        self.Mval_cc = Mval_cc
        self.Mvalf = Mvalf
        self.bmax = bmax
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.fbh_crit = fbh_crit

        # Parameters that were fit to N-body.
        self.zeta = zeta
        self.beta = beta
        self.nu = nu
        self.n = n
        self.ntrh = ntrh
        self.Rht = Rht

        # Some integration params.
        self.tend = tend
        self.dtout = dtout
        self.Mst_min = Mst_min
        self.integration_method = integration_method

        self.output = False  # Just a string in order to save the results.
        self.outfile = "cluster.txt"  # File to save the results if needed.

        # Conditions
        self.kick = kick
        self.tidal = tidal
        self.escapers = escapers
        # self.two_relaxations = two_relaxations
        # self.mass_segregation = mass_segregation
        # self.finite_escape_time = finite_escape_time
        # self.running_psi_exponent = running_psi_exponent
        # self.bh_exponent_correction = bh_exponent_correction
        # self.running_bh_ejection_rate = running_bh_ejection_rate
        # self.running_stellar_ejection_rate = running_stellar_ejection_rate

        self.Vc = Vc

        self.a_slopes = a_slopes
        self.m_breaks = m_breaks
        self.nbins = nbins

        self.FeH = np.log10(self.Z / self.Zsolar)

        self.imf = ssptools.masses.PowerLawIMF(m_breaks, a_slopes, N0=N0)
        self.m0 = self.imf.mmean

        self.M0 = self.m0 * N0
        self.rh0 = (3 * self.M0 / (8 * np.pi * rhoh))**(1. / 3)

        self.vesc0 = 50 * (self.M0 / 1e5)**(1. / 3) * (rhoh / 1e5)**(1. / 6)

        self.vesc0 *= self.fc  # Augment the value for different King models.

        if ibh_kwargs is None:
            ibh_kwargs = {}

        # Implement kicks for this IMF with such metallicity.
        self.ibh = ssptools.InitialBHPopulation.from_powerlaw(
            self.m_breaks, self.a_slopes, self.nbins, self.FeH, N0=self.N0,
            vesc=self.vesc0, natal_kicks=self.kick, **ibh_kwargs
        )

        self.Mbh0 = self.ibh.Mtot
        self.Nbh0 = self.ibh.Ntot
        self.mbh0 = self.Mbh0 / self.Nbh0

        # TODO not sure what should be re-comped after making BHs
        # total mass = initial mass - stellar mass lost + BH mass formed
        # But this is stellar mass actually, so let's just ignore it for now
        # self.M0 = self.M0 - self.ibh.Ms_lost
        # self.N0 = self.N0 - self.ibh.Ns_lost

        self.f0 = self.Mbh0 / self.M0

        self.Nbh0 = np.round(self.ibh.Ntot)
        self.Mst_lost = self.ibh.Ms_lost
        self.t_bhcreation = self.ibh.age

        # self.trh0 = self._trh(self.M0 + self.Mbh0, self.rh0,
        #                       self.Mbh0 / (self.Mbh0 + self.M0), 0)
        # self.tcc = self.ntrh * self.trh0

        # Initial relaxation is extracted as a function of the BH population so
        # that clusters with different metallicities and BH fractions can have
        # a different core collapse.
        self.psi0 = (self.a0 + self.a11 * self.a12
                     * self.f0**self.b1 * (self.mbh0 / self.m0)**(self.b2))
        self.trh0 = (0.138 * self.N0
                     / (self.psi0 * np.log(self.gamma * self.N0))
                     * np.sqrt(self.rh0**3 / (self.G * self.M0)))
        self.tcc = self.ntrh * self.trh0

        # if self.bh_exponent_correction:
        #     self.b4 = self.b4 * self.b2

        self.evolve(N0, rhoh)

    def deplete_BHMF(self, M_eject, M_BH, N_BH):
        '''
        Function to determine by how much the BHMF changes, given a particular
        mass loss.
        '''

        # Avoid altering initial BH bins.
        M_BH, N_BH = M_BH.copy(), N_BH.copy()

        # Remove BH starting from Heavy to Light.
        j = M_BH.size

        while M_eject != 0:
            j -= 1

            # Stop ejecting if trying to eject more mass than there is in BHs.
            if j < 0:
                break

            # Remove entirety of this mass bin.
            if M_BH[j] < M_eject:
                M_eject -= M_BH[j]
                M_BH[j] = 0
                N_BH[j] = 0
                continue

            # Remove required fraction of the last affected bin.
            else:
                m_BH_j = M_BH[j] / N_BH[j]
                M_BH[j] -= M_eject
                N_BH[j] -= M_eject / m_BH_j

                break

        return M_BH, N_BH

    def _mbh(self, Mbh):
        '''Average BH mass.'''

        # Determine amount of BH mass (total) to eject.
        M_eject = self.ibh.Mtot - Mbh

        # Deplete the initial BH MF based on this M_eject.
        M_BH, N_BH = self.deplete_BHMF(M_eject, self.ibh.M, self.ibh.N)
        return M_BH.sum() / N_BH.sum()

    def _rt(self, M):
        '''Tidal radius.'''
        O2 = (self.Vc * 1.023 / (self.rg * 1e3)) ** 2  # Angular vel. squared
        return (self.G * M / (2 * O2)) ** (1. / 3)

    def _mst(self, t):
        '''Average mass of stars, due to stellar evolution.
        It takes as an input either a value or an array for simplicity.
        Does not take into consideration the effect of tides.
        In that case, it should be derived from differential equations.
        '''
        if t > self.tsev:
            return self.m0 * (t / self.tsev)**(-self.nu)
        else:
            return self.m0

    def _psi(self, fbh, M, mbh, t):
        '''Friction term ψ in the relaxation.
        It is characteristic of mass spectrum, here due to black holes.
        We neglect O(2) corrections.
        '''

        # This means that BHs have been ejected so we have no psi
        if M * fbh < mbh:
            return self.a0

        # Number of particles
        Np = self._N(M, fbh, mbh, t)

        # Average mass
        mav = M / Np

        # Exponent.
        gamma = self.bmax
        # if self.running_exponent:
        #     gamma = self.bmax - (self.bmax-self.gamma0) * fbh / self.fbh_crit
        # else:
        #     gamma = self.bmax

        # Approximate form
        # Parameters a11, a12, b1 and b2 relate the properties within rh
        # to the global properties.
        # This statement is needed otherwise mbh gets nan values and cannot
        # give correct results.
        psi = (self.a0 + self.a11 * (self.a12)**(gamma - 1)
               * abs(fbh)**self.b1 * (mbh / mav)**((gamma - 1) * self.b2))

        # Complete expression if we include the contribution from stars as well
        # We have self.a0=1.
        # psi = (self.a0 * (mst / mav)**(self. b0 * (gamma - 1))
        #        * (1 - self. a11 * fbh**self.b1)
        #        + self.a11 * self.a12**(gamma - 1)
        #        * fbh**self.b1 * (mbh / mav)**(self.b2 * (gamma - 1)))

        return psi

    def _N(self, M, fbh, mbh, t):
        '''
        Number of particles, approximately stars. We write it as a function of
        the total mass and the black hole fraction. The number of black holes
        is a minor correction since typically 1 star in 1e3 becomes a black
        hole, so clusters with 1e6 have 1e3 black holes roughly speaking, a
        small correction.
        '''
        # Varying average mass for stars. Mass of BHs is constant for now.
        Np = M * ((1 - fbh) / self._mst(t))

        # Include BHs if we have them
        # Np += M * fbh / mbh if (M * fbh < mbh) else 0

        return Np

    def _trh(self, M, rh, fbh, mbh, t):
        '''
        Relaxation as defined by Spitzer. Here we consider the effect of mass
        spectrum due to black holes.
        '''
        Np = self._N(M, fbh, mbh, t)

        mav = M / Np

        if M > 0 and rh > 0:
            return (0.138 * np.sqrt(M * rh ** 3 / self.G)
                    / (mav * self._psi(fbh, M, mbh, t)
                       * np.log(self.gamma * Np)))
        else:
            return 1e-99

    def _trhstar(self, M, rh, fbh, mbh, t):
        '''Relaxation for stars depending on the assumptions.
        Here we include only one assumption.'''

        # Half mass relaxation.
        #  trh = self._trh(M, rh, fbh, mbh, t)

        # Average mass.
        Np = self._N(M, fbh, mbh, t)
        mav = M / Np

        # Relaxation for evaporation
        if M > 0 and rh > 0:

            trhstar = (0.138 * np.sqrt(M * rh ** 3 / self.G)
                       / (mav * np.log(self.gamma * Np)))
        else:
            trhstar = 1e-99

        # trhstar = trhstar if (self.two_relaxations) else trh
        return trhstar

    def _tcr(self, M, rh):
        '''Crossing time within the half-mass radius in Myrs.'''

        # Prefactor derived from the expression tcr = 1 / sqrt(G rhoh).
        # If the numerator is not 1, it needs to be changed.
        k = 2 * np.sqrt(2 * np.pi / 3) * self.tcross_index

        if M > 0 and rh > 0:
            return k * np.sqrt(rh**3 / (self.G * M))

        else:
            return 1e-99

    def _xi(self, rh, rt):
        '''Tides.
        Here one could perhaps approach tides differently, perhaps with an
        exponential function.'''
        # Power-law for generality.
        xi = 3 * self.zeta / 5 * (rh / rt / self.Rht)**self.n
        return xi

    def odes(self, t, y):
        '''The differential equations to be solved.'''

        Mst = y[0]  # Mass of stars.
        Mbh = y[1]  # Mass of black holes.
        rh = y[2]  # Half mass radius.

        # Mval = y[3]  # Parameter for mass segregation.
        Mval = self.Mval0

        # Total mass of the cluster.
        # It overestimates a bit initially because we assume Mbh>0 from start.
        M = Mst + Mbh

        # Fraction of black holes.
        # For the same reason it whould be considered after core collapse.
        fbh = Mbh / M

        tcc = self.tcc   # Core collapse.
        tsev = self.tsev  # Stellar evolution.
        tbh = self.t_bhcreation  # Time instance when black holes are created.

        # mst = self._mst(t)  # Average stellar mass
        mbh = self._mbh(Mbh)  # Extract the average BH mass
        if Mbh < mbh:
            Mbh = 0

        rt = self._rt(M)
        trh = self._trh(M, rh, fbh, mbh, t)  # Relaxation
        trhstar = self._trhstar(M, rh, fbh, mbh, t)  # Evaporation time scale
        # tcr = self._tcr(M, rh)  # Crossing time.
        alpha_c = 0

        # At first the derivatives are set equal to zero, then we build them up.
        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0
        # Mval_dot = 0

        # This parameter described mass segregation, used in the equation
        # for rh. The reason why it changes to a constant value after core
        # collapse is because of Henon's statement, here described by
        # parameter zeta.
        Mval = self.Mval0 if t < tcc else self.Mval_cc

        Mst_dotsev = 0

        # Stellar mass loss.

        if t >= tsev:

            # Stars lose mass due to stellar evolution.
            Mst_dotsev -= self.nu * Mst / t

            # The cluster expands for this reason. It is because we assume a
            # uniform distribution initially. With mass segregation up until
            # core collapse, we should get a better description.
            rh_dot -= (Mval - 2) * Mst_dotsev / M * rh
            # Evolution of parameter describing mass segregation
            # Mval_dot += Mval * (self.Mvalf - Mval) / trh

        if tcc < t < tbh:

            # Reduce Mbh because heavy stars are ejected. This must happen if
            # core collapse occurs due to stars.
            Mbh_dot -= self.alpha_c * M / trh

        # Add tidal mass loss.

        if (self.tidal):  # Check if we have tides.

            xi = self._xi(rh, rt)  # Tides.

            # Check if we have finite escape time from the Lagrange points.
            # if self.finite_escape_time:
            #     xi *= self.p * (trhstar / tcr) ** (1 - self.x)

            # Central stellar ejection rate. If we were to include ejections
            # before tcc, this expression needs to be modified.
            # if t >= tcc:
            #     if self.running_stellar_ejection_rate:
            #         alpha_c = self.alpha_c * (1 - 5 * xi / (3 * self.zeta))
            #     else:
            #         alpha_c = self.alpha_c

            # Mass loss rate of stars due to tides (and central ejections).
            # Mst_dot -= xi * Mst / trhstar + alpha_c * self.zeta * M / trh
            Mst_dot -= xi * Mst / trhstar + self.alpha_c * self.zeta * M / trh

        # Case of isolated cluster, it is described by central ejections only,
        # if present
        else:
            alpha_c += self.alpha_c if t >= tcc else 0
            Mst_dot -= alpha_c * self.zeta * M / trh

        # Impact of tidal mass loss to the size of the cluster.
        rh_dot += 2 * Mst_dot / M * rh

        # Correct the total stellar mass loss rate.
        Mst_dot += Mst_dotsev

        # Expansion due to energy loss. It is due to the fact that the cluster
        # has a negative heat capacity due to its gravitational nature.
        rh_dot += self.zeta * rh / trh if t >= tcc else 0

        # Effect of evaporating stars on the half mass radius.
        # Keep only evaporation, not ejections here so use xi and not xi_total.
        # # If escapers carry negative energy as they leave, the half mass
        # radius is expected to increase since it is similar to emitting
        # positive energy. The effect is proportional to tides.
        # if self.escapers:
        #     rh_dot += 6 * xi / self.r / trhstar * (1 - self.kin) * rh**2 / rt

        Mbh_dot_alpha = -alpha_c * self.zeta * M / trh if tcc <= t < tbh else 0

        # Check if we have BHs so that we can evolve them as well.
        if Mbh > 0 and t >= tcc:
            # Normalising with q0 and f0 should give the same BH ejection
            # initially for all clusters but then it evolves.

            # Varying ejection rate of black holes.
            # It becomes more difficult to eject them as fbh tends to zero.
            # Exponents may need to change if we use the properties within rh.
            # if self.running_bh_ejection_rate and (0 < fbh):
            #     beta = (self.beta * (abs(fbh) / self.fcc)**self.b3
            #             * (self.q0 * mbh / mst)**self.b4)
            # else:
            #     beta = self.beta

            # Ejection of BHs each relaxation
            # s = (fbh / 0.006)**0.1 if fbh < 0.006 else (0.006 / fbh)**0.064
            Mbh_dot -= self.beta * self.zeta * M / trh  # * s
            # If the tidal field was important, an additional
            # mass-loss mechanism would be needed.

            # Contraction since BHs are removed.
            rh_dot += 2 * Mbh_dot / M * rh

        # This happens because alpha_c, if present, participates already in rh,
        # it is not needed twice in the formalism, from stars and from BHs.
        Mbh_dot += Mbh_dot_alpha

        derivs = np.array([Mst_dot, Mbh_dot, rh_dot])

        # Return the derivatives in an array.
        return derivs

    # Extract the solution using the above differential equations.
    def evolve(self, N, rhoh):
        Mst = [self.M0]  # Initial Stellar mass.
        Mbh = [self.Mbh0]  # Initial black hole mass.
        rh = [self.rh0]  # Initial half-mass radius.
        # Mval = [self.Mval0]  # Initial parameter for mass segregation.

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

        self.t = np.array([x / 1e3 for x in sol.t])  # Time in Gyrs.
        self.Mst = sol.y[0]  # Stellar mass.
        self.Mbh = sol.y[1]  # Black hole mass.
        self.rh = sol.y[2]  # Half mass radius

        self.mbh = np.array([self._mbh(x) if y > self.tcc else self.mbh0
                             for (x, y) in zip(self.Mbh, sol.t)])

        self.Nbh = self.Mbh / self.mbh

        cbh = (self.Mbh < self.mbh)  # Where no black holes

        self.Mbh[cbh] = 0.
        self.mbh[cbh] = 0.
        self.Nbh[cbh] = 0.

        self.M = self.Mst + self.Mbh  # Total mass of the cluster.
        self.rt = self._rt(self.M)  # Tidal radius.
        self.fbh = self.Mbh / self.M  # BH fraction.

        # Friction term ψ.
        self.psi = np.array([self._psi(x, y, z, u) for (x, y, z, u)
                             in zip(self.fbh, self.M, self.mbh, sol.t)])

        # Average stellar mass over time.
        self.mst_sev = np.array([self._mst(x) for x in sol.t])

        # Number of components.
        self.Np = np.array([self._N(x, y, z, u) for (x, y, z, u)
                            in zip(self.M, self.fbh, self.mbh, sol.t)])

        # Average mass of cluster over time, includes BHs.
        # No significant change is expected given that Nbh <= O(1e3),
        # apart from the beginning where the difference is a few percent.
        self.mav = self.M / self.Np

        # Energy of the cluster at each time instance.
        self.E = -self.r * self.G * self.M ** 2 / (4 * self.rh)

        # Virial radius as a function of time. It is of course a constant
        # fraction of rh since r does not vary in this approach.
        self.rv = -self. G * self. M ** 2 / (4 * self.E)

        # Relaxation.
        self.trh = np.array(
            [self._trh(x, y, z, u, v) for (x, y, z, u, v)
             in zip(self.M, self.rh, self.fbh, self.mbh, sol.t)]
        )
        self.trhstar = np.array(
            [self._trhstar(x, y, z, u, v) for (x, y, z, u, v)
             in zip(self.M, self.rh, self.fbh, self.mbh, sol.t)]
        )
