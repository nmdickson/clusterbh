from __future__ import division
import numpy
import pylab as plt
from pylab import log, sqrt, pi, exp, log10
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1
from ssptools import evolve_mf



class clusterBH:
    def __init__(self, N, rhoh, **kwargs):
        self.G  = 0.004499 # pc^3 /Msun /Myr^2

        # Cluster ICs
        self.m0 = 0.606  # For Kroupa (2001) IMF 0.06-150 Msun
        self.N = N # Initial number of stars.
        self.fc = 1 # equation (50)
        self.rg = 8 # [kpc]
        self.Z = 0.0002 # Metalicity. 
        self.mbh = 20 # Average mass of black holes in solar masses.
        self.f = 1 # Prefactor that relates relaxation of stars with the half mass relaxation.
        self.Mval = 1.5 # Contribution of stellar evolution in half mass radius after core collapse.
        
        # Model parameters
        self.zeta = 0.1 # Energy loss per half mass relaxation.
        self.a0 = 1 # Fix zeroth order in ψ.
        self.a2 = 0 # Ignore 2nd order in ψ for now.
        self.kick = True # We include natal kicks.     
        self.tsev = 2 # In Myrs. Time instance when stars start evolving.
        self.n = 1.5 # Exponent in the power-law for tides.
        self.r = 1. # Ratio of the half mass radius over the Virial radius initially.
        self.alpha_c = 0.0065 # 0.65% Evaporation of stars initially, approximating tides.
        self.gamma = 0.02 # Parameter of Coulomb logarithm as used in the cmc models.
        self.kin = 0.9 # Kinetic term of escaping stars.
        
        # Parameters that were fit to N-body.
        self.ntrh = 3.21 # Number of initial relaxations in order to compute core collapse instance.
        self.beta = 0.0028 # Ejection rate of black holes from the core per relaxation.
        self.nu = 0.0823 # Mass loss rate of stars.
        self.a1 = 1.47 # Linear order in ψ.
     
        # BHMF
        
        self.alpha = -0.5
        self.mlo = 5
        self.mup = 50
        
        
        self.sigmans = 265 # km/s # Velocity dispersion.
        self.mns = 1.4 # Msun. Mass of Neutron stars

        # Some integration params.
        self.tend = 13.8e3 # Final time instance where we integrate to.
        self.dtout = 2 # Myr. Time step for integration.
        self.Mst_min = 100 # [Msun] Stop criterion.
        self.integration_method = "RK45" # Integration method.
        
        self.output = False # Just a string in order to save the results.
        self.outfile = "cluster.txt" # File to save the results if needed.

    
        # Mass loss mechanism
        self.tidal = True # We activate tides.
        self.escapers = False # Condition for escapers to carry negative energy as they escape the cluster due to a tidal field.
        self.Rht = 0.125 # Ratio of rh/rt to give correct Mdot [17/3/22]. It is not necessarily the final value of rh / rt.
        self.Vc = 220. # [km/s] circular velocity of singular isothermal galaxy
       
        
        # Natal Kicks.
        self.a_slope1 = -1.3 # Slope of mass function for the first interval.
        self.a_slope2 = -2.3 # Slope of mass function for the second interval.
        self.a_slope3 = -2.3 # Slope of mass function for the third interval.
        self.m_break1 = 0.08 # Lowest stellar mass of the cluster.
        self.m_break2 = 0.5 # Highest stellar mass in the first interval.
        self.m_break3 = 1. # Highest stellar mass in the second interval.
        self.m_break4 = 150. # Highest mass in the cluster.
        self.nbin1 = 5
        self.nbin2 = 5
        self.nbin3 = 20
   
        
        # Check input parameters. Afterwards we start computations.
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
            
                
       

        self.a_slopes = [self.a_slope1, self.a_slope2, self.a_slope3]  # 3-component IMF slopes.
        self.m_breaks = [self.m_break1, self.m_break2, self.m_break3, self.m_break4]  # Break masses.
        self.nbins = [self.nbin1, self.nbin2, self.nbin3]  # Number of bins.
        
        self.FeH = log10(self.Z / 0.014) # This is log10(Z / Zsolar).
        
        self.M0 = self.m0 * N # Total mass of stars (cluster) initially.
        self.rh0 = (3 * self.M0 / (8 * pi * rhoh)) ** (1./3) # Half mass radius initially.
        
        self.vesc0 = 50 * (self.M0 / 1e5) ** (1./3) * (rhoh / 1e5) ** (1./6) # Initial escape velocity as a function of mass and density.
                        
        self.vesc0 *= self.fc # Augment the value for different King models.

        # Implement kicks for this IMF with such metallicity.
        self.ibh = evolve_mf.InitialBHPopulation.from_IMF(self.m_breaks, self.a_slopes, self.nbins, self.FeH, N0=self.N, vesc=self.vesc0, natal_kicks=self.kick)
        self.Mbh0 = self.ibh.Mtot # Expected initial mass of black holes due to kicks.
       
        self.f0 = self.Mbh0 / self.M0 # Initial fraction of black holes. It should be close to 0.05 for poor metal clusters. 
        
        
        if (self.kick): # Check if we have kicks so that we can fix mb and compute the upper mass of black holes at each time instance.
            mb = (9 * pi / 2) ** (1./6) * self.sigmans * self.mns / self.vesc0
            self.mb = mb
            
        
        self.Nbh0 = numpy.round(self.ibh.Ntot) # Initial number of black holes.
        self.Mst_lost = self.ibh.Ms_lost # Mass of stars lost in order to form black holes.
        self.t_bhcreation = self.ibh.age # Time in Myrs that is required in order to form these black holes.

        self.trh0 = self._trh(self.M0 + self.Mbh0, self.rh0, self.Mbh0 / (self.Mbh0 + self.M0), 0) # Initial relaxation in Myrs. Because we start with black holes, they are included here so that trh is continuous. It is a small error which is practically absorbed in ntrh and corects itself after a few Myrs.
        self.tcc = self.ntrh * self.trh0 # Core collapse in Myrs.

        self.evolve(N, rhoh)

    def _rt(self, M): # Tidal radius.
        O2 = (self.Vc * 1.023 / (self.rg * 1e3)) ** 2 # Angular velocity squared.
        return (self.G * M / (2 * O2)) ** (1./3)

    # Average mass of stars, due to stellar evolution. It takes as an input either a value or an array for simplicity. This is really sloppy.    
    def _mst(self, t): # Does not take into consideration the effect of tides.
        return self.m0 * (t / self.tsev) ** (- self.nu) if t > self.tsev else self.m0
        
        
    # Friction term ψ in the relaxation. It is characteristic of mass spectrum, here due to black holes. We neglect O(2) corrections.
    def _psi(self, fbh): 
        psi = self.a0  + self.a1 * abs(fbh) / 0.01
        return psi

    # Number of particles, approximately stars. We write it as a function of the total mass and the black hole fraction. The number of black holes is a minor correction since typically 1 star in 1e3 becomes a black hole, so clusters with 1e6 have 1e3 black holes roughly speaking, a small correction. If mbh is needed, one can add a term M * fbh / mbh.    
    def _N(self, M, fbh, t): 
        return numpy.round(M * ( (1 - fbh) / self._mst(t))).astype(int) # Varying average mass for stars. Mass of black holes is constant for now.

    # Relaxation as defined by Spitzer. Here we consider the effect of mass spectrum due to black holes. When they vanish, we get unity, which is not entirely true.
    def _trh(self, M, rh, fbh, t):
        Np = self._N(M, fbh, t)
        if M > 0 and rh > 0: 
            return 0.138 * sqrt(M * rh ** 3 / self.G) / (self._mst(t) * self._psi(fbh) * log(self.gamma * Np)) 
        else:
            return 1e-99
   
    # We find the maximum value of the black hole mass at each time instance. We assume that heavy black holes are the only ones ejected from the core, a decent approximation.
    def find_mmax(self, Mbh):
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
            mmax_ = numpy.linspace(self.mlo, self.mup, Np)
            qml, qmb, qlb  = mmax_ / self.mlo, mmax_ / self.mb, self.mlo / self.mb

            A = self.Mbh0 / integr(self.mup, self.mup / self.mb, qlb) 

            Mbh_ = A * integr(mmax_, qmb, qlb)
            mmax = numpy.interp(Mbh, Mbh_, mmax_)
        else:
            # eq 51 in AG20
            mmax = (Mbh / self.Mbh0 * (self.mup ** a2 - self.mlo ** a2) + self.mlo ** a2) ** (1./a2)

        # TBD: Set to 0 when MBH = 0
        return mmax

    def _logcheck(self, t, y):
        return 0 if (y[0] > self.Mst_min) else -1
    
    
    # We construct the differential equations to be solved. 
    def odes(self, t, y):
        Mst = y[0] # Mass of stars.
        Mbh = y[1] # Mass of black holes.
        rh = y[2] # Half mass radius.

        
        M = Mst + Mbh # Total mass of the cluster. It overestimates a bit initially because we assume Mbh>0 from the start.
        fbh = Mbh / M # Fraction of black holes. For the same reason it whould be considered after core collapse.
        
        rt = self._rt(M)
        trh = self._trh(M, rh, fbh, t) # Relaxation
        tcc = self.tcc  # Core collapse.
        tsev = self.tsev # Stellar evolution.
        Np = self._N(M, fbh, t) # Number of particles (essentially only stars if black holes are neglected).
        psi = self._psi(fbh) # Friction term.
        tbh = self.t_bhcreation # Time instance when black holes are created.
        trhstar = trh * self.f * psi # If not needed, change to trh.

        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0 # At first the derivatives are set equal to zero, then we build them up.
        
        
        M_val = 3 # if t < tcc else self.Mval # This would be the term for mass segregation. Here it is kept constant and equal to 3.
        
        Mst_dotsev = 0
       
        # Stellar mass loss.
        
        if t >= tsev:
            Mst_dotsev -= self.nu * Mst / t # Stars lose mass due to stellar evolution.
            rh_dot -= (M_val - 2) *  Mst_dotsev / M * rh  # The cluster expands for this reason. It is because we assume a uniform distribution initially. With mass segregation up until core collapse, we should get a better description.
            
       
        if tcc < t < tbh:
            Mbh_dot -= self.alpha_c * M / trh # Reduce Mbh because heavy stars are ejected. This must happen if core collapse occurs due to stars.
       
        
       
        # Add tidal mass loss.
        
        if (self.tidal): # Check if we have tides.
            
            xi = 0.6 * self.zeta * (rh / rt / self.Rht) ** self.n # Power-law for generality. The user may choose any form they wish.
            
            
            xi_total = xi + self.alpha_c if t > tcc else xi # Total stellar mass loss rate is the combination of ejections and tides. alpha_c is treated as ejections, not a constant evaporation rate, and thus it is a property of the cluster and not the tidal field.
           
            Mst_dot -= xi * M / trhstar + (xi_total - xi) * M / trh  # Mass loss rate of stars due to tides. We add this to the already known expression for mass-loss rate due to stellar evolution. 
            
        rh_dot += 2 * Mst_dot / M * rh   # Note that we now subtract 2ν from the +ν term already in Mst_dot. This is an approximation since heavy stars residing in a shell around the core should contribute. This is captured better if we use a parameter that describes mass segregation.
        
        Mst_dot += Mst_dotsev  # Correct the total stellar mass loss rate.   
       
        #Keep only evaporation, not ejections here so use xi and not xi_total.   
        rh_dot += 6 * xi / self.r / trhstar * (1 - self.kin) * rh ** 2 / rt if (self.escapers) else 0 # If escapers carry negative energy as they leave, the half mass radius is expected to increase since it is similar to emitting positive energy. The effect is proportional to tides.
        
        
        # BH escape. It occurs after core collapse, in the balanced phase of the cluster.
        
        rh_dot += self.zeta * rh / trh  if t >= tcc else 0 # Expansion due to energy loss. It is due to the fact that the cluster has a negative heat capacity due to its gravitational nature.
            
        if Mbh > 0 and t >= tcc: # Check if we have black holes so that we can evolve them as well.
           Mbh_dot -= self.beta * M / trh # Ejection each relaxation.
           rh_dot += 2 * Mbh_dot / M * rh # Contraction since black holes are removed.
        
        
        
        derivs = [Mst_dot]
        derivs.append(Mbh_dot)
        derivs.append(rh_dot)

        return numpy.array(derivs, dtype=object) # Return the derivatives in an array.

   # Extract the solution using the above differential equations.
    def evolve(self, N, rhoh): 
        Mst = [self.M0] # Stellar mass initially.
        Mbh = [self.Mbh0] # Initial black hole mass.
        rh = [self.rh0] # Initial half mass radius.
     

        y = [Mst[0], Mbh[0], rh[0]] # Combine them in a multivariable. 

        def Mst_min_event(t, y):  # Event in order to stop when stars are lost.
            return y[0] - self.Mst_min

        Mst_min_event.terminal = True # Find solutions as long as the event holds.
        Mst_min_event.direction = -1 # Stop when we find a t such that the event is no longer True.

        t_eval = numpy.arange(0, self.tend, self.dtout) if self.dtout is not None else None
        # Solution.
        sol = solve_ivp(self.odes, [0, self.tend], y, method=self.integration_method, t_eval=t_eval)

        self.t = [x / 1e3 for x in sol.t] # Time in Gyrs.
        self.t = numpy.array(self.t)
        self.Mst = sol.y[0] # Stellar mass.
        self.Mbh = sol.y[1] # Black hole mass.
        self.rh = sol.y[2] # Half mass radius
      
        
        
        
        cbh = (self.Mbh>0)# Condition to see where the mass of black holes is present.     
        self.mmax = numpy.zeros_like(self.Mbh)
        self.mmax[cbh] = self.find_mmax(self.Mbh[cbh]) # Upper mass of black holes in the mass function at each instance.
        
        # Some derived quantities.
        self.M = self.Mst + self.Mbh  # Total mass of the cluster. We include black holes already so we subtract the mass needed to make them... If the plots are bad, neglect this.
        self.rt = self._rt(self.M) # Tidal radius.
        self.fbh = self.Mbh / self.M # Black hole fraction.
        self.psi = self._psi(self.fbh) # Friction term ψ.
        self.mst_av = [self._mst(x) for x in sol.t]# List of stellar average mass over time. 
        self.Np = [self._N(x, y, z) for (x, y, z) in zip(self.M, self.fbh, sol.t)] # Number of components. 
        self.Np = numpy.array(self.Np)
        self.mav = self.M / self.Np # Average mass of cluster over time, includes black holes. No significant change is expected given that Nbh <= O(1e3).

        self.E = -self.r * self.G * self.M ** 2 / (4 * self.rh) # Energy of the cluster at each time instance. 
        self.rv = -self. G * self. M ** 2 / (4 * self.E) # Virial radius as a function of time. It is of course a constant fraction of rh since r does not vary in this approach.   
        self.trh = [self._trh(x, y, z, u) for (x, y, z, u) in zip(self.M, self.rh, self.fbh, sol.t)] # Relaxation.
        self.trh = numpy.array(self.trh)
        if (self.output):
            f = open(self.outfile,"w")
            for i in range(len(self.t)):
                f.write("%12.5e %12.5e %12.5e %12.5e \n"%(self.t[i], self.Mbh[i],
                                                                self.M[i], self.rh[i]))
            f.close()


