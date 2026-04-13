"""
Code to compute the cooling coefficient for blackbody radiation 
around a binary with given mass and semi-major axis
"""

from logging import getLogger
from typing import NamedTuple
from math import sqrt

pi = 3.14159265359
cgs = dict(
		G = 6.6725985e-8,
		c = 2.99792458e10,
		kb = 1.38065812e-16,
		h = 6.62607015e-27, # erg s
		sigmab = 5.6705119e-5,
		mp = 1.6726e-24,
		kappa = 0.4,            # electron scattering
		pc = 3.085678e18,
		msun = 1.989e33,
		yr = 3.254e7,
	)

logger = getLogger(__name__)

class ShakuraSunyaevDisk(NamedTuple):
	"""The central mass, length scale (e.g. binary separation), alpha, and mach number (at r=3a)
	   for a Shakura-Sunyaev alpha-disk (Frank, King, & Raine 2002) 
		- for numerical reasons the disk Mach number at r=3a is supplied (as oppoed to the
		  accretion rate), and mdot as a fraction of the eddington rate is determined 
		  accordingly
	"""

	central_mass_msun : float
	length_scale_pc   : float
	mach_number_a     : float
	alpha             : float
	gamma             : float
	# fix_fedd          : int

	# -------------------------------------------------------------------------
	@property
	def _mass(self) -> float:
		return self.central_mass_msun * cgs['msun']

	@property
	def _length(self) -> float:
		return self.length_scale_pc * cgs['pc']
	
	@property
	def _time(self) -> float:
		return sqrt(self._length**3 / self._GM)  # Omega_b^-1

	@property
	def _GM(self) -> float:
		return self._mass * cgs['G']

	@property
	def _rschwz(self):
		return 2 * self._GM / cgs['c']**2

	@property
	def _accretion_efficiency(self):
		return 0.1
	
	@property
	def _eddington_rate(self) -> float:
		return 4 * pi * self._GM / cgs['kappa'] / cgs['c'] / self._accretion_efficiency

	@property
	def _eddington_fraction(self) -> float:
		"""Calculated for provided Mach number at r=3a

		   f_edd = 4pi * sqrt(2/3) * (mp^4 / kb^4 * sigmab / kappa * alpha)^(1/2) * (GM)^(7/4) * Mach(r)^-5 * r^(-1/4) / Mdot_edd
		"""
		f0 = 10.2604 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**0.5 * self.gamma**(-2.)
		return f0 * self.alpha**0.5 * self._GM**(7./4.) * self._length**(-1./4.) * self.mach_number_a**-5 / self._eddington_rate
	
	@property
	def _accretion_rate(self) -> float:
		# if self.fix_fedd > 0.0:
		# 	return self.fix_fedd * self._eddington_rate
		# else:
		# 	return self._eddington_fraction * self._eddington_rate
		return self._eddington_fraction * self._eddington_rate

	@property
	def _surface_density(self) -> float:
		"""Disk surface density scaling in cgs

		   Sigma = (32 / 3^6 / pi^3)^(1/5) * (mp^4 / kb^4 * sigmab / kappa)^(1/5) 
		   			* alpha^(-4/5) * (GM)^(1/5) * Mdot^(3/5) * r^(-3/5)
		"""
		s0 = 0.269274 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.) * self.gamma**(-4./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)

	@property
	def _surface_pressure(self) -> float:
		"""Disk pressure scaling in cgs

		   P = (1 / 3 / pi) * alpha^-1 * Mdot * (GM)^0.5 * r^(-3/2)
		"""
		return 0.106103 / self.gamma / self.alpha * self._accretion_rate * sqrt(self._GM) * self._length**(-3./2.)

	@property
	def _midplane_temperature(self) -> float:
		"""Disk midplane temperature scaling in cgs (for completeness)

		   T = (3 / 32 / pi^2)^(1/5) * (mp * kappa / kb / sigmab)^(1/5) 
		        * alpha^(-1/5) * (GM)^(3/10) * Mdot^(2/5) * r^(-9/10)
		"""
		t0 = 0.394035 * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.) * self.gamma**(-1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	@property
	def _mach_number(self) -> float:
		"""Disk Mach number profile (dimensionless)

		   Mach = 2^(1/2) * (pi^2 / 3)^(1/10) * (mp / kb)^(2/5) * (sigmab / kappa)^(1/10) * alpha^(1/10) * (GM)^(7/20) * Mdot^(-1/5) * r^(1/20)
		"""
		m0 = 1.593063 * (cgs['mp'] / cgs['kb'])**(2./5.) * (cgs['sigmab'] / cgs['kappa'])**(1./10.) * self.gamma**(-2./5.)
		return m0 * self.alpha**(1./10.) * self._GM**(7./20.) * self._accretion_rate**(-1./5.) * self._length**(-1./20.)
	

	# -------------------------------------------------------------------------
	@property
	def surface_density_coefficient(self) -> float:
		return self._surface_density / (self._mass / self._length**2)

	@property
	def surface_pressure_coefficient(self) -> float:
		return self._surface_pressure / (self._mass / self._time**2)	

	def surface_density_profile(self, r:float) -> float:
		return self.surface_density_coefficient * r**(-3./5.)

	def surface_pressure_profile(self, r:float) -> float:
		return self.surface_pressure_coefficient * r**(-3./2.)

	def optical_depth(self, r:float) -> float:
		return cgs['kappa'] * self._surface_density * r**(-3./5.)

	def mach_number_profile(self, r:float) -> float:
		return self._mach_number * r**(-1./20.)

	def effective_temperature_profile(self, r:float) -> float:
		return (3. * self._GM * self._accretion_rate / (8. * pi * cgs['sigmab']))**(1./4.) * (r * self._length)**(-3./4.)

	def viscous_time(self, r:float) -> float:
		mach = self.mach_number_profile(r)
		return 2. / 3 * mach**2 / self.alpha * r**(3./2.)
		# cs0 = self.gamma * self.surface_pressure_coefficient / self.surface_density_coefficient
		# return 2. / 3. / self.alpha / cs0 * r**(7./5.)

	# -------------------------------------------------------------------------
	@property
	def opacity(self):
		""" Give the gas opcaity in code units 

		    - right now only for electron scattering
		"""
		return cgs['kappa'] / (self._length**2 / self._mass)
	

	def cooling_coefficient(self) -> float:
		"""Assumes avg fluid particle mass is the proton mass
	
				P / Sigma = eps * (gamma - 1)
	
				deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4
	
				eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 
	
				cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb)^4 * (gamma - 1)
		"""
		mp_code = cgs['mp'] /  self._mass
		kb_code = cgs['kb'] / (self._mass * self._length**2 / self._time**2)
		# kappa_code  = cgs['kappa'] / (self._length**2 / self._mass)
		sigmab_code = cgs['sigmab'] / (self._mass / self._time**3)	
		qdot_coeff = 8. / 3. * sigmab_code / self.opacity * (mp_code / kb_code)**4 * (self.gamma - 1.)**4
		logger.info(f"density coefficient : {self.surface_density_coefficient:0.2e}")
		logger.info(f"pressure coefficient : {self.surface_pressure_coefficient:0.2e}")
		logger.info(f"implied eddington fraction : {self._eddington_fraction:0.2e}")
		logger.info(f"approximate optical depth : {self.optical_depth(1.):0.4f}")
		logger.info(f"cooling coefficient : {qdot_coeff:0.2e}")
		return qdot_coeff


# =============================================================================
if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	r = np.linspace(0.5, 10., 250)
	ss = ShakuraSunyaevDisk(
        	central_mass_msun=8e6, 
        	length_scale_pc=9.7e-4,
        	mach_number_a=21,
        	alpha=0.1,
        	gamma=5./3.,
        )
	print("fedd : ", ss._eddington_fraction)

	fcavity = 0.0001 + 0.9999 * np.exp(-((1.0 / r) ** 30))
	fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, figsize=[5,8])
	ax1.plot(r, ss.surface_density_profile(r) * fcavity, c='C0')
	ax1.plot(r, ss.surface_density_goodman() / (ss._mass / ss._length**2) * r**(-3./5.) * fcavity, c='C1', ls='--')
	ax1.plot(r, 0.057 * r**(-3./5.) * fcavity, c='C3')

	ax2.plot(r, ss.surface_pressure_profile(r) * fcavity, c='C0')
	ax2.plot(r, ss.surface_pressure_goodman() / (ss._mass / ss._time**2) * r**(-3./2.) * fcavity, c='C1', ls='--')
	ax2.plot(r, pi * 6.7e-5 * r**(-3./2.) * fcavity, c='C3')

	ax3.plot(r, ss.surface_pressure_profile(r) / ss.surface_density_profile(r), c='C0')
	ax3.plot(r, 6.7e-5 * r**(-3./2.) / (0.057 * r**(-3./5.)), c='C3')

	ax1.set_ylabel(r'$\Sigma$')
	ax2.set_ylabel(r'$P$')
	ax3.set_ylabel(r'$c_s^2$')
	ax3.set_xlabel(r'$r$')

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.1)
	plt.show()


