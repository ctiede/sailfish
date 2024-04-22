"""
Code to compute the cooling coefficient for blackbody radiation 
around a binary with given mass and semi-major axis
"""

from typing import NamedTuple
from math import sqrt

pi = 3.14159265359
cgs = dict(
		G = 6.6725985e-8,
		c = 2.99792458e10,
		kb = 1.38065812e-16,
		sigmab = 5.6705119e-5,
		mp = 1.6726e-24,
		kappa = 0.4,            # electron scattering
		pc = 3.085678e18,
		msun = 1.989e33,
	)

class ShakuraSunyaevDisk(NamedTuple):
	"""The central mass, length scale (e.g. binary separation), alpha, and mach number (at r=3a)
	   for a Shakura-Sunyaev alpha-disk (Frank, King, & Raine 2002) 
		- for numerical reasons the disk Mach number at r=3a is supplied (as oppoed to the
		  accretion rate), and mdot as a fraction of the eddington rate is determined 
		  accordingly
	"""

	central_mass_msun : float
	length_scale_pc   : float
	mach_number_3a    : float
	alpha             : float

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
	def _accretion_efficiency(self):
		return 0.1
	
	@property
	def _eddington_rate(self) -> float:
		return 4 * pi * self._GM / cgs['kappa'] / cgs['c'] / self._accretion_efficiency

	@property
	def _eddington_fraction(self) -> float:
		"""Calculated for provided Mach number at r=3a

		   f_edd = 10.26 * (mp^4 / kb^4 * sigmab / kappa * alpha)^(1/2) * (GM)^(7/4) * Mach(r)^-5 * r^(-1/4) / Mdot_edd
		"""
		f0 = 10.2604 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**0.5
		rm = 3 * self._length
		return (f0 * self.alpha**0.5 * self._GM**(7./4.) * rm**(-1./4.) * self.mach_number_3a**-5
		           / self._eddington_rate)
	
	@property
	def _accretion_rate(self) -> float:
		return self._eddington_fraction * self._eddington_rate

	@property
	def _surface_density(self) -> float:
		"""Disk surface density scaling in cgs

		   Sigma = (32 * 3^6 / pi^3)^(1/5) * (mp^4 / kb^4 * sigmab / kappa)^(1/5) 
		   			* alpha^(-4/5) * (GM)^(1/5) * Mdot^(3/5) * r^(-3/5)
		"""
		s0 = 0.269274 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)

	@property
	def _surface_pressure(self) -> float:
		"""Disk pressure scaling in cgs

		   P = (1 / 3 / pi) * alpha^-1 * Mdot * (GM)^0.5 * r^(-3/2)
		"""
		return 0.106103 / self.alpha * self._accretion_rate * sqrt(self._GM) * self._length**(-3./2.)

	@property
	def _midplane_temperature(self) -> float:
		"""Disk midplane temperature scaling in cgs (for completeness)

		   T = (3 / 32 / pi^2)^(1/5) * (mp * kappa / kb / sigmab)^(1/5) 
		        * alpha^(-1/5) * (GM)^(3/10) * Mdot^(2/5) * r^(-9/10)
		"""
		t0 = 0.394035 * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	# -------------------------------------------------------------------------
	@property
	def surface_density_coefficient(self) -> float:
		return self._surface_density / (self._mass / self._length**2)

	@property
	def surface_pressure_coefficient(self, r:float) -> float:
		return self._surface_pressure / (self._mass / self._time**2)

	def surface_density_profile(self, r:float) -> float:
		return self.surface_density_coefficient * r**(-3./5.)

	def surface_pressure_profile(self, r:float) -> float:
		return self.surface_pressure_coefficient * r**(-3./2.)

	# -------------------------------------------------------------------------
	def cooling_coefficient(self, gamma:float=5./3.) -> float:
		"""Assumes avg fluid particle mass is the proton mass
	
				P / Sigma = eps * (gamma - 1)
	
				deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4
	
				eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 
	
				cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb) * (gamma - 1)
		"""
		mp_code = cgs['mp'] /  self._mass
		kb_code = cgs['kb'] / (self._mass * self._length**2 / self._time**2)
		kappa_code  = cgs['kappa'] / (self._length**2 / self._mass)
		sigmab_code = cgs['sigmab'] / (self._mass / self._time**3)	
		return 8. / 3. * sigmab_code * (mp_code / kb_code)**4 * (gamma - 1.)**4 / kappa_code

	# Only for temporary testing
	# =============================================================================
	def surface_density_goodman(self):
		coeff = 2**(4./5.) / 3. / pi**(3./5.)
		s0 = coeff * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)

	def midplane_temperature_goodman(self):
		coeff = (1. / 16. / pi**2)**(1./5.)
		t0 = coeff * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	def surface_pressure_goodman(self):
		return cgs['kb'] / cgs['mp'] * self.midplane_temperature_goodman() * self.surface_density_goodman()


# =============================================================================
# class ShakuraSunyaevDisk(NamedTuple):
# 	"""The central mass, length scale (e.g. binary separation), alpha, and mach number (at r=3a)
# 	   for a Shakura-Sunyaev alpha-disk (Frank, King, & Raine 2002) 
# 		- for numerical reasons the disk Mach number at r=3a is supplied (as oppoed to the
# 		  accretion rate), and mdot as a fraction of the eddington rate is determined 
# 		  accordingly
# 	"""

# 	central_mass : float
# 	length_scale : float
# 	mach_number  : float
# 	alpha        : float

# 	# TODO : Can I do this with namedtuple? Or do I need a more clever way?
# 	#  - create a separate object for storing physical constants in code units?
# 	# -------------------------------------------------------------------------
# 	a = length_scale * cgs['pc']
# 	m = central_mass * cgs['msun']
# 	t = sqrt(a**3 / m / cgs['G'])
# 	G_code  = cgs['G'] / (self.a**3 / self.m / self.t**2)
# 	c_code  = cgs['c'] / (self.a / self.t)
# 	mp_code = cgs['mp'] /  self.m
# 	kb_code = cgs['kb'] / (self.m * self.a**2 / self.t**2)
# 	kappa_code  = cgs['kappa'] / (self.a**2 / self.m)
# 	sigmab_code = cgs['sigmab'] / (self.m / self.t**3)	
# 	eddington_rate_code = 4 * pi * self.G_code * M / self.kappa_code / self.c_code / accretion_efficiency

# 	# -------------------------------------------------------------------------
# 	@property
# 	def accretion_rate(self):
# 		return self.eddington_fraction * self.eddington_rate_code

# 	@property
# 	def eddington_fraction(self):
# 		"""Calculated for provided Mach number at r=3a"""
# 		f0 = 10.2604 * (mp_code**4 / kb_code**4 * sigmab_code / kappa_code)**0.5
# 		mach_at_3a = 3**(-1./4.) * self.mach_number**-5
# 		return f0 * self.alpha**0.5 * (G_code * M)**(7./4.) / self.eddington_rate_code * mach_at_3a

# 	# -------------------------------------------------------------------------
# 	def surface_density(self, r:float) -> float:
# 		# 0.269274 = (32 * 3^6 / pi^3)^(1/5)
# 		s0 = 0.269274 * (mp_code**4 / kb_code**4 * sigmab_code / kappa_code)**(1./5.)
# 		return s0 * self.alpha**(-4./5.) * (G_code * M)**(1./5.) * self.accretion_rate**(3./5.) * r**(-3./5.)

# 	def surface_pressure(self, r:float) -> float:
# 		# 0.106103 = 1 / 3 / pi
# 		return 0.106103 / self.alpha * self.accretion_rate * sqrt(G_code * M) * r**(-3./2.)

# 	def midplane_temperature(self, r:float) -> float:
# 		# 0.394035 = (3 / 32 / pi^2)^(1/5)
# 		t0 = 0.394035 * (mp_code * kappa_code / kb_code / sigmab_code)**(1./5.)
# 		return t0 * self.alpha**(-1./5.) * (G_code * M)**(3./10.) * self.accretion_rate**(2./5.) * r**(-9./10.)

# 	# -------------------------------------------------------------------------
# 	def compute_cooling_coefficient(gamma:float=5./3.) -> float:
# 		"""Assumes avg fluid particle mass is the proton mass
	
# 				P / Sigma = eps * (gamma - 1)
	
# 				deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4
	
# 				eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 
	
# 				cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb) * (gamma - 1)
# 		"""
# 		return 8. / 3. * self.sigmab_code * (self.mp_code / self.kb_code)**4 * (gamma - 1.)**4 / self.kappa_code

# =============================================================================

# G_cgs         = 6.6725985e-8
# c_cgs         = 2.99792458e10
# kb_cgs        = 1.38065812e-16
# sigmab_cgs    = 5.6705119e-5
# mp_cgs        = 1.6726e-24
# kappa_cgs     = 0.4            # electron scattering
# pc_to_cm      = 3.085678e18
# msun_to_grams = 1.989e33
# pi            = 3.14159265359

# def compute_cooling_coefficient(M_sun, a_pc, gamma=5./3.):
# 	"""Assumes avg fluid particle mass is the proton mass

# 		P / Sigma = eps * (gamma - 1)

# 		deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4

# 		eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 

# 		cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb) * (gamma - 1)
# 	"""
# 	a = a_pc * pc_to_cm
# 	m = M_sun * msun_to_grams
# 	t = sqrt(a**3 / m / G_cgs)
# 	mp = mp_cgs /  m
# 	kb = kb_cgs / (m * a**2 / t**2)
# 	kappa  = kappa_cgs / (a**2 / m)
# 	sigmab = sigmab_cgs / (m / t**3)
# 	return 8. / 3. * sigmab * (mp / kb)**4 * (gamma - 1.)**4 / kappa

if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	r = np.linspace(0.5, 20., 250)
	ss = ShakuraSunyaevDisk(
        	central_mass_msun=8e6, 
        	length_scale_pc=9.7e-4,
        	mach_number_3a=21.0,
        	alpha=0.1,
        )
	print("fedd : ", ss._eddington_fraction)
	fcavity = 0.0001 + 0.9999 * np.exp(-((1.0 / r) ** 30))

	fig, [ax1, ax2, ax3] = plt.subplots(3, 1)
	ax1.plot(r, ss.surface_density_profile(r) * fcavity, c='C0')
	ax1.plot(r, ss.surface_density_goodman() / (ss._mass / ss._length**2) * r**(-3./5.) * fcavity, c='C1', ls='--')
	ax1.plot(r, 0.057 * r**(-3./5.) * fcavity, c='C3')

	ax2.plot(r, ss.surface_pressure_profile(r) * fcavity, c='C0')
	ax2.plot(r, ss.surface_pressure_goodman() / (ss._mass / ss._time**2) * r**(-3./2.) * fcavity, c='C1', ls='--')
	ax2.plot(r, 6.7e-5 * r**(-3./2.) * fcavity, c='C3')

	ax3.plot(r, ss.surface_pressure_profile(r) / ss.surface_density_profile(r), c='C0')
	ax3.plot(r, 6.7e-5 * r**(-3./2.) / (0.057 * r**(-3./5.)), c='C3')

	ax2.set_xlabel(r'$r$')
	ax1.set_ylabel(r'$\Sigma$')
	ax2.set_ylabel(r'$P$')
	ax3.set_ylabel(r'$c_s^2$')

	plt.show()


