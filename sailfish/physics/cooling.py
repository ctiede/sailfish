"""
Code to compute the cooling coefficient for blackbody radiation 
around a binary with given mass and semi-major axis
"""

from math import sqrt

G_cgs         = 6.6725985e-8
kb_cgs        = 1.38065812e-16
sigmab_cgs    = 5.6705119e-5
mp_cgs        = 1.6726e-24
kappa_cgs     = 0.4            # electron scattering
pc_to_cm      = 3.085678e18
msun_to_grams = 1.989e33

def compute_cooling_coefficient(M_sun, a_pc, gamma=5./3.):
	"""Assumes avg fluid particle mass is the proton mass

		P / Sigma = eps * (gamma - 1)

		deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4

		eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 

		cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb) * (gamma - 1)
	"""
	a = a_pc * pc_to_cm
	m = M_sun * msun_to_grams
	t = sqrt(a**3 / m / G_cgs)
	mp_code  = mp_cgs /  m
	kb_code  = kb_cgs / (m * a**2 / t**2)
	sig_code = sigmab_cgs / (m / t**3)
	kap_code = kappa_cgs / (a**2 / m)
	return 8. / 3. * sig_code * (mp_code / kb_code)**4 * (gamma - 1.)**4 / kap_code
