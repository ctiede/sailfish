"""
Validation setups for various 3D solvers
"""

from sailfish.mesh import PlanarCartesian3DMesh
from sailfish.physics.circumbinary import EquationOfState
from sailfish.setup import Setup, param
from math import exp

__all__ = ["SphericalExplosion"]


class SphericalExplosion(Setup):
    """
    A cylindrical explosion in 3D planar geometry; isothermal only.

    This problem is useful for testing bare-bones setups with minimal physics.
    A spherical region of high density and pressure is initiated at the center
    of a square domain. In isothermal mode, the sound speed is set to 1
    everywhere.

    Currently this setup can only specify either the `cbdiso_2d`.
    """

    smooth = param(6.0, "k to smooth density enhancement, ~exp(-r^k) [0.0 for tophat]")
    # eos = param("isothermal", "EOS type: either isothermal or gamma-law")
    # use_dg = param(False, "use the DG solver (isothermal only)")

    @property
    def is_isothermal(self):
        return self.eos == "isothermal"

    # @property
    # def is_gamma_law(self):
    #     return self.eos == "gamma-law"

    def primitive(self, t, coords, primitive):
        x, y, z = coords
        r = (x * x + y * y + z * z) ** 0.5

        if self.smooth != 0.0:
            f = exp(-((r / 0.25) ** self.smooth))
        else:
            f = float(r < 0.25)

        primitive[0] = 0.1 + 0.9 * f

    def mesh(self, resolution):
        return PlanarCartesian3DMesh.centered_cube(1.0, resolution)

    @property
    def physics(self):
        return dict(eos_type=EquationOfState.GLOBALLY_ISOTHERMAL, sound_speed=1.0)

    @property
    def solver(self):
        return "cbdiso_3d" 

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_resolution(self):
        return 200

    @property
    def default_end_time(self):
        return 0.3

    def validate(self):
        return