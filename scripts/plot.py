import argparse
import pickle
import sys
import cmasher as cmr
import matplotlib.pyplot as plt

# sys.path.insert(1, "/Users/ctiede/Research/sailfish")
sys.path.insert(1, "/groups/astro/ctiede/sailfish-sweep")
import load_sfdata_cbdgam2d as cbdgam

# -----------------------------------------------------------------------------
text_width   = 7.1
column_width = 3.35225
def configure_matplotlib():
    plt.rc('xtick' , labelsize=8)
    plt.rc('ytick' , labelsize=8)
    plt.rc('axes'  , labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', family='DejaVu Sans', size=8)
    plt.rc('text', usetex=True)

def config_axes_negative(ax, leg=None, cbar=None):
    ax.spines['bottom'].set_color('white')
    ax.spines['top'   ].set_color('white' )
    ax.spines['right' ].set_color('white')
    ax.spines['left'  ].set_color('white')
    ax.tick_params(which='both', colors='white')
    # ax.tick_params(direction='in', which='both', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    if leg is not None:
        [txt.set_color("white") for txt in leg.get_texts()]
    if cbar is not None:
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# -----------------------------------------------------------------------------
def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as file:
        chkpt = pickle.load(file)

        if require_solver is not None and chkpt["solver"] != require_solver:
            raise ValueError(
                f"checkpoint is from a run with solver {chkpt['solver']}, "
                f"expected {require_solver}"
            )
        return chkpt

# -----------------------------------------------------------------------------
def main_srhd_1d():
    from sailfish.mesh import LogSphericalMesh

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    args = parser.parse_args()

    fig, ax = plt.subplots()

    for filename in args.checkpoints:
        chkpt = load_checkpoint(filename, require_solver="srhd_1d")

        mesh = chkpt["mesh"]
        x = mesh.zone_centers(chkpt["time"])
        rho = chkpt["primitive"][:, 0]
        vel = chkpt["primitive"][:, 1]
        pre = chkpt["primitive"][:, 2]
        ax.plot(x, rho, label=r"$\rho$")
        ax.plot(x, vel, label=r"$\Gamma \beta$")
        ax.plot(x, pre, label=r"$p$")

    if type(mesh) == LogSphericalMesh:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend()
    plt.show()

# -----------------------------------------------------------------------------
def main_srhd_2d():
    import numpy as np
    import sailfish

    fields = {
        "ur": lambda p: p[..., 1],
        "uq": lambda p: p[..., 2],
        "rho": lambda p: p[..., 0],
        "pre": lambda p: p[..., 3],
        "e": lambda p: p[..., 3] / p[..., 0] * 3.0,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="ur",
        choices=fields.keys(),
        help="which field to plot",
    )
    parser.add_argument(
        "--radial-coordinates",
        "-c",
        type=str,
        default="comoving",
        choices=["comoving", "proper"],
        help="plot in comoving or proper (time-independent) radial coordinates",
    )
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )

    args = parser.parse_args()

    for filename in args.checkpoints:
        fig, ax = plt.subplots()

        chkpt = load_checkpoint(filename, require_solver="srhd_2d")
        mesh = chkpt["mesh"]
        prim = chkpt["primitive"]

        t = chkpt["time"]
        r, q = np.meshgrid(mesh.radial_vertices(t), mesh.polar_vertices)
        z = r * np.cos(q)
        x = r * np.sin(q)
        f = fields[args.field](prim).T

        if args.radial_coordinates == "comoving":
            x[...] /= mesh.scale_factor(t)
            z[...] /= mesh.scale_factor(t)

        if args.log:
            f = np.log10(f)

        cm = ax.pcolormesh(
            x,
            z,
            f,
            edgecolors="none",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap="plasma",
        )

        ax.set_aspect("equal")
        # ax.set_xlim(0, 1.25)
        # ax.set_ylim(0, 1.25)
        fig.colorbar(cm)
        fig.suptitle(filename)

    plt.show()

# -----------------------------------------------------------------------------
def main_cbdiso_2d():
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "torque": None,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="sigma",
        choices=fields.keys(),
        help="which field to plot",
    )
    parser.add_argument("--poly", type=int, nargs=2, default=None)
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--scale-by-power",
        "-s",
        default=None,
        type=float,
        help="scale the field by the given power",
    )
    parser.add_argument(
        "--scale-by-radius",
        default=None,
        type=float,
        help="scale the field by 1 / r to given power",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )
    parser.add_argument(
        "--cmap",
        default=cmr.sunburst,
        help="colormap name",
    )
    parser.add_argument(
        "--radius",
        default=None,
        type=float,
        help="plot the domain out to this radius",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save PNG files instead of showing a window",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
    )
    parser.add_argument(
        "--fix-edges",
        action="store_true",
    )
    parser.add_argument(
        "--draw-binary",
        action="store_true",
    )
    parser.add_argument(
        "--draw-lindblad31-radius",
        action="store_true",
    )
    parser.add_argument(
        "--orbital-elements",
        '-oe',
        action="store_true",
    )
    parser.add_argument(
        "--velocity-vectors",
        '-v',
        action="store_true",
    )
    parser.add_argument(
        "--as-pdf",
        '-pdf',
        action="store_true",
    )
    parser.add_argument("-m", "--print-model-parameters", action="store_true")
    args = parser.parse_args()

    class TorqueCalculation:
        def __init__(self, mesh, masses):
            self.mesh = mesh
            self.masses = masses

        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            dx = mesh.dx
            dy = mesh.dy
            da = dx * dy
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]

            x1 = self[0].position_x
            y1 = self.masses[0].position_y
            x2 = self.masses[1].position_x
            y2 = self.masses[1].position_y
            m1 = self.masses[0].mass
            m2 = self.masses[1].mass
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length

            sigma = primitive[:, :, 0]
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2

            # forces on the gas
            fx1 = -sigma * da * m1 * delx1 / (delx1**2 + dely1**2 + rs1**2) ** 1.5
            fy1 = -sigma * da * m1 * dely1 / (delx1**2 + dely1**2 + rs1**2) ** 1.5
            fx2 = -sigma * da * m2 * delx2 / (delx2**2 + dely2**2 + rs2**2) ** 1.5
            fy2 = -sigma * da * m2 * dely2 / (delx2**2 + dely2**2 + rs2**2) ** 1.5

            t1 = x * fy1 - y * fx1
            t2 = x * fy2 - y * fx2
            t = t1 + t2
            print("total torque:", t.sum())
            return np.abs(t) ** 0.125 * np.sign(t)

    for filename in args.checkpoints:
        fig, ax = plt.subplots(figsize=[12, 9])
        chkpt = load_checkpoint(filename)
        mesh = chkpt["mesh"]
        fields["torque"] = TorqueCalculation(mesh, chkpt["point_masses"])

        if chkpt["solver"] == "cbdisodg_2d":
            prim = chkpt["primitive"]
            if args.poly is None:
                prim = chkpt["primitive"]
                f = fields[args.field](prim).T
            else:
                m, n = args.poly
                f = chkpt["solution"][:, :, 0, m, n].T
        else:
            # the cbdiso_2d solver uses primitive data as the solution array
            prim = chkpt["solution"]

        f = fields[args.field](prim).T

        if args.fix_edges:
            ni, nj = mesh.shape
            dx = mesh.dx
            dy = mesh.dy
            da = dx * dy
            dr = chkpt['model_parameters']['domain_radius']
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None].T
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :].T
            r = np.sqrt(x * x + y * y)
            f[r > dr-0.1] = np.nan

        if args.print_model_parameters:
            print(chkpt["model_parameters"])

        if args.scale_by_radius is not None:
            x1 = chkpt["point_masses"][0].position_x - x
            y1 = chkpt["point_masses"][0].position_y - y
            x2 = chkpt["point_masses"][1].position_x - x
            y2 = chkpt["point_masses"][1].position_y - y
            r1 = np.sqrt(x1 * x1 + y1 * y1 + 0.05**2)
            r2 = np.sqrt(x2 * x2 + y2 * y2 + 0.05**2)
            f = f / r1**args.scale_by_radius + f / r2**args.scale_by_radius
        if args.scale_by_power is not None:
            f = f**args.scale_by_power
        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        cm = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            extent=extent,
        )

        if args.draw_binary:
            x1 = chkpt["point_masses"][0].position_x
            y1 = chkpt["point_masses"][0].position_y
            x2 = chkpt["point_masses"][1].position_x
            y2 = chkpt["point_masses"][1].position_y
            ax.scatter([x1, x2], [y1, y2], s=0.1, c='k',)

        if args.draw_lindblad31_radius:
            x1 = chkpt["point_masses"][0].position_x
            y1 = chkpt["point_masses"][0].position_y
            t = np.linspace(0, 2 * np.pi, 1000)
            x = x1 + 0.3 * np.cos(t)
            y = y1 + 0.3 * np.sin(t)
            a = 1.0
            q = chkpt["model_parameters"]["mass_ratio"]
            # Eq. 1 in Franchini & Martin (2019; https://arxiv.org/pdf/1908.02776.pdf)
            r_res = 3 ** (-2 / 3) * (1 + q) ** (-1 / 3) * a
            ax.plot(x, y, ls="--", lw=0.75, c="w", alpha=1.0)

        if args.orbital_elements:
            import sailfish.physics.kepler as kepler
            m1 = chkpt['point_masses'][0]
            m2 = chkpt['point_masses'][1]
            m1 = kepler.PointMass(m1.mass, m1.position_x, m1.position_y, m1.velocity_x, m1.velocity_y)
            m2 = kepler.PointMass(m2.mass, m2.position_x, m2.position_y, m2.velocity_x, m2.velocity_y) 
            orbital_state = kepler.OrbitalState(primary=m1, secondary=m2)
            fig.suptitle('t={:.2f} orbits  :   e={:.3f}   q={:.3f} '.format(chkpt['time'] / 2. / np.pi, orbital_state.eccentricity, orbital_state.mass_ratio))
        else:
            if args.no_frame == False:
                fig.suptitle(filename)

        if args.velocity_vectors:
            mesh = chkpt['mesh']
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(mesh.ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(mesh.nj)])[None, :]
            vx = fields["vx"](prim).T
            vy = fields["vy"](prim).T
            xx, yy = np.meshgrid(x, y)
            skip = (slice(None, None, int(mesh.ni / 300)),) * 2
            ax.quiver(xx[skip], yy[skip], vx[skip], vy[skip], alpha=0.5, color='cyan', scale=80, headwidth=2, headlength=4)

        ax.set_aspect("equal")
        if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)
        if args.no_frame:
            ax.axis('off')
        else:
            fig.colorbar(cm)
            fig.subplots_adjust(
                left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
            )
        if args.save:
            if args.as_pdf:
                pngname = filename.replace(".pk", ".pdf")
            else:
                pngname = filename.replace(".pk", ".png")
            print(pngname)
            fig.savefig(pngname, dpi=400, pad_inches=0.0 if args.no_frame else 0.5, bbox_inches='tight', transparent=True if args.no_frame else False)
            plt.close()
    if not args.save:
        plt.show()

# -----------------------------------------------------------------------------
def main_cbdisodg_2d():
    main_cbdiso_2d()

# -----------------------------------------------------------------------------
def main_cbdgam_2d():
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pre": lambda p: p[:, :, 3],
        "eps": lambda p: p[:, :, 3] / p[:, :, 0] / (5./3. - 1.),
        "rho": None,
        "temperature": None,
        "peak-frequency": None,
        "optical-depth": None,
        "scale-height": None,
        "mach-number": None,
        "cooling-rate": None,
        "heating-rate": None,
        "compressional-heating": None,
        "vorticity": None,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="sigma",
        choices=fields.keys(),
        help="which field to plot",
    )
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )
    parser.add_argument(
        "--cmap",
        default=cmr.sunburst,
        help="colormap name",
    )
    parser.add_argument(
        "--radius",
        default=None,
        type=float,
        help="plot the domain out to this radius",
    )
    parser.add_argument(
        "--orbital-elements",
        '-oe',
        action="store_true",
    )
    parser.add_argument(
        "--draw-binary",
        action="store_true",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
    )
    parser.add_argument(
        "--fancy",
        action="store_true",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save PNG files instead of showing a window",
    )
    parser.add_argument(
        "--as-pdf",
        '-pdf',
        action="store_true",
    )
    parser.add_argument("-m", "--print-model-parameters", action="store_true")
    args = parser.parse_args()
    if args.fancy:
            configure_matplotlib()

    class Vorticity:
        def __init__(self, mesh):
            self.mesh = mesh

        def __call__(self, primitive):
            sig = primitive[:, :, 0]
            vx  = primitive[:, :, 1]
            vy  = primitive[:, :, 2]
            _, dvxdy = np.gradient(vx, mesh.dx, mesh.dy)
            dvydx, _ = np.gradient(vy, mesh.dx, mesh.dy)
            return np.abs(dvydx - dvxdy)

    class Temperature:
        def __init__(self, disk):
            self.disk = disk

        def __call__(self, primitive):
            from sailfish.physics.cooling import cgs
            disk = self.disk
            sigma = primitive[:, :, 0]
            press = primitive[:, :, 3]
            mp_code = cgs['mp'] /  disk._mass
            kb_code = cgs['kb'] / (disk._mass * disk._length**2 / disk._time**2)
            return (mp_code / kb_code) * press / sigma

    class PeakFrequency:
        def __init__(self, disk):
            self.disk = disk

        def __call__(self, primitive):
            from sailfish.physics.cooling import cgs
            disk = self.disk
            sigma = primitive[:, :, 0]
            press = primitive[:, :, 3]
            mp_code = cgs['mp'] /  disk._mass
            kb_code = cgs['kb'] / (disk._mass * disk._length**2 / disk._time**2)
            tempk = (mp_code / kb_code) * press / sigma
            teff = tempk * (disk._eddington_fraction)**(-1./4.)
            return 5.879e10 * teff # Hz

    class OpticalDepth:
        def __init__(self, disk):
            self.disk = disk

        def __call__(self, primitive):
            kappa = self.disk.opacity
            sigma = primitive[:, :, 0]
            return kappa * sigma

    class ScaleHeight:
        def __init__(self, mesh, masses, logh, gamma=5./3.):
            self.mesh = mesh
            self.masses = masses
            self.gamma = gamma
            self.logh = logh

        def __call__(self, primitive):
            print('h0 / a: ', logh)
            mesh = self.mesh
            ni, nj = mesh.shape
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            r = np.sqrt(x * x + y * y + 1e-12)
            x1 = self.masses[0].position_x
            y1 = self.masses[0].position_y
            x2 = self.masses[1].position_x
            y2 = self.masses[1].position_y
            m1 = self.masses[0].mass
            m2 = self.masses[1].mass
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2
            dr1 = np.sqrt(delx1**2 + dely1**2 + rs1**2)
            dr2 = np.sqrt(delx2**2 + dely2**2 + rs2**2)
            omegasq1 = m1 * dr1**(-3.)
            omegasq2 = m2 * dr2**(-3.)
            omegatilde = np.sqrt(omegasq1 + omegasq2)
            sigma = primitive[:, :, 0]
            pres = primitive[:, :, 3]
            h = np.sqrt(self.gamma * pres / sigma) / omegatilde
            return h / r

    class CoolingRate:
        def __init__(self, disk, dx, gamma=5./3.):
            self.dx = dx
            self.disk = disk
            self.gamma = gamma

        def __call__(self, primitive):
            """ Qdot = Sigma * depsilon / dt  [ergs / s]

                depsilon / dt = - 8 / 3 sigma T^4 / (tau Sigma)
                              = - 8 / 3 (sigma / kappa) (mp / kb)^4 P^4 / Sigma^6
                              = - beta / (gamma - 1)^4 P^4 / Sigma^6
            """
            disk = self.disk
            sigma = primitive[:, :, 0]
            press = primitive[:, :, 3]
            beta = disk.cooling_coefficient()
            qdot = beta / (self.gamma - 1.)**4 * press**4 / sigma**5
            units = disk._mass / disk._time**3
            # units = disk._mass * disk._length**2 / disk._time**3
            return qdot * units # * self.dx**2

    class HeatingRate:
        def __init__(self, mesh, masses, disk, gamma=5./3.):
            self.mesh = mesh
            self.masses = masses
            self.disk = disk
            self.gamma = gamma

        def __call__(self, primitive):
            disk = self.disk
            mesh = self.mesh
            ni, nj = mesh.shape
            x = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            r = np.sqrt(x * x + y * y + 1e-12)
            dx = mesh.dx
            dy = mesh.dx # uniform grid
            sig = primitive[:, :, 0]
            vx  = primitive[:, :, 1]
            vy  = primitive[:, :, 2]
            pre = primitive[:, :, 3]
            x1 = self.masses[0].position_x
            y1 = self.masses[0].position_y
            x2 = self.masses[1].position_x
            y2 = self.masses[1].position_y
            m1 = self.masses[0].mass
            m2 = self.masses[1].mass
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2
            dr1 = np.sqrt(delx1**2 + dely1**2 + rs1**2)
            dr2 = np.sqrt(delx2**2 + dely2**2 + rs2**2)
            omegasq1 = m1 * dr1**(-3.)
            omegasq2 = m2 * dr2**(-3.)
            omegatilde = np.sqrt(omegasq1 + omegasq2)
            h = np.sqrt(self.gamma * pre / sig) / omegatilde
            cs2 = pre / sig * self.gamma
            nu = disk.alpha * np.sqrt(cs2) * h
            dvxdx, dvxdy = np.gradient(vx, dx, dy)
            dvydx, dvydy = np.gradient(vy, dx, dy)
            sxx = 4.0 / 3.0 * dvxdx - 2.0 / 3.0 * dvydy;
            syy =-2.0 / 3.0 * dvxdx + 4.0 / 3.0 * dvydy;
            sxy = 1.0 / 1.0 * dvydx + 1.0 / 1.0 * dvxdy;
            heat = nu * sig * (sxx * dvxdx + syy * dvydy + sxy * (dvxdy + dvydx))
            heat[heat == 0.0] += 1e-12
            units = disk._mass / disk._time**3
            # units = disk._mass * disk._length**2 / disk._time**3
            return heat * units # * dx * dy 

    class CompressionHeating:
        def __init__(self, mesh, disk):
            self.mesh = mesh
            self.disk = disk

        def __call__(self, primitive):
            mesh = self.mesh
            disk = self.disk
            vx  = primitive[:, :, 1]
            vy  = primitive[:, :, 2]
            pre = primitive[:, :, 3]
            dvxdx, _ = np.gradient(vx, mesh.dx, mesh.dy)
            _, dvydy = np.gradient(vy, mesh.dx, mesh.dy)
            units = disk._mass / disk._time**3
            heat = -pre * (dvxdx + dvydy) * units
            heat[heat <= 0.0] = 1e-10 # Only keep the heating (no cooling)
            return heat

    for filename in args.checkpoints:
        if args.fancy:
            fig, ax = plt.subplots(figsize=[1.2 * column_width, column_width])
        else:
            fig, ax = plt.subplots(figsize=[12, 9])
        chkpt = load_checkpoint(filename, require_solver="cbdgam_2d")
        mesh = chkpt["mesh"]
        prim = chkpt["solution"]
        logh = np.log10(1. / chkpt['model_parameters']['mach_at_a'])
        fields["vorticity"] = Vorticity(mesh)
        fields["temperature"] = Temperature(cbdgam.get_ss_disk_model(filename))
        fields["scale-height"] = ScaleHeight(mesh, chkpt["point_masses"], logh)
        fields["mach-number"] =  ScaleHeight(mesh, chkpt["point_masses"], logh)
        fields["cooling-rate"] = CoolingRate(cbdgam.get_ss_disk_model(filename), mesh.dx)
        fields["heating-rate"] = HeatingRate(mesh, chkpt['point_masses'], cbdgam.get_ss_disk_model(filename))
        fields["optical-depth"] = OpticalDepth(cbdgam.get_ss_disk_model(filename))
        fields["compressional-heating"] = CompressionHeating(mesh, cbdgam.get_ss_disk_model(filename))
        fields["peak-frequency"] = PeakFrequency(cbdgam.get_ss_disk_model(filename))
        f = fields[args.field](prim).T

        cmap = eval(args.cmap)
        if args.field == 'sigma':
            s0 = cbdgam.get_ss_disk_model(filename).surface_density_coefficient
            f = f / s0
        if args.field == 'mach-number':
            f = 1. / f
        if args.field == 'peak-frequency':
            cmap.set_over('red')
            sig = prim[:,:,0].T
            tau = sig * cbdgam.get_ss_disk_model(filename).opacity
            f[tau < 10.0] = 1e10
            print(cbdgam.get_ss_disk_model(filename).surface_density_coefficient)
            print(cbdgam.get_ss_disk_model(filename).surface_pressure_coefficient)
        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        cm = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=cmap,
            extent=extent,
        )
        ax.set_aspect("equal")

        if args.draw_binary:
            x1 = chkpt["point_masses"][0].position_x
            y1 = chkpt["point_masses"][0].position_y
            x2 = chkpt["point_masses"][1].position_x
            y2 = chkpt["point_masses"][1].position_y
            ax.scatter([x1, x2], [y1, y2], s=2, c='w')

        if args.orbital_elements:
            import sailfish.physics.kepler as kepler
            m1 = chkpt['point_masses'][0]
            m2 = chkpt['point_masses'][1]
            m1 = kepler.PointMass(m1.mass, m1.position_x, m1.position_y, m1.velocity_x, m1.velocity_y)
            m2 = kepler.PointMass(m2.mass, m2.position_x, m2.position_y, m2.velocity_x, m2.velocity_y) 
            orbital_state = kepler.OrbitalState(primary=m1, secondary=m2)
            fig.suptitle('t={:.2f} orbits  :   e={:.3f}   q={:.2e} '.format(chkpt['time'] / 2. / np.pi, orbital_state.eccentricity, orbital_state.mass_ratio))
        else:
            if not args.no_frame and not args.fancy:
                fig.suptitle(filename)

        cblabel = {'sigma'        : r'$\log_{10}(\Sigma / \Sigma_0)$',
                   'temperature'  : r'$\log_{10}(T)$ [K]',
                   'cooling-rate' : r'$\log_{10}(\dot Q)$ [ergs / s / cm$^2$]',
                   'heating-rate' : r'$\log_{10}(\tau\, \nabla v)$ [ergs / s / cm$^2$]',
                   'scale-height' : r'$\log_{10}(h/r)$',
                   'mach-number'  : r'$\log_{10}(\mathcal{M})$',
                   'peak-frequency' : r'$\log_{10}(\nu_{\rm peak})$',
                  }
        basecol = 'w' if args.negative else 'k'
        if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)
        if args.no_frame:
            ax.axis('off')
        else:
            cbar = fig.colorbar(cm, pad=0.01)
            cbar.ax.tick_params(size=0)
            if args.field in cblabel:
                cbar.set_label(cblabel[args.field], color=basecol)
            if args.field == 'mach-number':
                cbar.ax.axhline(np.log10(1. / 10**logh), color='w', lw=1.2, alpha=0.7)
            ax.set_xlabel(r'$x / a$')
            ax.set_ylabel(r'$y / a$')
        

        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )
        #fig.suptitle(filename)

        if args.negative:
            config_axes_negative(ax, cbar=cbar)

        if args.save:
            trans = (args.no_frame) or (args.negative)
            pngname = filename.replace(".pk", ".pdf") if args.as_pdf else filename.replace(".pk", ".png")
            print(pngname)
            fig.savefig(pngname, dpi=400, bbox_inches='tight', pad_inches=0.0 if args.no_frame else 0.05, transparent=trans)
            plt.close()

    if not args.save:
        plt.show()


if __name__ == "__main__":
    for arg in sys.argv:
        if arg.endswith(".pk"):
            chkpt = load_checkpoint(arg)
            if chkpt["solver"] == "srhd_1d":
                print("plotting for srhd_1d solver")
                exit(main_srhd_1d())
            if chkpt["solver"] == "srhd_2d":
                print("plotting for srhd_2d solver")
                exit(main_srhd_2d())
            if chkpt["solver"] == "cbdiso_2d":
                print("plotting for cbdiso_2d solver")
                exit(main_cbdiso_2d())
            if chkpt["solver"] == "cbdisodg_2d":
                print("plotting for cbdisodg_2d solver")
                exit(main_cbdisodg_2d())
            if chkpt["solver"] == "cbdgam_2d":
                print("plotting for cbdgam_2d solver")
                exit(main_cbdgam_2d())
            else:
                print(f"Unknown solver {chkpt['solver']}")
