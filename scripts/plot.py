#!/usr/bin/env python3
import cmasher as cmr
import argparse
import pickle
import sys

sys.path.insert(1, "/Users/ctiede/Research/sailfish")
# sys.path.insert(1, "/home/cwt271/sailfish-up-to-date")


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as file:
        chkpt = pickle.load(file)

        if require_solver is not None and chkpt["solver"] != require_solver:
            raise ValueError(
                f"checkpoint is from a run with solver {chkpt['solver']}, "
                f"expected {require_solver}"
            )
        return chkpt


def main_srhd_1d():
    import matplotlib.pyplot as plt
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


def main_srhd_2d():
    import matplotlib.pyplot as plt
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


def main_cbdiso_2d():
    import matplotlib.pyplot as plt
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pressure": None,
        "torque"  : None,
        "jadvect" : None,
        "jspecific": None,
        "gradp-r" : None,
        "gradp-p" : None,
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
        "--circle",
        default=None,
        type=float,
        help="draw a circle at given radius",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save PNG files instead of showing a window",
    )
    parser.add_argument(
        "--draw-lindblad31-radius",
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

            x1 = self.masses[0].position_x
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
            # return np.abs(t) ** 0.125 * np.sign(t)
            return t / da

    class Pressure:
        def __init__(self, mesh, masses, mach_number):
            self.mesh = mesh 
            self.masses = masses
            self.mach_number = mach_number

        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            x  = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y  = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            
            m1  = self.masses[0].mass
            m2  = self.masses[1].mass
            x1  = self.masses[0].position_x
            y1  = self.masses[0].position_y
            x2  = self.masses[1].position_x
            y2  = self.masses[1].position_y
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length

            sigma = primitive[:, :, 0]
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2

            phi1 = -m1 / (delx1**2 + dely1**2 + rs1**2) ** 0.5
            phi2 = -m2 / (delx2**2 + dely2**2 + rs2**2) ** 0.5
            cs2 = -(phi1 + phi2) / self.mach_number ** 2
            p = sigma * cs2
            return p
            # return np.abs(p) ** 0.125 * np.sign(p)

    class PressureGradients:
        def __init__(self, mesh, masses, mach_number, direction):
            self.mesh = mesh 
            self.masses = masses
            self.direction = direction
            self.mach_number = mach_number

        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            dx = mesh.dx
            dy = mesh.dy
            da = dx * dy
            x  = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y  = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            
            m1  = self.masses[0].mass
            m2  = self.masses[1].mass
            x1  = self.masses[0].position_x
            y1  = self.masses[0].position_y
            x2  = self.masses[1].position_x
            y2  = self.masses[1].position_y
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length

            sigma = primitive[:, :, 0]
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2

            phi1 = -m1 / (delx1**2 + dely1**2 + rs1**2) ** 0.5
            phi2 = -m2 / (delx2**2 + dely2**2 + rs2**2) ** 0.5
            cs2 = -(phi1 + phi2) / self.mach_number ** 2
            p = sigma * cs2
            dpx = -np.gradient(p, dx, axis=0) * da
            dpy = -np.gradient(p, dy, axis=1) * da
            if self.direction == 'r':
                dp = (x * dpx + y * dpy) / np.sqrt(x * x + y * y)
            elif self.direction == 'phi':
                dp = x * dpy - y * dpx
            else:
                print('invalid pressure gradient direction')
            # return np.abs(dp * self.mach_number**2) ** 0.125 * np.sign(dp)
            return dp 

    class AdvectedAngularMomentum:
        def __init__(self, mesh):
            self.mesh = mesh

        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            x  = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y  = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            da = mesh.dx * mesh.dy
            sig = primitive[:, :, 0]
            vx  = primitive[:, :, 1]
            vy  = primitive[:, :, 2]
            delv2 = vy * vy - vx * vx
            delx2 =  x *  x -  y *  y
            j = sig * da * (x * y * delv2 + vx * vy * delx2) / np.sqrt(x * x + y * y)
            return j
            # return np.abs(j) ** 0.125 * np.sign(j)

    class SpecificAngularMomentumSurplus:
        def __init__(self, mesh):
            self.mesh = mesh

        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            x  = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y  = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            r  = np.sqrt(x * x + y * y)
            da = mesh.dx * mesh.dy
            vx = primitive[:, :, 1]
            vy = primitive[:, :, 2]
            return (x * vy - y * vx) - (x * x + y * y) ** 0.25

    for filename in args.checkpoints:
        fig, ax = plt.subplots(figsize=[12, 9])
        chkpt = load_checkpoint(filename)
        mesh = chkpt["mesh"]
        fields["pressure" ]  = Pressure(mesh, chkpt["point_masses"], chkpt['model_parameters']['mach_number'])
        fields["torque" ]  = TorqueCalculation(mesh, chkpt["point_masses"])
        fields["gradp-r"]  = PressureGradients(mesh, chkpt["point_masses"], chkpt['model_parameters']['mach_number'], 'r')
        fields["gradp-p"]  = PressureGradients(mesh, chkpt["point_masses"], chkpt['model_parameters']['mach_number'], 'phi')
        fields["jadvect"]  = AdvectedAngularMomentum(mesh)
        fields["jspecific"] = SpecificAngularMomentumSurplus(mesh)

        clabel = r'$\Sigma$'
        if args.field == 'pressure':
            clabel = r'$P$'
        elif args.field == 'torque':
            clabel = 'torque'
        elif args.field == 'gradp-r':
            clabel = r'$(\nabla P)_r$'
            # clabel = r'$(\nabla P)_r \cdot \mathcal{M}^2$')
        elif args.field == 'gradp-p':
            clabel = r'$(\nabla P)_\phi$'
            # clabel = r'$(\nabla P)_\phi \cdot \mathcal{M}^2$'
        elif args.field == 'jadvect':
            clabel = r'$j_{\rm adv}$'
        elif args.field == 'jspecific':
            clabel = r'$j - j_{\rm kep}$'
        else:
            clabel=None

        if chkpt["solver"] == "cbdisodg_2d":
            prim = chkpt["primitive"]
        else:
            # the cbdiso_2d solver uses primitive data as the solution array
            prim = chkpt["solution"]
        f = fields[args.field](prim).T

        if args.print_model_parameters:
            print(chkpt["model_parameters"])

        if args.poly is None:
            prim = chkpt["solution"]
            f = fields[args.field](prim).T
        else:
            m, n = args.poly
            f = chkpt["solution"][:, :, 0, m, n].T
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

        if args.circle is not None:
            c = plt.Circle((0., 0.), args.circle, color='r', fill=False)
            p1 = chkpt["point_masses"][0]
            p2 = chkpt["point_masses"][1]
            # ax.scatter([-0.5, 0.5], [0., 0.], s=8, color='cyan')
            ax.scatter([p1.position_x, p2.position_x], [p1.position_y, p2.position_y], s=8, color='cyan')
            ax.add_artist(c)

        ax.set_aspect("equal")
        if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)
        fig.colorbar(cm, label=clabel)
        fig.suptitle(filename)
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )
        if args.save:
            pngname = filename.split('/')[-1].replace(".pk", ".png")
            print(pngname)
            fig.savefig(pngname, dpi=400)
            plt.close()
    if not args.save:
        plt.show()


def main_cbdisodg_2d():
    main_cbdiso_2d()


def main_cbdgam_2d():
    import matplotlib.pyplot as plt
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pre": lambda p: p[:, :, 3],
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

    args = parser.parse_args()

    for filename in args.checkpoints:
        fig, ax = plt.subplots(figsize=[10, 10])
        chkpt = load_checkpoint(filename, require_solver="cbdgam_2d")
        mesh = chkpt["mesh"]
        prim = chkpt["solution"]
        f = fields[args.field](prim).T

        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
        cm = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap="magma",
            extent=extent,
        )
        ax.set_aspect("equal")
        fig.colorbar(cm)
        fig.suptitle(filename)

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
