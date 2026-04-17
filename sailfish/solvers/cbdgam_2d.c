/*
MODULE: cbdgam_2d

DESCRIPTION: Energy-conserving solver for a binary accretion problem in 2D
  planar cartesian coordinates.

TODO:
    + add plm_theta as a solver option (currently it's hard-coded)
*/


// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 4
#define PLM_THETA 1.5
#define GRAD_MAX 1e6
#define HRMAX 1.0


// ============================ MATH ==========================================
// ============================================================================
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

PRIVATE double plm_gradient_scalar(double yl, double y0, double yr)
{
    double gl = (y0 - yl);
    double gr = (yr - y0);
    double g0 = (yr - yl);
    double gmax = max3(fabs(gl), fabs(gr), fabs(g0));
    double ymin = min3(fabs(yl), fabs(yr), fabs(y0));
    double grel = gmax / max2(ymin, 1e-12);
    double theta = (grel > GRAD_MAX) ? 1.0 : PLM_THETA;
    double a = gl * theta;
    double b = g0 * 0.5;
    double c = gr * theta;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

PRIVATE void plm_gradient(double *yl, double *y0, double *yr, double *g)
{
    for (int q = 0; q < NCONS; ++q)
    {
        g[q] = plm_gradient_scalar(yl[q], y0[q], yr[q]);
    }
}

// PRIVATE void fallback_if_dangerous(double *prim, const double *pcc, double dfloor, double pfloor, double vceil)
// {
//     int fallback = 0;
//     if (prim[0] < dfloor || !isfinite(prim[0]) || prim[3] < pfloor || !isfinite(prim[3])) 
//     {
//         fallback = 1;
//     }
//     if (fabs(prim[1]) > vceil || fabs(prim[2]) > vceil) 
//     {
//         fallback = 1;
//     }
//     if (fallback) 
//     {
//         printf("Fallback required in primitive reconstruction\n");
//         prim[0] = pcc[0];
//         prim[1] = pcc[1];
//         prim[2] = pcc[2];
//         prim[3] = pcc[3];
//     }
// }

// PRIVATE void check_conserved(const double *cons, int i, int j, const char *label) {
//     double rho = cons[0];
//     double px  = cons[1];
//     double py  = cons[2];
//     double en  = cons[3];

//     int bad = 0;

//     if (!isfinite(rho) || rho <= 0.0) {
//         printf("[BAD CONSERVED] %s at (%d, %d): rho = %.4e\n", label, i, j, rho);
//         bad = 1;
//     }
//     if (!isfinite(en) || en <= 0.0) {
//         printf("[BAD CONSERVED] %s at (%d, %d): energy = %.4e\n", label, i, j, en);
//         bad = 1;
//     }
//     if (!isfinite(px) || !isfinite(py)) {
//         printf("[BAD CONSERVED] %s at (%d, %d): momentum = (%.4e, %.4e)\n", label, i, j, px, py);
//         bad = 1;
//     }

//     if (bad) {
//         double ke = 0.5 * (px * px + py * py) / rho;
//         printf("    total energy = %.4e, kinetic = %.4e, internal = %.4e\n", en, ke, en - ke);
//     }
// }


// ============================ INTERNAL STRUCTS ==============================
// ============================================================================
struct PointMass {
    double x;
    double y;
    double vx;
    double vy;
    double mass;
    double softening_length;
    double sink_rate;
    double sink_radius;
    int sink_model;
};

struct PointMassList {
    struct PointMass masses[2];
};

struct KeplerianBuffer {
    double surface_density;
    double surface_pressure;
    double central_mass;
    double driving_rate;
    double outer_radius;
    double onset_width;
    int is_enabled;
};


// ============================ GRAVITY =======================================
// ============================================================================
PRIVATE double omega_tilde(
    struct PointMassList *mass_list,
    double x1,
    double y1)
{
    if (mass_list->masses[0].mass == 0.0 && mass_list->masses[1].mass == 0.0)
    {
        return 1.0;
    }
    double omegatilde2 = 0.0;

    for (int p = 0; p < 2; ++p)
    {
        if (mass_list->masses[p].mass > 0.0)
        {
            double x0 = mass_list->masses[p].x;
            double y0 = mass_list->masses[p].y;
            double mp = mass_list->masses[p].mass;

            double dx = x1 - x0;
            double dy = y1 - y0;
            double r2 = dx * dx + dy * dy + 1e-12;
            double r  = sqrt(r2);
            omegatilde2 += mp * pow(r, -3.0);
        }
    }
    return sqrt(omegatilde2);
}

PRIVATE double disk_height(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double *prim,
    double gamma_law_index)
{
    double sigma = prim[0];
    double pres  = prim[3];
    double omega = omega_tilde(mass_list, x1, y1);
    return sqrt(gamma_law_index * pres / sigma) / omega;
}

PRIVATE void point_mass_source_term(
    struct PointMass *mass,
    double x1,
    double y1,
    double dt,
    double *prim,
    double h,
    double *delta_cons,
    int constant_softening,
    double gamma_law_index)
{
    double x0 = mass->x;
    double y0 = mass->y;
    double sigma = prim[0];
    double pres  = prim[3];
    double eps = pres / sigma / (gamma_law_index - 1.0);
    double dx = x1 - x0;
    double dy = y1 - y0;
    double r2 = dx * dx + dy * dy;
    double dr = sqrt(r2);
    double r_sink = mass->sink_radius;
    double r_soft;

    if (constant_softening)
    {
        r_soft = mass->softening_length;
    }
    else if (dr > r_sink)
    {
        r_soft = 0.5 * h;
    }
    else
    {
        double transition = pow(1.0 - pow(dr / r_sink, 2.0), 2.0);
        r_soft = transition * r_sink + (1.0 - transition) * 0.5 * h;
    }
    // if (dr < 1.0 * r_sink)
    // {
    //     sink_rate = mass->sink_rate * pow(1.0 - pow(dr / r_sink, 2.0), 2.0);
    // }
    // double sink_rate = (dr < r_sink) ? mass->sink_rate * pow(1.0 - pow(dr / r_sink, 4.0), 2.0) : 0.0; //for testing ss-setup

    double sink_rate = (dr < 4.0 * r_sink) ? mass->sink_rate * exp(-pow(dr / r_sink, 4.0)) : 0.0;
    double fgrav_numerator = sigma * mass->mass * pow(r2 + r_soft * r_soft, -1.5);
    double fx = -fgrav_numerator * dx;
    double fy = -fgrav_numerator * dy;
    if (mass->sink_model == 4) {
        // Central sink instead of around each point mass
        // More localized at sink boundary : only consider in sink radius * 1.5 + steeper drop-off
        // Divide two because will be applied twice
        double r = sqrt(x1 * x1 + y1 * y1);
        sink_rate = (r < 1.5 * r_sink) ? mass->sink_rate / 2. * exp(-pow( r / r_sink, 8.0)) : 0.0; 
    }
    double mdot = sigma * sink_rate * -1.0;

    switch (mass->sink_model)
    {
        case 1: // acceleration-free
        {
            double vx = prim[1];
            double vy = prim[2];
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * prim[1] + dt * fx;
            delta_cons[2] = dt * mdot * prim[2] + dt * fy;
            delta_cons[3] = dt * (mdot * eps + 0.5 * mdot * (vx * vx + vy * vy)) + dt * (fx * vx + fy * vy);
            break;
        }
        case 2: // torque-free
        {
            double vx = prim[1];
            double vy = prim[2];
            double vx0 = mass->vx;
            double vy0 = mass->vy;
            double rhatx = dx / (dr + 1e-12);
            double rhaty = dy / (dr + 1e-12);
            double dvdotrhat = (vx - vx0) * rhatx + (vy - vy0) * rhaty;
            double vxstar = dvdotrhat * rhatx + vx0;
            double vystar = dvdotrhat * rhaty + vy0;
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * vxstar + dt * fx;
            delta_cons[2] = dt * mdot * vystar + dt * fy;
            delta_cons[3] = dt * (mdot * eps + 0.5 * mdot * (vxstar * vxstar + vystar * vystar)) + dt * (fx * vx + fy * vy);
            // NEED EXTRA TERM OR ELSE SINK DOES WORK
            double phatx = -dy / (dr + 1e-12);
            double phaty =  dx / (dr + 1e-12);
            double dvphi = (vx - vx0) * phatx + (vy - vy0) * phaty;
            // double delta =  dt * sink_rate;
            // delta_cons[3] += dt / (delta - 1.0) * 0.5 * mdot * dvphi * dvphi;
            delta_cons[3] -= dt * 0.5 * mdot * dvphi * dvphi;
            break;
        }
        case 3: // force-free
        {
            double vx = prim[1];
            double vy = prim[2];
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * fx;
            delta_cons[2] = dt * fy;
            delta_cons[3] = dt * (fx * vx + fy * vy);
            break;
        }
        case 4: // central excision
        { // Use standard sink -- e.g. eat mass and ang-mom bc no orbital velocity
            double vx = prim[1];
            double vy = prim[2];
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * prim[1] + dt * fx;
            delta_cons[2] = dt * mdot * prim[2] + dt * fy;
            delta_cons[3] = dt * (mdot * eps + 0.5 * mdot * (vx * vx + vy * vy)) + dt * (fx * vx + fy * vy);
            break;
        }
        default: // sink is inactive
        {
            delta_cons[0] = 0.0;
            delta_cons[1] = 0.0;
            delta_cons[2] = 0.0;
            delta_cons[3] = 0.0;
            break;
        }
    }
}

PRIVATE void point_masses_source_term(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double dt,
    double *prim,
    double h,
    double *cons,
    int constant_softening,
    double gamma_law_index)
{
    for (int p = 0; p < 2; ++p)
    {
        double delta_cons[NCONS];
        point_mass_source_term(&mass_list->masses[p], x1, y1, dt, prim, h, delta_cons, constant_softening, gamma_law_index);

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] += delta_cons[q];
        }
    }
}


// ============================ EOS AND BUFFER ================================
// ============================================================================
PRIVATE double sound_speed_squared(
    double gamma_law_index,
    const double *prim)
{
    return prim[3] / prim[0] * gamma_law_index;
}

PRIVATE void buffer_source_term(
    struct KeplerianBuffer *buffer,
    double xc,
    double yc,
    double dt,
    double *cons,
    double alpha,
    double gamma_law_index)
{
    if (buffer->is_enabled)
    {
        double rc = sqrt(xc * xc + yc * yc);
        double surface_density = buffer->surface_density;
        double surface_pressure = buffer->surface_pressure;
        double central_mass = buffer->central_mass;
        double driving_rate = buffer->driving_rate;
        double outer_radius = buffer->outer_radius;
        double onset_width = buffer->onset_width;
        double onset_radius = outer_radius - onset_width;

        if (rc > onset_radius)
        {
            double omega_outer = sqrt(central_mass * pow(onset_radius, -3.0));
            //double buffer_rate = driving_rate * omega_outer * max2(rc, 1.0);
            double buffer_rate = driving_rate * omega_outer * (rc - onset_radius) / (outer_radius - onset_radius);

            double invr = 1.0 / rc;
            double cosphi = xc * invr;
            double sinphi = yc * invr;
            double dm = -3.0 * alpha * gamma_law_index * surface_pressure / omega_outer;
            double vr = dm / (2.0 * rc * surface_density); // factors of pi cancel
            double vp = sqrt(central_mass / rc);
            double vx = vr * cosphi - vp * sinphi;
            double vy = vr * sinphi + vp * cosphi;
            double px = surface_density * vx;
            double py = surface_density * vy;
            double kinetic_energy = 0.5 * (px * px + py * py) / surface_density;
            double energy = surface_pressure / (gamma_law_index - 1.0) + kinetic_energy;
            double u0[NCONS] = {surface_density, px, py, energy};

            for (int q = 0; q < NCONS; ++q)
            {
                cons[q] -= (cons[q] - u0[q]) * buffer_rate * dt;
            }
        }
    }
}

PRIVATE void shear_strain(
    const double *gx,
    const double *gy,
    double dx,
    double dy,
    double *s)
{
    double sxx = 4.0 / 3.0 * gx[1] / dx - 2.0 / 3.0 * gy[2] / dy;
    double syy =-2.0 / 3.0 * gx[1] / dx + 4.0 / 3.0 * gy[2] / dy;
    double sxy = 1.0 / 1.0 * gx[2] / dx + 1.0 / 1.0 * gy[1] / dy;
    double syx = sxy;
    s[0] = sxx;
    s[1] = sxy;
    s[2] = syx;
    s[3] = syy;
}


// ============================ HYDRO =========================================
// ============================================================================
PRIVATE void beta_cooling_source_term(
    double beta,
    struct PointMassList *mass_list,
    double temp0,
    double xc,
    double yc, 
    double dt,
    double *prim,
    double *cons,
    double gamma_law_index)
{
    // RIGHT NOW IS NOT HANDLED ELEGANTLY 
    // TODO : CLEAN WAY TO PICK BETWEEN RADIATIVE AND BETA COOLING
    double r = sqrt(xc * xc + yc * yc + 1e-12);
    double h = disk_height(mass_list, xc, yc, prim, gamma_law_index);
    double cs2 = sound_speed_squared(gamma_law_index, prim);
    double tcool = beta * h / sqrt(cs2); // (beta / omega_eff)
    double temp = prim[3] / prim[0]; // I think these should both be in code units 
    double temp_ref = temp0 * pow(r, -9. / 10.); // TODO : Generalize this to problem generator
    // double qdot = (tcool > 0.0) ? prim[0] * (temp - temp_ref) / (gamma_law_index - 1.) / tcool : 0.0;
    double qdot = (beta > 0.0) ? prim[0] * max2(temp - temp_ref, 0.0) / (gamma_law_index - 1.) / tcool : 0.0;
    cons[3] -= dt * qdot;
}


PRIVATE void cooling_term(
    double cooling_coefficient,
    double opacity,
    double mach_ceiling,
    double dt,
    double *prim,
    double *cons,
    double gamma_law_index)
{
    double gamma = gamma_law_index;
    double sigma = prim[0];
    double eps = prim[3] / prim[0] / (gamma - 1.0);
    double eps_cooled = eps * pow(1.0 + 3.0 * cooling_coefficient * pow(sigma, -2.0) * pow(eps, 3.0) * dt, -1.0 / 3.0);
    double vx = prim[1];
    double vy = prim[2];

    double ek = 0.5 * (vx * vx + vy * vy);
    eps_cooled = max2(eps_cooled, 2.0 * ek / gamma / (gamma - 1.0) * pow(mach_ceiling, -2.0));

    // Ignored for now
    // int tau_flag = (sigma * opacity >= 1) ? 1: 0;

    cons[3] += sigma * (eps_cooled - eps);
}

PRIVATE void conserved_to_primitive(
    const double *cons,
    double *prim,
    struct PointMassList *mass_list,
    double xc,
    double yc, 
    double velocity_ceiling,
    double density_floor,
    double pressure_floor,
    double gamma_law_index)
{
    double gamma = gamma_law_index;
    double pres  = max2(pressure_floor, (cons[3] - 0.5 * (cons[1] * cons[1] + cons[2] * cons[2]) / cons[0]) * (gamma - 1.0));
    double vx = sign(cons[1]) * min2(fabs(cons[1] / cons[0]), velocity_ceiling);
    double vy = sign(cons[2]) * min2(fabs(cons[2] / cons[0]), velocity_ceiling);
    double rho = cons[0];

    if (cons[0] < density_floor)
    {
        rho = density_floor;
        pres = pressure_floor;
        // Induced some funny behavior in sinks....
        // but may still be important in cavity
        // TODO: test in binary setup
        vx = 0.0;
        vy = 0.0;
    }
 
    // TEMP: Check for precission loss in energy equation
    // double epsilon = 2.2204460492503131e-16; // <float.h> -> DBL_EPSILON, double precession limit
    // double internal_energy = cons[3] - 0.5 * (cons[1] * cons[1] + cons[2] * cons[2]) / cons[0];
    // if (fabs(internal_energy) < epsilon * fabs(cons[3])) {
    //     printf("Pressure floor is adding heat, e = %e\n", internal_energy);
    // }
    
    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = pres;

    double r = sqrt(xc * xc + yc * yc + 1e-12);
    double h = disk_height(mass_list, xc, yc, prim, gamma_law_index);
    if (h / r > HRMAX) {
    	double omega_tilde = sqrt(gamma_law_index * pres / rho) / h;
    	prim[3] = rho / gamma_law_index * pow(r * omega_tilde * HRMAX, 2);
    }
}

PRIVATE void primitive_to_conserved(const double *prim, double *cons, double gamma_law_index)
{
    double gamma = gamma_law_index;
    double rho = prim[0];
    double vx = prim[1];
    double vy = prim[2];
    double pres = prim[3];
    double px = vx * rho;
    double py = vy * rho;
    double en = pres / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = en;
}

PRIVATE double primitive_to_velocity(const double *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

PRIVATE void primitive_to_flux(
    const double *prim,
    const double *cons,
    double *flux,
    int direction)
{
    double vn = primitive_to_velocity(prim, direction);
    double pressure = prim[3];

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * (cons[3] + pressure);
}


PRIVATE double primitive_max_wavespeed(const double *prim, double cs2)
{
    double cs = sqrt(cs2);
    double vx = prim[1];
    double vy = prim[2];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    double ay = max2(fabs(vy - cs), fabs(vy + cs));
    return max2(ax, ay);
}

PRIVATE void primitive_to_outer_wavespeeds(
    const double *prim,
    double *wavespeeds,
    double cs2,
    int direction)
{
    double cs = sqrt(cs2);
    double vn = primitive_to_velocity(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

PRIVATE void riemann_hlle(const double *pl, const double *pr, double *flux, int direction, double gamma_law_index)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];
    double cs2l = sound_speed_squared(gamma_law_index, pl);
    double cs2r = sound_speed_squared(gamma_law_index, pr);

    primitive_to_conserved(pl, ul, gamma_law_index);
    primitive_to_conserved(pr, ur, gamma_law_index);
    primitive_to_flux(pl, ul, fl, direction);
    primitive_to_flux(pr, ur, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2l, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2r, direction);

    const double am = min3(0.0, al[0], ar[0]);
    const double ap = max3(0.0, al[1], ar[1]);

    // upwinding
    if (am >= 0.0) { for (int q=0; q<NCONS; ++q) flux[q] = fl[q]; return; }
    if (ap <= 0.0) { for (int q=0; q<NCONS; ++q) flux[q] = fr[q]; return; }

    double apam = ap * am;
    double inv_denom = 1.0 / (ap - am);
    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * apam) * inv_denom;
    }
    return;
}


// ============================ PUBLIC API ====================================
// ============================================================================
PUBLIC void cbdgam_2d_advance_rk(
    int ni,
    int nj,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double *conserved_rk, // :: $.shape == (ni + 4, nj + 4, 4)
    double *primitive_rd, // :: $.shape == (ni + 4, nj + 4, 4)
    double *primitive_wr, // :: $.shape == (ni + 4, nj + 4, 4)
    double gamma_law_index,
    double buffer_surface_density,
    double buffer_surface_pressure,
    double buffer_central_mass,
    double buffer_driving_rate,
    double buffer_outer_radius,
    double buffer_onset_width,
    int buffer_is_enabled,
    double x1, // point mass 1
    double y1,
    double vx1,
    double vy1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double x2, // point mass 2
    double y2,
    double vx2,
    double vy2,
    double mass2,
    double softening_length2,
    double sink_rate2,
    double sink_radius2,
    int sink_model2,
    double alpha, // other
    double beta,
    double temp0,
    double a,
    double dt,
    double velocity_ceiling,
    double cooling_coefficient,
    double opacity,
    double mach_ceiling,
    double density_floor,
    double pressure_floor,
    int constant_softening)
{
    struct KeplerianBuffer buffer = {
        buffer_surface_density,
        buffer_surface_pressure,
        buffer_central_mass,
        buffer_driving_rate,
        buffer_outer_radius,
        buffer_onset_width,
        buffer_is_enabled
    };
    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;

    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    FOR_EACH_2D(ni, nj)
    {
        double xl = patch_xl + (i + 0.0) * dx;
        double xc = patch_xl + (i + 0.5) * dx;
        double xr = patch_xl + (i + 1.0) * dx;
        double yl = patch_yl + (j + 0.0) * dy;
        double yc = patch_yl + (j + 0.5) * dy;
        double yr = patch_yl + (j + 1.0) * dy;

        // ------------------------------------------------------------------------
        //                 tj
        //
        //      +-------+-------+-------+
        //      |       |       |       |
        //      |  lr   |  rj   |   rr  |
        //      |       |       |       |
        //      +-------+-------+-------+
        //      |       |       |       |
        //  ki  |  li  -|+  c  -|+  ri  |  ti
        //      |       |       |       |
        //      +-------+-------+-------+
        //      |       |       |       |
        //      |  ll   |  lj   |   rl  |
        //      |       |       |       |
        //      +-------+-------+-------+
        //
        //                 kj
        // ------------------------------------------------------------------------

        int ncc = (i     + ng) * si + (j     + ng) * sj;
        int nli = (i - 1 + ng) * si + (j     + ng) * sj;
        int nri = (i + 1 + ng) * si + (j     + ng) * sj;
        int nlj = (i     + ng) * si + (j - 1 + ng) * sj;
        int nrj = (i     + ng) * si + (j + 1 + ng) * sj;
        int nki = (i - 2 + ng) * si + (j     + ng) * sj;
        int nti = (i + 2 + ng) * si + (j     + ng) * sj;
        int nkj = (i     + ng) * si + (j - 2 + ng) * sj;
        int ntj = (i     + ng) * si + (j + 2 + ng) * sj;
        int nll = (i - 1 + ng) * si + (j - 1 + ng) * sj;
        int nlr = (i - 1 + ng) * si + (j + 1 + ng) * sj;
        int nrl = (i + 1 + ng) * si + (j - 1 + ng) * sj;
        int nrr = (i + 1 + ng) * si + (j + 1 + ng) * sj;

        double *un = &conserved_rk[ncc];
        double *pcc = &primitive_rd[ncc];
        double *pli = &primitive_rd[nli];
        double *pri = &primitive_rd[nri];
        double *plj = &primitive_rd[nlj];
        double *prj = &primitive_rd[nrj];
        double *pki = &primitive_rd[nki];
        double *pti = &primitive_rd[nti];
        double *pkj = &primitive_rd[nkj];
        double *ptj = &primitive_rd[ntj];
        double *pll = &primitive_rd[nll];
        double *plr = &primitive_rd[nlr];
        double *prl = &primitive_rd[nrl];
        double *prr = &primitive_rd[nrr];

        double plip[NCONS];
        double plim[NCONS];
        double prip[NCONS];
        double prim[NCONS];
        double pljp[NCONS];
        double pljm[NCONS];
        double prjp[NCONS];
        double prjm[NCONS];

        double gxli[NCONS];
        double gxri[NCONS];
        double gyli[NCONS];
        double gyri[NCONS];
        double gxlj[NCONS];
        double gxrj[NCONS];
        double gylj[NCONS];
        double gyrj[NCONS];
        double gxcc[NCONS];
        double gycc[NCONS];

        plm_gradient(pki, pli, pcc, gxli);
        plm_gradient(pli, pcc, pri, gxcc);
        plm_gradient(pcc, pri, pti, gxri);
        plm_gradient(pkj, plj, pcc, gylj);
        plm_gradient(plj, pcc, prj, gycc);
        plm_gradient(pcc, prj, ptj, gyrj);
        plm_gradient(pll, pli, plr, gyli);
        plm_gradient(prl, pri, prr, gyri);
        plm_gradient(pll, plj, prl, gxlj);
        plm_gradient(plr, prj, prr, gxrj);

        for (int q = 0; q < NCONS; ++q)
        {
            plim[q] = pli[q] + 0.5 * gxli[q];
            plip[q] = pcc[q] - 0.5 * gxcc[q];
            prim[q] = pcc[q] + 0.5 * gxcc[q];
            prip[q] = pri[q] - 0.5 * gxri[q];

            pljm[q] = plj[q] + 0.5 * gylj[q];
            pljp[q] = pcc[q] - 0.5 * gycc[q];
            prjm[q] = pcc[q] + 0.5 * gycc[q];
            prjp[q] = prj[q] - 0.5 * gyrj[q];
        }

        double fli[NCONS];
        double fri[NCONS];
        double flj[NCONS];
        double frj[NCONS];
        double ucc[NCONS];

        riemann_hlle(plim, plip, fli, 0, gamma_law_index);
        riemann_hlle(prim, prip, fri, 0, gamma_law_index);
        riemann_hlle(pljm, pljp, flj, 1, gamma_law_index);
        riemann_hlle(prjm, prjp, frj, 1, gamma_law_index);

        if (alpha > 0.0)
        {
            double sli[4];
            double sri[4];
            double slj[4];
            double srj[4];
            double scc[4];

            shear_strain(gxli, gyli, dx, dy, sli);
            shear_strain(gxri, gyri, dx, dy, sri);
            shear_strain(gxlj, gylj, dx, dy, slj);
            shear_strain(gxrj, gyrj, dx, dy, srj);
            shear_strain(gxcc, gycc, dx, dy, scc);
            double omegali = omega_tilde(&mass_list, xl, yc);
            double omegari = omega_tilde(&mass_list, xr, yc);
            double omegalj = omega_tilde(&mass_list, xc, yl);
            double omegarj = omega_tilde(&mass_list, xc, yr);

            double mulim = alpha * gamma_law_index * plim[3] / omegali; // dynamic viscosity \mu = \nu\Sigma
            double mulip = alpha * gamma_law_index * plip[3] / omegali;
            double murim = alpha * gamma_law_index * prim[3] / omegari;
            double murip = alpha * gamma_law_index * prip[3] / omegari;
            double muljm = alpha * gamma_law_index * pljm[3] / omegalj;
            double muljp = alpha * gamma_law_index * pljp[3] / omegalj;
            double murjm = alpha * gamma_law_index * prjm[3] / omegarj;
            double murjp = alpha * gamma_law_index * prjp[3] / omegarj;

            fli[1] -= 0.5 * (mulim * sli[0] + mulip * scc[0]); // tau_xx
            fli[2] -= 0.5 * (mulim * sli[1] + mulip * scc[1]); // tau_xy
            fri[1] -= 0.5 * (murim * scc[0] + murip * sri[0]); // tau_xx
            fri[2] -= 0.5 * (murim * scc[1] + murip * sri[1]); // tau_xy
            flj[1] -= 0.5 * (muljm * slj[2] + muljp * scc[2]); // tau_yx
            flj[2] -= 0.5 * (muljm * slj[3] + muljp * scc[3]); // tau_yy
            frj[1] -= 0.5 * (murjm * scc[2] + murjp * srj[2]); // tau_yx
            frj[2] -= 0.5 * (murjm * scc[3] + murjp * srj[3]); // tau_yy
            fli[3] -= 0.5 * (mulim * (sli[0] * plim[1] + sli[1] * plim[2]) + mulip * (scc[0] * plip[1] + scc[1] * plip[2])); // x-left face:  v_x tau_xx + v_y tau_xy
            fri[3] -= 0.5 * (murim * (scc[0] * prim[1] + scc[1] * prim[2]) + murip * (sri[0] * prip[1] + sri[1] * prip[2])); // x-right face: v_x tau_xx + v_y tau_xy
            flj[3] -= 0.5 * (muljm * (slj[2] * pljm[1] + slj[3] * pljm[2]) + muljp * (scc[2] * pljp[1] + scc[3] * pljp[2])); // y-left face:  v_x tau_yx + v_y tau_yy
            frj[3] -= 0.5 * (murjm * (scc[2] * prjm[1] + scc[3] * prjm[2]) + murjp * (srj[2] * prjp[1] + srj[3] * prjp[2])); // y-right face: v_x tau_yx + v_y tau_yy
        }

        double hcc = disk_height(&mass_list, xc, yc, pcc, gamma_law_index);
        primitive_to_conserved(pcc, ucc, gamma_law_index);
        buffer_source_term(&buffer, xc, yc, dt, ucc, alpha, gamma_law_index);
        point_masses_source_term(&mass_list, xc, yc, dt, pcc, hcc, ucc, constant_softening, gamma_law_index);
        cooling_term(cooling_coefficient, opacity, mach_ceiling, dt, pcc, ucc, gamma_law_index);
        beta_cooling_source_term(beta, &mass_list, temp0, xc, yc,  dt, pcc, ucc, gamma_law_index);

        for (int q = 0; q < NCONS; ++q)
        {
            ucc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
            ucc[q] = (1.0 - a) * ucc[q] + a * un[q];
        }

        double *pout = &primitive_wr[ncc];
        conserved_to_primitive(ucc, pout, &mass_list, xc, yc, velocity_ceiling, density_floor, pressure_floor, gamma_law_index);
    }
}

PUBLIC void cbdgam_2d_wavespeed(
    int ni,
    int nj,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *wavespeed, // :: $.shape == (ni + 4, nj + 4)
    double gamma_law_index)
{
    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;
    int ti = nj + 2 * ng;
    int tj = 1;

    FOR_EACH_2D(ni, nj)
    {
        int np = (i + ng) * si + (j + ng) * sj;
        int na = (i + ng) * ti + (j + ng) * tj;

        double *pc = &primitive[np];
        double cs2 = sound_speed_squared(gamma_law_index, pc);
        double a = primitive_max_wavespeed(pc, cs2);
        wavespeed[na] = a;
    }
}

PUBLIC void cbdgam_2d_primitive_to_conserved(
    int ni,
    int nj,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *conserved, // :: $.shape == (ni + 4, nj + 4, 4)
    double gamma_law_index)
{
    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    FOR_EACH_2D(ni, nj)
    {
        int n = (i + ng) * si + (j + ng) * sj;

        double *pc = &primitive[n];
        double *uc = &conserved[n];
        primitive_to_conserved(pc, uc, gamma_law_index);
    }
}

PUBLIC void cbdgam_2d_point_mass_source_term(
    int ni,
    int nj,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double x1, // point mass 1
    double y1,
    double vx1,
    double vy1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double x2, // point mass 2
    double y2,
    double vx2,
    double vy2,
    double mass2,
    double softening_length2,
    double sink_rate2,
    double sink_radius2,
    int sink_model2,
    int which_mass, // :: $ in [1, 2]
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *cons_rate, // :: $.shape == (ni + 4, nj + 4, 4)
    int constant_softening,
    double gamma_law_index)
{
    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;

    FOR_EACH_2D(ni, nj)
    {
        int ncc = (i + ng) * si + (j + ng) * sj;

        double xc = patch_xl + (i + 0.5) * dx;
        double yc = patch_yl + (j + 0.5) * dy;
        double *pc = &primitive[ncc];
        double *uc = &cons_rate[ncc];
        double h = disk_height(&mass_list, xc, yc, pc, gamma_law_index);
        point_mass_source_term(&mass_list.masses[which_mass - 1], xc, yc, 1.0, pc, h, uc, constant_softening, gamma_law_index);
    }
}


// ======================== HLLC future option ================================
// ============================================================================
PRIVATE void riemann_hllc(const double *pl, const double *pr, double *flux, int direction, double gamma_law_index)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    double cs2l = sound_speed_squared(gamma_law_index, pl);
    double cs2r = sound_speed_squared(gamma_law_index, pr);

    primitive_to_conserved(pl, ul, gamma_law_index);
    primitive_to_conserved(pr, ur, gamma_law_index);
    primitive_to_flux(pl, ul, fl, direction);
    primitive_to_flux(pr, ur, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2l, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2r, direction);

    const double SL = min3(0.0, al[0], ar[0]);
    const double SR = max3(0.0, al[1], ar[1]);

    // Supersonic upwinding
    if (SL >= 0.0) { for (int q = 0; q < NCONS; ++q) flux[q] = fl[q]; return; }
    if (SR <= 0.0) { for (int q = 0; q < NCONS; ++q) flux[q] = fr[q]; return; }

    // Normal / tangential primitive variables
    const double rhoL = pl[0];
    const double rhoR = pr[0];
    const double pL   = pl[3];
    const double pR   = pr[3];
    const double unL = pl[1 + direction];
    const double unR = pr[1 + direction];
    const double utL = pl[1 + (1 - direction)];
    const double utR = pr[1 + (1 - direction)];
    const double EL = ul[3];
    const double ER = ur[3];

    // Contact wave speed
    const double denom = rhoL * (SL - unL) - rhoR * (SR - unR);
    if (fabs(denom) < 1e-14) { riemann_hlle(pl, pr, flux, direction, gamma_law_index); return; }

    // Star pressure estimate
    const double SM = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) / denom;
    const double pStarL = pL + rhoL * (SL - unL) * (SM - unL);
    const double pStarR = pR + rhoR * (SR - unR) * (SM - unR);
    const double pStar  = 0.5 * (pStarL + pStarR);

    // Fall back if star region looks nonphysical
    if (!(pStar >= 0.0)) { riemann_hlle(pl, pr, flux, direction, gamma_law_index); return; }

    // Left star state
    const double facL = rhoL * (SL - unL) / (SL - SM);
    double usL[NCONS];
    usL[0] = facL;
    if (direction == 0)
    {
        usL[1] = facL * SM;
        usL[2] = facL * utL;
    }
    else
    {
        usL[1] = facL * utL;
        usL[2] = facL * SM;
    }
    usL[3] = ((SL - unL) * EL - pL * unL + pStar * SM) / (SL - SM);

    // Right star state
    const double facR = rhoR * (SR - unR) / (SR - SM);
    double usR[NCONS];
    usR[0] = facR;
    if (direction == 0)
    {
        usR[1] = facR * SM;
        usR[2] = facR * utR;
    }
    else
    {
        usR[1] = facR * utR;
        usR[2] = facR * SM;
    }
    usR[3] = ((SR - unR) * ER - pR * unR + pStar * SM) / (SR - SM);

    // HLLC flux
    if (SM >= 0.0)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = fl[q] + SL * (usL[q] - ul[q]);
        }
    }
    else
    {
        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = fr[q] + SR * (usR[q] - ur[q]);
        }
    }
}