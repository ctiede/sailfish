/*
MODULE: cbdiso_3d

DESCRIPTION: Isothermal solver for a binary accretion problem in 2D planar
  cartesian coordinates.
*/

// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 4
#define PLM_THETA 1.8


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
    double a = (y0 - yl) * PLM_THETA;
    double b = (yr - yl) * 0.5;
    double c = (yr - y0) * PLM_THETA;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

PRIVATE void plm_gradient(double *yl, double *y0, double *yr, double *g)
{
    for (int q = 0; q < NCONS; ++q)
    {
        g[q] = plm_gradient_scalar(yl[q], y0[q], yr[q]);
    }
}


// ============================ INTERNAL STRUCTS ==============================
// ============================================================================
struct PointMass {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
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
    double central_mass;
    double driving_rate;
    double outer_radius;
    double onset_width;
    int is_enabled;
};


// ============================ GRAVITY =======================================
// ============================================================================
PRIVATE double gravitational_potential(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double z1)
{
    double phi = 0.0;

    for (int p = 0; p < 2; ++p)
    {
        if (mass_list->masses[p].mass > 0.0)
        {
            double x0 = mass_list->masses[p].x;
            double y0 = mass_list->masses[p].y;
            double z0 = mass_list->masses[p].z;
            double mp = mass_list->masses[p].mass;
            double rs = mass_list->masses[p].softening_length;

            double dx = x1 - x0;
            double dy = y1 - y0;
            double dz = z1 - z0;
            double r2 = dx * dx + dy * dy + dz * dz;
            double r2_softened = r2 + rs * rs;

            phi -= mp / sqrt(r2_softened);
        }
    }
    return phi;
}

PRIVATE void point_mass_source_term(
    struct PointMass *mass,
    double x1,
    double y1,
    double z1,
    double dt,
    double *prim,
    double *delta_cons)
{
    double x0 = mass->x;
    double y0 = mass->y;
    double z0 = mass->z;
    double sigma = prim[0];
    double dx = x1 - x0;
    double dy = y1 - y0;
    double dz = z1 - z0;
    double r2 = dx * dx + dy * dy + dz * dz;
    double dr = sqrt(r2);
    double r_sink = mass->sink_radius;
    double r_soft = mass->softening_length;

    double fgrav_numerator = sigma * mass->mass * pow(r2 + r_soft * r_soft, -1.5);
    double fx = -fgrav_numerator * dx;
    double fy = -fgrav_numerator * dy;
    double fz = -fgrav_numerator * dz;
    double sink_rate = (dr < 4.0 * r_sink) ? mass->sink_rate * exp(-pow(dr / r_sink, 4.0)) : 0.0;
    double mdot = 0.0;

    if (sink_rate > 0.0)
    {
        mdot = -sink_rate * sigma;
    }
    else if (sink_rate < 0.0)
    {
        mdot = -sink_rate; // add constant M-dot for uniform sink.
    }

    // gravitational force
    delta_cons[0] += 0.0;
    delta_cons[1] += fx * dt;
    delta_cons[2] += fy * dt;
    delta_cons[3] += fz * dt;

    switch (mass->sink_model)
    {
        case 1: // acceleration-free
        {
            delta_cons[0] += dt * mdot;
            delta_cons[1] += dt * mdot * prim[1];
            delta_cons[2] += dt * mdot * prim[2];
            delta_cons[3] += dt * mdot * prim[3];
            break;
        }
        case 2: // torque-free
        {
            double vx = prim[1];
            double vy = prim[2];
            double vz = prim[3];
            double vx0 = mass->vx;
            double vy0 = mass->vy;
            double vz0 = mass->vz;
            double rhatx = dx / (dr + 1e-12);
            double rhaty = dy / (dr + 1e-12);
            double rhatz = dz / (dr + 1e-12);
            double dvdotrhat = (vx - vx0) * rhatx + (vy - vy0) * rhaty + (vz - vz0) * rhatz;
            double vxstar = dvdotrhat * rhatx + vx0;
            double vystar = dvdotrhat * rhaty + vy0;
            double vzstar = dvdotrhat * rhatz + vz0;
            delta_cons[0] += dt * mdot;
            delta_cons[1] += dt * mdot * vxstar;
            delta_cons[2] += dt * mdot * vystar;
            delta_cons[3] += dt * mdot * vzstar;
            break;
        }
        case 3: // force-free
        {
            delta_cons[0] += dt * mdot;
            delta_cons[1] += 0.0;
            delta_cons[2] += 0.0;
            delta_cons[3] += 0.0;
            break;
        }
        default: // sink is inactive
        {
            delta_cons[0] += 0.0;
            delta_cons[1] += 0.0;
            delta_cons[2] += 0.0;
            delta_cons[3] += 0.0;
            break;
        }
    }
}

PRIVATE void point_masses_source_term(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double z1,
    double dt,
    double *prim,
    double *delta_cons)
{
    for (int p = 0; p < 2; ++p)
    {
        point_mass_source_term(&mass_list->masses[p], x1, y1, z1, dt, prim, delta_cons);
    }
}


// ============================ EOS AND BUFFER ================================
// ============================================================================
PRIVATE double sound_speed_squared(
    double cs2,
    double mach_squared,
    int eos_type,
    double x,
    double y,
    double z,
    struct PointMassList *mass_list)
{
    switch (eos_type)
    {
        case 1: // globally isothermal
            return cs2;
        case 2: // locally Isothermal
            return -gravitational_potential(mass_list, x, y, z) / mach_squared;
        default:
            return 1.0; // WARNING
    }
}

PRIVATE void buffer_source_term(
    struct KeplerianBuffer *buffer,
    double xc,
    double yc,
    double zc,
    double dt,
    double *cons,
    double *delta_cons)
{
    if (buffer->is_enabled)
    {
        double rc = sqrt(xc * xc + yc * yc + zc * zc);
        double surface_density = buffer->surface_density;
        double central_mass = buffer->central_mass;
        double driving_rate = buffer->driving_rate;
        double outer_radius = buffer->outer_radius;
        double onset_width = buffer->onset_width;
        double onset_radius = outer_radius - onset_width;

        if (rc > onset_radius)
        {
            double v_kep = sqrt(central_mass / rc);
            double px = surface_density * (-yc / rc) * v_kep;
            double py = surface_density * (+xc / rc) * v_kep;
            double pz = surface_density * (+zc / rc) * v_kep;  // CHECK
            double u0[NCONS] = {surface_density, px, py, pz};
            double omega_outer = sqrt(central_mass * pow(onset_radius, -3.0));
            double buffer_rate = driving_rate * omega_outer * (rc - onset_radius) / (outer_radius - onset_radius);

            for (int q = 0; q < NCONS; ++q)
            {
                delta_cons[q] -= (cons[q] - u0[q]) * buffer_rate * dt;
            }
        }
    }
}

PRIVATE void shear_strain( //CHECK
    const double *gx,
    const double *gy,
    const double *gz,
    double dx,
    double dy,
    double dz,
    double *s)
{
    double sxx = 4.0 / 3.0 * gx[1] / dx - 2.0 / 3.0 * gy[2] / dy - 2.0 / 3.0 * gz[3] / dz;
    double syy =-2.0 / 3.0 * gx[1] / dx + 4.0 / 3.0 * gy[2] / dy - 2.0 / 3.0 * gz[3] / dz;
    double szz =-2.0 / 3.0 * gx[1] / dx - 2.0 / 3.0 * gy[2] / dy + 4.0 / 3.0 * gz[3] / dz;
    double sxy = 1.0 / 1.0 * gx[2] / dx + 1.0 / 1.0 * gy[1] / dy;
    double sxz = 1.0 / 1.0 * gx[3] / dx + 1.0 / 1.0 * gz[1] / dz;
    double syz = 1.0 / 1.0 * gy[3] / dy + 1.0 / 1.0 * gz[2] / dz;
    double syx = sxy;
    double szx = sxz;
    double szy = syz;
    s[0] = sxx;
    s[1] = sxy;
    s[2] = sxz;
    s[3] = syx;
    s[4] = syy;
    s[5] = syz;
    s[6] = szx;
    s[7] = szy;
    s[8] = szz;
}


// ============================ HYDRO =========================================
// ============================================================================
PRIVATE void conserved_to_primitive(
    const double *cons,
    double *prim,
    double velocity_ceiling,
    double density_floor)
{
    double rho = max2(cons[0], density_floor);
    double px = cons[1];
    double py = cons[2];
    double pz = cons[3];
    double vx = sign(px) * min2(fabs(px / rho), velocity_ceiling);
    double vy = sign(py) * min2(fabs(py / rho), velocity_ceiling);
    double vz = sign(pz) * min2(fabs(pz / rho), velocity_ceiling);

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = vz;
}

PRIVATE void primitive_to_conserved(
    const double *prim,
    double *cons)
{
    double rho = prim[0];
    double vx = prim[1];
    double vy = prim[2];
    double vz = prim[3];
    double px = vx * rho;
    double py = vy * rho;
    double pz = vz * rho;

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = pz;
}

PRIVATE double primitive_to_velocity(
    const double *prim,
    int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        case 2: return prim[3];
        default: return 0.0;
    }
}

PRIVATE void primitive_to_flux(
    const double *prim,
    const double *cons,
    double *flux,
    double cs2,
    int direction)
{
    double vn = primitive_to_velocity(prim, direction);
    double rho = prim[0];
    double pressure = rho * cs2;

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * cons[3] + pressure * (direction == 2);
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

PRIVATE double primitive_max_wavespeed(
    const double *prim,
    double cs2)
{
    double cs = sqrt(cs2);
    double vx = prim[1];
    double vy = prim[2];
    double vz = prim[3];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    double ay = max2(fabs(vy - cs), fabs(vy + cs));
    double az = max2(fabs(vz - cs), fabs(vz + cs));
    return max3(ax, ay, az);
}

PRIVATE void riemann_hlle(
    const double *pl,
    const double *pr,
    double *flux,
    double cs2,
    int direction)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux(pl, ul, fl, cs2, direction);
    primitive_to_flux(pr, ur, fr, cs2, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2, direction);

    const double am = min3(0.0, al[0], ar[0]);
    const double ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}


// ============================ PUBLIC API ====================================
// ============================================================================
// TODO : figure out this in 3D
PUBLIC void cbdiso_3d_advance_rk(
    int ni,
    int nj,
    int nk,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double patch_zl,
    double patch_zr,
    double *conserved_rk, // :: $.shape == (ni + 4, nj + 4, 4)
    double *primitive_rd, // :: $.shape == (ni + 4, nj + 4, 4)
    double *primitive_wr, // :: $.shape == (ni + 4, nj + 4, 4)
    double buffer_surface_density,
    double buffer_central_mass,
    double buffer_driving_rate,
    double buffer_outer_radius,
    double buffer_onset_width,
    int buffer_is_enabled,
    double x1, // point mass 1
    double y1,
    double z1, 
    double vx1,
    double vy1,
    double vz1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double x2, // point mass 2
    double y2,
    double z2,
    double vx2,
    double vy2,
    double vz2,
    double mass2,
    double softening_length2,
    double sink_rate2,
    double sink_radius2,
    int sink_model2,
    double cs2, // equation of state
    double mach_squared,
    int eos_type,
    double nu, // kinematic viscosity coefficient
    double a, // RK parameter
    double dt, // timestep
    double velocity_ceiling,
    double density_floor)
{
    struct KeplerianBuffer buffer = {
        buffer_surface_density,
        buffer_central_mass,
        buffer_driving_rate,
        buffer_outer_radius,
        buffer_onset_width,
        buffer_is_enabled
    };
    struct PointMass m1 = {x1, y1, z1, vx1, vy1, vz1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, z2, vx2, vy2, vz2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;
    double dz = (patch_zr - patch_zl) / nk;

    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng) * (nk + 2 * ng); // index striding
    int sj = NCONS * (nk + 2 * ng);
    int sk = NCONS;

    FOR_EACH_3D(ni, nj, nk)
    {
        double xl = patch_xl + (i + 0.0) * dx;
        double xc = patch_xl + (i + 0.5) * dx;
        double xr = patch_xl + (i + 1.0) * dx;
        double yl = patch_yl + (j + 0.0) * dy;
        double yc = patch_yl + (j + 0.5) * dy;
        double yr = patch_yl + (j + 1.0) * dy;
        double zl = patch_zl + (k + 0.0) * dz;
        double zc = patch_zl + (k + 0.5) * dz;
        double zr = patch_zl + (k + 1.0) * dz;

        // --------------------------------------------------------------------------------------------------------------------------------------
        //                   tj                 slice                                   slice                                                 
        //                                      above:                                  below:                                                 
        //      +-------+-------+-------+               +-------+-------+-------+               +-------+-------+-------+          
        //      |       |       |       |               |       |       |       |               |       |       |       |         
        //      |   lr  |   rj  |   rr  |               |       |  rjt  |       |               |       |  rjb  |       |      
        //      |       |       |       |               |       |       |       |               |       |       |       |       
        //      +-------+-------+-------+               +-------+-------+-------+               +-------+-------+-------+        above `rk` : tk
        //      |       |       |       |               |       |       |       |               |       |       |       |      
        //  bi  |  li  -|+  c  -|+  ri  |  ti           |  lit  |   rk  |  rit  |               |  lib  |   lk  |  rib  |      
        //      |       |       |       |               |       |       |       |               |       |       |       |       
        //      +-------+-------+-------+               +-------+-------+-------+               +-------+-------+-------+        below `lk` : bk
        //      |       |       |       |               |       |       |       |               |       |       |       |      
        //      |   ll  |   lj  |   rl  |               |       |  ljt  |       |               |       |  ljb  |       |      
        //      |       |       |       |               |       |       |       |               |       |       |       |      
        //      +-------+-------+-------+               +-------+-------+-------+               +-------+-------+-------+      
        //                                                                                                                     
        //                  bj                                                                                                    
        // --------------------------------------------------------------------------------------------------------------------------------------

        // stencil array indices
        int ncc = (i     + ng) * si + (j     + ng) * sj + (k     + ng) * sk;
        int nli = (i - 1 + ng) * si + (j     + ng) * sj + (k     + ng) * sk;
        int nri = (i + 1 + ng) * si + (j     + ng) * sj + (k     + ng) * sk;
        int nlj = (i     + ng) * si + (j - 1 + ng) * sj + (k     + ng) * sk;
        int nrj = (i     + ng) * si + (j + 1 + ng) * sj + (k     + ng) * sk;
        int nlk = (i     + ng) * si + (j     + ng) * sj + (k - 1 + ng) * sk;
        int nrk = (i     + ng) * si + (j     + ng) * sj + (k + 1 + ng) * sk;

        int nbi = (i - 2 + ng) * si + (j     + ng) * sj + (k     + ng) * sk;
        int nti = (i + 2 + ng) * si + (j     + ng) * sj + (k     + ng) * sk;
        int nbj = (i     + ng) * si + (j - 2 + ng) * sj + (k     + ng) * sk;
        int ntj = (i     + ng) * si + (j + 2 + ng) * sj + (k     + ng) * sk;
        int nbk = (i     + ng) * si + (j     + ng) * sj + (k - 2 + ng) * sk;
        int ntk = (i     + ng) * si + (j     + ng) * sj + (k + 2 + ng) * sk;
        
        int nll = (i - 1 + ng) * si + (j - 1 + ng) * sj + (k     + ng) * sk;
        int nlr = (i - 1 + ng) * si + (j + 1 + ng) * sj + (k     + ng) * sk;
        int nrl = (i + 1 + ng) * si + (j - 1 + ng) * sj + (k     + ng) * sk;
        int nrr = (i + 1 + ng) * si + (j + 1 + ng) * sj + (k     + ng) * sk;

        int nlit = (i - 1 + ng) * si + (j     + ng) * sj + (k + 1 + ng) * sk;
        int nrit = (i - 1 + ng) * si + (j     + ng) * sj + (k + 1 + ng) * sk;
        int nljt = (i     + ng) * si + (j - 1 + ng) * sj + (k + 1 + ng) * sk;
        int nrjt = (i     + ng) * si + (j + 1 + ng) * sj + (k + 1 + ng) * sk;
        int nlib = (i - 1 + ng) * si + (j     + ng) * sj + (k - 1 + ng) * sk;
        int nrib = (i - 1 + ng) * si + (j     + ng) * sj + (k - 1 + ng) * sk;
        int nljb = (i     + ng) * si + (j - 1 + ng) * sj + (k - 1 + ng) * sk;
        int nrjb = (i     + ng) * si + (j + 1 + ng) * sj + (k - 1 + ng) * sk;

        // stencil prims
        double *un = &conserved_rk[nccc];

        double *pcc = &primitive_rd[ncc];
        double *pli = &primitive_rd[nli];
        double *pri = &primitive_rd[nri];
        double *plj = &primitive_rd[nlj];
        double *prj = &primitive_rd[nrj];
        double *plk = &primitive_rd[nlk];
        double *prk = &primitive_rd[nrk];

        double *plit = &primitive_rd[nlit];
        double *prit = &primitive_rd[nrit];
        double *pljt = &primitive_rd[nljt];
        double *prjt = &primitive_rd[nrjt];
        double *plib = &primitive_rd[nlib];
        double *prib = &primitive_rd[nrib];
        double *pljb = &primitive_rd[nljb];
        double *prjb = &primitive_rd[nrjb];

        double *pbi = &primitive_rd[nbi];
        double *pti = &primitive_rd[nti];
        double *pbj = &primitive_rd[nbj];
        double *ptj = &primitive_rd[ntj];
        double *pbk = &primitive_rd[nbk];
        double *ptk = &primitive_rd[ntk];
        
        double *pll = &primitive_rd[nll];
        double *plr = &primitive_rd[nlr];
        double *prl = &primitive_rd[nrl];
        double *prr = &primitive_rd[nrr];

        // interpolated prims at faces
        double plip[NCONS];
        double plim[NCONS];
        double prip[NCONS];
        double prim[NCONS];
        double pljp[NCONS];
        double pljm[NCONS];
        double prjp[NCONS];
        double prjm[NCONS];
        double plkp[NCONS];
        double plkm[NCONS];
        double prkp[NCONS];
        double prkm[NCONS];

        // stencil gradients
        double gxli[NCONS];
        double gxri[NCONS];
        double gyli[NCONS];
        double gyri[NCONS];
        double gzli[NCONS];
        double gzri[NCONS];
        double gxlj[NCONS];
        double gxrj[NCONS];
        double gylj[NCONS];
        double gyrj[NCONS];
        double gzlj[NCONS];
        double gzrj[NCONS];
        double gxlk[NCONS];
        double gxrk[NCONS];
        double gylk[NCONS];
        double gyrk[NCONS];
        double gzlk[NCONS];
        double gzrk[NCONS];
        double gxcc[NCONS];
        double gycc[NCONS];
        double gzcc[NCONS];

        // calculate gradinet
        plm_gradient(pli , pcc, pri , gxcc);
        plm_gradient(pbi , pli, pcc , gxli);
        plm_gradient(pcc , pri, pti , gxri);
        plm_gradient(pll , plj, prl , gxlj);
        plm_gradient(plr , prj, prr , gxrj);
        plm_gradient(plib, plk, prib, gxlk);
        plm_gradient(plit, prk, prit, gxrk);
        plm_gradient(plj , pcc, prj , gycc);
        plm_gradient(pll , pli, plr , gyli);
        plm_gradient(prl , pri, prr , gyri);
        plm_gradient(pbj , plj, pcc , gylj);
        plm_gradient(pcc , prj, ptj , gyrj);
        plm_gradient(pljb, plk, prjb, gylk);
        plm_gradient(pljt, prk, prjt, gyrk);
        plm_gradient(plk , pcc, prk , gzcc);
        plm_gradient(plib, pli, plit, gzli);
        plm_gradient(prib, pri, prit, gzri);
        plm_gradient(pljb, plj, pljt, gzlj);
        plm_gradient(prjb, prj, prjt, gzrj);
        plm_gradient(pbk , plk, pcc , gzlk);
        plm_gradient(pcc , prk, ptk , gzrk);

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

            plkm[q] = plk[q] + 0.5 * gylk[q];
            plkp[q] = pcc[q] - 0.5 * gycc[q];
            prkm[q] = pcc[q] + 0.5 * gycc[q];
            prkp[q] = prk[q] - 0.5 * gyrk[q];
        }

        double fli[NCONS];
        double fri[NCONS];
        double flj[NCONS];
        double frj[NCONS];
        double flk[NCONS];
        double frk[NCONS];
        double ucc[NCONS];

        double cs2li = sound_speed_squared(cs2, mach_squared, eos_type, xl, yc, zc, &mass_list);
        double cs2ri = sound_speed_squared(cs2, mach_squared, eos_type, xr, yc, zc, &mass_list);
        double cs2lj = sound_speed_squared(cs2, mach_squared, eos_type, xc, yl, zc, &mass_list);
        double cs2rj = sound_speed_squared(cs2, mach_squared, eos_type, xc, yr, zc, &mass_list);
        double cs2lk = sound_speed_squared(cs2, mach_squared, eos_type, xc, yc, zl, &mass_list);
        double cs2rk = sound_speed_squared(cs2, mach_squared, eos_type, xc, yc, zr, &mass_list);

        riemann_hlle(plim, plip, fli, cs2li, 0);
        riemann_hlle(prim, prip, fri, cs2ri, 0);
        riemann_hlle(pljm, pljp, flj, cs2lj, 1);
        riemann_hlle(prjm, prjp, frj, cs2rj, 1);
        riemann_hlle(plkm, plkp, flk, cs2lk, 2);
        riemann_hlle(prkm, prkp, frk, cs2rk, 2);

        if (nu > 0.0)
        {
            double sli[9];
            double sri[9];
            double slj[9];
            double srj[9];
            double slk[9];
            double srk[9];
            double scc[9];

            shear_strain(gxli, gyli, gzli, dx, dy, dz, sli);
            shear_strain(gxri, gyri, gzri, dx, dy, dz, sri);
            shear_strain(gxlj, gylj, gzlj, dx, dy, dz, slj);
            shear_strain(gxrj, gyrj, gzrj, dx, dy, dz, srj);
            shear_strain(gxlk, gylk, gzlk, dx, dy, dz, slk);
            shear_strain(gxrk, gyrk, gzrk, dx, dy, dz, srk);
            shear_strain(gxcc, gycc, gzcc, dx, dy, dz, scc);

            fli[1] -= 0.5 * nu * (pli[0] * sli[0] + pcc[0] * scc[0]); // x-x
            fli[2] -= 0.5 * nu * (pli[0] * sli[1] + pcc[0] * scc[1]); // x-y
            fli[3] -= 0.5 * nu * (pli[0] * sli[2] + pcc[0] * scc[2]); // x-z
            fri[1] -= 0.5 * nu * (pcc[0] * scc[0] + pri[0] * sri[0]); // x-x
            fri[2] -= 0.5 * nu * (pcc[0] * scc[1] + pri[0] * sri[1]); // x-y
            fri[3] -= 0.5 * nu * (pcc[0] * scc[2] + pri[0] * sri[2]); // x-z

            flj[1] -= 0.5 * nu * (plj[0] * slj[3] + pcc[0] * scc[3]); // y-x
            flj[2] -= 0.5 * nu * (plj[0] * slj[4] + pcc[0] * scc[4]); // y-y
            flj[3] -= 0.5 * nu * (plj[0] * slj[5] + pcc[0] * scc[5]); // y-z
            frj[1] -= 0.5 * nu * (pcc[0] * scc[3] + prj[0] * srj[3]); // y-x
            frj[2] -= 0.5 * nu * (pcc[0] * scc[4] + prj[0] * srj[4]); // y-y
            frj[2] -= 0.5 * nu * (pcc[0] * scc[5] + prj[0] * srj[5]); // y-z

            flk[1] -= 0.5 * nu * (plk[0] * slk[6] + pcc[0] * scc[6]); // z-x
            flk[2] -= 0.5 * nu * (plk[0] * slk[7] + pcc[0] * scc[7]); // z-y
            flk[3] -= 0.5 * nu * (plk[0] * slk[8] + pcc[0] * scc[8]); // z-z
            frk[1] -= 0.5 * nu * (pcc[0] * scc[6] + prk[0] * srk[6]); // z-x
            frk[2] -= 0.5 * nu * (pcc[0] * scc[7] + prk[0] * srk[7]); // z-y
            frk[2] -= 0.5 * nu * (pcc[0] * scc[8] + prk[0] * srk[8]); // z-z
        }
        double delta_cons[4] = {0.0, 0.0, 0.0, 0.0};
        primitive_to_conserved(pcc, ucc);
        buffer_source_term(&buffer, xc, yc, zc, dt, ucc, delta_cons);
        point_masses_source_term(&mass_list, xc, yc, zc, dt, pcc, delta_cons);

        for (int q = 0; q < NCONS; ++q)
        {
            delta_cons[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy + (frk[q] - flk[q]) / dz) * dt;
        }
        for (int q = 0; q < NCONS; ++q)
        {
            ucc[q] += delta_cons[q];
            ucc[q] = (1.0 - a) * ucc[q] + a * un[q];
        }
        conserved_to_primitive(ucc, &primitive_wr[ncc], velocity_ceiling, density_floor);
    }
}


PUBLIC void cbdiso_3d_primitive_to_conserved(
    int ni,
    int nj,
    int nk,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *conserved) // :: $.shape == (ni + 4, nj + 4, 4)
{
    int ng = 2; 
    int si = NCONS * (nj + 2 * ng) * (nk + 2 * ng); 
    int sj = NCONS * (nk + 2 * ng);
    int sk = NCONS;

    FOR_EACH_3D(ni, nj, nk)
    {
        int n = (i + ng) * si + (j + ng) * sj + (k + ng) * sk;

        double *pc = &primitive[n];
        double *uc = &conserved[n];
        primitive_to_conserved(pc, uc);
    }
}

PUBLIC void cbdiso_3d_point_mass_source_term(
    int ni,
    int nj,
    int nk,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double patch_zl,
    double patch_zr,
    double x1, // particle
    double y1,
    double z1,
    double vx1,
    double vy1,
    double vz1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *cons_rate) // :: $.shape == (ni + 4, nj + 4, 4)
{
    struct PointMass m1 = {x1, y1, z1, vx1, vy1, vz1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};

    int ng = 2; 
    int si = NCONS * (nj + 2 * ng) * (nk + 2 * ng); 
    int sj = NCONS * (nk + 2 * ng);
    int sk = NCONS;

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;
    double dz = (patch_zr - patch_zl) / nk;

    FOR_EACH_3D(ni, nj, nk)
    {
        int ncc = (i + ng) * si + (j + ng) * sj + (k + ng) * sk;

        double xc = patch_xl + (i + 0.5) * dx;
        double yc = patch_yl + (j + 0.5) * dy;
        double zc = patch_zl + (k + 0.5) * dz;
        double *pc = &primitive[ncc];
        double *uc = &cons_rate[ncc];
        point_mass_source_term(&m1, xc, yc, zc, 1.0, pc, uc);
    }
}

PUBLIC void cbdiso_3d_wavespeed(
    int ni, // mesh
    int nj,
    int nk,
    double patch_xl,
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double patch_zl,
    double patch_zr,
    double soundspeed2, // equation of state
    double mach_squared,
    int eos_type,
    double x1, // point mass 1
    double y1,
    double z1,
    double vx1,
    double vy1,
    double vz1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double x2, // point mass 2
    double y2,
    double z2,
    double vx2,
    double vy2,
    double vz2,
    double mass2,
    double softening_length2,
    double sink_rate2,
    double sink_radius2,
    int sink_model2,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *wavespeed) // :: $.shape == (ni + 4, nj + 4)
{
    struct PointMass m1 = {x1, y1, z1, vx1, vy1, vz1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, z2, vx2, vy2, vz2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    int ng = 2; 
    int si = NCONS * (nj + 2 * ng) * (nk + 2 * ng); 
    int sj = NCONS * (nk + 2 * ng);
    int sk = NCONS;
    int ti = (nj + 2 * ng) * (nk + 2 * ng));
    int tj = nk + 2 * ng;
    int tk = 1;
    double dx = (patch_xr - patch_xl)/ni;
    double dy = (patch_yr - patch_yl)/nj;
    double dz = (patch_zr - patch_zl)/nk;

    FOR_EACH_3D(ni, nj, nk)
    {
        int np = (i + ng) * si + (j + ng) * sj + (k + ng) * sk;
        int na = (i + ng) * ti + (j + ng) * tj + (k + ng) * tk;

        double x = patch_xl + (i + 0.5) * dx;
        double y = patch_yl + (j + 0.5) * dy;
        double z = patch_zl + (k + 0.5) * dz;

        double *pc = &primitive[np];
        double cs2 = sound_speed_squared(soundspeed2, mach_squared, eos_type, x, y, z, &mass_list);
        double a = primitive_max_wavespeed(pc, cs2);
        wavespeed[na] = a;
    }
}