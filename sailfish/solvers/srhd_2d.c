/*
MODULE: srhd_2d

AUTHOR: Jonathan Zrake

DESCRIPTION:
  Solves relativistic hydrodynamics in 2D spherical-polar coordinates.
*/


// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 4
#define PLM_THETA 2.0
#define ADIABATIC_GAMMA (4.0 / 3.0)


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


// ============================ HYDRO =========================================
// ============================================================================
PRIVATE double primitive_to_gamma_beta_squared(const double *prim)
{
    const double u1 = prim[1];
    const double u2 = prim[2];
    return u1 * u1 + u2 * u2;
}

PRIVATE double primitive_to_lorentz_factor(const double *prim)
{
    return sqrt(1.0 + primitive_to_gamma_beta_squared(prim));
}

PRIVATE double primitive_to_gamma_beta_component(const double *prim, int direction)
{
    switch (direction)
    {
        case 1: return prim[1];
        case 2: return prim[2];
    }
    return 0.0;
}

PRIVATE double primitive_to_beta_component(const double *prim, int direction)
{
    const double w = primitive_to_lorentz_factor(prim);

    switch (direction)
    {
        case 1: return prim[1] / w;
        case 2: return prim[2] / w;
    }
    return 0.0;
}

PRIVATE double primitive_to_enthalpy_density(const double* prim)
{
    const double rho = prim[0];
    const double pre = prim[3];
    return rho + pre * (1.0 + 1.0 / (ADIABATIC_GAMMA - 1.0));
}

PRIVATE void primitive_to_conserved(const double *prim, double *cons, double dv)
{
    const double rho = prim[0];
    const double u1 = prim[1];
    const double u2 = prim[2];
    const double pre = prim[3];

    const double w = primitive_to_lorentz_factor(prim);
    const double h = primitive_to_enthalpy_density(prim) / rho;
    const double m = rho * w;

    cons[0] = dv * m;
    cons[1] = dv * m * h * u1;
    cons[2] = dv * m * h * u2;
    cons[3] = dv * m * (h * w - 1.0) - dv * pre;
    // cons[4] = dv * m * prim[3];
}

PRIVATE void conserved_to_primitive(double *cons, double *prim, double dv, double r, double q)
{
    const double newton_iter_max = 500;
    const double error_tolerance = 1e-12 * (cons[0] + cons[3]) / dv;
    const double gm              = ADIABATIC_GAMMA;
    const double m               = cons[0] / dv;
    const double tau             = cons[3] / dv;
    const double s1              = cons[1] / dv;
    const double s2              = cons[2] / dv;
    const double ss              = s1 * s1 + s2 * s2;
    int iteration                = 0;
    double p                     = prim[3];
    double w0;
    double f;

    while (1) {
        const double et = tau + p + m;
        const double b2 = min2(ss / et / et, 1.0 - 1e-10);
        const double w2 = 1.0 / (1.0 - b2);
        const double w  = sqrt(w2);
        const double e  = (tau + m * (1.0 - w) + p * (1.0 - w2)) / (m * w);
        const double d  = m / w;
        const double h  = 1.0 + e + p / d;
        const double a2 = gm * p / (d * h);
        const double g  = b2 * a2 - 1.0;

        f  = d * e * (gm - 1.0) - p;
        p -= f / g;

        if (fabs(f) < error_tolerance || iteration == newton_iter_max) {
            w0 = w;
            break;
        }
        iteration += 1;
    }

    prim[0] = m / w0;
    prim[1] = w0 * cons[1] / dv / (tau + m + p);
    prim[2] = w0 * cons[2] / dv / (tau + m + p);
    prim[3] = p;
    // prim[4] = cons[4] / cons[0];

    double mach_ceiling = 500.0;
    double u_squared = prim[1] * prim[1] + prim[2] * prim[2];
    double e = prim[3] / prim[0] * 3.0;
    double emin = u_squared / (1.0 + u_squared) / pow(mach_ceiling, 2.0);

    if (e < emin) {
        prim[3] = prim[0] * emin * (ADIABATIC_GAMMA - 1.0);
        // primitive_to_conserved(prim, cons, dv);
    }

    #if (EXEC_MODE != EXEC_GPU)
    if (iteration == newton_iter_max) {
        printf(
            "[FATAL] srhd_2d_conserved_to_primitive reached max "
            "iteration at position (%.3f %.3f) "
            "cons = [%.3e %.3e %.3e %.3e] error = %.3e\n", r, q, cons[0], cons[1], cons[2], cons[3], f);
        exit(1);
    }
    if (cons[3] <= 0.0) {
        printf(
            "[FATAL] srhd_2d_conserved_to_primitive found non-positive "
            "or NaN total energy tau=%.5e at position (%.3f %.3f)\n", cons[3], r, q);
        exit(1);
    }
    if (prim[3] <= 0.0 || prim[3] != prim[3]) {
        printf(
            "[FATAL] srhd_2d_conserved_to_primitive found non-positive "
            "or NaN pressure p=%.5e at position (%.3f %.3f)\n", prim[3], r, q);
        exit(1);
    }
    #endif
}

PRIVATE void primitive_to_flux(const double *prim, const double *cons, double *flux, int direction)
{
    const double vn = primitive_to_beta_component(prim, direction);
    const double pre = prim[3];
    // const double s = prim[4]; // scalar concentration

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pre * (direction == 1);
    flux[2] = vn * cons[2] + pre * (direction == 2);
    flux[3] = vn * cons[3] + pre * vn;
    // flux[4] = vn * cons[0] * s;
}

PRIVATE double primitive_to_sound_speed_squared(const double *prim)
{
    const double pre = prim[3];
    const double rho_h = primitive_to_enthalpy_density(prim);
    return ADIABATIC_GAMMA * pre / rho_h;
}

PRIVATE void primitive_to_outer_wavespeeds(const double *prim, double *wavespeeds, int direction)
{
    const double a2 = primitive_to_sound_speed_squared(prim);
    const double un = primitive_to_gamma_beta_component(prim, direction);
    const double uu = primitive_to_gamma_beta_squared(prim);
    const double vv = uu / (1.0 + uu);
    const double v2 = un * un / (1.0 + uu);
    const double vn = sqrt(v2);
    const double k0 = sqrt(a2 * (1.0 - vv) * (1.0 - vv * a2 - v2 * (1.0 - a2)));

    wavespeeds[0] = (vn * (1.0 - a2) - k0) / (1.0 - vv * a2);
    wavespeeds[1] = (vn * (1.0 - a2) + k0) / (1.0 - vv * a2);
}

PRIVATE void riemann_hlle(const double *pl, const double *pr, double v_face, double *flux, int direction)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    primitive_to_conserved(pl, ul, 1.0);
    primitive_to_conserved(pr, ur, 1.0);
    primitive_to_flux(pl, ul, fl, direction);
    primitive_to_flux(pr, ur, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, direction);
    primitive_to_outer_wavespeeds(pr, ar, direction);

    const double am = min2(al[0], ar[0]);
    const double ap = max2(al[1], ar[1]);

    if (v_face < am)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = fl[q] - v_face * ul[q];
        }
    }
    else if (v_face > ap)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = fr[q] - v_face * ur[q];
        }
    }
    else
    {    
        for (int q = 0; q < NCONS; ++q)
        {
            double u_hll = (ur[q] * ap - ul[q] * am + (fl[q] - fr[q]))           / (ap - am);
            double f_hll = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
            flux[q] = f_hll - v_face * u_hll;
        }
    }
}


// ============================ GEOMETRY ======================================
// ============================================================================
PRIVATE double face_area(double r0, double r1, double q0, double q1)
{
    double R0 = r0 * sin(q0);
    double R1 = r1 * sin(q1);
    double z0 = r0 * cos(q0);
    double z1 = r1 * cos(q1);
    double dR = R1 - R0;
    double dz = z1 - z0;
    return M_PI * (R0 + R1) * sqrt(dR * dR + dz * dz);
}

PRIVATE double cell_volume(double r0, double r1, double q0, double q1)
{
    return -(r1 * r1 * r1 - r0 * r0 * r0) * (cos(q1) - cos(q0)) * 2.0 * M_PI / 3.0;
}

PRIVATE void geometric_source_terms(double r0, double r1, double q0, double q1, const double *prim, double *source)
{
    double ur = prim[1];
    double uq = prim[2];
    double up = 0.0;
    double pg = prim[3];
    double rhoh = primitive_to_enthalpy_density(prim);

    double dcosq = cos(q1) - cos(q0);
    double dsinq = sin(q1) - sin(q0);
    double dr2 = r1 * r1 - r0 * r0;

    // The forumulas are A8 and A9 from Zhang & MacFadyen (2006), integrated
    // over the cell volume with finite radial and polar extent.
    // 
    // https://iopscience.iop.org/article/10.1086/500792/pdf
    double srdot = -M_PI * dr2 * dcosq * (rhoh * (uq * uq + up * up) + 2 * pg);
    double sqdot = +M_PI * dr2 * (dcosq * rhoh * ur * uq + dsinq * (pg + rhoh * up * up));

    source[0] = 0.0;
    source[1] = srdot;
    source[2] = sqdot;
    source[3] = 0.0;
    // source[4] = 0.0;
}


// ============================ KERNELS =======================================
// ============================================================================


/**
 * Converts an array of primitive data to an array of conserved data.
 */
PUBLIC void srhd_2d_primitive_to_conserved(
    int ni,
    int nj,
    double *face_positions,  // :: $.shape == (ni + 1,)
    double *primitive,       // :: $.shape == (ni + 4, nj, 4)
    double *conserved,       // :: $.shape == (ni + 4, nj, 4)
    double scale_factor)     // :: $ > 0.0
{
    int ng = 2; // number of guard zones in the radial direction
    int si = NCONS * nj;
    int sj = NCONS;
    double dq = M_PI / nj; // polar zone spacing (domain is pole-to-pole)

    FOR_EACH_2D(ni, nj)
    {
        int n = (i + ng) * si + j * sj;
        double *p = &primitive[n];
        double *u = &conserved[n];
        double x0 = face_positions[i];
        double x1 = face_positions[i + 1];
        double r0 = x0 * scale_factor;
        double r1 = x1 * scale_factor;
        double q0 = dq * (j + 0);
        double q1 = dq * (j + 1);
        double dv = cell_volume(r0, r1, q0, q1);
        primitive_to_conserved(p, u, dv);
    }
}


/**
 * Converts an array of conserved data to an array of primitive data.
 */
PUBLIC void srhd_2d_conserved_to_primitive(
    int ni,
    int nj,
    double *face_positions,  // :: $.shape == (ni + 1,)
    double *conserved,       // :: $.shape == (ni + 4, nj, 4)
    double *primitive,       // :: $.shape == (ni + 4, nj, 4)
    double scale_factor)     // :: $ > 0.0
{
    int ng = 2; // number of guard zones in the radial direction
    int si = NCONS * nj;
    int sj = NCONS;
    double dq = M_PI / nj; // polar zone spacing (domain is pole-to-pole)

    FOR_EACH_2D(ni, nj)
    {
        int n = (i + ng) * si + j * sj;
        double *p = &primitive[n];
        double *u = &conserved[n];
        double x0 = face_positions[i];
        double x1 = face_positions[i + 1];
        double r0 = x0 * scale_factor;
        double r1 = x1 * scale_factor;
        double q0 = dq * (j + 0);
        double q1 = dq * (j + 1);
        double dv = cell_volume(r0, r1, q0, q1);
        conserved_to_primitive(u, p, dv, r0, q0);
    }
}


/**
 * Updates an array of primitive data by advancing it a single Runge-Kutta
 * step.
 */
PUBLIC void srhd_2d_advance_rk(
    int ni,          // number of zones, not including guard zones
    int nj,
    double *face_positions, // :: $.shape == (ni + 1,)
    double *conserved_rk,   // :: $.shape == (ni + 4, nj, 4)
    double *primitive_rd,   // :: $.shape == (ni + 4, nj, 4)
    double *conserved_rd,   // :: $.shape == (ni + 4, nj, 4)
    double *conserved_wr,   // :: $.shape == (ni + 4, nj, 4)
    double a0,              // scale factor at t=0
    double adot,            // scale factor derivative
    double time,            // current time
    double rk_param,        // runge-kutta parameter
    double dt)              // timestep size
{
    int ng = 2; // number of guard zones in the radial direction
    int si = NCONS * nj;
    int sj = NCONS;
    double dq = M_PI / nj; // polar zone spacing (domain is pole-to-pole)

    FOR_EACH_2D(ni, nj)
    {
        double x0 = face_positions[i];
        double x1 = face_positions[i + 1];
        double r0 = x0 * (a0 + adot * time);
        double r1 = x1 * (a0 + adot * time);
        double q0 = dq * (j + 0);
        double q1 = dq * (j + 1);

        double *urk = &conserved_rk[(i + 0 + ng) * si + (j + 0) * sj];
        double *urd = &conserved_rd[(i + 0 + ng) * si + (j + 0) * sj];
        double *uwr = &conserved_wr[(i + 0 + ng) * si + (j + 0) * sj];
        double *pcc = &primitive_rd[(i + 0 + ng) * si + (j + 0) * sj];
        double *pli = &primitive_rd[(i - 1 + ng) * si + (j + 0) * sj];
        double *pri = &primitive_rd[(i + 1 + ng) * si + (j + 0) * sj];
        double *pki = &primitive_rd[(i - 2 + ng) * si + (j + 0) * sj];
        double *pti = &primitive_rd[(i + 2 + ng) * si + (j + 0) * sj];
        double *plj = &primitive_rd[(i + 0 + ng) * si + max2(j - 1, 0) * sj];
        double *prj = &primitive_rd[(i + 0 + ng) * si + min2(j + 1, nj - 1) * sj];
        double *pkj = &primitive_rd[(i + 0 + ng) * si + max2(j - 2, 0) * sj];
        double *ptj = &primitive_rd[(i + 0 + ng) * si + min2(j + 2, nj - 1) * sj];

        double plip[NCONS];
        double plim[NCONS];
        double prip[NCONS];
        double prim[NCONS];
        double pljp[NCONS];
        double pljm[NCONS];
        double prjp[NCONS];
        double prjm[NCONS];
        double grli[NCONS];
        double grri[NCONS];
        double grcc[NCONS];
        double gqlj[NCONS];
        double gqrj[NCONS];
        double gqcc[NCONS];
        double fli[NCONS];
        double fri[NCONS];
        double flj[NCONS];
        double frj[NCONS];
        double sources[NCONS];

        plm_gradient(pki, pli, pcc, grli);
        plm_gradient(pli, pcc, pri, grcc);
        plm_gradient(pcc, pri, pti, grri);
        plm_gradient(pkj, plj, pcc, gqlj);
        plm_gradient(plj, pcc, prj, gqcc);
        plm_gradient(pcc, prj, ptj, gqrj);

        for (int q = 0; q < NCONS; ++q)
        {
            plim[q] = pli[q] + 0.5 * grli[q];
            plip[q] = pcc[q] - 0.5 * grcc[q];
            prim[q] = pcc[q] + 0.5 * grcc[q];
            prip[q] = pri[q] - 0.5 * grri[q];
            pljm[q] = plj[q] + 0.5 * gqlj[q];
            pljp[q] = pcc[q] - 0.5 * gqcc[q];
            prjm[q] = pcc[q] + 0.5 * gqcc[q];
            prjp[q] = prj[q] - 0.5 * gqrj[q];
        }

        double da_r0 = face_area(r0, r0, q0, q1);
        double da_r1 = face_area(r1, r1, q0, q1);
        double da_q0 = face_area(r0, r1, q0, q0);
        double da_q1 = face_area(r0, r1, q1, q1);

        riemann_hlle(plim, plip, x0 * adot, fli, 1);
        riemann_hlle(prim, prip, x1 * adot, fri, 1);
        riemann_hlle(pljm, pljp, 0.0, flj, 2);
        riemann_hlle(prjm, prjp, 0.0, frj, 2);
        geometric_source_terms(x0, x1, q0, q1, pcc, sources);

        for (int q = 0; q < NCONS; ++q)
        {
            uwr[q] = urd[q] + (
                fli[q] * da_r0 - fri[q] * da_r1 +
                flj[q] * da_q0 - frj[q] * da_q1 + sources[q]
            ) * dt;
            uwr[q] = (1.0 - rk_param) * uwr[q] + rk_param * urk[q];
        }
    }
}
