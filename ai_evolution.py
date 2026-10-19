import os
import re
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from numba import njit, uint64

# ==============================================================================
# 1. PARAMETERS FROM Basic-2026xxxx.txt & SIMULATION SETTINGS
# ==============================================================================
BASIC_CONFIG = {
    "name": "2026xxxx",
    "t": 0.0,
    "a": 2.257,
    "b": 1.385,
    "c": 0.753,
    "sigma": 0.5,
    "dm_profile": "NFW",
    "baryon_profile": "Plummer",
    "rmin": 0.005,
    "rmax": 500.0,
    "shell_number": 160,
    "extra_shell": 10,
    "baryon_Plummer_mass_norm": 0.1,
    "baryon_Plummer_ars": 0.1,
}

SIM_CONFIG = {
    "total_steps": 10000,
    "save_steps": 100,
    "epsilon": 0.001,
    "mc_sweeps_per_step": 200,
    "density_threshold": 1e30,
    "output_file": f"result_{BASIC_CONFIG['name']}.txt",
}


# ==============================================================================
# 2. ANALYTICAL PROFILES & MONTE CARLO ENERGY KERNELS
# ==============================================================================
@njit(inline="always", fastmath=True)
def MbaryonP(myr, mymass, myscale):
    return mymass * (1.0 + (myscale ** 2) / (myr ** 2)) ** -1.5

@njit(inline="always", fastmath=True)
def rho_nfw(r):
    return 1.0 / (r * (1.0 + r) ** 2)

@njit(inline="always", fastmath=True)
def xorshift64(state):
    x = state[0]
    x ^= (x << uint64(13))
    x ^= (x >> uint64(7))
    x ^= (x << uint64(17))
    state[0] = x
    return x

@njit(inline="always", fastmath=True)
def rand_uniform(state):
    return (xorshift64(state) >> uint64(11)) * (1.0 / 9007199254740992.0)

@njit(inline="always", fastmath=True)
def I_plummer_kernel(r, a_s):
    """Exact antiderivative of r^2 / sqrt(r^2 + a_s^2) dr for Plummer potential."""
    if r <= 0.0:
        return 0.0
    sq = np.sqrt(r * r + a_s * a_s)
    return 0.5 * r * sq - 0.5 * (a_s * a_s) * np.arcsinh(r / a_s)

@njit(inline="always", fastmath=True)
def eval_shell_energy_mc(ra, rb, m_in, dm, a_ent, m_norm, a_s):
    """
    Computes exact dimensionless total energy (Internal + SIDM Gravity + Baryon Gravity)
    of a spherical shell bounded by [ra, rb] at constant dm and adiabatic variable a_ent.
    """
    ra2 = ra * ra
    ra3 = ra2 * ra
    rb2 = rb * rb
    rb3 = rb2 * rb

    d2 = rb2 - ra2
    d3 = rb3 - ra3
    d5 = rb3 * rb2 - ra3 * ra2
    inv_d3 = 1.0 / d3

    # 1. Internal energy: U_i = dm_i * u_i = 1.5 * dm_i * A_i * rho_i^(2/3)
    rho = 3.0 * dm * inv_d3
    u_int = 1.5 * dm * a_ent * (rho ** (2.0 / 3.0))

    # 2. SIDM gravitational energy (interior enclosed mass + shell self-gravity)
    w_sidm_int = -1.5 * m_in * dm * d2 * inv_d3
    w_sidm_self = -3.0 * (dm * dm) * (0.2 * d5 - 0.5 * ra3 * d2) * (inv_d3 * inv_d3)

    # 3. Static Plummer baryon gravitational potential energy
    w_bary = -3.0 * dm * m_norm * (I_plummer_kernel(rb, a_s) - I_plummer_kernel(ra, a_s)) * inv_d3

    return u_int + w_sidm_int + w_sidm_self + w_bary


# ==============================================================================
# 3. INITIAL CONDITION GENERATOR (EXACT VARIATIONAL EQUILIBRIUM)
# ==============================================================================
def generate_initial_profiles(cfg):
    """
    Constructs initial dimensionless profiles RList, RhoList, MList, uList, LList
    from Basic-2026xxxx.txt in exact variational hydrostatic balance (dE/dR_i = 0).
    """
    n_tot = cfg["shell_number"]
    n_extra = cfg["extra_shell"]
    no_layers = n_tot - n_extra

    rmin, rmax = cfg["rmin"], cfg["rmax"]
    a, b, c, sigma = cfg["a"], cfg["b"], cfg["c"], cfg["sigma"]
    mass_norm = cfg["baryon_Plummer_mass_norm"]
    scale_norm = cfg["baryon_Plummer_ars"]

    # 1. Logarithmic radial grid
    r_full = np.logspace(np.log10(rmin), np.log10(rmax), n_tot)

    # 2. Shell midpoint NFW density
    rho_full = np.zeros(n_tot)
    rho_full[0] = rho_nfw(0.5 * r_full[0])
    for i in range(1, n_tot):
        rho_full[i] = rho_nfw(0.5 * (r_full[i] + r_full[i - 1]))

    # 3. Enclosed SIDM mass M_i and individual shell masses dm_i
    m_full = np.zeros(n_tot)
    dm_full = np.zeros(n_tot)
    dm_full[0] = (1.0 / 3.0) * (r_full[0] ** 3) * rho_full[0]
    m_full[0] = dm_full[0]
    for i in range(1, n_tot):
        dm_full[i] = (1.0 / 3.0) * (r_full[i] ** 3 - r_full[i - 1] ** 3) * rho_full[i]
        m_full[i] = m_full[i - 1] + dm_full[i]

    # 4. Inward integration of variational gravitational gradient dW/dR_i
    p_full = np.zeros(n_tot)
    mhy_end = m_full[-1] + MbaryonP(r_full[-1], mass_norm, scale_norm)
    p_full[-1] = mhy_end * rho_full[-1] / (4.0 * r_full[-1])

    eps = 1e-7
    for i in range(n_tot - 2, -1, -1):
        ra = 0.0 if i == 0 else r_full[i - 1]
        r_curr = r_full[i]
        rb = r_full[i + 1]
        m_in_left = 0.0 if i == 0 else m_full[i - 1]
        m_in_right = m_full[i]
        dr = eps * r_curr

        w_plus = (
            eval_shell_energy_mc(ra, r_curr + dr, m_in_left, dm_full[i], 0.0, mass_norm, scale_norm) +
            eval_shell_energy_mc(r_curr + dr, rb, m_in_right, dm_full[i + 1], 0.0, mass_norm, scale_norm)
        )
        w_minus = (
            eval_shell_energy_mc(ra, r_curr - dr, m_in_left, dm_full[i], 0.0, mass_norm, scale_norm) +
            eval_shell_energy_mc(r_curr - dr, rb, m_in_right, dm_full[i + 1], 0.0, mass_norm, scale_norm)
        )
        dW_dR_i = (w_plus - w_minus) / (2.0 * dr)
        p_full[i] = p_full[i + 1] + dW_dR_i / (r_curr * r_curr)

    # 5. Specific internal energy u = 3/2 * P / rho and velocity dispersion v = sqrt(2/3 * u)
    u_full = 1.5 * p_full / rho_full
    v_full = np.sqrt((2.0 / 3.0) * u_full)

    # 6. Initial luminosity profile L(r)
    l_full = np.zeros(n_tot)
    sigma_2 = sigma * sigma
    v0, v1 = v_full[0], v_full[1]
    l_full[0] = (
        -(u_full[1] - u_full[0]) / r_full[1] * (r_full[0] ** 2) * a * b * c * sigma *
        (rho_full[0] * (v0 ** 3) / (a * c * sigma_2 * rho_full[0] * (v0 ** 2) + b) +
         rho_full[1] * (v1 ** 3) / (a * c * sigma_2 * rho_full[1] * (v1 ** 2) + b))
    )
    for i in range(1, n_tot - 1):
        vi, vi1 = v_full[i], v_full[i + 1]
        l_full[i] = (
            -(u_full[i + 1] - u_full[i]) / (r_full[i + 1] - r_full[i - 1]) * (r_full[i] ** 2) * a * b * c * sigma *
            (rho_full[i] * (vi ** 3) / (a * c * sigma_2 * rho_full[i] * (vi ** 2) + b) +
             rho_full[i + 1] * (vi1 ** 3) / (a * c * sigma_2 * rho_full[i + 1] * (vi1 ** 2) + b))
        )

    # 7. Truncate the 10 extra buffer shells (note: l_full[no_layers - 1] is already computed above!)
    return (
        r_full[:no_layers].copy(),
        rho_full[:no_layers].copy(),
        m_full[:no_layers].copy(),
        u_full[:no_layers].copy(),
        l_full[:no_layers].copy(),
    )


# ==============================================================================
# 4. IMPLICIT CONDUCTION & BIDIRECTIONAL ANTITHETIC MC RELAXATION KERNELS
# ==============================================================================
@njit(fastmath=True)
def numba_conduction_coefficients(LList, MList, uList, deltaUcoeff, NoLayers):
    deltaUcoeff[0] = -LList[1] / (MList[1] * uList[0])
    for i in range(1, NoLayers - 1):
        deltaUcoeff[i] = -(LList[i + 1] - LList[i - 1]) / (MList[i + 1] - MList[i - 1]) / uList[i]
    deltaUcoeff[NoLayers - 1] = 0.0


@njit(fastmath=True)
def numba_fill_implicit_matrix(C, ab, u, M, r, RhoList, deltat, a, b, c, sigma):
    NoLayers = len(u)
    dt = deltat

    # 1. Compute vi and kappa inline
    vi = np.zeros(NoLayers)
    for i in range(NoLayers):
        val = u[i]
        if val < 1e-50:
            val = 1e-50
        vi[i] = np.sqrt(2.0 / 3.0) * np.sqrt(val)

    kappa = np.zeros(NoLayers)
    a_c_sig2 = a * c * sigma * sigma
    ab_const = a * b * c * sigma
    for i in range(NoLayers):
        rho = RhoList[i]
        v_val = vi[i]
        denom = a_c_sig2 * rho * v_val * v_val + b
        if denom == 0.0:
            kappa[i] = 0.0
        else:
            kappa[i] = ab_const * (rho * v_val**3 / denom)

    # Clear arrays quickly
    C.fill(0.0)
    ab.fill(0.0)

    # 2. Exact original inner boundary (index 0)
    r_plus_0 = 0.5 * (r[0] + r[1])
    kappa_plus_0 = 0.5 * (kappa[0] + kappa[1])

    den_0 = M[1] * (r[1] - r[0])
    alpha_plus_0 = 0.0
    if den_0 != 0.0:
        alpha_plus_0 = dt * kappa_plus_0 * (r_plus_0**2) / den_0

    ab[1, 0] = 1.0 + alpha_plus_0
    ab[0, 1] = -alpha_plus_0
    C[0] = u[0]

    # 3. Interior layers (i = 1 to NoLayers - 2)
    for i in range(1, NoLayers - 1):
        r_plus = 0.5 * (r[i] + r[i + 1])
        r_minus = 0.5 * (r[i] + r[i - 1])
        kappa_plus = 0.5 * (kappa[i] + kappa[i + 1])
        kappa_minus = 0.5 * (kappa[i] + kappa[i - 1])

        dM = M[i + 1] - M[i - 1]
        den_plus = dM * (r[i + 1] - r[i])
        den_minus = dM * (r[i] - r[i - 1])

        alpha_plus = 0.0
        if den_plus != 0.0:
            alpha_plus = dt * kappa_plus * (r_plus**2) / den_plus

        alpha_minus = 0.0
        if den_minus != 0.0:
            alpha_minus = dt * kappa_minus * (r_minus**2) / den_minus

        ab[0, i + 1] = -alpha_plus                      # Superdiagonal (row i, col i+1)
        ab[1, i] = 1.0 + alpha_plus + alpha_minus       # Main diagonal (row i, col i)
        ab[2, i - 1] = -alpha_minus                     # Subdiagonal   (row i, col i-1)
        C[i] = u[i]

    # 4. Outer boundary (index NoLayers - 1, held at fixed boundary temperature)
    last = NoLayers - 1
    ab[1, last] = 1.0
    C[last] = u[last]


@njit(fastmath=True)
def numba_mc_relaxation_step(RList, RhoList, pList, MList, dmList, aList,
                             mc_step_sizes, mass_norm, scale_norm, n_sweeps, rng_state):
    """
    Bidirectional Antithetic Adaptive Monte Carlo Relaxation:
    Minimizes the total energy functional E = sum(U_i + W_sidm_i + W_bary_i)
    with respect to interior shell radii RList[0 ... NoLayers-2] holding dmList and aList fixed.
    """
    NoLayers = len(RList)
    n_inner = NoLayers - 1

    # Cache current shell energies
    E_shells = np.zeros(NoLayers, dtype=np.float64)
    E_shells[0] = eval_shell_energy_mc(
        0.0, RList[0], 0.0, dmList[0], aList[0], mass_norm, scale_norm
    )
    for i in range(1, NoLayers):
        E_shells[i] = eval_shell_energy_mc(
            RList[i - 1], RList[i], MList[i - 1], dmList[i], aList[i], mass_norm, scale_norm
        )

    # Ensure step sizes start above a sensible minimum floor at the start of the step
    for i in range(n_inner):
        r_prev = 0.0 if i == 0 else RList[i - 1]
        min_start = 1e-5 * (RList[i] - r_prev)
        if mc_step_sizes[i] < min_start:
            mc_step_sizes[i] = min_start

    for sweep in range(n_sweeps):
        # Alternate forward (outward) and backward (inward) sweeps
        forward = (sweep & 1) == 0

        for idx in range(n_inner):
            i = idx if forward else (n_inner - 1 - idx)

            r_prev = 0.0 if i == 0 else RList[i - 1]
            r_curr = RList[i]
            r_next = RList[i + 1]

            # Draw random step magnitude in [0.1 * step_size, 1.0 * step_size] with random sign
            u_rand = rand_uniform(rng_state)
            mag = (0.1 + 0.9 * u_rand) * mc_step_sizes[i]
            dr = mag if rand_uniform(rng_state) < 0.5 else -mag

            m_in_left = 0.0 if i == 0 else MList[i - 1]
            m_in_right = MList[i]
            e_old = E_shells[i] + E_shells[i + 1]

            accepted = False
            r_trial = r_curr + dr
            if r_prev < r_trial < r_next:
                e_new1 = eval_shell_energy_mc(
                    r_prev, r_trial, m_in_left, dmList[i], aList[i], mass_norm, scale_norm
                )
                e_new2 = eval_shell_energy_mc(
                    r_trial, r_next, m_in_right, dmList[i + 1], aList[i + 1], mass_norm, scale_norm
                )
                if (e_new1 + e_new2) < e_old:
                    RList[i] = r_trial
                    E_shells[i] = e_new1
                    E_shells[i + 1] = e_new2
                    accepted = True

            # Antithetic trial (-dr) if +dr went uphill
            if not accepted:
                r_trial = r_curr - dr
                if r_prev < r_trial < r_next:
                    e_new1 = eval_shell_energy_mc(
                        r_prev, r_trial, m_in_left, dmList[i], aList[i], mass_norm, scale_norm
                    )
                    e_new2 = eval_shell_energy_mc(
                        r_trial, r_next, m_in_right, dmList[i + 1], aList[i + 1], mass_norm, scale_norm
                    )
                    if (e_new1 + e_new2) < e_old:
                        RList[i] = r_trial
                        E_shells[i] = e_new1
                        E_shells[i + 1] = e_new2
                        accepted = True

            # Adapt step size based on whether the trial bracketed or overshot the minimum
            if accepted:
                max_cap = 0.05 * (r_next - r_prev)
                new_step = mc_step_sizes[i] * 1.2
                mc_step_sizes[i] = new_step if new_step < max_cap else max_cap
            else:
                min_floor = 1e-11 * r_curr
                new_step = mc_step_sizes[i] * 0.7
                mc_step_sizes[i] = new_step if new_step > min_floor else min_floor

    # Recompute exact RhoList and pList from the relaxed radii and conserved (dmList, aList)
    RhoList[0] = 3.0 * dmList[0] / (RList[0] ** 3)
    pList[0] = aList[0] * (RhoList[0] ** (5.0 / 3.0))
    for i in range(1, NoLayers):
        RhoList[i] = 3.0 * dmList[i] / (RList[i] ** 3 - RList[i - 1] ** 3)
        pList[i] = aList[i] * (RhoList[i] ** (5.0 / 3.0))


@njit(fastmath=True)
def numba_update_profiles(RList, RhoList, MList, MhyList, uList, vList, aList, LList,
                          mass_norm, scale_norm, a, b, c, sigma):
    NoLayers = len(RList)
    sigma_2 = sigma * sigma

    for i in range(NoLayers - 1):
        MhyList[i] = MList[i] + MbaryonP(RList[i], mass_norm, scale_norm)

    uList[:] = 1.5 * aList * (RhoList ** (2.0 / 3.0))
    vList[:] = np.sqrt(2.0 / 3.0) * np.sqrt(np.maximum(uList, 1e-50))

    R0 = RList[0]
    R0_2 = R0 * R0
    v0 = vList[0]
    v0_2 = v0 * v0
    v0_3 = v0_2 * v0
    v1 = vList[1]
    v1_2 = v1 * v1
    v1_3 = v1_2 * v1

    LList[0] = (
        -(uList[1] - uList[0]) / RList[1] * R0_2 * a * b * c * sigma *
        (RhoList[0] * v0_3 / (a * c * sigma_2 * RhoList[0] * v0_2 + b) +
         RhoList[1] * v1_3 / (a * c * sigma_2 * RhoList[1] * v1_2 + b))
    )

    for i in range(1, NoLayers - 1):
        Ri = RList[i]
        Ri_2 = Ri * Ri
        Ri1 = RList[i + 1]
        Ri_1 = RList[i - 1]

        vi = vList[i]
        vi_2 = vi * vi
        vi_3 = vi_2 * vi

        vi1 = vList[i + 1]
        vi1_2 = vi1 * vi1
        vi1_3 = vi1_2 * vi1

        LList[i] = (
            -(uList[i + 1] - uList[i]) / (Ri1 - Ri_1) * Ri_2 * a * b * c * sigma *
            (RhoList[i] * vi_3 / (a * c * sigma_2 * RhoList[i] * vi_2 + b) +
             RhoList[i + 1] * vi1_3 / (a * c * sigma_2 * RhoList[i + 1] * vi1_2 + b))
        )


# ==============================================================================
# 5. SNAPSHOT FILE EXPORT & READER FUNCTIONS
# ==============================================================================
def write_snapshot_header(file_handle, state, basic_cfg, sim_cfg, initial_time):
    """Writes the 14-line metadata header matching my_evolution_v1.ipynb."""
    file_handle.write(
        f"Input profile name:{basic_cfg['name']}\n"
        f"Initial time:{initial_time}\n"
        f"Number of layers:{state.NoLayers}\n"
        f"Initial r inner most:{state.RList[0]}\n"
        f"Initial r outer most:{state.RList[state.NoLayers - 1]}\n"
        f"Initial rho inner most:{state.RhoList[0]}\n"
        f"Initial rho outer most:{state.RhoList[state.NoLayers - 1]}\n"
        f"Total step:{sim_cfg['total_steps']}\n"
        f"Save step:{sim_cfg['save_steps']}\n"
        f"Abs(delta u/u):{sim_cfg['epsilon']}\n"
        f"Cross section (sigma):{basic_cfg['sigma']}\n"
        f"Conduction parameter (a,b,c):{basic_cfg['a']},{basic_cfg['b']},{basic_cfg['c']}\n"
        f"Baryon parameter (mass_norm,scale_norm):{basic_cfg['baryon_Plummer_mass_norm']},{basic_cfg['baryon_Plummer_ars']}\n"
        f"time, step, SIDM radius, SIDM density, SIDM enclosed mass, SIDM internal energy, SIDM luminosity\n"
    )


def write_single_snapshot(file_handle, t_val, step_val, r_arr, rho_arr, m_arr, u_arr, l_arr):
    """Appends a single time snapshot block (time, step, and 5 bracketed arrays) to file."""
    file_handle.write(
        f"{t_val} {step_val}\n"
        f"{np.array2string(r_arr.T)}\n"
        f"{np.array2string(rho_arr.T)}\n"
        f"{np.array2string(m_arr.T)}\n"
        f"{np.array2string(u_arr.T)}\n"
        f"{np.array2string(l_arr.T)}\n"
    )


def export_snapshots_to_file(snapshots, output_filepath, basic_cfg, sim_cfg):
    """
    Standalone function to export a list of recorded snapshots to a text file
    in the exact format read by read_gravothermal_output() in my_evolution_v1.ipynb.
    """
    if not snapshots:
        raise ValueError("Snapshot list is empty.")

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    snap0 = snapshots[0]
    no_layers = len(snap0["r"])

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(
            f"Input profile name:{basic_cfg['name']}\n"
            f"Initial time:{snap0['time']}\n"
            f"Number of layers:{no_layers}\n"
            f"Initial r inner most:{snap0['r'][0]}\n"
            f"Initial r outer most:{snap0['r'][-1]}\n"
            f"Initial rho inner most:{snap0['rho'][0]}\n"
            f"Initial rho outer most:{snap0['rho'][-1]}\n"
            f"Total step:{sim_cfg['total_steps']}\n"
            f"Save step:{sim_cfg['save_steps']}\n"
            f"Abs(delta u/u):{sim_cfg['epsilon']}\n"
            f"Cross section (sigma):{basic_cfg['sigma']}\n"
            f"Conduction parameter (a,b,c):{basic_cfg['a']},{basic_cfg['b']},{basic_cfg['c']}\n"
            f"Baryon parameter (mass_norm,scale_norm):{basic_cfg['baryon_Plummer_mass_norm']},{basic_cfg['baryon_Plummer_ars']}\n"
            f"time, step, SIDM radius, SIDM density, SIDM enclosed mass, SIDM internal energy, SIDM luminosity\n"
        )
        for snap in snapshots:
            write_single_snapshot(
                f, snap["time"], snap["step"],
                snap["r"], snap["rho"], snap["M"], snap["u"], snap["L"]
            )

    print(f"Successfully exported {len(snapshots)} snapshots to: {output_filepath}")


def read_gravothermal_output(filename):
    """Reads a saved snapshot file back into a list of dictionaries."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_data = "".join(lines[14:])
    array_strings = re.findall(r"\[(.*?)\]", raw_data, re.DOTALL)
    time_step_text = re.sub(r"\[.*?\]", "", raw_data, flags=re.DOTALL)
    time_lines = [line.strip() for line in time_step_text.split("\n") if line.strip()]

    snapshots = []
    for i, t_line in enumerate(time_lines):
        t_val, step_val = map(float, t_line.split())
        idx = i * 5
        snapshots.append({
            "time": t_val,
            "step": int(step_val),
            "r": np.fromstring(array_strings[idx], sep=" "),
            "rho": np.fromstring(array_strings[idx + 1], sep=" "),
            "M": np.fromstring(array_strings[idx + 2], sep=" "),
            "u": np.fromstring(array_strings[idx + 3], sep=" "),
            "L": np.fromstring(array_strings[idx + 4], sep=" "),
        })
    return snapshots


# ==============================================================================
# 6. SIMULATION STATE & SIMULATOR CLASSES
# ==============================================================================
class SimulationState:
    def __init__(self, basic_cfg):
        self.cfg = basic_cfg

    def initialize(self):
        self.RList, self.RhoList, self.MList, self.uList, self.LList = generate_initial_profiles(self.cfg)
        self.NoLayers = len(self.RList)

        self.dmList = np.zeros(self.NoLayers)
        self.dmList[0] = self.MList[0]
        self.dmList[1:] = self.MList[1:] - self.MList[:-1]

        self.MhyList = np.zeros(self.NoLayers)
        self.vList = np.sqrt(2.0 / 3.0) * np.sqrt(self.uList)
        self.pList = (2.0 / 3.0) * self.RhoList * self.uList
        self.aList = (2.0 / 3.0) * (self.RhoList ** (-2.0 / 3.0)) * self.uList

        mass_norm = self.cfg["baryon_Plummer_mass_norm"]
        scale_norm = self.cfg["baryon_Plummer_ars"]
        for i in range(self.NoLayers - 1):
            self.MhyList[i] = self.MList[i] + MbaryonP(self.RList[i], mass_norm, scale_norm)

        print("Number of layers:", self.NoLayers)
        print(f"Initial r inner most: {self.RList[0]:.6g}")
        print(f"Initial r next to outer most: {self.RList[-2]:.10g}")
        print(f"Initial rho inner most: {self.RhoList[0]:.10g}")
        print(f"Initial rho next to outer most: {self.RhoList[-2]:.9e}")

    def check_abnormal_state(self, density_threshold):
        if self.RhoList[0] > density_threshold:
            print("Rho reaches threshold!")
            return True
        if self.RList[0] < 0:
            print("R[0] is negative!")
            return True
        if np.isnan(self.RList[0]):
            print("R is nan!")
            return True
        return False


class GravothermalMCSimulator:
    def __init__(self, state, basic_cfg, sim_cfg):
        self.state = state
        self.bcfg = basic_cfg
        self.scfg = sim_cfg
        self.total_time = basic_cfg["t"]
        self.snapshots = []

    def initialize(self):
        self.state.initialize()
        NoLayers = self.state.NoLayers

        self.deltaUcoeff = np.zeros(NoLayers)
        self.C = np.zeros(NoLayers)
        self.ab = np.zeros((3, NoLayers))

        # Persistent per-shell adaptive step sizes and PRNG state for MC relaxation
        self.mc_step_sizes = 1e-4 * self.state.RList[:-1].copy()
        self.rng_state = np.array([88172645463325252], dtype=np.uint64)

    def performConductionStep(self):
        NoLayers = self.state.NoLayers
        epsilon = self.scfg["epsilon"]
        a, b, c, sigma = self.bcfg["a"], self.bcfg["b"], self.bcfg["c"], self.bcfg["sigma"]

        numba_conduction_coefficients(
            self.state.LList, self.state.MList, self.state.uList, self.deltaUcoeff, NoLayers
        )
        deltat = epsilon / np.max(np.abs(self.deltaUcoeff))

        numba_fill_implicit_matrix(
            self.C, self.ab, self.state.uList, self.state.MList,
            self.state.RList, self.state.RhoList, deltat, a, b, c, sigma
        )

        self.state.uList[:] = solve_banded((1, 1), self.ab, self.C)

        self.state.pList[:] = (2.0 / 3.0) * (self.state.RhoList * self.state.uList)
        self.state.aList[:] = (2.0 / 3.0) * (self.state.RhoList ** (-2.0 / 3.0)) * self.state.uList

        return deltat

    def performMCRelaxationStep(self):
        numba_mc_relaxation_step(
            self.state.RList, self.state.RhoList, self.state.pList,
            self.state.MList, self.state.dmList, self.state.aList,
            self.mc_step_sizes,
            self.bcfg["baryon_Plummer_mass_norm"], self.bcfg["baryon_Plummer_ars"],
            self.scfg["mc_sweeps_per_step"], self.rng_state
        )

    def updateProfiles(self):
        numba_update_profiles(
            self.state.RList, self.state.RhoList, self.state.MList, self.state.MhyList,
            self.state.uList, self.state.vList, self.state.aList, self.state.LList,
            self.bcfg["baryon_Plummer_mass_norm"], self.bcfg["baryon_Plummer_ars"],
            self.bcfg["a"], self.bcfg["b"], self.bcfg["c"], self.bcfg["sigma"]
        )

    def record_snapshot(self, step):
        self.snapshots.append({
            "time": self.total_time,
            "step": step,
            "r": self.state.RList.copy(),
            "rho": self.state.RhoList.copy(),
            "M": self.state.MList.copy(),
            "u": self.state.uList.copy(),
            "L": self.state.LList.copy(),
        })

    def run(self):
        self.initialize()
        self.record_snapshot(0)

        total_steps = self.scfg["total_steps"]
        current_save_step = self.scfg["save_steps"]
        density_threshold = self.scfg["density_threshold"]

        print("Implicit Conduction + Bidirectional Antithetic MC Relaxation starts!")
        t_start = time.perf_counter()

        for tstep in range(1, total_steps + 1):
            try:
                deltat = self.performConductionStep()
                self.total_time += deltat

                self.performMCRelaxationStep()
                self.updateProfiles()

                if self.state.RhoList[0] >= 1e6 and current_save_step != 1:
                    current_save_step = 1
                    print("Density threshold 1e6 reached! Save frequency increased to every step.")

                if self.state.check_abnormal_state(density_threshold):
                    break

                if (tstep % current_save_step) == 0:
                    self.record_snapshot(tstep)
                    print(f"Saved at time= {self.total_time:.16f} , step= {tstep}")

            except Exception as e:
                print(f"Error during simulation step {tstep}: {e}")
                traceback.print_exc()
                break

        t_elapsed = time.perf_counter() - t_start
        print(f"Evolution ends. Computation time: {t_elapsed:.4f}s")

        # Export all recorded snapshots to the output file
        export_snapshots_to_file(
            self.snapshots, self.scfg["output_file"], self.bcfg, self.scfg
        )
        return self.snapshots


# ==============================================================================
# 7. MAIN EXECUTION & PLOTTING FROM SAVED SNAPSHOT FILE
# ==============================================================================
def plot_gravothermal_evolution(snapshots):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

    for i, snap in enumerate(snapshots):
        axes[0].plot(snap["r"], snap["u"], color=colors[i], lw=1.5)
        axes[1].plot(snap["r"], snap["L"], color=colors[i], lw=1.5)
        axes[2].plot(snap["r"], snap["rho"], color=colors[i], lw=1.5)

    axes[0].set_title("Internal Energy ($u$) vs Radius", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Radius ($r$)", fontsize=12)
    axes[0].set_ylabel("Internal Energy ($u$)", fontsize=12)
    axes[0].set_xscale("log")
    axes[0].grid(True, which="both", ls=":", alpha=0.6)

    axes[1].set_title("Luminosity ($L$) vs Radius", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Radius ($r$)", fontsize=12)
    axes[1].set_ylabel("Luminosity Flux ($L$)", fontsize=12)
    axes[1].set_xscale("log")
    axes[1].grid(True, which="both", ls=":", alpha=0.6)

    axes[2].set_title(r"Density ($\rho$) vs Radius", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Radius ($r$)", fontsize=12)
    axes[2].set_ylabel(r"Density ($\rho$)", fontsize=12)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].grid(True, which="both", ls=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    state = SimulationState(BASIC_CONFIG)
    sim = GravothermalMCSimulator(state, BASIC_CONFIG, SIM_CONFIG)
    sim.run()

    # Load snapshots back from the exported file and plot
    loaded_snapshots = read_gravothermal_output(SIM_CONFIG["output_file"])
    print(f"Successfully loaded {len(loaded_snapshots)} time snapshots from {SIM_CONFIG['output_file']}.")
    plot_gravothermal_evolution(loaded_snapshots)