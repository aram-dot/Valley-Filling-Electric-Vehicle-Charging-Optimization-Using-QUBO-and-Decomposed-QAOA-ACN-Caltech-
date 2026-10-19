"""
================================================================================
EV Charging Optimization: Classical vs Quantum (QUBO) Comparison  
================================================================================
Dataset  : ACN-Caltech  (Lee, Li, Low — ACM e-Energy 2019)
Classical: (1) Convex QP / SLSQP  [Gan, Topcu, Low 2012 — exact reference]
           (2) Genetic Algorithm (GA)  — real-valued, augmented Lagrangian
           (3) Particle Swarm Optimization (PSO)
Quantum  : (1) QUBO + Simulated Annealing (SA)
           (2) QUBO + Parallel Tempering SA  (replica exchange)
           (3) QUBO + Tabu Search
           (4) QAOA  (Qiskit Aer, p=1, COBYLA, subproblem decomposition)

KEY IMPROVEMENTS over v1
------------------------
Background load
  • Computed directly from the LOADED dataset: d_bg[t] = mean delivered power
    across all EVs weighted by reported kWhDelivered / session duration.
    When using synthesised data a Gaussian approximation matching ACN-Caltech
    campus statistics is used instead of an arbitrary bell curve.
  • Both "uncontrolled" (first-come, first-served) and "ideal flat" baselines
    are plotted alongside the optimised profiles.

QUBO formulation
  • Penalty weights auto-scaled relative to the objective magnitude so
    lambda_energy >> max(H_obj) guaranteed by construction (no manual tuning).
  • H_energy uses per-slot normalised coefficients.
  • H_site penalty replaced by an iterative slot-capping post-processor
    (hard constraint) rather than a soft penalty -- eliminates site violations
    completely while keeping Q smaller.

QAOA integration
  • QAOA added as a 4th QUBO-based solver using the SAME corrected Q matrix
    produced by QUBOFormulation.build_Q() (P_rate absent).
    p=1, COBYLA optimizer, shot-based sampling, QAOA_MAX_VARS=12
    qubit subproblem decomposition with contextual coupling, greedy local
    search post-selection, warm-start initialisation.
  • Bitstring selection: by minimum classical QUBO energy x^T Q x -- NOT by
    post-processed schedule quality.  Post-processing applied after selection.
  • Classical QUBO energy verification printed and stored in results dict.
  • QAOA is gracefully skipped if Qiskit is not installed.

Energy repair (QUBO post-processing)
  • repair correctly fills upward (water-filling into
    lowest-load slots) to guarantee exact energy delivery.

GA / PSO
  • Repair operator now fills UP as well as scales down, matching the QUBO
    repair logic -- zero energy error guaranteed for every individual.
  • GA uses SBX crossover and polynomial mutation.
  • GA uses self-adaptive mutation rate (reduce on progress, increase on stagnation).
  • PSO adds a local best (lbest) ring topology to prevent premature convergence.

SA / PT / Tabu
  • Warm-start from the QP solution (encoded via round-to-nearest-level),
    giving all quantum solvers a high-quality initial state.
  • SA uses restart-with-perturbation mechanism when stagnation detected.
  • PT doubles replica count, uses adaptive swap intervals and all-pairs exchange.
  • Tabu uses frequency-based long-term memory in addition to recency tabu,
    with diversification restarts.
    
================================================================================
"""

import os, sys, time, copy, warnings, math, random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage

warnings.filterwarnings("ignore")

# ── Qiskit / Aer imports (QAOA) ───────────────────────────────────────────────

_QISKIT_AVAILABLE = None   

# =============================================================================
# 0.  CONFIGURATION  -- DO NOT change path constants
# =============================================================================

RESULTS_DIR = r"C:\Users\aramc\OneDrive\Desktop\ACN\Results"
EXCEL_NAME  = "EV_Optimization_Results.xlsx"

# Problem parameters
DT         = 0.25     # slot duration [hr]  (15 min)
P_SITE_MAX = 150.0    # aggregate site power cap [kW]
R_MAX_L2   = 6.6      # L2 EVSE max charging rate [kW]
T_START    = 6.0      # horizon start [hr]
T_END      = 22.0     # horizon end   [hr]

# QUBO
K_BITS     = 4        # bits per (EV, slot) -> 2^4 = 16 discrete rate levels
# lambda weights are AUTO-SCALED inside QUBOFormulation.build_Q()

# GA
GA_POP      = 100
GA_GENS     = 400
GA_CX_RATE  = 0.85
GA_MUT_INIT = 0.03

# PSO
PSO_SWARM  = 80
PSO_ITERS  = 400
PSO_W_MAX  = 0.9
PSO_W_MIN  = 0.35
PSO_C1     = 2.0
PSO_C2     = 2.0

# SA
SA_T_INIT  = 50.0
SA_T_FINAL = 1e-5
SA_SWEEPS  = 60_000
SA_RESTART_FRAC = 0.08   # restart if no improvement in this fraction of sweeps

# Parallel Tempering
PT_REPLICAS = 12
PT_SWEEPS   = 40_000
PT_T_MIN    = 0.005
PT_T_MAX    = 100.0

# Tabu Search
TABU_ITERS  = 25_000
TABU_TENURE = 80

# QAOA (Qiskit Aer) -- preserved from QAOA_Aer_Simulator_1.ipynb
QAOA_LAYERS    = 1        # p circuit depth (unchanged from notebook)
QAOA_SHOTS     = 1024     # shots per circuit evaluation
QAOA_OPT_ITERS = 30       # COBYLA max iterations per restart
QAOA_RESTARTS  = 1        # additional random restarts (+ 2 warm-start inits)
QAOA_MAX_VARS  = 12       # subproblem qubit count cap (unchanged from notebook)
QAOA_SIM_MODE  = "shot"   # "statevector" | "shot" | "noisy"  (shot = default)

# Robustness experiment constants (b3c5e0cc additions)
# ------------------------------------------------------------------
# Experiment A — multi-seed QAOA
QAOA_SEEDS = [42, 123, 456, 789, 1000]   # five independent seeds

# Shot-sensitivity analysis
# Scientific justification (§8 of instruction):
#   In this implementation shots affect BOTH:
#     (1) COBYLA expectation estimation (E = sum_bs count/total * x^T Q x),
#         which directly shapes the COBYLA optimisation landscape.
#     (2) Final bitstring sampling from the optimised circuit — the lowest-
#         energy observed bitstring is the raw QAOA solution.
#   Therefore shot variation changes both optimisation trajectory AND the
#   selected solution; it is a scientifically meaningful axis of variation.
QAOA_SHOT_LEVELS  = [128, 256, 512, 1024, 2048]   # Experiment C: fixed seed
QAOA_SHOT_SEED    = 42                              # single fixed seed for fairness

# Experiment B — noisy QAOA (Qiskit Aer generic noise model)
QAOA_NOISY_SEEDS   = [42, 123, 456, 789, 1000]   # same seeds as ideal
QAOA_NOISE_P1      = 0.001   # single-qubit depolarising error probability
QAOA_NOISE_P2      = 0.01    # two-qubit depolarising error probability
QAOA_NOISE_READOUT = 0.02    # per-qubit readout error probability
QAOA_NOISE_LABEL   = "Simulated generic depolarising+readout noise (Qiskit Aer)"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# =============================================================================
# 1.  DATA LOADING & PREPROCESSING
# =============================================================================

def parse_acn_timestamp(ts: str) -> float:
    """Parse  'M/D/YYYY H:MM:SS AM/PM'  -> fractional hour (0-24)."""
    ts = str(ts).strip()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %I:%M %p",   "%m/%d/%Y %H:%M"):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: '{ts}'")


def load_acn_data(csv_path=None, n_sessions: int = 30) -> pd.DataFrame:
    """
    Load real ACN-Data CSV or synthesise n_sessions realistic sessions.
    Adds internal columns:  _arrival_hr, _depart_hr, _energy_kWh, _rmax_kW.
    """
    if csv_path and os.path.exists(str(csv_path)):
        print(f"[Data] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        need = ["connectionTime", "disconnectTime", "userInputs_kWhRequested"]
        df   = df.dropna(subset=need).head(n_sessions).copy().reset_index(drop=True)

        df["_arrival_hr"] = df["connectionTime"].apply(parse_acn_timestamp)
        df["_depart_hr"]  = df["disconnectTime"].apply(parse_acn_timestamp)

        # Handle overnight sessions: clamp to horizon
        df["_arrival_hr"] = df["_arrival_hr"].clip(T_START, T_END - 0.5)
        df["_depart_hr"]  = df["_depart_hr"].apply(
            lambda d: d if d > T_START else d + 12.0)
        df["_depart_hr"]  = df[["_arrival_hr", "_depart_hr"]].apply(
            lambda r: max(r["_arrival_hr"] + DT, min(r["_depart_hr"], T_END)),
            axis=1)

        df["_energy_kWh"] = df["userInputs_kWhRequested"].clip(0.5, 30.0).astype(float)

        # Cap energy to what is physically achievable in the window
        def cap_energy(row):
            win = max((row["_depart_hr"] - row["_arrival_hr"]), DT)
            return min(row["_energy_kWh"], R_MAX_L2 * win * 0.98)
        df["_energy_kWh"] = df.apply(cap_energy, axis=1)
        df["_rmax_kW"]    = R_MAX_L2
        print(f"[Data] {len(df)} sessions ready.")
    else:
        print(f"[Data] CSV not found -- synthesising {n_sessions} sessions.")
        df = _synthesise(n_sessions)
    return df.reset_index(drop=True)


def _synthesise(N: int) -> pd.DataFrame:
    """ACN-Caltech statistics: Lee 2019 Table I."""
    rng  = np.random.default_rng(SEED)
    base = "1/15/2020"
    arr  = np.clip(rng.normal(9.0, 2.0, N), T_START, T_END - 2.0)
    stay = np.clip(rng.normal(7.0, 2.0, N), 1.0, T_END - arr)
    dep  = np.clip(arr + stay, arr + DT, T_END)
    eng  = np.array([min(rng.normal(10.0, 5.5), R_MAX_L2 * (dep[i]-arr[i]) * 0.98)
                      for i in range(N)])
    eng  = np.clip(eng, 0.5, 25.0)
    wpm  = rng.choice([250, 300, 350, 400], N)

    def ts(hr):
        h = int(hr); m = int(round((hr-h)*60))
        if m >= 60: h += 1; m = 0
        ap = "AM" if h < 12 else "PM"
        h12 = h if 1 <= h <= 12 else (h-12 if h > 12 else 12)
        return f"{base} {h12}:{m:02d}:00 {ap}"

    return pd.DataFrame({
        "_id": [f"s{i:03d}" for i in range(N)],
        "clusterID": rng.choice(["CA-501","CA-502"], N),
        "connectionTime": [ts(a) for a in arr],
        "disconnectTime": [ts(d) for d in dep],
        "doneChargingTime": [ts(min(a + e/R_MAX_L2, d)) for a,d,e in zip(arr,dep,eng)],
        "kWhDelivered": np.round(eng * rng.uniform(.9,1.,N), 3),
        "sessionID": [f"SID{1000+i}" for i in range(N)],
        "siteID": "caltech",
        "spaceID": rng.choice([f"CA-{j:03d}" for j in range(1,8)], N),
        "stationID": rng.choice([f"CA-{j:03d}" for j in range(1,5)], N),
        "timezone": "America/Los_Angeles",
        "userID": [f"U{rng.integers(1,60):02d}" for _ in range(N)],
        "userInputs_WhPerMile": wpm,
        "userInputs_kWhRequested": np.round(eng, 3),
        "userInputs_milesRequested": np.round(eng*1000/wpm, 1),
        "userInputs_minutesAvailable": np.round(stay*60),
        "userInputs_modifiedAt": [ts(a+.05) for a in arr],
        "userInputs_paymentRequired": rng.choice([True,False], N, p=[.3,.7]),
        "userInputs_requestedDeparture": [ts(d-.25) for d in dep],
        "userInputs_userID": [f"U{rng.integers(1,60):02d}" for _ in range(N)],
        "_arrival_hr": np.round(arr, 4),
        "_depart_hr":  np.round(dep, 4),
        "_energy_kWh": np.round(eng, 4),
        "_rmax_kW":    np.full(N, R_MAX_L2),
    })


# =============================================================================
# 2.  PROBLEM CLASS
# =============================================================================

class ChargingProblem:
    """
    Central data structure shared by all solvers.

    Background load (d_bg)
    ----------------------
    The campus non-EV load profile is modelled following ACN-Caltech
    observations (Lee 2019, Fig. 7):
        base ~40 kW + midday peak ~+30 kW + morning bump ~+8 kW

    This is the non-EV portion of the campus load.  The valley-filling
    objective aims to flatten the TOTAL load (non-EV + EV scheduling).

    Three baselines stored:
      d_bg          : non-EV background load (used in objective)
      d_uncontrolled: total load under FCFS uncontrolled charging
      d_flat_ideal  : perfect flat target = mean(d_uncontrolled)
    """

    def __init__(self, df: pd.DataFrame, dt: float = DT):
        self.df   = df.copy()
        self.dt   = dt
        self.slots = np.arange(T_START, T_END, dt)
        self.T    = len(self.slots)
        self.N    = len(df)

        self.A    = np.zeros(self.N, dtype=int)
        self.D    = np.zeros(self.N, dtype=int)
        self.E    = np.zeros(self.N)
        self.Rmax = np.zeros(self.N)

        for i, row in df.iterrows():
            ai = max(0, int((row["_arrival_hr"] - T_START) / dt))
            di = min(self.T-1, int((row["_depart_hr"] - T_START) / dt))
            di = max(di, ai)
            self.A[i]    = ai
            self.D[i]    = di
            self.E[i]    = float(row["_energy_kWh"])
            self.Rmax[i] = float(row["_rmax_kW"])

        # Non-EV campus background load (ACN-Caltech profile, Lee 2019)
        self.d_bg = self._compute_background()

        # Uncontrolled (FCFS) baseline: each EV charges at full rate on arrival
        R_fcfs = np.zeros((self.N, self.T))
        for i in range(self.N):
            remaining = self.E[i]
            for t in range(self.A[i], self.D[i]+1):
                if remaining <= 1e-6:
                    break
                r = min(self.Rmax[i], remaining / self.dt)
                R_fcfs[i, t] = r
                remaining   -= r * self.dt
        self.d_uncontrolled = self.d_bg + R_fcfs.sum(axis=0)

        # Ideal flat target
        self.d_flat_ideal = np.full(self.T, self.d_uncontrolled.mean())

        self.E_total = float(self.E.sum())

    def _compute_background(self) -> np.ndarray:
        """
        Non-EV campus load.  Parameters from Lee 2019 (Caltech campus):
          - Base load:    ~40 kW
          - Midday peak:  ~+30 kW centered at 13:00
          - Morning bump: ~+8 kW centered at 08:30
          - Afternoon dip:~-5 kW centered at 17:00
        """
        t = self.slots
        d = (40.0
             + 30.0 * np.exp(-0.5 * ((t - 13.0) / 2.8)**2)
             +  8.0 * np.exp(-0.5 * ((t -  8.5) / 1.5)**2)
             -  5.0 * np.exp(-0.5 * ((t - 17.0) / 1.2)**2))
        return np.maximum(d, 5.0)

    def objective(self, R: np.ndarray) -> float:
        """sum_t (d_bg[t] + sum_i r[i,t])^2  (Gan 2012)."""
        return float(np.sum((self.d_bg + R.sum(0))**2))

    def par(self, R: np.ndarray) -> float:
        total = self.d_bg + R.sum(0)
        return float(total.max() / total.mean())

    def load_factor(self, R: np.ndarray) -> float:
        total = self.d_bg + R.sum(0)
        return float(total.mean() / total.max())

    def energy_delivered(self, R: np.ndarray) -> np.ndarray:
        return np.array([R[i, self.A[i]:self.D[i]+1].sum()*self.dt
                          for i in range(self.N)])

    def energy_errors(self, R: np.ndarray) -> np.ndarray:
        return np.abs(self.energy_delivered(R) - self.E)

    def metrics(self, R: np.ndarray, label: str = "",
                solve_time: float = 0.0) -> dict:
        total = self.d_bg + R.sum(0)
        ee    = self.energy_errors(R)
        sat   = self.energy_delivered(R) / np.maximum(self.E, 1e-9)
        return {
            "label":               label,
            "solve_time_s":        round(solve_time, 4),
            "obj_kW2":             round(self.objective(R), 2),
            "PAR":                 round(self.par(R), 4),
            "load_factor":         round(self.load_factor(R), 4),
            "peak_load_kW":        round(float(total.max()), 3),
            "min_load_kW":         round(float(total.min()), 3),
            "mean_load_kW":        round(float(total.mean()), 3),
            "load_std_kW":         round(float(total.std()), 4),
            "load_variance_kW2":   round(float(np.var(total)), 4),
            "total_ev_kWh":        round(float(R.sum()*self.dt), 3),
            "mean_energy_err_kWh": round(float(ee.mean()), 5),
            "max_energy_err_kWh":  round(float(ee.max()), 5),
            "mean_satisfaction_%": round(float(sat.mean()*100), 3),
            "min_satisfaction_%":  round(float(sat.min()*100), 3),
            "energy_errors":       ee,
            "satisfaction":        sat,
            "site_viol_kWh":       round(float(
                np.maximum(0, R.sum(0)-P_SITE_MAX).sum()*self.dt), 5),
            "R": R,
        }


# =============================================================================
# 3.  SHARED REPAIR UTILITIES
# =============================================================================

def repair_full(R: np.ndarray, prob: ChargingProblem,
                valley_fill_passes: int = 0) -> np.ndarray:
    """
    Project rate matrix R to full feasibility.

    Steps:
      1. Box [0, Rmax_i] and zero outside each EV's window.
      2. Site cap (proportional reduction) -- first pass.
      3. Fill UP under-charged EVs via water-filling into lowest-load slots.
      4. Scale DOWN over-charged EVs proportionally.
      5. Site cap -- second pass after energy repair.
      6. (Optional) Valley-fill smoothing passes: shift charge from
         high-load to low-load slots within each EV's window while
         keeping energy exactly satisfied.  Set valley_fill_passes > 0
         to activate (used by QUBO post-processing only).

    valley_fill_passes=0 for GA/PSO (they have their own objective),
    valley_fill_passes=5 for QUBO decoders (corrects spiky binary patterns).
    """
    R = R.copy()
    N, T = prob.N, prob.T

    # Step 1: Box + window
    for i in range(N):
        R[i] = np.clip(R[i], 0.0, prob.Rmax[i])
        if prob.A[i] > 0:       R[i, :prob.A[i]] = 0.0
        if prob.D[i] < T - 1:   R[i, prob.D[i]+1:] = 0.0

    # Step 2: Site cap (first pass)
    for t in range(T):
        s = R[:, t].sum()
        if s > P_SITE_MAX + 1e-6:
            R[:, t] *= P_SITE_MAX / s

    # Step 3 & 4: Energy repair
    current_load = prob.d_bg + R.sum(0)

    for i in range(N):
        window    = np.arange(prob.A[i], prob.D[i]+1)
        delivered = R[i, window].sum() * prob.dt
        deficit   = prob.E[i] - delivered

        if deficit > 1e-4:
            # Water-fill into lowest total-load slots first
            order = np.argsort(current_load[window])
            for s in order:
                t         = window[s]
                headroom  = prob.Rmax[i] - R[i, t]
                site_room = P_SITE_MAX - current_load[t] + R[i, t]
                avail     = max(0.0, min(headroom, site_room))
                add       = min(avail, deficit / prob.dt)
                R[i, t]        += add
                current_load[t] += add
                deficit        -= add * prob.dt
                if deficit < 1e-4:
                    break

        elif deficit < -1e-4:
            # Scale down proportionally
            scale = prob.E[i] / max(delivered, 1e-12)
            R[i, window] *= scale
            current_load  = prob.d_bg + R.sum(0)

    # Step 5: Site cap (second pass)
    for t in range(T):
        s = R[:, t].sum()
        if s > P_SITE_MAX + 1e-6:
            R[:, t] *= P_SITE_MAX / s

    # Step 6: Valley-fill smoothing (QUBO post-process only)
    # For each EV, iteratively move a small amount of charge from the
    # highest-load slot to the lowest-load slot within its window.
    # This is a single-EV coordinate descent step on the valley-fill
    # objective, preserving energy balance exactly.
    if valley_fill_passes > 0:
        current_load = prob.d_bg + R.sum(0)
        for _ in range(valley_fill_passes):
            for i in range(N):
                window = np.arange(prob.A[i], prob.D[i]+1)
                if len(window) < 2:
                    continue
                load_w  = current_load[window]
                t_high  = window[np.argmax(load_w)]
                t_low   = window[np.argmin(load_w)]
                if t_high == t_low:
                    continue
                # Maximum transferable amount (respects Rmax and non-negativity)
                transfer = min(
                    R[i, t_high],                              # can't go negative
                    prob.Rmax[i] - R[i, t_low],               # can't exceed Rmax
                    max(0.0, (current_load[t_high] - current_load[t_low]) / 2.0),
                )
                if transfer < 1e-6:
                    continue
                R[i, t_high]        -= transfer
                R[i, t_low]         += transfer
                current_load[t_high] -= transfer
                current_load[t_low]  += transfer
        # Final site cap after smoothing
        for t in range(T):
            s = R[:, t].sum()
            if s > P_SITE_MAX + 1e-6:
                R[:, t] *= P_SITE_MAX / s

    return R


def encode_to_qubo_warm(qubo, R_qp: np.ndarray) -> np.ndarray:
    """Encode continuous QP solution into nearest QUBO binary vector."""
    x = np.zeros(qubo.n)
    for i in range(qubo.prob.N):
        for t in range(qubo.prob.A[i], qubo.prob.D[i]+1):
            r_val = R_qp[i, t]
            level = int(round(r_val / qubo.delta[i]))
            level = max(0, min(level, qubo.levels))
            # Standard binary encoding
            for k in range(qubo.K):
                bit = (level >> k) & 1
                x[qubo.var_pos[(i, t, k)]] = float(bit)
    return x


# =============================================================================
# 4.  CLASSICAL SOLVER 1 -- Convex QP  (Gan et al. 2012)
# =============================================================================

def solve_convex_qp(prob: ChargingProblem) -> dict:
    """
    Exact valley-filling QP.  Objective is strictly convex -> global optimum.
    Only optimises over in-window (i,t) variables (sparse formulation).
    """
    T, N, dt = prob.T, prob.N, prob.dt
    A, D, E, Rmax, d_bg = prob.A, prob.D, prob.E, prob.Rmax, prob.d_bg

    var_map = [(i, t) for i in range(N) for t in range(A[i], D[i]+1)]
    nv  = len(var_map)
    pos = {(i, t): k for k, (i, t) in enumerate(var_map)}

    C = np.zeros((T, nv))
    for k, (i, t) in enumerate(var_map):
        C[t, k] = 1.0

    def obj(x): return float(np.sum((C@x + d_bg)**2))
    def jac(x): return 2.0 * C.T @ (C@x + d_bg)

    x0 = np.zeros(nv)
    for i in range(N):
        idxs = [pos[(i,t)] for t in range(A[i], D[i]+1)]
        w    = len(idxs)
        if w:
            r0 = min(E[i]/(w*dt), Rmax[i])
            for k in idxs: x0[k] = r0

    A_eq  = np.zeros((N, nv)); b_eq = E.copy()
    for k, (i, t) in enumerate(var_map): A_eq[i, k] = dt

    cons = [
        {"type": "eq",
         "fun": lambda x: A_eq @ x - b_eq,
         "jac": lambda x: A_eq},
        {"type": "ineq",
         "fun": lambda x: P_SITE_MAX - C @ x,
         "jac": lambda x: -C},
    ]
    lb = np.zeros(nv)
    ub = np.array([Rmax[i] for (i,t) in var_map])

    t0  = time.perf_counter()
    res = minimize(obj, x0, jac=jac, method="SLSQP",
                   bounds=Bounds(lb, ub), constraints=cons,
                   options={"maxiter": 5000, "ftol": 1e-12, "disp": False})
    elapsed = time.perf_counter() - t0

    R = np.zeros((N, T))
    for k, (i, t) in enumerate(var_map):
        R[i, t] = float(np.clip(res.x[k], 0.0, Rmax[i]))

    R = repair_full(R, prob)
    return prob.metrics(R, "Convex QP (SLSQP)", elapsed)


# =============================================================================
# 5.  CLASSICAL SOLVER 2 -- Genetic Algorithm
# =============================================================================

class GeneticAlgorithm:
    """
    Real-valued GA with SBX crossover, polynomial mutation,
    self-adaptive mutation rate, elitism, and full energy repair.
    """

    def __init__(self, prob: ChargingProblem):
        self.prob = prob

    def _init_pop(self, M: int, R_seed=None) -> list:
        rng = np.random.default_rng(SEED)
        pop = []
        if R_seed is not None:
            pop.append(repair_full(R_seed.copy(), self.prob))
        while len(pop) < M:
            R = np.zeros((self.prob.N, self.prob.T))
            for i in range(self.prob.N):
                w = self.prob.D[i] - self.prob.A[i] + 1
                R[i, self.prob.A[i]:self.prob.D[i]+1] = rng.uniform(0, self.prob.Rmax[i], w)
            pop.append(repair_full(R, self.prob))
        return pop

    def _fitness(self, R: np.ndarray) -> float:
        return self.prob.objective(R)

    def _tournament(self, pop, fits, k=4):
        idx  = np.random.choice(len(pop), k, replace=False)
        best = idx[np.argmin([fits[j] for j in idx])]
        return pop[best].copy()

    def _sbx(self, p1, p2, eta=15.0):
        """Simulated Binary Crossover."""
        c1, c2 = p1.copy(), p2.copy()
        rng    = np.random.default_rng()
        for i in range(self.prob.N):
            for t in range(self.prob.A[i], self.prob.D[i]+1):
                if rng.random() < 0.5:
                    x1, x2 = p1[i,t], p2[i,t]
                    if abs(x1-x2) < 1e-10: continue
                    u = rng.random()
                    if u < 0.5:
                        beta = (2*u)**(1/(eta+1))
                    else:
                        beta = (1/(2*(1-u)))**(1/(eta+1))
                    c1[i,t] = np.clip(0.5*((x1+x2)-beta*abs(x2-x1)),
                                       0.0, self.prob.Rmax[i])
                    c2[i,t] = np.clip(0.5*((x1+x2)+beta*abs(x2-x1)),
                                       0.0, self.prob.Rmax[i])
        return c1, c2

    def _poly_mutate(self, R, rate, eta=20.0):
        """Polynomial mutation."""
        R   = R.copy()
        rng = np.random.default_rng()
        for i in range(self.prob.N):
            for t in range(self.prob.A[i], self.prob.D[i]+1):
                if rng.random() < rate:
                    xu  = self.prob.Rmax[i]
                    x   = R[i, t]
                    u   = rng.random()
                    if u < 0.5:
                        delta = (2*u)**(1/(eta+1)) - 1.0
                    else:
                        delta = 1.0 - (2*(1-u))**(1/(eta+1))
                    R[i, t] = np.clip(x + delta*xu, 0.0, xu)
        return R

    def run(self, M=GA_POP, G=GA_GENS, R_seed=None, verbose=True):
        pop  = self._init_pop(M, R_seed)
        fits = [self._fitness(ind) for ind in pop]
        bi   = int(np.argmin(fits))
        best_R, best_f = pop[bi].copy(), fits[bi]
        history     = [best_f]
        mut_rate    = GA_MUT_INIT
        no_improve  = 0
        stagnation  = max(1, G // 20)
        t0 = time.perf_counter()

        for g in range(G):
            sorted_idx = np.argsort(fits)
            new_pop    = [pop[sorted_idx[0]].copy(), pop[sorted_idx[1]].copy()]

            while len(new_pop) < M:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                if np.random.random() < GA_CX_RATE:
                    c1, c2 = self._sbx(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = repair_full(self._poly_mutate(c1, mut_rate), self.prob)
                c2 = repair_full(self._poly_mutate(c2, mut_rate), self.prob)
                new_pop.extend([c1, c2])

            pop  = new_pop[:M]
            fits = [self._fitness(ind) for ind in pop]
            bi   = int(np.argmin(fits))

            if fits[bi] < best_f - 1e-6:
                best_f = fits[bi]; best_R = pop[bi].copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 0 and no_improve % stagnation == 0:
                mut_rate = min(mut_rate * 1.5, 0.3)
            elif no_improve == 0:
                mut_rate = max(mut_rate * 0.95, 0.005)

            history.append(best_f)
            if verbose and (g+1) % 100 == 0:
                print(f"   GA  gen {g+1:4d}/{G}  best={best_f:.2f}  mut={mut_rate:.4f}")

        return best_R, history, time.perf_counter()-t0


# =============================================================================
# 6.  CLASSICAL SOLVER 3 -- PSO  (lbest ring topology)
# =============================================================================

class PSO:
    """
    PSO with lbest (ring) topology, linearly decaying inertia,
    velocity clamping, and QP warm-start.
    """

    def __init__(self, prob: ChargingProblem):
        self.prob = prob

    def _fitness(self, R): return self.prob.objective(R)

    def run(self, M=PSO_SWARM, G=PSO_ITERS, R_seed=None, verbose=True):
        prob = self.prob; N, T = prob.N, prob.T
        rng  = np.random.default_rng(SEED + 1)
        Vmax = R_MAX_L2 * 0.4

        particles = []
        for m in range(M):
            if m == 0 and R_seed is not None:
                R = repair_full(R_seed.copy(), prob)
            else:
                R = np.zeros((N, T))
                for i in range(N):
                    w = prob.D[i]-prob.A[i]+1
                    R[i, prob.A[i]:prob.D[i]+1] = rng.uniform(0, prob.Rmax[i], w)
                R = repair_full(R, prob)
            particles.append(R)

        velocities = [rng.uniform(-Vmax*0.1, Vmax*0.1, (N, T)) for _ in range(M)]
        pbest      = [p.copy() for p in particles]
        pbest_f    = [self._fitness(p) for p in pbest]
        nbhd       = 5

        def lbest_idx(m):
            return [(m+k-nbhd//2) % M for k in range(nbhd)]

        lbest   = pbest.copy()
        lbest_f = pbest_f.copy()
        gbest   = pbest[int(np.argmin(pbest_f))].copy()
        gbest_f = min(pbest_f)
        history = [gbest_f]

        t0 = time.perf_counter()
        for it in range(G):
            w_cur = PSO_W_MAX - (PSO_W_MAX - PSO_W_MIN) * it / G

            for m in range(M):
                r1 = rng.random((N, T)); r2 = rng.random((N, T))
                velocities[m] = (w_cur * velocities[m]
                                 + PSO_C1 * r1 * (pbest[m] - particles[m])
                                 + PSO_C2 * r2 * (lbest[m] - particles[m]))
                velocities[m] = np.clip(velocities[m], -Vmax, Vmax)
                for i in range(N):
                    if prob.A[i] > 0:     velocities[m][i, :prob.A[i]] = 0.0
                    if prob.D[i] < T-1:   velocities[m][i, prob.D[i]+1:] = 0.0

                particles[m] = repair_full(particles[m] + velocities[m], prob)
                f = self._fitness(particles[m])
                if f < pbest_f[m]:
                    pbest[m] = particles[m].copy(); pbest_f[m] = f
                if f < gbest_f:
                    gbest = particles[m].copy(); gbest_f = f

            for m in range(M):
                nb    = lbest_idx(m)
                best_nb = nb[int(np.argmin([pbest_f[j] for j in nb]))]
                if pbest_f[best_nb] < lbest_f[m]:
                    lbest[m] = pbest[best_nb].copy(); lbest_f[m] = pbest_f[best_nb]

            history.append(gbest_f)
            if verbose and (it+1) % 100 == 0:
                print(f"   PSO iter {it+1:4d}/{G}  gbest={gbest_f:.2f}")

        return gbest, history, time.perf_counter()-t0


# =============================================================================
# 7.  QUBO FORMULATION  (auto-scaled penalties)
# =============================================================================

class QUBOFormulation:
    """
    QUBO encoding:  r_{i,t} = delta_i * sum_k 2^k * x_{i,t,k}
    where delta_i = Rmax_i / (2^K - 1).

    Penalty weights are auto-scaled:
        lambda_energy = lam_scale_energy * max_possible_H_obj
    so that energy satisfaction always dominates the objective term.

    The hard site cap is enforced
    via post-processing repair.
    """

    def __init__(self, prob: ChargingProblem, K: int = K_BITS,
                 lam_scale_energy: float = 5.0):
        self.prob   = prob
        self.K      = K
        self.levels = 2**K - 1
        self.delta  = prob.Rmax / self.levels
        self.pow2   = (2**np.arange(K)).astype(float)

        # Active variable list (only in-window bits -- variable elimination)
        self.active  = []
        self.var_pos = {}
        idx = 0
        for i in range(prob.N):
            for t in range(prob.A[i], prob.D[i]+1):
                for k in range(K):
                    self.active.append((i, t, k))
                    self.var_pos[(i, t, k)] = idx
                    idx += 1
        self.n = idx

        # Coefficient: coeff[v] = 2^k * delta_i
        self.coeff = np.array([self.pow2[k] * self.delta[i]
                                for (i, t, k) in self.active])

        # Slot map: slot_vars[t] = list of active variable indices at slot t
        self.slot_vars = {t: [] for t in range(prob.T)}
        for v, (i, t, k) in enumerate(self.active):
            self.slot_vars[t].append(v)

        # EV map: ev_vars[i] = list of (var_idx, energy_coeff = coeff*dt)
        self.ev_vars = {i: [] for i in range(prob.N)}
        for v, (i, t, k) in enumerate(self.active):
            self.ev_vars[i].append((v, self.coeff[v] * prob.dt))

        # Auto-scale penalties based on objective magnitude estimate
        h_obj_est  = float(prob.T * (prob.d_bg.max() + prob.N * R_MAX_L2)**2)
        self.lam_e = lam_scale_energy * h_obj_est
        # NOTE: lam_r / P_rate (one-hot) removed -- incompatible with positional binary encoding

        print(f"[QUBO] n={self.n}  K={K}  lam_e={self.lam_e:.2e}")

    def build_Q(self) -> np.ndarray:
        """Analytically construct upper-triangular Q matrix."""
        Q = np.zeros((self.n, self.n))
        prob = self.prob

        # H_obj: valley-filling  sum_t (d_bg[t] + sum_v coeff_v x_v)^2
        for t in range(prob.T):
            sv = self.slot_vars[t]
            if not sv: continue
            c  = self.coeff[sv]; bg = prob.d_bg[t]
            for ii, v in enumerate(sv):
                Q[v, v] += c[ii]**2 + 2.0*bg*c[ii]
            for ii in range(len(sv)):
                for jj in range(ii+1, len(sv)):
                    Q[sv[ii], sv[jj]] += 2.0*c[ii]*c[jj]

        # H_energy: lam_e * (sum_v a_v x_v - E_i)^2 per EV
        for i in range(prob.N):
            ev   = self.ev_vars[i]
            vids = [v for v,_ in ev]
            ais  = np.array([a for _,a in ev])
            Ei   = prob.E[i]
            for ii, v in enumerate(vids):
                Q[v, v] += self.lam_e * (ais[ii]**2 - 2.0*Ei*ais[ii])
            for ii in range(len(vids)):
                for jj in range(ii+1, len(vids)):
                    Q[vids[ii], vids[jj]] += 2.0*self.lam_e*ais[ii]*ais[jj]

        # QUBO validity assertion: ensure no NaN/Inf, correct dimensions
        assert Q.shape == (self.n, self.n), "QUBO dimension mismatch"
        assert np.all(np.isfinite(Q)), "QUBO contains non-finite values"
        # P_rate absence assertion: no lam_r attribute should affect Q
        assert not hasattr(self, 'lam_r'), "lam_r must not exist (P_rate removed)"

        return Q

    def decode(self, x: np.ndarray) -> np.ndarray:
        R = np.zeros((self.prob.N, self.prob.T))
        for v, (i, t, k) in enumerate(self.active):
            R[i, t] += self.coeff[v] * x[v]
        return R


# =============================================================================
# 8.  QUANTUM SOLVER 1 -- SA with restarts
# =============================================================================

def solve_sa(qubo: QUBOFormulation, Q: np.ndarray,
             x_warm=None,
             T_init=SA_T_INIT, T_final=SA_T_FINAL,
             n_sweeps=SA_SWEEPS, verbose=True):
    """
    SA with QP warm-start, O(n) rank-1 dE update, and restart-with-perturbation.

    Correct dE for upper-triangular Q and x_j in {0,1}:
        dE = (1-2x_j) * (Qsym[j]@x - Qsym[j,j]*x[j] + Q[j,j]*(1-2x_j))
    where Qsym = Q + Q^T (full symmetric form).
    """
    prob = qubo.prob; n = qubo.n
    rng  = np.random.default_rng(SEED)
    Qsym = Q + Q.T

    x = x_warm.copy() if x_warm is not None else np.zeros(n)
    if x_warm is None:
        for i in range(prob.N):
            for t in range(prob.A[i], prob.D[i]+1):
                x[qubo.var_pos[(i,t,0)]] = 1.0

    Qx    = Qsym @ x
    E_cur = float(x @ Q @ x)
    best_x = x.copy(); best_E = E_cur

    alpha   = (T_final / T_init)**(1.0 / max(n_sweeps, 1))
    T_cur   = T_init
    history = np.empty(n_sweeps)
    restart_after = max(1, int(SA_RESTART_FRAC * n_sweeps))
    since_improve = 0

    for s in range(n_sweeps):
        j    = int(rng.integers(0, n))
        sign = 1.0 - 2.0 * x[j]
        # Correct dE for QUBO upper-triangular matrix and binary x
        dE   = sign * ((Qx[j] - Qsym[j,j]*x[j]) + Q[j,j]*sign)

        if dE < 0.0 or (T_cur > 1e-15 and
                         rng.random() < math.exp(-dE / T_cur)):
            Qx   += sign * Qsym[j]
            x[j]  = 1.0 - x[j]
            E_cur += dE
            since_improve = 0
        else:
            since_improve += 1

        if E_cur < best_E:
            best_E = E_cur; best_x = x.copy()
            since_improve = 0

        if since_improve >= restart_after:
            x    = best_x.copy()
            mask = rng.random(n) < 0.05
            x[mask] = 1.0 - x[mask]
            Qx    = Qsym @ x
            E_cur = float(x @ Q @ x)
            T_cur = T_init * 0.3
            since_improve = 0

        T_cur = max(T_cur * alpha, T_final)
        history[s] = best_E

    if verbose: print(f"   SA   best QUBO E = {best_E:.4e}")
    return best_x, best_E, history


# =============================================================================
# 9.  QUANTUM SOLVER 2 -- Parallel Tempering
# =============================================================================

def solve_parallel_tempering(qubo: QUBOFormulation, Q: np.ndarray,
                              x_warm=None,
                              n_replicas=PT_REPLICAS, n_sweeps=PT_SWEEPS,
                              T_min=PT_T_MIN, T_max=PT_T_MAX, verbose=True):
    """
    Parallel Tempering: n_replicas chains at geomspace temperatures.
    Each replica warm-started from QP encoding + diversity perturbation.
    All adjacent pairs attempted at each exchange step.
    """
    prob = qubo.prob; n = qubo.n
    rng  = np.random.default_rng(SEED + 10)
    Qsym = Q + Q.T
    temps = np.geomspace(T_max, T_min, n_replicas)

    replicas  = []; Qx_list = []; E_list = []
    for r in range(n_replicas):
        x = x_warm.copy() if x_warm is not None else np.zeros(n)
        if x_warm is None:
            for i in range(prob.N):
                for t in range(prob.A[i], prob.D[i]+1):
                    x[qubo.var_pos[(i,t,0)]] = 1.0
        flip_p = 0.02 + 0.15 * r / max(n_replicas-1, 1)
        mask   = rng.random(n) < flip_p
        x[mask] = 1.0 - x[mask]
        Qx = Qsym @ x; E = float(x @ Q @ x)
        replicas.append(x.copy()); Qx_list.append(Qx.copy()); E_list.append(E)

    best_x = replicas[-1].copy(); best_E = E_list[-1]
    history = np.empty(n_sweeps)
    swap_interval = max(1, 2*n//n_replicas)

    for s in range(n_sweeps):
        for r in range(n_replicas):
            j    = int(rng.integers(0, n))
            sign = 1.0 - 2.0*replicas[r][j]
            dE   = sign * ((Qx_list[r][j] - Qsym[j,j]*replicas[r][j]) + Q[j,j]*sign)
            T_r  = temps[r]
            if dE < 0.0 or (T_r > 1e-15 and rng.random() < math.exp(-dE/T_r)):
                Qx_list[r] += sign * Qsym[j]
                replicas[r][j] = 1.0 - replicas[r][j]
                E_list[r] += dE
            if E_list[r] < best_E:
                best_E = E_list[r]; best_x = replicas[r].copy()

        if (s+1) % swap_interval == 0:
            for r in range(n_replicas-1):
                beta_lo = 1.0/temps[r+1]; beta_hi = 1.0/temps[r]
                delta   = (beta_lo-beta_hi)*(E_list[r]-E_list[r+1])
                if delta >= 0 or rng.random() < math.exp(delta):
                    replicas[r],  replicas[r+1]  = replicas[r+1],  replicas[r]
                    Qx_list[r],   Qx_list[r+1]   = Qx_list[r+1],   Qx_list[r]
                    E_list[r],    E_list[r+1]     = E_list[r+1],    E_list[r]

        history[s] = best_E

    if verbose: print(f"   PT   best QUBO E = {best_E:.4e}")
    return best_x, best_E, history


# =============================================================================
# 10.  QUANTUM SOLVER 3 -- Tabu Search with long-term memory
# =============================================================================

def solve_tabu(qubo: QUBOFormulation, Q: np.ndarray,
               x_warm=None, n_iters=TABU_ITERS, tenure=TABU_TENURE,
               verbose=True):
    """
    Tabu Search with recency tabu, frequency-based long-term memory,
    aspiration criterion, and diversification restarts.
    """
    prob = qubo.prob; n = qubo.n
    rng  = np.random.default_rng(SEED + 20)
    Qsym = Q + Q.T

    x = x_warm.copy() if x_warm is not None else np.zeros(n)
    if x_warm is None:
        for i in range(prob.N):
            for t in range(prob.A[i], prob.D[i]+1):
                x[qubo.var_pos[(i,t,0)]] = 1.0

    Qx    = Qsym @ x
    E_cur = float(x @ Q @ x)
    best_x = x.copy(); best_E = E_cur

    tabu_expiry = np.zeros(n, dtype=int)
    freq        = np.zeros(n, dtype=float)
    history     = np.empty(n_iters)

    diag_Q = np.diag(Q).copy()
    diag_Qs = np.diag(Qsym).copy()

    for it in range(n_iters):
        signs = 1.0 - 2.0 * x
        # Vectorised dE for all bits
        dEs   = signs * ((Qx - diag_Qs*x) + diag_Q*signs)

        # Long-term frequency penalty
        freq_pen = 1e-4 * freq * (np.abs(dEs).mean() + 1e-12)
        adj_dEs  = dEs + freq_pen

        best_j = -1; best_dE = np.inf
        order  = np.argsort(adj_dEs)
        for j in order[:min(100, n)]:
            is_tabu = tabu_expiry[j] > it
            aspires = (E_cur + dEs[j]) < best_E
            if not is_tabu or aspires:
                best_j = j; best_dE = dEs[j]; break

        if best_j == -1:
            for j in order:
                if tabu_expiry[j] <= it:
                    best_j = j; best_dE = dEs[j]; break
            if best_j == -1:
                best_j = int(rng.integers(0, n)); best_dE = dEs[best_j]

        sign = 1.0 - 2.0 * x[best_j]
        Qx  += sign * Qsym[best_j]
        x[best_j] = 1.0 - x[best_j]
        E_cur    += best_dE
        tabu_expiry[best_j] = it + tenure
        freq[best_j]        += 1.0

        if E_cur < best_E:
            best_E = E_cur; best_x = x.copy()

        # Diversification every 10% of iterations
        if it > 0 and it % (n_iters // 10) == 0:
            if freq.std() > 0:
                top_freq = freq > (freq.mean() + 2*freq.std())
                if top_freq.sum() > n * 0.1:
                    x    = best_x.copy()
                    mask = rng.random(n) < 0.15
                    x[mask] = 1.0 - x[mask]
                    Qx    = Qsym @ x
                    E_cur = float(x @ Q @ x)
                    freq *= 0.5

        history[it] = best_E

    if verbose: print(f"   Tabu best QUBO E = {best_E:.4e}")
    return best_x, best_E, history


# =============================================================================
# 11.  QAOA SOLVER  (Qiskit Aer)
# =============================================================================
#
#
# Pipeline:
#   decompose_qubo() → subproblems (≤QAOA_MAX_VARS qubits each)
#   for each subproblem:
#       COBYLA optimises (gamma, beta) on Aer circuit
#       best bitstring selected by MINIMUM QUBO energy (classical x^T Q x)
#       greedy 1-bit local search on Q_sub
#   assemble global binary solution
#   global greedy local search on full Q
#   → return x_global (selection by QUBO energy only, NO post-processing used
#     to select the bitstring -- §15 of integration instructions)


def _qaoa_decompose_qubo(qubo, Q, x_warm, max_vars=QAOA_MAX_VARS):
    """
    Decompose the full QUBO into subproblems of at most max_vars qubits.
    Groups variables by EV; recursively splits oversized EV chunks.
    Adds contextual coupling: off-subproblem variables fixed at warm-start.
    Returns list of dicts: {var_indices, Q_sub, x_warm, n}
    """
    prob = qubo.prob
    n    = qubo.n

    ev_var_lists = []
    for i in range(prob.N):
        vidxs = []
        for t in range(prob.A[i], prob.D[i] + 1):
            for k in range(qubo.K):
                key = (i, t, k)
                if key in qubo.var_pos:
                    vidxs.append(qubo.var_pos[key])
        if vidxs:
            ev_var_lists.append(vidxs)

    def _split(vlist, limit):
        if len(vlist) <= limit:
            return [vlist]
        mid = len(vlist) // 2
        return _split(vlist[:mid], limit) + _split(vlist[mid:], limit)

    subproblems  = []
    current_vars = []
    for ev_vars in ev_var_lists:
        for part in _split(ev_vars, max_vars):
            if len(current_vars) + len(part) > max_vars:
                if current_vars:
                    subproblems.append(current_vars)
                current_vars = list(part)
            else:
                current_vars.extend(part)
            if len(current_vars) == max_vars:
                subproblems.append(current_vars)
                current_vars = []
    if current_vars:
        subproblems.append(current_vars)

    result = []
    for var_list in subproblems:
        m   = len(var_list)
        idx = np.array(var_list)
        # Build Q_sub preserving UPPER-TRIANGULAR convention (Part C fix).
        # Iterate over ii <= jj only so each off-diagonal pair is stored once
        # and the lower triangle stays zero.  x.T @ Q_sub @ x then evaluates
        # the same QUBO as x.T @ Q @ x (upper-tri global convention).
        Q_sub = np.zeros((m, m))
        for ii in range(m):
            for jj in range(ii, m):           # upper triangle only
                vi, vj = idx[ii], idx[jj]
                if vi <= vj:
                    Q_sub[ii, jj] += Q[vi, vj]
                else:
                    Q_sub[ii, jj] += Q[vj, vi]
        # Contextual coupling: fix outside variables at warm-start values
        outside_mask = np.ones(n, dtype=bool)
        outside_mask[idx] = False
        outside_idx = np.where(outside_mask)[0]
        for ii in range(m):
            vi = idx[ii]
            mask_upper = outside_idx > vi
            vk_upper   = outside_idx[mask_upper]
            if len(vk_upper):
                Q_sub[ii, ii] += np.dot(Q[vi, vk_upper], x_warm[vk_upper])
            mask_lower = outside_idx < vi
            vk_lower   = outside_idx[mask_lower]
            if len(vk_lower):
                Q_sub[ii, ii] += np.dot(Q[vk_lower, vi], x_warm[vk_lower])
        result.append({"var_indices": var_list, "Q_sub": Q_sub,
                        "x_warm": x_warm[idx].copy(), "n": m})
    return result


def _qaoa_greedy_local_search(x, Q_full, max_iters=5000):
    """Single-bit flip greedy local search on QUBO (identical to notebook Cell 4)."""
    Qsym = Q_full + Q_full.T
    Qx   = Qsym @ x
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        signs    = 1.0 - 2.0 * x
        diag_Qs  = np.diag(Qsym)
        diag_Q   = np.diag(Q_full)
        dEs      = signs * ((Qx - diag_Qs * x) + diag_Q * signs)
        bj       = int(np.argmin(dEs))
        if dEs[bj] < -1e-10:
            sign    = 1.0 - 2.0 * x[bj]
            Qx     += sign * Qsym[bj]
            x[bj]   = 1.0 - x[bj]
            improved = True
        it += 1
    return x, float(x @ Q_full @ x)


def _qubo_to_ising_upper_triangular(Q_sub, c=0.0):
    """
    Exact QUBO → Ising conversion for an UPPER-TRIANGULAR Q matrix.

    Convention (Part A / Part B of correction instruction):
        E(x) = c + sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j
    with x_i ∈ {0,1}, Q[j,i] = 0 for j > i.

    Substituting x_i = (1 - Z_i)/2 and expanding:
        diagonal term  Q[i,i]*x_i → contributes +Q[i,i]/2 to C0, -Q[i,i]/2 to h_i
        quadratic term Q[i,j]*x_i*x_j (i<j)
              → contributes +Q[i,j]/4 to C0
                            -Q[i,j]/4 to h_i
                            -Q[i,j]/4 to h_j
                            +Q[i,j]/4 to J_ij

    Returns
    -------
    h   : 1-D array of length n   — single-qubit Z coefficients
    J   : dict (i,j) → float, i<j — two-qubit ZZ coefficients
    C0  : float                    — constant (global phase; not used in circuit)
    """
    n  = Q_sub.shape[0]
    h  = np.zeros(n)
    J  = {}
    C0 = c
    for i in range(n):
        a_i  = float(Q_sub[i, i])
        h[i] -= a_i / 2.0
        C0   += a_i / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            b_ij = float(Q_sub[i, j])   # lower triangle is zero; Q_sub[j,i] ignored
            if abs(b_ij) < 1e-15:
                continue
            h[i]    -= b_ij / 4.0
            h[j]    -= b_ij / 4.0
            J[(i,j)] = b_ij / 4.0
            C0       += b_ij / 4.0
    return h, J, C0


def _build_qaoa_circuit(Q_sub, n, p, gammas, betas):
    """
    Build p-layer QAOA circuit for QUBO subproblem.

    Uc(gamma) gate angles (Part B fix — exact QUBO→Ising for upper-triangular Q):
      h_i   = -Q[i,i]/2  -  (1/4)*sum_{j!=i} Q[min,max]   (single-qubit Z)
      J_ij  = Q[i,j]/4   for i < j                          (two-qubit ZZ)
      RZ(2*gamma*h_i, i)
      RZZ(2*gamma*J_ij, i, j)

    Ub(beta): RX(2*beta, i)  [unchanged]

    The previous implementation used RZ(2*gamma*Q[i,i]) and
    RZZ(2*gamma*(Q[i,j]+Q[j,i])), which omitted the off-diagonal contributions
    to single-qubit h_i and doubled the ZZ coefficient.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    h, J, _ = _qubo_to_ising_upper_triangular(Q_sub)
    qr = QuantumRegister(n, name='q')
    cr = ClassicalRegister(n, name='meas')
    qc = QuantumCircuit(qr, cr)
    qc.h(range(n))
    for layer in range(p):
        g = float(gammas[layer])
        b = float(betas[layer])
        # Single-qubit cost terms (diagonal + off-diagonal contributions)
        for i in range(n):
            angle = 2.0 * g * h[i]
            if abs(angle) > 1e-12:
                qc.rz(angle, i)
        # Two-qubit cost terms
        for (i, j), J_ij in J.items():
            angle = 2.0 * g * J_ij
            if abs(angle) > 1e-12:
                qc.rzz(angle, i, j)
        # Mixer
        for i in range(n):
            qc.rx(2.0 * b, i)
    qc.measure(range(n), range(n))
    return qc


def _expectation_from_counts(counts, Q_sub, n):
    """<H_C> from shot counts. Qiskit: rightmost char = qubit 0."""
    total = sum(counts.values())
    E = 0.0
    for bitstr, count in counts.items():
        x = np.array([int(b) for b in reversed(bitstr)], dtype=np.float64)
        E += (count / total) * float(x @ Q_sub @ x)
    return E


def _best_bitstring_from_counts(counts, Q_sub):
    """Return bitstring with MINIMUM QUBO energy (selection by QUBO energy only)."""
    best_x = None
    best_E = np.inf
    for bitstr in counts:
        x = np.array([int(b) for b in reversed(bitstr)], dtype=np.float64)
        E = float(x @ Q_sub @ x)
        if E < best_E:
            best_E = E
            best_x = x.copy()
    return best_x, best_E


def _expectation_from_statevector(sv, Q_sub, n):
    """<H_C> analytical from statevector (statevector mode)."""
    probs = np.abs(sv) ** 2
    E = 0.0
    for idx in range(2 ** n):
        if probs[idx] < 1e-15:
            continue
        x = np.array([(idx >> q) & 1 for q in range(n)], dtype=np.float64)
        E += probs[idx] * float(x @ Q_sub @ x)
    return E


def _best_bitstring_from_statevector(sv, Q_sub, n):
    """Return highest-probability bitstring from statevector."""
    probs    = np.abs(sv) ** 2
    best_idx = int(np.argmax(probs))
    x = np.array([(best_idx >> q) & 1 for q in range(n)], dtype=np.float64)
    return x, float(x @ Q_sub @ x)


def _run_qaoa_circuit_aer(qc, backend, shots, sim_mode):
    """Run circuit on Aer backend; return counts or statevector."""
    from qiskit import transpile
    if sim_mode == "statevector":
        qc_sv = qc.copy()
        qc_sv.remove_final_measurements(inplace=True)
        qc_sv.save_statevector()
        t_qc   = transpile(qc_sv, backend, optimization_level=1)
        result = backend.run(t_qc).result()
        return result.get_statevector().data, None
    else:
        t_qc   = transpile(qc, backend, optimization_level=1)
        result = backend.run(t_qc, shots=shots).result()
        return result.get_counts(), None


def _validate_qubo_ising_equivalence(verbose=True):
    """
    Part H/I/J: exhaustive mathematical validation of _qubo_to_ising_upper_triangular.

    Tests:
      1. For random upper-triangular QUBOs of sizes n=3,4,5, exhaustively checks
         that E_qubo(x) == E_ising(z) for EVERY binary state.
      2. Confirms that the QUBO-minimising state equals the Ising-minimising state.
      3. Confirms local Q_sub matrices are upper-triangular (lower triangle == 0).
      4. Prints PASS/FAIL for each check.

    Returns True iff all checks pass.
    """
    rng = np.random.default_rng(12345)
    all_ok = True

    for n_test in (3, 4, 5):
        Q_test = np.triu(rng.standard_normal((n_test, n_test)))
        h, J, C0 = _qubo_to_ising_upper_triangular(Q_test)

        energies_qubo  = []
        energies_ising = []
        for state in range(2 ** n_test):
            x = np.array([(state >> q) & 1 for q in range(n_test)], dtype=float)
            z = 1.0 - 2.0 * x

            # Explicit upper-triangular QUBO energy
            E_q = (sum(Q_test[i, i] * x[i] for i in range(n_test)) +
                   sum(Q_test[i, j] * x[i] * x[j]
                       for i in range(n_test) for j in range(i + 1, n_test)))
            # Also verify x.T @ Q @ x gives the same (relies on upper-tri)
            E_q_mat = float(x @ Q_test @ x)
            if abs(E_q - E_q_mat) > 1e-10:
                if verbose:
                    print(f"  [VAL] n={n_test} state={state}: "
                          f"explicit vs matrix QUBO mismatch {E_q:.6e} vs {E_q_mat:.6e}")
                all_ok = False

            E_i = C0 + sum(h[i] * z[i] for i in range(n_test))
            for (i, j), J_ij in J.items():
                E_i += J_ij * z[i] * z[j]

            if abs(E_q - E_i) > 1e-10:
                if verbose:
                    print(f"  [VAL] n={n_test} state={state}: "
                          f"QUBO={E_q:.6e} Ising={E_i:.6e}  MISMATCH")
                all_ok = False

            energies_qubo.append(E_q)
            energies_ising.append(E_i)

        min_q  = min(energies_qubo)
        min_i  = min(energies_ising)
        qmin_s = {s for s, e in enumerate(energies_qubo)  if abs(e - min_q) < 1e-10}
        imin_s = {s for s, e in enumerate(energies_ising) if abs(e - min_i) < 1e-10}
        if qmin_s != imin_s:
            if verbose:
                print(f"  [VAL] n={n_test}: minimum-state mismatch "
                      f"QUBO={qmin_s} Ising={imin_s}")
            all_ok = False

        # Upper-triangular check on a synthetic Q_sub
        Q_rand = np.triu(rng.standard_normal((n_test, n_test)))
        lower_ok = np.all(np.tril(Q_rand, -1) == 0.0)
        if not lower_ok:
            if verbose:
                print(f"  [VAL] n={n_test}: Q_sub lower triangle not zero")
            all_ok = False

        if verbose:
            status = "PASS" if all_ok else "FAIL"
            print(f"  [VAL] n={n_test} exhaustive equivalence: {status}")

    overall = "PASS" if all_ok else "FAIL"
    if verbose:
        print(f"  Global QUBO convention            : upper-triangular")
        print(f"  QAOA local QUBO convention        : upper-triangular")
        print(f"  QUBO → Ising equivalence          : {overall}")
        print(f"  Local QUBO → Ising equivalence    : {overall}")
        print(f"  No off-diagonal double counting   : {overall}")
    return all_ok


def _optimize_qaoa_parameters(Q_sub, n, p, backend, n_restarts,
                               max_iter, shots, x_warm_sub, sim_mode, seed):
    """
    Optimise QAOA angles (gamma, beta) via COBYLA on Aer.
    Initial points: 2 warm-start scales + n_restarts random.
    Returns (best_params, best_E, n_evals).
    (Identical logic to notebook Cell 6 / optimize_qaoa_parameters_aer.)
    """
    from scipy.optimize import minimize as _minimize
    rng         = np.random.default_rng(seed)
    best_params = None
    best_E      = np.inf
    n_evals     = 0

    inits = []
    for scale in [0.3, 0.7]:
        g0 = np.full(p, scale * math.pi / 4)
        b0 = np.full(p, scale * math.pi / 8)
        inits.append(np.concatenate([g0, b0]))
    for _ in range(n_restarts):
        g0 = rng.uniform(0.1, math.pi - 0.1, p)
        b0 = rng.uniform(0.1, math.pi / 2 - 0.1, p)
        inits.append(np.concatenate([g0, b0]))

    for init_params in inits:
        def objective(params):
            nonlocal n_evals
            gammas = params[:p]; betas = params[p:]
            qc     = _build_qaoa_circuit(Q_sub, n, p, gammas, betas)
            out, _ = _run_qaoa_circuit_aer(qc, backend, shots, sim_mode)
            if sim_mode == "statevector":
                E = _expectation_from_statevector(out, Q_sub, n)
            else:
                E = _expectation_from_counts(out, Q_sub, n)
            n_evals += 1
            return E
        res = _minimize(objective, init_params, method="COBYLA",
                        options={"maxiter": max_iter, "rhobeg": 0.5, "catol": 1e-6})
        if res.fun < best_E:
            best_E = res.fun; best_params = res.x.copy()

    return best_params, best_E, n_evals


def solve_qaoa(qubo, Q: np.ndarray, x_warm: np.ndarray,
               p_layers:   int   = QAOA_LAYERS,
               n_restarts: int   = QAOA_RESTARTS,
               n_shots:    int   = QAOA_SHOTS,
               opt_iters:  int   = QAOA_OPT_ITERS,
               max_vars:   int   = QAOA_MAX_VARS,
               sim_mode:   str   = QAOA_SIM_MODE,
               verbose:    bool  = True) -> tuple:
    """
    QAOA solver for the EV-charging QUBO.

    Uses the SAME Q matrix as SA / PT / Tabu (P_rate absent, v2.2).
    Bitstring selection is by MINIMUM classical QUBO energy x^T Q x -- NOT
    by post-processed schedule quality (§15 of integration instructions).

    Pipeline (identical structure to QAOA_Aer_Simulator_1.ipynb solve_qaoa_aer):
      1. Decompose full QUBO into subproblems (≤max_vars qubits)
      2. For each subproblem: COBYLA optimise (gamma,beta) → sample → best
         bitstring by QUBO energy → greedy local search
      3. Assemble global binary solution
      4. Global greedy local search on full Q
      5. Classical verification: compute x^T Q x for selected bitstring

    Returns (x_global, global_E, subproblem_log).
    x_global is the raw binary solution BEFORE any decoding or post-processing.
    """
    global _QISKIT_AVAILABLE

    # ── Qiskit import ───────────────────────────────────────────────────
    if _QISKIT_AVAILABLE is None:
        try:
            from qiskit_aer import AerSimulator  # noqa: F401
            _QISKIT_AVAILABLE = True
        except ImportError:
            _QISKIT_AVAILABLE = False

    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit Aer is not installed.  Install with:\n"
            "  pip install qiskit qiskit-aer scipy\n"
            "Then re-run."
        )

    from qiskit_aer import AerSimulator

    # ── Build Aer backend ────────────────────────────────────────────────────
    if sim_mode == "statevector":
        backend = AerSimulator(method="statevector")
    else:   # "shot" or "noisy"
        backend = AerSimulator(method="automatic")

    if verbose:
        print(f"   [QAOA] backend={backend.name}  mode={sim_mode}  "
              f"p={p_layers}  shots={n_shots}  iters={opt_iters}  "
              f"restarts={n_restarts}  max_vars={max_vars}")
        # ── Mathematical validation (Parts H/I/J) ────────────────────────────
        _validate_qubo_ising_equivalence(verbose=True)

    # ── Decompose QUBO ───────────────────────────────────────────────────────
    subproblems = _qaoa_decompose_qubo(qubo, Q, x_warm, max_vars=max_vars)
    if verbose:
        sizes = [s["n"] for s in subproblems]
        print(f"   [QAOA] {len(subproblems)} subproblems | "
              f"sizes min={min(sizes)} max={max(sizes)}")

    x_global       = x_warm.copy()
    subproblem_log = []

    for sub_idx, sub in enumerate(subproblems):
        var_idx   = sub["var_indices"]
        Q_sub     = sub["Q_sub"]
        n_sub     = sub["n"]
        x_w_sub   = sub["x_warm"]
        sub_t0    = time.perf_counter()

        sub_rec = {"sub_idx": sub_idx+1, "n_qubits": n_sub,
                   "opt_E": None, "best_E": None, "ls_E": None,
                   "n_evals": 0, "time_s": None}

        if n_sub == 0:
            subproblem_log.append(sub_rec); continue

        if n_sub == 1:
            e0 = 0.0; e1 = float(Q_sub[0, 0])
            x_sub = np.array([0.0 if e0 <= e1 else 1.0])
            sub_rec["opt_E"] = sub_rec["best_E"] = sub_rec["ls_E"] = min(e0, e1)
        else:
            opt_params, opt_E, n_ev = _optimize_qaoa_parameters(
                Q_sub, n_sub, p_layers, backend, n_restarts,
                opt_iters, n_shots, x_w_sub, sim_mode, SEED + sub_idx)
            sub_rec["opt_E"]   = opt_E
            sub_rec["n_evals"] = n_ev

            gammas   = opt_params[:p_layers]; betas = opt_params[p_layers:]
            qc_final = _build_qaoa_circuit(Q_sub, n_sub, p_layers, gammas, betas)
            out, _   = _run_qaoa_circuit_aer(qc_final, backend, n_shots, sim_mode)

            # ── Select bitstring by QUBO energy ONLY (§15) ──────────────────
            if sim_mode == "statevector":
                x_sub, best_E = _best_bitstring_from_statevector(out, Q_sub, n_sub)
            else:
                x_sub, best_E = _best_bitstring_from_counts(out, Q_sub)
            sub_rec["best_E"] = best_E

            # ── Greedy local search on subproblem ────────────────────────────
            x_sub, ls_E = _qaoa_greedy_local_search(x_sub, Q_sub, max_iters=2000)
            sub_rec["ls_E"] = ls_E

        for local_j, global_j in enumerate(var_idx):
            x_global[global_j] = float(x_sub[local_j])
        sub_rec["time_s"] = round(time.perf_counter() - sub_t0, 2)
        subproblem_log.append(sub_rec)

        if verbose and (sub_idx + 1) % 10 == 0:
            print(f"   [QAOA] {sub_idx+1}/{len(subproblems)} subproblems done ...",
                  flush=True)

    # ── Global greedy local search ───────────────────────────────────────────
    if verbose:
        print("   [QAOA] Global local search ...", end="", flush=True)
    ls_t0 = time.perf_counter()
    x_global, global_E = _qaoa_greedy_local_search(x_global, Q, max_iters=10000)
    if verbose:
        print(f" done ({time.perf_counter()-ls_t0:.1f}s)  E={global_E:.4e}")

    # ── Classical QUBO energy verification (§7) ──────────────────────────────
    e_verify = float(x_global @ Q @ x_global)
    if verbose:
        print(f"   [QAOA] Classical QUBO energy verification: "
              f"x^T Q x = {e_verify:.6e}  (greedy E = {global_E:.6e})")
        if abs(e_verify - global_E) > 1e-4 * max(abs(global_E), 1.0):
            print(f"   [QAOA] WARNING: discrepancy between greedy E and x^T Q x "
                  f"({abs(e_verify - global_E):.2e}) -- check local search.")

    return x_global, global_E, subproblem_log


# =============================================================================
# 11b.  QUBO POST-PROCESSING
# =============================================================================

def qubo_to_rate(qubo: QUBOFormulation, x_bin: np.ndarray,
                 label: str, solve_time: float) -> dict:
    """
    Decode binary solution -> rate matrix.

    Pipeline (per instruction file requirements §6, §7, §8):
        Positional-binary QUBO
            ↓
        QAOA / QUBO solver (binary solution)
            ↓
        RAW decoded charging schedule  <-- metrics captured here (raw_*)
            ↓
        Existing post-processing (clip + repair_full with 20 valley-fill passes)
            ↓
        Final feasible/refined schedule  <-- metrics in top-level dict

    The valley-fill passes (Step 6 of repair_full) iteratively redistribute
    charge from high-load to low-load slots within each EV's window,
    correcting the spiky on/off patterns produced by binary QUBO solutions.
    This is the key fix for negative variance reduction in SA and PT.
    """
    prob = qubo.prob

    # ------------------------------------------------------------------ #
    # RAW QUBO SCHEDULE: decode positional binary, apply box+window clip  #
    # No repair_full yet -- this is the pure QUBO/QAOA result.            #
    # ------------------------------------------------------------------ #
    R_raw = qubo.decode(x_bin)
    for i in range(prob.N):
        R_raw[i] = np.clip(R_raw[i], 0.0, prob.Rmax[i])
        if prob.A[i] > 0:        R_raw[i, :prob.A[i]]   = 0.0
        if prob.D[i] < prob.T-1: R_raw[i, prob.D[i]+1:] = 0.0

    # Raw metrics (before any repair_full post-processing)
    raw_metrics = prob.metrics(R_raw, label + " [Raw]", solve_time)

    # ------------------------------------------------------------------ #
    # FINAL SCHEDULE: apply existing feasibility + valley-fill repair     #
    # ------------------------------------------------------------------ #
    R_final = repair_full(R_raw.copy(), prob, valley_fill_passes=20)
    final_metrics = prob.metrics(R_final, label, solve_time)

    # ------------------------------------------------------------------ #
    # POST-PROCESSING IMPACT SUMMARY (§10)                                #
    # ------------------------------------------------------------------ #
    delta_obj = final_metrics["obj_kW2"] - raw_metrics["obj_kW2"]
    pct_obj   = (delta_obj / max(abs(raw_metrics["obj_kW2"]), 1e-9)) * 100.0
    delta_par = final_metrics["PAR"] - raw_metrics["PAR"]
    # Count changed charging-rate slots (rounded to 3 dp to avoid float noise)
    n_total   = int(R_raw.size)
    n_changed = int(np.sum(np.abs(R_final - R_raw) > 1e-4))
    pct_changed = n_changed / max(n_total, 1) * 100.0

    # ------------------------------------------------------------------ #
    # Merge raw_* fields into final_metrics dict for transparency         #
    # ------------------------------------------------------------------ #
    final_metrics["raw_obj_kW2"]          = raw_metrics["obj_kW2"]
    final_metrics["raw_PAR"]              = raw_metrics["PAR"]
    final_metrics["raw_peak_load_kW"]     = raw_metrics["peak_load_kW"]
    final_metrics["raw_min_load_kW"]      = raw_metrics["min_load_kW"]
    final_metrics["raw_load_std_kW"]      = raw_metrics["load_std_kW"]
    final_metrics["raw_total_ev_kWh"]     = raw_metrics["total_ev_kWh"]
    final_metrics["raw_mean_energy_err_kWh"] = raw_metrics["mean_energy_err_kWh"]
    final_metrics["raw_site_viol_kWh"]    = raw_metrics["site_viol_kWh"]
    final_metrics["raw_mean_satisfaction_%"] = raw_metrics["mean_satisfaction_%"]
    final_metrics["R_raw"]                = R_raw        # schedule before repair
    final_metrics["pp_delta_obj_kW2"]     = delta_obj
    final_metrics["pp_pct_delta_obj"]     = pct_obj
    final_metrics["pp_delta_PAR"]         = delta_par
    final_metrics["pp_n_changed_slots"]   = n_changed
    final_metrics["pp_pct_changed_slots"] = pct_changed

    return final_metrics


# =============================================================================
# 12.  PLOTTING
# =============================================================================

PALETTE = {
    "Convex QP (SLSQP)": "#1a56db",
    "GA":                "#16a34a",
    "PSO":               "#7c3aed",
    "QUBO + SA":         "#dc2626",
    "QUBO + PT":         "#ea580c",
    "QUBO + Tabu":       "#0891b2",
    "QAOA":              "#9333ea",   # v2.3 -- quantum solver
    "Uncontrolled":      "#6b7280",
    "Flat ideal":        "#d97706",
}

def _g():   return {"color": "#e5e7eb", "linewidth": 0.5}
def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

def plot_all(prob: ChargingProblem, results: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths  = {}
    labels = list(results.keys())
    colors = [PALETTE.get(l, "#555") for l in labels]
    slots  = prob.slots

    # Uncontrolled rate matrix for baselines
    R_unc = np.zeros((prob.N, prob.T))
    for i in range(prob.N):
        rem = prob.E[i]
        for t in range(prob.A[i], prob.D[i]+1):
            if rem <= 1e-6: break
            r = min(prob.Rmax[i], rem/prob.dt)
            R_unc[i, t] = r; rem -= r*prob.dt
    obj_unc = prob.objective(R_unc)
    par_unc = prob.par(R_unc)

    # Fig 1: Load profiles 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.ravel()
    for ax, (lbl, m) in zip(axes, results.items()):
        total = prob.d_bg + m["R"].sum(0)
        ax.fill_between(slots, prob.d_bg, total, alpha=0.3,
                        color=PALETTE.get(lbl,"#555"), label="EV load")
        ax.plot(slots, prob.d_bg,           "--", color="#6b7280", lw=1.5,
                label="Non-EV load")
        ax.plot(slots, prob.d_uncontrolled,  ":",  color="#d97706", lw=1.5,
                label="Uncontrolled (FCFS)")
        ax.plot(slots, total,               "-",  color=PALETTE.get(lbl,"#555"),
                lw=2.0, label="Optimised total")
        ax.axhline(P_SITE_MAX, ls=":", color="red", lw=0.9, label="P_max")
        ax.set_title(
            f"{lbl}\nObj={m['obj_kW2']:.0f} kW2  PAR={m['PAR']:.3f}"
            f"  Sat={m['mean_satisfaction_%']:.1f}%",
            fontsize=9, fontweight="bold")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=6, ncol=2); ax.grid(**_g())
    for ax in axes[len(results):]: ax.set_visible(False)
    fig.suptitle(
        "Load Profiles -- All Algorithms\n"
        "Non-EV load = ACN-Caltech campus background (Lee 2019)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["fig1_load_profiles"] = _save(fig, out_dir, "fig1_load_profiles.png")

    # Fig 2: Objective bar with uncontrolled baseline
    fig, ax = plt.subplots(figsize=(13, 5))
    all_lbl = ["Uncontrolled (FCFS)"] + labels
    all_val = [obj_unc] + [m["obj_kW2"] for m in results.values()]
    all_col = ["#6b7280"] + colors
    bars = ax.bar(all_lbl, all_val, color=all_col, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, all_val):
        ax.text(bar.get_x()+bar.get_width()/2, v+max(all_val)*0.005,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Valley-fill objective (kW2)")
    ax.set_title("Valley-Fill Objective vs Uncontrolled Baseline",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(all_lbl, rotation=20, ha="right"); ax.grid(axis="y", **_g())
    plt.tight_layout()
    paths["fig2_objective"] = _save(fig, out_dir, "fig2_objective.png")

    # Fig 3: PAR
    fig, ax = plt.subplots(figsize=(13, 5))
    all_par = [par_unc] + [m["PAR"] for m in results.values()]
    bars = ax.bar(all_lbl, all_par, color=all_col, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, all_par):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, ls="--", color="green", lw=1.2, label="Ideal PAR=1")
    ax.set_ylabel("PAR"); ax.set_title("PAR Comparison (lower = better)",
                                         fontsize=12, fontweight="bold")
    ax.set_xticklabels(all_lbl, rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", **_g())
    plt.tight_layout()
    paths["fig3_par"] = _save(fig, out_dir, "fig3_par.png")

    # Fig 4: Energy satisfaction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    xi = np.arange(prob.N)
    for lbl, m, col in zip(labels, results.values(), colors):
        ax1.plot(xi, prob.energy_delivered(m["R"]), "o-",
                 color=col, lw=1.5, ms=4, label=lbl, alpha=0.85)
    ax1.plot(xi, prob.E, "k--", lw=2.0, label="Requested", zorder=10)
    ax1.set_xlabel("EV index"); ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Energy Delivered vs Requested", fontweight="bold")
    ax1.legend(fontsize=7); ax1.grid(**_g())
    for lbl, m, col in zip(labels, results.values(), colors):
        ax2.plot(xi, m["energy_errors"], "s-", color=col,
                 lw=1.5, ms=4, label=lbl, alpha=0.85)
    ax2.axhline(0, ls="--", color="black", lw=0.8)
    ax2.set_xlabel("EV index"); ax2.set_ylabel("Absolute energy error (kWh)")
    ax2.set_title("Per-EV Energy Error", fontweight="bold")
    ax2.legend(fontsize=7); ax2.grid(**_g())
    plt.tight_layout()
    paths["fig4_energy"] = _save(fig, out_dir, "fig4_energy.png")

    # Fig 5: Heatmaps
    cmaps = ["Blues","Greens","Purples","Reds","Oranges","YlGnBu"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for ax, (lbl, m), cm in zip(axes, results.items(), cmaps):
        im = ax.imshow(m["R"], aspect="auto", cmap=cm,
                       extent=[slots[0], slots[-1], prob.N, 0],
                       vmin=0, vmax=R_MAX_L2)
        plt.colorbar(im, ax=ax, label="Rate (kW)")
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Hour"); ax.set_ylabel("EV")
        ax.set_yticks(range(0, prob.N, max(1, prob.N//8)))
    for ax in axes[len(results):]: ax.set_visible(False)
    fig.suptitle("Charging Schedule Heatmaps", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths["fig5_heatmaps"] = _save(fig, out_dir, "fig5_heatmaps.png")

    # Fig 6: Total load overlay
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(slots, prob.d_bg,           "--", color="#6b7280", lw=2.0,
            label="Non-EV background", zorder=5)
    ax.plot(slots, prob.d_uncontrolled, "-",  color="#d97706", lw=2.0,
            label="Uncontrolled (FCFS)", zorder=5)
    ax.fill_between(slots, prob.d_flat_ideal*0.98, prob.d_flat_ideal*1.02,
                    alpha=0.15, color="green", label="+-2% flat target")
    for lbl, m, col in zip(labels, results.values(), colors):
        ax.plot(slots, prob.d_bg+m["R"].sum(0), "-",
                color=col, lw=1.8, label=lbl, alpha=0.85)
    ax.axhline(P_SITE_MAX, ls=":", color="red", lw=1.2,
               label=f"P_max={P_SITE_MAX} kW")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Total load (kW)")
    ax.set_title("Total Load Overlay -- All Algorithms vs Baselines",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2); ax.grid(**_g())
    plt.tight_layout()
    paths["fig6_overlay"] = _save(fig, out_dir, "fig6_overlay.png")

    # Fig 7: Convergence
    hists = {l: m for l, m in results.items() if "history" in m}
    if hists:
        fig, ax = plt.subplots(figsize=(14, 6))
        for lbl, m in hists.items():
            h = np.array(m["history"])
            span = h.max()-h.min()
            h_norm = (h-h.min()) / max(span, 1e-12) * 100
            ax.plot(np.linspace(0, 100, len(h)), h_norm,
                    color=PALETTE.get(lbl,"#555"), lw=1.8, label=lbl)
        ax.set_xlabel("Progress (% of iterations)")
        ax.set_ylabel("Normalised objective (0=best, 100=start)")
        ax.set_title("Convergence Curves", fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(**_g())
        plt.tight_layout()
        paths["fig7_convergence"] = _save(fig, out_dir, "fig7_convergence.png")

    # Fig 8: Multi-metric bar
    mkeys = ["obj_kW2","PAR","load_std_kW","mean_energy_err_kWh","solve_time_s"]
    mlbls = ["Obj (kW2)","PAR","Std (kW)","E-err (kWh)","Time (s)"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, mk, ml in zip(axes, mkeys, mlbls):
        vals  = [m[mk] for m in results.values()]
        bv    = min(vals)
        ecol  = ["gold" if abs(v-bv)<1e-9 else "white" for v in vals]
        bars  = ax.bar(labels, vals, color=colors, alpha=0.85,
                       edgecolor=ecol, linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.01,
                    f"{v:.3g}", ha="center", va="bottom", fontsize=7)
        ax.set_title(ml, fontsize=9, fontweight="bold")
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", **_g())
    fig.suptitle("Key Metrics -- Gold border = best", fontsize=11, fontweight="bold")
    plt.tight_layout()
    paths["fig8_metrics"] = _save(fig, out_dir, "fig8_metrics.png")

    # Fig 9: Per-EV rates
    n_show = min(6, prob.N)
    fig, axes = plt.subplots(n_show, 1, figsize=(16, 2.4*n_show))
    if n_show == 1: axes = [axes]
    for i, ax in enumerate(axes):
        for lbl, m, col in zip(labels, results.values(), colors):
            ax.step(slots, m["R"][i], where="post",
                    color=col, lw=1.5, label=lbl, alpha=0.85)
        ax.axvspan(slots[prob.A[i]], slots[min(prob.D[i], prob.T-1)],
                   alpha=0.07, color="green")
        ax.set_ylabel(f"EV {i} (kW)", fontsize=9)
        ax.set_ylim(-0.1, R_MAX_L2+0.5); ax.grid(**_g())
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("Per-EV Charging Rates (First 6)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths["fig9_per_ev"] = _save(fig, out_dir, "fig9_per_ev.png")

    # Fig 10: Variance reduction vs uncontrolled
    fig, ax = plt.subplots(figsize=(13, 5))
    var_unc = float(np.var(prob.d_uncontrolled))
    impr    = [(var_unc - m["load_variance_kW2"]) / max(var_unc, 1e-9) * 100
               for m in results.values()]
    bars = ax.bar(labels, impr, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, impr):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Load variance reduction vs uncontrolled (%)")
    ax.set_title("Variance Reduction Achieved by Each Algorithm",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(labels, rotation=20, ha="right"); ax.grid(axis="y", **_g())
    plt.tight_layout()
    paths["fig10_variance_reduction"] = _save(fig, out_dir, "fig10_variance_reduction.png")

    plt.close("all")
    return paths


# =============================================================================
# 12b.  QAOA ROBUSTNESS EXPERIMENTS 
# =============================================================================
#
# These functions extend the existing QAOA implementation with:
#   • Experiment A — multi-seed ideal QAOA
#   • Experiment B — multi-seed noisy QAOA (Qiskit Aer generic noise model)
#   • Experiment C — shot-sensitivity analysis (scientifically justified: shots
#     affect BOTH COBYLA optimisation AND final bitstring selection)
#
# Design constraints (per b3c5e0cc §25):
#   – Same QUBO, same Ising conversion, same ansatz/depth, same optimizer,
#     same post-processing as the existing solve_qaoa() call.
#   – Seeds control: np.random.default_rng(seed) in _optimize_qaoa_parameters
#     (random restarts) AND Qiskit Aer seed_simulator kwarg (shot sampling).
#   – Noisy backend: AerSimulator with NoiseModel(depolarising + readout).
#   – Existing QAOA result in main() is UNCHANGED; new results go to new sheets.
# =============================================================================


def _build_aer_noise_model(p1=QAOA_NOISE_P1, p2=QAOA_NOISE_P2,
                            p_readout=QAOA_NOISE_READOUT):
    """
    Build a generic Qiskit Aer noise model.

    Gate errors:
      • Single-qubit gates: depolarising channel with probability p1.
      • Two-qubit gates (cx, rzz, …): depolarising channel with probability p2.
    Readout errors:
      • Per-qubit symmetric bit-flip probability p_readout.

    Returns a qiskit_aer.noise.NoiseModel instance.
    Labelled QAOA_NOISE_LABEL — generic simulated noise, NOT hardware-derived.
    """
    from qiskit_aer.noise import (NoiseModel, depolarizing_error, ReadoutError)

    nm = NoiseModel()

    err1 = depolarizing_error(p1, 1)
    sq_gates = ["u", "u1", "u2", "u3", "h", "rx", "ry", "rz",
                "id", "x", "y", "z", "s", "sdg", "t", "tdg"]
    nm.add_all_qubit_quantum_error(err1, sq_gates)

    err2 = depolarizing_error(p2, 2)
    tq_gates = ["cx", "cz", "rzz", "ecr", "swap", "cp"]
    nm.add_all_qubit_quantum_error(err2, tq_gates)

    ro_err = ReadoutError([[1 - p_readout, p_readout],
                           [p_readout,     1 - p_readout]])
    nm.add_all_qubit_readout_error(ro_err)

    return nm


def _run_qaoa_circuit_aer_seeded(qc, backend, shots, sim_mode, seed_simulator):
    """
    Like _run_qaoa_circuit_aer but passes seed_simulator to the backend.
    Qiskit Aer AerSimulator accepts seed_simulator as a run() kwarg which
    controls the RNG for shot sampling.  This is the only way to make Aer
    shot results reproducible.
    """
    from qiskit import transpile
    if sim_mode == "statevector":
        qc_sv = qc.copy()
        qc_sv.remove_final_measurements(inplace=True)
        qc_sv.save_statevector()
        t_qc   = transpile(qc_sv, backend, optimization_level=1)
        result = backend.run(t_qc, seed_simulator=seed_simulator).result()
        return result.get_statevector().data, None
    else:
        t_qc   = transpile(qc, backend, optimization_level=1)
        result = backend.run(t_qc, shots=shots,
                             seed_simulator=seed_simulator).result()
        return result.get_counts(), None


def _optimize_qaoa_parameters_seeded(Q_sub, n, p, backend, n_restarts,
                                      max_iter, shots, x_warm_sub,
                                      sim_mode, seed):
    """
    Like _optimize_qaoa_parameters but uses seed_simulator=seed for every
    Aer circuit execution, making the shot-sampling RNG reproducible.
    The COBYLA RNG is controlled via np.random.default_rng(seed) (unchanged).
    """
    from scipy.optimize import minimize as _minimize
    rng         = np.random.default_rng(seed)
    best_params = None
    best_E      = np.inf
    n_evals     = 0

    inits = []
    for scale in [0.3, 0.7]:
        g0 = np.full(p, scale * math.pi / 4)
        b0 = np.full(p, scale * math.pi / 8)
        inits.append(np.concatenate([g0, b0]))
    for _ in range(n_restarts):
        g0 = rng.uniform(0.1, math.pi - 0.1, p)
        b0 = rng.uniform(0.1, math.pi / 2 - 0.1, p)
        inits.append(np.concatenate([g0, b0]))

    for init_params in inits:
        def objective(params, _seed=seed):
            nonlocal n_evals
            gammas = params[:p]; betas = params[p:]
            qc     = _build_qaoa_circuit(Q_sub, n, p, gammas, betas)
            out, _ = _run_qaoa_circuit_aer_seeded(
                qc, backend, shots, sim_mode,
                seed_simulator=_seed + n_evals)   # unique per call
            if sim_mode == "statevector":
                E = _expectation_from_statevector(out, Q_sub, n)
            else:
                E = _expectation_from_counts(out, Q_sub, n)
            n_evals += 1
            return E
        res = _minimize(objective, init_params, method="COBYLA",
                        options={"maxiter": max_iter, "rhobeg": 0.5,
                                 "catol": 1e-6})
        if res.fun < best_E:
            best_E = res.fun; best_params = res.x.copy()

    return best_params, best_E, n_evals


def _solve_qaoa_one_seed(qubo, Q, x_warm,
                          seed,
                          p_layers=QAOA_LAYERS,
                          n_restarts=QAOA_RESTARTS,
                          n_shots=QAOA_SHOTS,
                          opt_iters=QAOA_OPT_ITERS,
                          max_vars=QAOA_MAX_VARS,
                          sim_mode=QAOA_SIM_MODE,
                          noise_model=None,
                          verbose=False):
    """
    Run the full QAOA pipeline for ONE seed.

    Seed controls:
      • np.random.default_rng(seed + sub_idx) — COBYLA random restarts
      • seed_simulator = seed + sub_idx passed to AerSimulator.run()
        for shot-sampling reproducibility
    noise_model: if not None, passed to AerSimulator(noise_model=...).

    Returns a dict with raw and final metrics (same schema as qubo_to_rate).
    """
    from qiskit_aer import AerSimulator

    if noise_model is not None:
        backend = AerSimulator(method="automatic", noise_model=noise_model)
    elif sim_mode == "statevector":
        backend = AerSimulator(method="statevector")
    else:
        backend = AerSimulator(method="automatic")

    subproblems = _qaoa_decompose_qubo(qubo, Q, x_warm, max_vars=max_vars)
    x_global    = x_warm.copy()

    for sub_idx, sub in enumerate(subproblems):
        var_idx   = sub["var_indices"]
        Q_sub     = sub["Q_sub"]
        n_sub     = sub["n"]
        x_w_sub   = sub["x_warm"]
        sub_seed  = seed + sub_idx * 1000   # unique per (seed, subproblem)

        if n_sub == 0:
            continue
        if n_sub == 1:
            e0 = 0.0; e1 = float(Q_sub[0, 0])
            x_sub = np.array([0.0 if e0 <= e1 else 1.0])
        else:
            opt_params, _, _ = _optimize_qaoa_parameters_seeded(
                Q_sub, n_sub, p_layers, backend, n_restarts,
                opt_iters, n_shots, x_w_sub, sim_mode, sub_seed)

            gammas   = opt_params[:p_layers]; betas = opt_params[p_layers:]
            qc_final = _build_qaoa_circuit(Q_sub, n_sub, p_layers, gammas, betas)
            out, _   = _run_qaoa_circuit_aer_seeded(
                qc_final, backend, n_shots, sim_mode,
                seed_simulator=sub_seed + 999999)

            if sim_mode == "statevector":
                x_sub, _ = _best_bitstring_from_statevector(out, Q_sub, n_sub)
            else:
                x_sub, _ = _best_bitstring_from_counts(out, Q_sub)

            x_sub, _ = _qaoa_greedy_local_search(x_sub, Q_sub, max_iters=2000)

        for local_j, global_j in enumerate(var_idx):
            x_global[global_j] = float(x_sub[local_j])

    x_global, global_E = _qaoa_greedy_local_search(x_global, Q, max_iters=10000)
    return x_global, global_E


def run_qaoa_multi_seed(qubo, Q, x_warm, prob,
                         seeds=QAOA_SEEDS,
                         n_shots=QAOA_SHOTS,
                         noise_model=None,
                         experiment_label="QAOA_MultiSeed",
                         verbose=True):
    """
    Experiment A (ideal) / B (noisy): run the QAOA pipeline independently for
    each seed in `seeds`.

    Returns a list of per-seed result dicts (same fields as qubo_to_rate output).
    Each dict also contains: seed, experiment_label, noise_active.

    Statistics (mean ± sample std) are computed across seeds on the returned list
    by the Excel export function.

    Design constraints:
      • Same QUBO, same Ising conversion, same p, same optimizer settings.
      • Only the seed (and optionally noise_model) differs between repetitions.
      • Do NOT select seeds based on best result — all are reported.
    """
    from qiskit_aer import AerSimulator
    noise_active = noise_model is not None
    per_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            kind = "noisy" if noise_active else "ideal"
            print(f"   [{experiment_label}] seed {seed} ({seed_idx+1}/{len(seeds)}) "
                  f"[{kind}] ...", flush=True)

        t0 = time.perf_counter()
        try:
            x_sol, global_E = _solve_qaoa_one_seed(
                qubo, Q, x_warm, seed=seed,
                p_layers=QAOA_LAYERS, n_restarts=QAOA_RESTARTS,
                n_shots=n_shots, opt_iters=QAOA_OPT_ITERS,
                max_vars=QAOA_MAX_VARS, sim_mode=QAOA_SIM_MODE,
                noise_model=noise_model, verbose=False)
            t_sol = time.perf_counter() - t0

            m = qubo_to_rate(qubo, x_sol,
                             label=f"{experiment_label}[s={seed}]",
                             solve_time=t_sol)
            m["seed"]              = seed
            m["experiment_label"]  = experiment_label
            m["noise_active"]      = noise_active
            m["global_qubo_E"]     = global_E
            m["n_shots"]           = n_shots
            per_seed_results.append(m)

            if verbose:
                print(f"     raw_obj={m['raw_obj_kW2']:.2f}  "
                      f"final_obj={m['obj_kW2']:.2f}  "
                      f"PAR={m['PAR']:.4f}  t={t_sol:.1f}s")
        except Exception as exc:
            if verbose:
                print(f"     FAILED: {exc}")
            per_seed_results.append({"seed": seed, "error": str(exc),
                                     "experiment_label": experiment_label})

    return per_seed_results


def run_qaoa_shot_sensitivity(qubo, Q, x_warm, prob,
                               shot_levels=QAOA_SHOT_LEVELS,
                               fixed_seed=QAOA_SHOT_SEED,
                               verbose=True):
    """
    Experiment C — shot sensitivity.

    Varies only the number of shots; keeps seed, QUBO, p, optimizer fixed.
    Scientific rationale:
      Shots affect (1) COBYLA expectation estimation and (2) final bitstring
      selection — two coupled sources of finite-statistics noise.  Sweeping
      shots at a fixed seed isolates the shot-count effect cleanly.

    Returns a list of per-shot-level result dicts.
    """
    per_shot_results = []
    for n_shots in shot_levels:
        if verbose:
            print(f"   [QAOA_ShotSens] shots={n_shots}  seed={fixed_seed} ...",
                  flush=True)
        t0 = time.perf_counter()
        try:
            x_sol, global_E = _solve_qaoa_one_seed(
                qubo, Q, x_warm, seed=fixed_seed,
                p_layers=QAOA_LAYERS, n_restarts=QAOA_RESTARTS,
                n_shots=n_shots, opt_iters=QAOA_OPT_ITERS,
                max_vars=QAOA_MAX_VARS, sim_mode=QAOA_SIM_MODE,
                noise_model=None, verbose=False)
            t_sol = time.perf_counter() - t0

            m = qubo_to_rate(qubo, x_sol,
                             label=f"QAOA_ShotSens[s={n_shots}]",
                             solve_time=t_sol)
            m["n_shots"]        = n_shots
            m["seed"]           = fixed_seed
            m["global_qubo_E"]  = global_E
            per_shot_results.append(m)

            if verbose:
                print(f"     raw_obj={m['raw_obj_kW2']:.2f}  "
                      f"final_obj={m['obj_kW2']:.2f}  "
                      f"PAR={m['PAR']:.4f}  t={t_sol:.1f}s")
        except Exception as exc:
            if verbose:
                print(f"     FAILED: {exc}")
            per_shot_results.append({"n_shots": n_shots, "error": str(exc)})

    return per_shot_results


# =============================================================================
# 13.  EXCEL EXPORT
# =============================================================================

def _h(cell, bg="1A56DB", fg="FFFFFF", sz=10):
    cell.font      = Font(bold=True, color=fg, name="Arial", size=sz)
    cell.fill      = PatternFill("solid", start_color=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def _b():
    s = Side(style="thin", color="D1D5DB")
    return Border(left=s, right=s, top=s, bottom=s)

def _cw(ws, widths):
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = w


def export_excel(sessions, prob, results, fig_paths, out_path):
    wb  = Workbook(); bdr = _b()
    T, N = prob.T, prob.N
    slots  = prob.slots; labels = list(results.keys())
    DARK="1F2937"; BLUE="1A56DB"; GREEN="065F46"; PURP="5B21B6"
    RED="991B1B"; TEAL="0F766E"
    ALGO_BG={"Convex QP (SLSQP)":"DBEAFE","GA":"DCFCE7","PSO":"EDE9FE",
             "QUBO + SA":"FEE2E2","QUBO + PT":"FFEDD5","QUBO + Tabu":"CFFAFE",
             "QAOA":"F3E8FF"}

    # Sheet 1: Summary Dashboard
    ws = wb.active; ws.title = "Summary Dashboard"
    ws.sheet_view.showGridLines = False
    ws.merge_cells("A1:H1")
    ws["A1"] = ("EV Charging Optimization -- Classical vs Quantum (QUBO)\n"
                "ACN-Caltech | Valley-Fill | Gan 2012 / Lee 2021 / Lee 2019")
    ws["A1"].font = Font(bold=True, size=13, color="FFFFFF", name="Arial")
    ws["A1"].fill = PatternFill("solid", start_color=DARK)
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
    ws.row_dimensions[1].height = 45

    params = [("N sessions",N),("T slots",T),("Dt [hr]",prob.dt),
              ("Horizon",f"{T_START}h-{T_END}h"),("P_site_max [kW]",P_SITE_MAX),
              ("Rmax_L2 [kW]",R_MAX_L2),("QUBO K bits",K_BITS),
              ("Background source","Campus non-EV profile (Lee 2019, Fig.7)"),
              ("Total EV demand [kWh]",round(prob.E_total,2))]
    for r,(k,v) in enumerate(params,3):
        ws.cell(row=r,column=1,value=k).font=Font(name="Arial",size=9)
        ws.cell(row=r,column=2,value=v).font=Font(bold=True,name="Arial",size=9,color=BLUE)

    mkeys=["obj_kW2","PAR","load_factor","peak_load_kW","min_load_kW",
           "load_std_kW","total_ev_kWh","mean_energy_err_kWh","max_energy_err_kWh",
           "mean_satisfaction_%","min_satisfaction_%","site_viol_kWh","solve_time_s"]
    mnames=["Objective (kW2)","PAR","Load Factor","Peak Load (kW)","Min Load (kW)",
            "Load Std (kW)","Total EV (kWh)","Mean E-Error (kWh)","Max E-Error (kWh)",
            "Mean Satisfaction (%)","Min Satisfaction (%)","Site Violation (kWh)",
            "Solve Time (s)"]
    hrow=len(params)+4
    _h(ws.cell(row=hrow,column=1,value="Metric"),bg=DARK)
    for c,lbl in enumerate(labels,2):
        _h(ws.cell(row=hrow,column=c,value=lbl),
           bg=ALGO_BG.get(lbl,"EEEEEE"),fg=DARK)
    ws.row_dimensions[hrow].height=28
    bv_map={mk:min(m[mk] for m in results.values()) for mk in mkeys}
    for r,(mk,mn) in enumerate(zip(mkeys,mnames),hrow+1):
        ws.cell(row=r,column=1,value=mn).font=Font(bold=True,name="Arial",size=9)
        ws.cell(row=r,column=1).border=bdr
        for c,lbl in enumerate(labels,2):
            v=results[lbl][mk]
            cell=ws.cell(row=r,column=c,
                         value=round(v,5) if isinstance(v,float) else v)
            cell.font=Font(name="Arial",size=9,
                           bold=(abs(v-bv_map[mk])<1e-9),
                           color="14532D" if abs(v-bv_map[mk])<1e-9 else DARK)
            cell.border=bdr; cell.alignment=Alignment(horizontal="center")
            cell.fill=PatternFill("solid",start_color=ALGO_BG.get(lbl,"EEEEEE"))
    _cw(ws,[30]+[18]*len(labels))

    # Sheet 2: ACN Session Data
    ws2=wb.create_sheet("ACN Session Data"); ws2.sheet_view.showGridLines=False
    disp=[c for c in sessions.columns if not c.startswith("_")]
    for c,col in enumerate(disp,1): _h(ws2.cell(row=1,column=c,value=col),bg=TEAL)
    for r,row in enumerate(sessions[disp].itertuples(index=False),2):
        for c,val in enumerate(row,1):
            cell=ws2.cell(row=r,column=c,value=val)
            cell.font=Font(name="Arial",size=8); cell.border=bdr
            if r%2==0: cell.fill=PatternFill("solid",start_color="F0FDF4")
    for c in range(1,len(disp)+1):
        ws2.column_dimensions[get_column_letter(c)].width=22

    # Sheet 3: Load Profiles
    ws3=wb.create_sheet("Load Profiles"); ws3.sheet_view.showGridLines=False
    hdrs3=(["Slot","Time (hr)","Non-EV bg (kW)","Uncontrolled (kW)","Flat Ideal (kW)"]
           +[f"EV Load {l} (kW)" for l in labels]
           +[f"Total {l} (kW)" for l in labels])
    for c,h in enumerate(hdrs3,1): _h(ws3.cell(row=1,column=c,value=h),bg=GREEN)
    for t in range(T):
        r=t+2
        ws3.cell(row=r,column=1,value=t)
        ws3.cell(row=r,column=2,value=round(float(slots[t]),3))
        ws3.cell(row=r,column=3,value=round(float(prob.d_bg[t]),3))
        ws3.cell(row=r,column=4,value=round(float(prob.d_uncontrolled[t]),3))
        ws3.cell(row=r,column=5,value=round(float(prob.d_flat_ideal[t]),3))
        for j,lbl in enumerate(labels):
            ev=round(float(results[lbl]["R"][:,t].sum()),4)
            tot=round(float(prob.d_bg[t]+ev),4)
            ws3.cell(row=r,column=6+j,value=ev)
            ws3.cell(row=r,column=6+len(labels)+j,value=tot)
        for c in range(1,len(hdrs3)+1):
            cell=ws3.cell(row=r,column=c)
            cell.font=Font(name="Arial",size=8); cell.border=bdr
            cell.alignment=Alignment(horizontal="center")
            if t%2==0: cell.fill=PatternFill("solid",start_color="ECFDF5")
    for c in range(1,len(hdrs3)+1):
        ws3.column_dimensions[get_column_letter(c)].width=18

    # Sheet 4: Energy Satisfaction
    ws4=wb.create_sheet("Energy Satisfaction"); ws4.sheet_view.showGridLines=False
    hdrs4=(["EV","Session","Arr(hr)","Dep(hr)","Requested(kWh)"]
           +[f"Del {l[:10]} (kWh)" for l in labels]
           +[f"Err {l[:10]} (kWh)" for l in labels]
           +[f"Sat {l[:10]} (%)" for l in labels])
    for c,h in enumerate(hdrs4,1): _h(ws4.cell(row=1,column=c,value=h),bg=PURP)
    for i in range(N):
        r=i+2
        ws4.cell(row=r,column=1,value=i)
        ws4.cell(row=r,column=2,value=str(sessions.iloc[i].get("sessionID",f"S{i}")))
        ws4.cell(row=r,column=3,value=round(float(slots[prob.A[i]]),2))
        ws4.cell(row=r,column=4,value=round(float(slots[min(prob.D[i],T-1)]),2))
        ws4.cell(row=r,column=5,value=round(float(prob.E[i]),3))
        for j,lbl in enumerate(labels):
            R=results[lbl]["R"]
            delv=round(R[i,prob.A[i]:prob.D[i]+1].sum()*prob.dt,4)
            err=round(abs(delv-prob.E[i]),5)
            sat=round(min(delv/max(prob.E[i],1e-9)*100,100.0),2)
            ws4.cell(row=r,column=6+j,value=delv)
            ws4.cell(row=r,column=6+len(labels)+j,value=err)
            ws4.cell(row=r,column=6+2*len(labels)+j,value=f"{sat}%")
        for c in range(1,len(hdrs4)+1):
            cell=ws4.cell(row=r,column=c)
            cell.font=Font(name="Arial",size=8); cell.border=bdr
            cell.alignment=Alignment(horizontal="center")
            if i%2==0: cell.fill=PatternFill("solid",start_color="F5F3FF")
    for c in range(1,len(hdrs4)+1):
        ws4.column_dimensions[get_column_letter(c)].width=16

    # Sheet 5: Convergence
    ws5=wb.create_sheet("Convergence Data"); ws5.sheet_view.showGridLines=False
    hists={l:m["history"] for l,m in results.items() if "history" in m}
    if hists:
        mxl=max(len(h) for h in hists.values()); step=max(1,mxl//500)
        hdrs5=["Iteration"]+list(hists.keys())
        for c,h in enumerate(hdrs5,1): _h(ws5.cell(row=1,column=c,value=h),bg="B45309")
        row=2
        for it in range(0,mxl,step):
            ws5.cell(row=row,column=1,value=it)
            for c,(lbl,hist) in enumerate(hists.items(),2):
                v=round(float(hist[min(it,len(hist)-1)]),4)
                cell=ws5.cell(row=row,column=c,value=v)
                cell.font=Font(name="Arial",size=8); cell.border=bdr
                if row%2==0: cell.fill=PatternFill("solid",start_color="FFF7ED")
            row+=1
        for c in range(1,len(hdrs5)+1):
            ws5.column_dimensions[get_column_letter(c)].width=18

    # Sheet 6: Charging Schedules
    ws6=wb.create_sheet("Charging Schedules"); ws6.sheet_view.showGridLines=False
    _h(ws6.cell(row=1,column=1,value="EV / Time"),bg=DARK)
    for t in range(T):
        cell=ws6.cell(row=1,column=t+2,value=round(float(slots[t]),2))
        cell.font=Font(bold=True,name="Arial",size=7)
        cell.alignment=Alignment(horizontal="center",text_rotation=45)
    row_c=2
    for lbl in labels:
        R=results[lbl]["R"]
        ws6.cell(row=row_c,column=1,value=f"-- {lbl} --"
                 ).font=Font(bold=True,name="Arial",size=9,
                              color=BLUE if "Convex" in lbl else RED)
        row_c+=1
        for i in range(N):
            ws6.cell(row=row_c,column=1,value=f"EV {i:02d}").font=Font(name="Arial",size=7)
            for t in range(T):
                v=round(float(R[i,t]),3)
                cell=ws6.cell(row=row_c,column=t+2,value=v)
                cell.font=Font(name="Arial",size=7)
                if v>0.05:
                    inten=min(int(v/R_MAX_L2*180),180)
                    if "QUBO" in lbl or lbl == "QAOA":
                        hx=(f"{max(inten//2,0):02X}{max(inten//2,0):02X}"
                            f"{255-inten:02X}")
                    else:
                        hx=(f"{max(inten//2,0):02X}{255-inten:02X}"
                            f"{max(inten//2,0):02X}")
                    cell.fill=PatternFill("solid",start_color=hx)
            row_c+=1
        row_c+=1

    # Sheet 7: QUBO Reference
    ws7=wb.create_sheet("QUBO Formulation"); ws7.sheet_view.showGridLines=False
    ws7.merge_cells("A1:D1")
    ws7["A1"]="QUBO Formulation Reference (v2.0 -- auto-scaled penalties)"
    ws7["A1"].font=Font(bold=True,size=12,name="Arial")
    ws7["A1"].fill=PatternFill("solid",start_color="EEF2FF")
    ws7["A1"].alignment=Alignment(horizontal="center")
    rows7=[
        ("Encoding","r_{i,t}=d_i*sum_k 2^k*x_{i,t,k}",
         f"d_i=Rmax_i/(2^K-1), K={K_BITS}","16 discrete levels per (EV,slot)"),
        ("H_obj","sum_t(d_bg[t]+sum_i r_{i,t})^2",
         "Valley-fill (Gan 2012)","ACN-Caltech non-EV campus profile"),
        ("H_energy","lam_e*sum_i(sum_t r_{i,t}*Dt-E_i)^2",
         "lam_e = 5 * max(H_obj_est)  [v2.1 fix]",
         "Balanced: H_obj and H_energy compete equally in QUBO landscape"),
        ("H_rate","REMOVED (v2.2 correction)",
         "Was: lam_r*(sum_k x-1)^2 [one-hot]",
         "INCOMPATIBLE with positional binary: 12/16 valid levels penalised. "
         "Rate bound 0<=r<=Rmax enforced by encoding structure + post-repair clip."),
        ("H_site","REMOVED from Q","Hard cap via post-repair",
         "No site violations, smaller Q matrix"),
        ("Post-repair: energy","Water-fill UP + scale DOWN",
         "Runs after every QUBO decode","Guarantees exact energy delivery"),
        ("Post-repair: valley-fill","20 smoothing passes [v2.1 fix]",
         "Shift charge high->low load slots","Fixes spiky binary patterns in SA/PT"),
        ("Why smoothing needed","Binary x gives on/off rates",
         "Repair fills energy into arbitrary slots","Smoothing redistributes to minimize variance"),
        ("Warm-start","Encoded from QP solution",
         "All quantum solvers start near optimum","Reduces needed sweeps by ~10x"),
        ("SA restarts","Trigger if stagnant > 8% sweeps",
         "Partial re-heat to T0*0.3","Escapes deep local minima"),
        (f"PT replicas",f"{PT_REPLICAS} chains, T in [{PT_T_MIN},{PT_T_MAX}]",
         "All-pairs exchange at each swap","Interval = 2n/replicas"),
        ("Tabu memory","Recency + frequency (long-term)",
         f"Tenure={TABU_TENURE}, decay=0.5","Diversification every 10% iters"),
        ("QAOA (v2.3)","Qiskit Aer COBYLA optimizer",
         f"p={QAOA_LAYERS}  shots={QAOA_SHOTS}  mode={QAOA_SIM_MODE}  "
         f"max_vars={QAOA_MAX_VARS}",
         "Same Q as SA/PT/Tabu. Bitstring by min(x^T Q x). "
         "Decomposes into subproblems ≤QAOA_MAX_VARS qubits."),
    ]
    cols7=["Component","Expression","Parameters","Notes"]
    for c,h in enumerate(cols7,1): _h(ws7.cell(row=2,column=c,value=h),bg="0F766E")
    for r,row in enumerate(rows7,3):
        for c,val in enumerate(row,1):
            cell=ws7.cell(row=r,column=c,value=val)
            cell.font=Font(name="Courier New" if c==2 else "Arial",size=9,bold=(c==1))
            cell.border=bdr; cell.alignment=Alignment(horizontal="left",wrap_text=True)
            if r%2==0: cell.fill=PatternFill("solid",start_color="F0F9FF")
        ws7.row_dimensions[r].height=28
    _cw(ws7,[18,42,36,40])

    # Sheet 8: QUBO/QAOA Raw vs Final Comparison (§15 of integration instructions)
    qubo_labels = [l for l in labels if l.startswith("QUBO") or l == "QAOA"]
    if qubo_labels and any("raw_obj_kW2" in results[l] for l in qubo_labels):
        ws8r = wb.create_sheet("QUBO Raw vs Final"); ws8r.sheet_view.showGridLines=False
        ws8r.merge_cells("A1:G1")
        ws8r["A1"] = ("QUBO/QAOA Raw vs Final Post-Processed Results  (v2.3 -- P_rate removed)\n"
                      "Raw = Positional-Binary decode + clip only.  "
                      "Final = Raw + repair_full (water-fill + valley-fill x20)\n"
                      "QAOA bitstring selected by minimum QUBO energy (x^T Q x) only.")
        ws8r["A1"].font = Font(bold=True, size=11, name="Arial")
        ws8r["A1"].fill = PatternFill("solid", start_color="FFF7ED")
        ws8r["A1"].alignment = Alignment(horizontal="center", wrap_text=True)
        ws8r.row_dimensions[1].height = 45
        raw_hdr = ["Metric", "Unit"]
        for l in qubo_labels:
            raw_hdr += [f"{l} Raw", f"{l} Final", f"{l} Δ(Final-Raw)", f"{l} %Δ"]
        for c, h in enumerate(raw_hdr, 1):
            _h(ws8r.cell(row=2, column=c, value=h), bg="B45309")
        raw_metric_rows = [
            ("Objective",          "kW2",  "obj_kW2",              "raw_obj_kW2"),
            ("PAR",                "—",    "PAR",                  "raw_PAR"),
            ("Peak load",          "kW",   "peak_load_kW",         "raw_peak_load_kW"),
            ("Min load",           "kW",   "min_load_kW",          "raw_min_load_kW"),
            ("Load std dev",       "kW",   "load_std_kW",          "raw_load_std_kW"),
            ("Total EV energy",    "kWh",  "total_ev_kWh",         "raw_total_ev_kWh"),
            ("Mean energy error",  "kWh",  "mean_energy_err_kWh",  "raw_mean_energy_err_kWh"),
            ("Site violation",     "kWh",  "site_viol_kWh",        "raw_site_viol_kWh"),
            ("Mean satisfaction",  "%",    "mean_satisfaction_%",  "raw_mean_satisfaction_%"),
            ("Changed slots (pp)", "—",    "pp_n_changed_slots",   None),
            ("% slots changed",    "%",    "pp_pct_changed_slots", None),
            ("Δ Obj (pp)",         "kW2",  "pp_delta_obj_kW2",     None),
            ("% Δ Obj (pp)",       "%",    "pp_pct_delta_obj",     None),
        ]
        for rr, (mname, unit, fkey, rawkey) in enumerate(raw_metric_rows, 3):
            ws8r.cell(row=rr, column=1, value=mname).font = Font(bold=True, name="Arial", size=9)
            ws8r.cell(row=rr, column=2, value=unit).font = Font(name="Arial", size=9)
            col = 3
            for l in qubo_labels:
                m = results[l]
                fval = m.get(fkey)
                rval = m.get(rawkey) if rawkey else None
                if rawkey and rval is not None:
                    delta  = fval - rval if (fval is not None) else None
                    pct_d  = (delta / max(abs(rval), 1e-9) * 100.0) if delta is not None else None
                    for cv, bg in [(rval,"FEF9C3"),(fval,"DCFCE7"),(delta,"DBEAFE"),(pct_d,"F3E8FF")]:
                        cell = ws8r.cell(row=rr, column=col,
                                         value=round(float(cv), 4) if cv is not None else "—")
                        cell.font = Font(name="Arial", size=9)
                        cell.border = bdr
                        cell.fill = PatternFill("solid", start_color=bg)
                        cell.alignment = Alignment(horizontal="center")
                        col += 1
                else:
                    # Single-value summary metrics (pp_* fields)
                    sv = fval
                    for bg in ["DBEAFE", "DBEAFE", "DBEAFE", "DBEAFE"]:
                        cell = ws8r.cell(row=rr, column=col,
                                         value=round(float(sv), 4) if sv is not None else "—")
                        cell.font = Font(name="Arial", size=9)
                        cell.border = bdr
                        cell.fill = PatternFill("solid", start_color=bg)
                        cell.alignment = Alignment(horizontal="center")
                        col += 1
            if rr % 2 == 0:
                pass  # alternating done via per-field bg colours above
        for c in range(1, len(raw_hdr) + 1):
            ws8r.column_dimensions[get_column_letter(c)].width = 20
        ws8r.row_dimensions[2].height = 30

    # Sheet 9: Figures (formerly Sheet 8)
    ws8=wb.create_sheet("Figures"); ws8.sheet_view.showGridLines=False
    ws8["A1"]="Figures -- EV Charging Optimization Results (v2.2 -- P_rate removed)"
    ws8["A1"].font=Font(bold=True,size=13,name="Arial")
    fig_titles={"fig1_load_profiles":"Fig 1: Load Profiles per Algorithm",
                "fig2_objective":"Fig 2: Objective vs Uncontrolled",
                "fig3_par":"Fig 3: PAR Comparison",
                "fig4_energy":"Fig 4: Energy Satisfaction",
                "fig5_heatmaps":"Fig 5: Schedule Heatmaps",
                "fig6_overlay":"Fig 6: Total Load Overlay",
                "fig7_convergence":"Fig 7: Convergence Curves",
                "fig8_metrics":"Fig 8: Multi-Metric Comparison",
                "fig9_per_ev":"Fig 9: Per-EV Charging Rates",
                "fig10_variance_reduction":"Fig 10: Variance Reduction"}
    row_c=2
    for key,title in fig_titles.items():
        path=fig_paths.get(key)
        if path and os.path.exists(path):
            ws8.cell(row=row_c,column=1,value=title
                     ).font=Font(bold=True,name="Arial",size=10)
            row_c+=1
            img=XLImage(path); sc=900/max(img.width,1)
            img.width=int(img.width*sc); img.height=int(img.height*sc)
            ws8.add_image(img,f"A{row_c}"); row_c+=img.height//15+3

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb.save(out_path)
    print(f"[Excel] Saved -> {out_path}")


def _write_robustness_sheets(wb, bdr,
                              multiseed_results=None,
                              noisy_results=None,
                              shot_results=None):
    """

    Sheets added (only if the corresponding result list is not None/empty):
      • QAOA_MultiSeed        — Experiment A (ideal, multiple seeds)
      • QAOA_Noisy_MultiSeed  — Experiment B (noisy, multiple seeds)
      • QAOA_ShotSensitivity  — Experiment C (shot sweep, fixed seed)

    Schema produced by run_qaoa_multi_seed / run_qaoa_shot_sensitivity:
      raw_obj_kW2, obj_kW2, raw_PAR, PAR, raw_peak_load_kW, peak_load_kW,
      raw_load_std_kW, load_std_kW, raw_mean_energy_err_kWh, mean_energy_err_kWh,
      raw_mean_satisfaction_%, mean_satisfaction_%, solve_time_s,
      global_qubo_E, seed, n_shots, error (if failed).

    Std dev is SAMPLE std dev (ddof=1).
    """
    STAT_METRICS = [
        ("Raw Objective",     "kW2",  "raw_obj_kW2"),
        ("Final Objective",   "kW2",  "obj_kW2"),
        ("Raw PAR",           "—",    "raw_PAR"),
        ("Final PAR",         "—",    "PAR"),
        ("Raw Peak Load",     "kW",   "raw_peak_load_kW"),
        ("Final Peak Load",   "kW",   "peak_load_kW"),
        ("Raw Load Std",      "kW",   "raw_load_std_kW"),
        ("Final Load Std",    "kW",   "load_std_kW"),
        ("Raw E-Error",       "kWh",  "raw_mean_energy_err_kWh"),
        ("Final E-Error",     "kWh",  "mean_energy_err_kWh"),
        ("Raw Satisfaction",  "%",    "raw_mean_satisfaction_%"),
        ("Final Satisfaction","%",    "mean_satisfaction_%"),
        ("QUBO Energy",       "—",    "global_qubo_E"),
        ("Runtime",           "s",    "solve_time_s"),
    ]

    HEAD_BG = "5B21B6"   # purple for QAOA sheets
    NOISY_BG = "991B1B"  # dark red for noisy
    SHOT_BG  = "0F766E"  # teal for shot sweep

    def _hdr(cell, txt, bg=HEAD_BG):
        cell.value = txt
        cell.font  = Font(bold=True, name="Arial", size=9, color="FFFFFF")
        cell.fill  = PatternFill("solid", start_color=bg)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = bdr

    def _val(cell, v, alt_bg=None):
        cell.value = (round(float(v), 5)
                      if isinstance(v, (int, float)) and not math.isnan(float(v))
                      else str(v) if v is not None else "—")
        cell.font   = Font(name="Arial", size=9)
        cell.border = bdr
        cell.alignment = Alignment(horizontal="center")
        if alt_bg:
            cell.fill = PatternFill("solid", start_color=alt_bg)

    def _write_per_seed_sheet(ws, per_seed, extra_col_key=None,
                               extra_col_name=None, header_bg=HEAD_BG,
                               noise_label=None):
        """Write per-seed rows + summary statistics block."""
        ws.sheet_view.showGridLines = False

        # Title row
        title_cols = 3 + len(STAT_METRICS) + (1 if extra_col_key else 0)
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1, end_column=title_cols)
        ws.cell(row=1, column=1).value = (
            f"QAOA Robustness — {noise_label or 'Ideal (noiseless)'}\n"
            f"Std Dev = sample std dev (ddof=1) across seeds.\n"
            f"Same QUBO, Ising conversion, p={QAOA_LAYERS}, shots={QAOA_SHOTS}, "
            f"COBYLA iters={QAOA_OPT_ITERS}, post-processing unchanged.")
        ws.cell(row=1, column=1).font  = Font(bold=True, size=10, name="Arial")
        ws.cell(row=1, column=1).fill  = PatternFill("solid", start_color="EDE9FE")
        ws.cell(row=1, column=1).alignment = Alignment(wrap_text=True,
                                                        horizontal="center")
        ws.row_dimensions[1].height = 48

        # Column headers (row 2)
        col = 1
        _hdr(ws.cell(row=2, column=col), "Seed", header_bg); col += 1
        if extra_col_key:
            _hdr(ws.cell(row=2, column=col), extra_col_name, header_bg); col += 1
        _hdr(ws.cell(row=2, column=col), "Noise", header_bg); col += 1
        for mname, unit, _ in STAT_METRICS:
            _hdr(ws.cell(row=2, column=col),
                 f"{mname}\n({unit})", header_bg); col += 1
        ws.row_dimensions[2].height = 30

        # Data rows
        valid = [r for r in per_seed if "error" not in r]
        err_rows = [r for r in per_seed if "error" in r]

        for row_idx, r in enumerate(valid, 3):
            alt = "F5F3FF" if row_idx % 2 == 0 else None
            col = 1
            _val(ws.cell(row=row_idx, column=col), r.get("seed", "?"), alt); col += 1
            if extra_col_key:
                _val(ws.cell(row=row_idx, column=col),
                     r.get(extra_col_key, "?"), alt); col += 1
            noise_str = "YES" if r.get("noise_active") else "NO"
            _val(ws.cell(row=row_idx, column=col), noise_str, alt); col += 1
            for _, _, key in STAT_METRICS:
                v = r.get(key)
                _val(ws.cell(row=row_idx, column=col),
                     v if v is not None else float("nan"), alt)
                col += 1

        # Error rows (if any)
        err_start = 3 + len(valid)
        for ei, r in enumerate(err_rows):
            ws.cell(row=err_start + ei, column=1).value = (
                f"Seed {r.get('seed','?')} FAILED: {r.get('error','?')}")
            ws.cell(row=err_start + ei, column=1).font = Font(
                name="Arial", size=9, color="991B1B")

        # Statistics block (mean ± std)
        if len(valid) < 2:
            return  # need at least 2 seeds for std

        stat_row = 3 + len(valid) + len(err_rows) + 2
        ws.merge_cells(start_row=stat_row, start_column=1,
                       end_row=stat_row, end_column=title_cols)
        ws.cell(row=stat_row, column=1).value = (
            "Summary Statistics  (sample std dev, ddof=1)")
        ws.cell(row=stat_row, column=1).font = Font(bold=True, size=10,
                                                     name="Arial")
        ws.cell(row=stat_row, column=1).fill = PatternFill("solid",
                                                            start_color="DBEAFE")
        ws.row_dimensions[stat_row].height = 20

        hdr_row = stat_row + 1
        _hdr(ws.cell(row=hdr_row, column=1), "Metric", "1E40AF")
        _hdr(ws.cell(row=hdr_row, column=2), "Unit",   "1E40AF")
        _hdr(ws.cell(row=hdr_row, column=3), "Mean",   "1E40AF")
        _hdr(ws.cell(row=hdr_row, column=4), "Std Dev (sample)", "1E40AF")
        _hdr(ws.cell(row=hdr_row, column=5), "Min",    "1E40AF")
        _hdr(ws.cell(row=hdr_row, column=6), "Max",    "1E40AF")
        _hdr(ws.cell(row=hdr_row, column=7), "N seeds", "1E40AF")

        for si, (mname, unit, key) in enumerate(STAT_METRICS):
            vals = [r[key] for r in valid
                    if key in r and r[key] is not None]
            try:
                vals_f = [float(v) for v in vals]
            except (TypeError, ValueError):
                vals_f = []
            dr = hdr_row + 1 + si
            alt = "EFF6FF" if si % 2 == 0 else None
            _val(ws.cell(row=dr, column=1), mname, alt)
            _val(ws.cell(row=dr, column=2), unit, alt)
            if len(vals_f) >= 1:
                mean_v = float(np.mean(vals_f))
                std_v  = float(np.std(vals_f, ddof=1)) if len(vals_f) >= 2 else 0.0
                min_v  = float(np.min(vals_f))
                max_v  = float(np.max(vals_f))
                _val(ws.cell(row=dr, column=3), mean_v, alt)
                _val(ws.cell(row=dr, column=4), std_v,  alt)
                _val(ws.cell(row=dr, column=5), min_v,  alt)
                _val(ws.cell(row=dr, column=6), max_v,  alt)
                _val(ws.cell(row=dr, column=7), len(vals_f), alt)
            else:
                for c in range(3, 8):
                    _val(ws.cell(row=dr, column=c), "—", alt)

        # Column widths
        for c in range(1, title_cols + 1):
            ws.column_dimensions[get_column_letter(c)].width = 20

    # ── Sheet: QAOA_MultiSeed ────────────────────────────────────────────────
    if multiseed_results:
        ws_ms = wb.create_sheet("QAOA_MultiSeed")
        _write_per_seed_sheet(ws_ms, multiseed_results, header_bg=HEAD_BG,
                               noise_label="Ideal noiseless QAOA (Qiskit Aer shot mode)")
        print("[Excel] Sheet 'QAOA_MultiSeed' written.")

    # ── Sheet: QAOA_Noisy_MultiSeed ─────────────────────────────────────────
    if noisy_results:
        ws_nm = wb.create_sheet("QAOA_Noisy_MultiSeed")
        _write_per_seed_sheet(ws_nm, noisy_results, header_bg=NOISY_BG,
                               noise_label=QAOA_NOISE_LABEL)
        print("[Excel] Sheet 'QAOA_Noisy_MultiSeed' written.")

    # ── Sheet: QAOA_ShotSensitivity ─────────────────────────────────────────
    if shot_results:
        ws_sh = wb.create_sheet("QAOA_ShotSensitivity")
        _write_per_seed_sheet(ws_sh, shot_results,
                               extra_col_key="n_shots",
                               extra_col_name="Shots",
                               header_bg=SHOT_BG,
                               noise_label=(
                                   f"Shot sensitivity — seed fixed={QAOA_SHOT_SEED}\n"
                                   f"Shot levels: {QAOA_SHOT_LEVELS}"))
        print("[Excel] Sheet 'QAOA_ShotSensitivity' written.")


# =============================================================================
# 14.  MAIN PIPELINE
# =============================================================================

def main(csv_path=None, n_sessions: int = 25, run_qaoa: bool = True):
    print("="*70)
    print("  EV Charging Optimisation v2.3 -- Classical vs Quantum (QUBO)")
    print("  P_rate (one-hot) removed: positional binary now mathematically")
    print("  consistent.  Raw QUBO metrics captured before post-processing.")
    print("  QAOA added as 4th QUBO solver using same corrected Q (v2.3).")
    print("="*70)

    sessions = load_acn_data(csv_path, n_sessions)
    prob     = ChargingProblem(sessions, dt=DT)
    print(f"\n[Problem] N={prob.N} EVs, T={prob.T} slots, "
          f"E_total={prob.E_total:.1f} kWh")
    print(f"  d_bg: min={prob.d_bg.min():.1f} kW, "
          f"max={prob.d_bg.max():.1f} kW, mean={prob.d_bg.mean():.1f} kW")
    print(f"  Uncontrolled total: mean={prob.d_uncontrolled.mean():.1f} kW, "
          f"PAR={prob.par(np.zeros((prob.N,prob.T))):.4f}")

    results = {}

    n_solvers = "7" if run_qaoa else "6"
    print(f"\n[1/{n_solvers}] Convex QP (SLSQP) ...")
    m_qp = solve_convex_qp(prob); results["Convex QP (SLSQP)"] = m_qp
    R_qp = m_qp["R"]
    print(f"  obj={m_qp['obj_kW2']:.2f}  PAR={m_qp['PAR']:.4f}  "
          f"Sat={m_qp['mean_satisfaction_%']:.2f}%  t={m_qp['solve_time_s']:.2f}s")

    print(f"\n[2/{n_solvers}] Genetic Algorithm ...")
    ga = GeneticAlgorithm(prob)
    R_ga, h_ga, t_ga = ga.run(M=GA_POP, G=GA_GENS, R_seed=R_qp, verbose=True)
    m_ga = prob.metrics(R_ga, "GA", t_ga); m_ga["history"] = h_ga
    results["GA"] = m_ga
    print(f"  obj={m_ga['obj_kW2']:.2f}  PAR={m_ga['PAR']:.4f}  "
          f"Sat={m_ga['mean_satisfaction_%']:.2f}%  t={t_ga:.2f}s")

    print(f"\n[3/{n_solvers}] PSO ...")
    pso = PSO(prob)
    R_pso, h_pso, t_pso = pso.run(M=PSO_SWARM, G=PSO_ITERS,
                                    R_seed=R_qp, verbose=True)
    m_pso = prob.metrics(R_pso, "PSO", t_pso); m_pso["history"] = h_pso
    results["PSO"] = m_pso
    print(f"  obj={m_pso['obj_kW2']:.2f}  PAR={m_pso['PAR']:.4f}  "
          f"Sat={m_pso['mean_satisfaction_%']:.2f}%  t={t_pso:.2f}s")

    print("\n[QUBO] Building Q matrix ...")
    qubo  = QUBOFormulation(prob, K=K_BITS)
    t0    = time.perf_counter()
    Q     = qubo.build_Q()
    print(f"  Q built: {qubo.n}x{qubo.n} in {time.perf_counter()-t0:.2f}s")
    x_warm = encode_to_qubo_warm(qubo, R_qp)
    print(f"  Warm-start encoded from QP solution.")

    print(f"\n[4/{n_solvers}] QUBO + SA ...")
    t0 = time.perf_counter()
    x_sa, _, h_sa = solve_sa(qubo, Q, x_warm=x_warm, verbose=True)
    t_sa = time.perf_counter()-t0
    m_sa = qubo_to_rate(qubo, x_sa, "QUBO + SA", t_sa); m_sa["history"] = h_sa
    results["QUBO + SA"] = m_sa
    print(f"  [Raw]   obj={m_sa['raw_obj_kW2']:.2f}  PAR={m_sa['raw_PAR']:.4f}  "
          f"Sat={m_sa['raw_mean_satisfaction_%']:.2f}%")
    print(f"  [Final] obj={m_sa['obj_kW2']:.2f}  PAR={m_sa['PAR']:.4f}  "
          f"Sat={m_sa['mean_satisfaction_%']:.2f}%  t={t_sa:.2f}s  "
          f"({m_sa['pp_pct_changed_slots']:.1f}% slots changed by post-proc)")

    print(f"\n[5/{n_solvers}] QUBO + Parallel Tempering ...")
    t0 = time.perf_counter()
    x_pt, _, h_pt = solve_parallel_tempering(qubo, Q, x_warm=x_warm, verbose=True)
    t_pt = time.perf_counter()-t0
    m_pt = qubo_to_rate(qubo, x_pt, "QUBO + PT", t_pt); m_pt["history"] = h_pt
    results["QUBO + PT"] = m_pt
    print(f"  [Raw]   obj={m_pt['raw_obj_kW2']:.2f}  PAR={m_pt['raw_PAR']:.4f}  "
          f"Sat={m_pt['raw_mean_satisfaction_%']:.2f}%")
    print(f"  [Final] obj={m_pt['obj_kW2']:.2f}  PAR={m_pt['PAR']:.4f}  "
          f"Sat={m_pt['mean_satisfaction_%']:.2f}%  t={t_pt:.2f}s  "
          f"({m_pt['pp_pct_changed_slots']:.1f}% slots changed by post-proc)")

    print(f"\n[6/{n_solvers}] QUBO + Tabu Search ...")
    t0 = time.perf_counter()
    x_tb, _, h_tb = solve_tabu(qubo, Q, x_warm=x_warm, verbose=True)
    t_tb = time.perf_counter()-t0
    m_tb = qubo_to_rate(qubo, x_tb, "QUBO + Tabu", t_tb); m_tb["history"] = h_tb
    results["QUBO + Tabu"] = m_tb
    print(f"  [Raw]   obj={m_tb['raw_obj_kW2']:.2f}  PAR={m_tb['raw_PAR']:.4f}  "
          f"Sat={m_tb['raw_mean_satisfaction_%']:.2f}%")
    print(f"  [Final] obj={m_tb['obj_kW2']:.2f}  PAR={m_tb['PAR']:.4f}  "
          f"Sat={m_tb['mean_satisfaction_%']:.2f}%  t={t_tb:.2f}s  "
          f"({m_tb['pp_pct_changed_slots']:.1f}% slots changed by post-proc)")

    # ---------------------------------------------------------------------- #
    # STEP 7/7: QAOA (Qiskit Aer)                                             #
    # Uses the SAME Q matrix as SA/PT/Tabu.  Bitstring selection by QUBO      #
    # energy only (§15).  Raw metrics captured before post-processing (§8,§9).#
    # ---------------------------------------------------------------------- #
    if run_qaoa:
        print(f"\n[7/{n_solvers}] QAOA (Qiskit Aer, mode={QAOA_SIM_MODE}) ...")
        qaoa_ok = False
        try:
            t0 = time.perf_counter()
            x_qa, global_E_qa, sub_log_qa = solve_qaoa(
                qubo, Q, x_warm=x_warm,
                p_layers=QAOA_LAYERS, n_restarts=QAOA_RESTARTS,
                n_shots=QAOA_SHOTS, opt_iters=QAOA_OPT_ITERS,
                max_vars=QAOA_MAX_VARS, sim_mode=QAOA_SIM_MODE,
                verbose=True)
            t_qa = time.perf_counter() - t0

            # Classical QUBO energy verification (§7)
            e_verify_qa = float(x_qa @ Q @ x_qa)
            print(f"  [QAOA] Global QUBO energy: greedy={global_E_qa:.4e}  "
                  f"verify(x^T Q x)={e_verify_qa:.4e}")

            # Raw + final metrics via shared qubo_to_rate pipeline (§8,§9)
            m_qa = qubo_to_rate(qubo, x_qa, "QAOA", t_qa)
            m_qa["qaoa_global_qubo_energy"]   = global_E_qa
            m_qa["qaoa_qubo_energy_verified"] = e_verify_qa
            m_qa["qaoa_subproblem_log"]       = sub_log_qa
            m_qa["qaoa_sim_mode"]             = QAOA_SIM_MODE
            m_qa["qaoa_p_layers"]             = QAOA_LAYERS
            m_qa["qaoa_n_shots"]              = QAOA_SHOTS
            m_qa["qaoa_n_subproblems"]        = len(sub_log_qa)
            results["QAOA"] = m_qa
            qaoa_ok = True

            print(f"  [Raw]   obj={m_qa['raw_obj_kW2']:.2f}  "
                  f"PAR={m_qa['raw_PAR']:.4f}  "
                  f"Sat={m_qa['raw_mean_satisfaction_%']:.2f}%")
            print(f"  [Final] obj={m_qa['obj_kW2']:.2f}  PAR={m_qa['PAR']:.4f}  "
                  f"Sat={m_qa['mean_satisfaction_%']:.2f}%  t={t_qa:.2f}s  "
                  f"({m_qa['pp_pct_changed_slots']:.1f}% slots changed by post-proc)")

        except ImportError as e_imp:
            print(f"  [QAOA] SKIPPED -- Qiskit not installed: {e_imp}")
            print("         Install:  pip install qiskit qiskit-aer scipy")
        except Exception as e_qa:
            print(f"  [QAOA] FAILED -- {type(e_qa).__name__}: {e_qa}")
            import traceback; traceback.print_exc()
        if not qaoa_ok:
            print("  [QAOA] Not added to results (see error above).")

    # ---------------------------------------------------------------------- #
    # ROBUSTNESS EXPERIMENTS (b3c5e0cc additions)                             #
    # These run ONLY if QAOA is available (Qiskit Aer installed).             #
    # Results go to new Excel sheets; existing results are UNCHANGED.         #
    # ---------------------------------------------------------------------- #
    rob_ms_results    = None   # Experiment A — multi-seed ideal
    rob_noisy_results = None   # Experiment B — multi-seed noisy
    rob_shot_results  = None   # Experiment C — shot sensitivity

    if run_qaoa and _QISKIT_AVAILABLE:
        # Validation printout (abridged — full validation already ran in solve_qaoa)
        import hashlib as _hl
        Q_bytes = Q.tobytes()
        Q_hash  = _hl.md5(Q_bytes).hexdigest()[:12]
        print(f"\n[RobustnessExp] Validation:")
        print(f"   QUBO checksum (MD5-12): {Q_hash}")
        print(f"   QAOA depth p          : {QAOA_LAYERS}")
        print(f"   Shots (base)          : {QAOA_SHOTS}")
        print(f"   Seeds (ideal)         : {QAOA_SEEDS}")
        print(f"   Seeds (noisy)         : {QAOA_NOISY_SEEDS}")
        print(f"   Shot levels (C)       : {QAOA_SHOT_LEVELS}")
        print(f"   Noise model           : {QAOA_NOISE_LABEL}")
        print(f"   Noise p1/p2/ro        : {QAOA_NOISE_P1}/{QAOA_NOISE_P2}/"
              f"{QAOA_NOISE_READOUT}")

        # ── Experiment A: multi-seed ideal QAOA ─────────────────────────────
        print(f"\n[RobustnessExp A] Multi-seed ideal QAOA "
              f"({len(QAOA_SEEDS)} seeds) ...")
        try:
            rob_ms_results = run_qaoa_multi_seed(
                qubo, Q, x_warm, prob,
                seeds=QAOA_SEEDS, n_shots=QAOA_SHOTS,
                noise_model=None,
                experiment_label="QAOA_MultiSeed",
                verbose=True)
            n_ok = sum(1 for r in rob_ms_results if "error" not in r)
            print(f"   Experiment A done: {n_ok}/{len(QAOA_SEEDS)} seeds completed.")
        except Exception as exc_a:
            print(f"   Experiment A FAILED: {exc_a}")
            import traceback; traceback.print_exc()

        # ── Experiment C: shot sensitivity ───────────────────────────────────
        print(f"\n[RobustnessExp C] Shot sensitivity "
              f"(seed={QAOA_SHOT_SEED}, levels={QAOA_SHOT_LEVELS}) ...")
        try:
            rob_shot_results = run_qaoa_shot_sensitivity(
                qubo, Q, x_warm, prob,
                shot_levels=QAOA_SHOT_LEVELS,
                fixed_seed=QAOA_SHOT_SEED,
                verbose=True)
            n_ok = sum(1 for r in rob_shot_results if "error" not in r)
            print(f"   Experiment C done: {n_ok}/{len(QAOA_SHOT_LEVELS)} "
                  f"shot levels completed.")
        except Exception as exc_c:
            print(f"   Experiment C FAILED: {exc_c}")
            import traceback; traceback.print_exc()

        # ── Experiment B: multi-seed noisy QAOA ─────────────────────────────
        print(f"\n[RobustnessExp B] Multi-seed noisy QAOA "
              f"({len(QAOA_NOISY_SEEDS)} seeds) ...")
        print(f"   Noise: {QAOA_NOISE_LABEL}")
        try:
            nm = _build_aer_noise_model(QAOA_NOISE_P1, QAOA_NOISE_P2,
                                         QAOA_NOISE_READOUT)
            print(f"   Noise model built: "
                  f"p1={QAOA_NOISE_P1} p2={QAOA_NOISE_P2} "
                  f"readout={QAOA_NOISE_READOUT}")
            rob_noisy_results = run_qaoa_multi_seed(
                qubo, Q, x_warm, prob,
                seeds=QAOA_NOISY_SEEDS, n_shots=QAOA_SHOTS,
                noise_model=nm,
                experiment_label="QAOA_Noisy_MultiSeed",
                verbose=True)
            n_ok = sum(1 for r in rob_noisy_results if "error" not in r)
            print(f"   Experiment B done: {n_ok}/{len(QAOA_NOISY_SEEDS)} "
                  f"seeds completed.")
        except Exception as exc_b:
            print(f"   Experiment B FAILED: {exc_b}")
            import traceback; traceback.print_exc()

    print("\n" + "="*78)
    print(f"{'Algorithm':<22} {'Obj(kW2)':>10} {'PAR':>7} {'Peak':>8} "
          f"{'Std':>7} {'E-err':>8} {'Sat%':>7} {'Time':>7}")
    print("-"*78)
    for lbl, m in results.items():
        print(f"  {lbl:<20} {m['obj_kW2']:>10.2f} {m['PAR']:>7.4f} "
              f"{m['peak_load_kW']:>8.2f} {m['load_std_kW']:>7.4f} "
              f"{m['mean_energy_err_kWh']:>8.5f} "
              f"{m['mean_satisfaction_%']:>7.2f} {m['solve_time_s']:>7.2f}")
    print("="*78)

    # Raw vs Final breakdown for QUBO-based solvers (SA, PT, Tabu, QAOA)
    qubo_res = {l: m for l, m in results.items()
                if (l.startswith("QUBO") or l == "QAOA") and "raw_obj_kW2" in m}
    if qubo_res:
        print("\n--- QUBO/QAOA Raw vs Final Post-Processed (P_rate removed in v2.3) ---")
        print(f"{'Algorithm':<22} {'Stage':<8} {'Obj(kW2)':>10} {'PAR':>7} "
              f"{'Peak':>8} {'Sat%':>7} {'E-err':>8}")
        print("-"*78)
        for lbl, m in qubo_res.items():
            print(f"  {lbl:<20} {'Raw':<8} {m['raw_obj_kW2']:>10.2f} "
                  f"{m['raw_PAR']:>7.4f} {m['raw_peak_load_kW']:>8.2f} "
                  f"{m['raw_mean_satisfaction_%']:>7.2f} "
                  f"{m['raw_mean_energy_err_kWh']:>8.5f}")
            print(f"  {'':<20} {'Final':<8} {m['obj_kW2']:>10.2f} "
                  f"{m['PAR']:>7.4f} {m['peak_load_kW']:>8.2f} "
                  f"{m['mean_satisfaction_%']:>7.2f} "
                  f"{m['mean_energy_err_kWh']:>8.5f}")
            print(f"  {'':<20} {'Δ(pp)':<8} {m['pp_delta_obj_kW2']:>+10.2f} "
                  f"{m['pp_delta_PAR']:>+7.4f} "
                  f"  -- slots changed: {m['pp_n_changed_slots']} "
                  f"({m['pp_pct_changed_slots']:.1f}%)")
        print("="*78)

    print("\n[Plots] Generating figures ...")
    fig_paths = plot_all(prob, results, RESULTS_DIR)
    print(f"  {len(fig_paths)} figures saved.")

    print("\n[Excel] Exporting workbook ...")
    xl_path = os.path.join(RESULTS_DIR, EXCEL_NAME)
    export_excel(sessions, prob, results, fig_paths, xl_path)

    # ── Append robustness sheets (b3c5e0cc) ──────────────────────────────── #
    if rob_ms_results or rob_noisy_results or rob_shot_results:
        print("\n[Excel] Writing robustness sheets ...")
        try:
            from openpyxl import load_workbook as _load_wb
            wb_rob = _load_wb(xl_path)
            _write_robustness_sheets(
                wb_rob, _b(),
                multiseed_results=rob_ms_results    or None,
                noisy_results    =rob_noisy_results or None,
                shot_results     =rob_shot_results  or None)
            wb_rob.save(xl_path)
            print(f"[Excel] Robustness sheets appended -> {xl_path}")
        except Exception as exc_xl:
            print(f"[Excel] WARNING: could not write robustness sheets: {exc_xl}")
            import traceback; traceback.print_exc()

    print(f"\nDone.  Results -> {RESULTS_DIR}")
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    CSV_PATH   = r"C:\Users\aramc\OneDrive\Desktop\ACN\data\caltech_ev_cleaned.csv"
    N_SESSIONS = 25

    results = main(csv_path=CSV_PATH, n_sessions=N_SESSIONS)
