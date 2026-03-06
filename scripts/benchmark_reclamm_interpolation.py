"""Benchmark reClAMM range shift interpolation: current vs optimal midpoint.

Compares total arb loss during a range shift under different interpolation methods:
  Geometric VB  -- exponential decay of overvalued virtual (what contracts do)
  Linear VB     -- uniform steps in VB
  Linear Z      -- uniform steps in Z = sqrt(P)*VA - VB/sqrt(P)  (optimal, from note)
  Optimal 2-step -- exact midpoint via quadratic formula (Section 5 of note)
  Brute-force optimal -- JAX gradient-optimised Z-target sequence

Key result: per-step loss ~ (DeltaZ)^2 / (4X).  Equal Z-increments minimise
total loss, analogous to TFMM optimal intermediate for G3M weight changes.

Usage:
    cd <repo-root>
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-reclamm
    python scripts/benchmark_reclamm_interpolation.py
"""

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize

jax.config.update("jax_enable_x64", True)


# ── Core reClAMM mechanics ─────────────────────────────────────────────────


def compute_VA_from_VB(RA, RB, VB, Q):
    """Contract rule (eq 15): VA = RA*(VB + RB) / ((Q-1)*VB - RB)."""
    return RA * (VB + RB) / ((Q - 1) * VB - RB)


def compute_Z(VA, VB, P):
    """Z = sqrt(P)*VA - VB/sqrt(P)  (eq 12)."""
    sqP = np.sqrt(P)
    return sqP * VA - VB / sqP


def pool_value(RA, RB, P):
    """Real pool value: P*RA + RB  (eq 3)."""
    return P * RA + RB


def micro_step(RA, RB, VA_new, VB_new, P):
    """Virtual-balance update then arb to equilibrium Y/X = P.

    Returns (RA_new, RB_new, arb_loss).
    """
    val_before = pool_value(RA, RB, P)
    X = RA + VA_new
    Y = RB + VB_new
    L = X * Y
    X_eq = np.sqrt(L / P)
    Y_eq = P * X_eq
    RA_new = X_eq - VA_new
    RB_new = Y_eq - VB_new
    return RA_new, RB_new, val_before - pool_value(RA_new, RB_new, P)


def solve_VB_for_Z(RA, RB, Z_star, Q, P):
    """Solve quadratic for VB achieving Z(VB) = Z_star.

    Derived by substituting VA = RA*(VB+RB)/((Q-1)*VB-RB) into
    Z = sqrt(P)*VA - VB/sqrt(P), then collecting terms in VB.

    NOTE: The research note (eq 28) has a sign error: the RB/sqrt(P)
    term in b should be positive, not negative.  Re-derived here from
    scratch.

    Returns the physically valid root (VB > RB/(Q-1), positive).
    Raises ValueError if no valid root exists.
    """
    sqP = np.sqrt(P)
    a = -(Q - 1) / sqP
    b = sqP * RA + RB / sqP - (Q - 1) * Z_star  # +RB/sqP, not minus
    c = sqP * RA * RB + Z_star * RB
    disc = b * b - 4 * a * c
    if disc < -1e-6:
        raise ValueError(f"negative discriminant: {disc:.4e}")
    disc = max(disc, 0.0)
    sd = np.sqrt(disc)
    r1, r2 = (-b + sd) / (2 * a), (-b - sd) / (2 * a)
    floor = RB / (Q - 1) + 1e-12
    ok = [r for r in (r1, r2) if r > floor]
    if not ok:
        raise ValueError(f"no valid root: r1={r1:.4f}, r2={r2:.4f}, floor={floor:.4f}")
    return min(ok)


# ── Interpolation methods ──────────────────────────────────────────────────


def run_shift(RA, RB, VA_stale, VB_start, VB_end, Q, P, N, schedule):
    """Execute N-step range shift (B overvalued, VB decreasing).

    schedule: "geometric" | "linear_VB" | "linear_Z"

    VA_stale: the current (possibly stale) VA -- used only for Z_start
    in the linear_Z schedule.  All micro-steps compute VA from the
    contract rule with current reserves.
    """
    # For linear_Z, precompute Z endpoints using contract-rule VA
    if schedule == "linear_Z":
        VA_start_cr = compute_VA_from_VB(RA, RB, VB_start, Q)
        Z0 = compute_Z(VA_start_cr, VB_start, P)
        VA_end_approx = compute_VA_from_VB(RA, RB, VB_end, Q)
        Z_end = compute_Z(VA_end_approx, VB_end, P)

    total_loss = 0.0
    RA_c, RB_c = RA, RB

    for i in range(1, N + 1):
        frac = i / N
        if schedule == "geometric":
            VB_i = VB_start * (VB_end / VB_start) ** frac
        elif schedule == "linear_VB":
            VB_i = VB_start + frac * (VB_end - VB_start)
        elif schedule == "linear_Z":
            Z_i = Z0 + frac * (Z_end - Z0)
            VB_i = solve_VB_for_Z(RA_c, RB_c, Z_i, Q, P)
        else:
            raise ValueError(schedule)

        VA_i = compute_VA_from_VB(RA_c, RB_c, VB_i, Q)
        RA_c, RB_c, loss = micro_step(RA_c, RB_c, VA_i, VB_i, P)
        total_loss += loss

    return total_loss, RA_c, RB_c


def run_shift_optimal_2step(RA, RB, VA_stale, VB_start, VB_end, Q, P):
    """Exact 2-step optimal midpoint (Section 5 of the note).

    Computes Z* = (Z_start + Z_end) / 2, solves quadratic for VB_mid.
    """
    VA_start_cr = compute_VA_from_VB(RA, RB, VB_start, Q)
    Z0 = compute_Z(VA_start_cr, VB_start, P)
    VA_end_approx = compute_VA_from_VB(RA, RB, VB_end, Q)
    Z2 = compute_Z(VA_end_approx, VB_end, P)
    Z_star = (Z0 + Z2) / 2.0

    # Step 1: jump to Z-midpoint
    VB_mid = solve_VB_for_Z(RA, RB, Z_star, Q, P)
    VA_mid = compute_VA_from_VB(RA, RB, VB_mid, Q)
    RA1, RB1, loss1 = micro_step(RA, RB, VA_mid, VB_mid, P)

    # Step 2: jump to endpoint
    VA_end = compute_VA_from_VB(RA1, RB1, VB_end, Q)
    RA2, RB2, loss2 = micro_step(RA1, RB1, VA_end, VB_end, P)

    return loss1 + loss2, RA2, RB2


# ── Scenario setup ─────────────────────────────────────────────────────────


def setup_centered_pool(P, price_ratio, R_scale=10000.0):
    """Centered pool at price P with contract-rule-consistent virtuals.

    Returns (RA, RB, VA, VB, Q).
    """
    Q = np.sqrt(price_ratio)
    q4 = price_ratio ** 0.25

    RA = R_scale
    RB = P * R_scale
    VA = RA / (q4 - 1)
    VB = RB / (q4 - 1)

    return RA, RB, VA, VB, Q


def setup_decentered_pool(P_init, P_final, price_ratio, R_scale=10000.0):
    """Centered pool at P_init, arb to P_final, then refresh virtuals.

    The refresh applies the contract rule to get consistent (VA, VB) at
    the post-arb reserves, then arbs once more. This gives a decentered
    but fully consistent state (equilibrium + contract rule).

    Returns (RA, RB, VA, VB, Q).
    """
    Q = np.sqrt(price_ratio)
    q4 = price_ratio ** 0.25

    RA0 = R_scale
    RB0 = P_init * R_scale
    VA0 = RA0 / (q4 - 1)
    VB0 = RB0 / (q4 - 1)

    # Arb to P_final (L preserved, virtuals stale)
    X0 = RA0 + VA0
    Y0 = RB0 + VB0
    L = X0 * Y0
    X_new = np.sqrt(L / P_final)
    Y_new = np.sqrt(L * P_final)
    RA = X_new - VA0
    RB = Y_new - VB0

    # Refresh: apply contract rule for current VB, then arb
    VB = VB0
    VA = compute_VA_from_VB(RA, RB, VB, Q)
    RA, RB, _ = micro_step(RA, RB, VA, VB, P_final)

    return RA, RB, VA, VB, Q


# ── JAX-differentiable versions for brute-force optimisation ──────────────


def _compute_VA_from_VB_jax(RA, RB, VB, Q):
    return RA * (VB + RB) / ((Q - 1) * VB - RB)


def _compute_Z_jax(VA, VB, P):
    sqP = jnp.sqrt(P)
    return sqP * VA - VB / sqP


def _pool_value_jax(RA, RB, P):
    return P * RA + RB


def _micro_step_jax(RA, RB, VA, VB, P):
    val_before = _pool_value_jax(RA, RB, P)
    X = RA + VA
    Y = RB + VB
    L = X * Y
    X_eq = jnp.sqrt(L / P)
    Y_eq = P * X_eq
    RA_new = X_eq - VA
    RB_new = Y_eq - VB
    return RA_new, RB_new, val_before - _pool_value_jax(RA_new, RB_new, P)


def _solve_VB_for_Z_jax(RA, RB, Z_star, Q, P):
    sqP = jnp.sqrt(P)
    a = -(Q - 1) / sqP
    b = sqP * RA + RB / sqP - (Q - 1) * Z_star
    c = sqP * RA * RB + Z_star * RB
    disc = jnp.maximum(b * b - 4 * a * c, 1e-30)
    sd = jnp.sqrt(disc)
    r1 = (-b + sd) / (2 * a)
    r2 = (-b - sd) / (2 * a)
    floor = RB / (Q - 1) + 1e-8
    return jnp.where(r2 > floor, r2, r1)


def _z_targets_from_raw(raw_params, Z_start, Z_end):
    """Map unconstrained params -> sorted Z targets via softplus gaps."""
    gaps = jax.nn.softplus(raw_params)
    gaps = gaps / jnp.sum(gaps) * (Z_end - Z_start)
    return Z_start + jnp.cumsum(gaps)


def _make_loss_fn(N):
    """Build a JIT-compiled loss function for a given N (unrolled loop)."""

    def total_loss(raw_params, RA, RB, Q, P, Z_start, Z_end):
        Z_all = _z_targets_from_raw(raw_params, Z_start, Z_end)
        RA_c, RB_c = RA, RB
        total = 0.0
        for i in range(N):
            VB_i = _solve_VB_for_Z_jax(RA_c, RB_c, Z_all[i], Q, P)
            VA_i = _compute_VA_from_VB_jax(RA_c, RB_c, VB_i, Q)
            RA_c, RB_c, loss = _micro_step_jax(RA_c, RB_c, VA_i, VB_i, P)
            total = total + loss
        return total

    return jax.jit(jax.value_and_grad(total_loss))


def optimise_z_targets(RA, RB, Q, P, Z_start, Z_end, N, verbose=False):
    """Find the Z-target sequence minimising total arb loss.

    Returns (optimal_loss, optimal_Z_targets_array_of_length_N).
    """
    loss_and_grad_fn = _make_loss_fn(N)
    RA_j = jnp.float64(RA)
    RB_j = jnp.float64(RB)
    Q_j = jnp.float64(Q)
    P_j = jnp.float64(P)
    Zs_j = jnp.float64(Z_start)
    Ze_j = jnp.float64(Z_end)

    def objective(x):
        val, grad = loss_and_grad_fn(
            jnp.array(x, dtype=jnp.float64), RA_j, RB_j, Q_j, P_j, Zs_j, Ze_j
        )
        return float(val), np.array(grad, dtype=np.float64)

    x0 = np.zeros(N)  # softplus(0) = ln2, uniform gaps → linear Z init
    result = scipy_minimize(objective, x0, jac=True, method="L-BFGS-B")

    optimal_Z = np.array(
        _z_targets_from_raw(jnp.array(result.x), Zs_j, Ze_j)
    )
    if verbose:
        print(f"    N={N}: loss={result.fun:.6f}  "
              f"nit={result.nit}  success={result.success}")
    return result.fun, optimal_Z


# ── Experiments ────────────────────────────────────────────────────────────


def main():
    # --- Scenario: centered pool, moderate VB decay ---
    P = 2.0               # token A costs 2 units of token B
    price_ratio = 4.0     # rho, so Q = sqrt(4) = 2
    R_scale = 10000.0
    decay_fraction = 0.90  # VB_end = 0.90 * VB_start (10% decay)

    RA, RB, VA, VB, Q = setup_centered_pool(P, price_ratio, R_scale)
    VB_start = VB
    VB_end = VB * decay_fraction

    # Diagnostics
    C = min(RA * VB, RB * VA) / max(RA * VB, RB * VA)
    is_above = RA * VB > RB * VA
    X = RA + VA
    print("=" * 72)
    print(f"Scenario: centered pool at P={P}, price_ratio={price_ratio}, Q={Q:.4f}")
    print(f"  RA={RA:.2f}  RB={RB:.2f}  VA={VA:.2f}  VB={VB:.2f}")
    print(f"  Effective X={X:.2f}  Pool value = {pool_value(RA, RB, P):.2f}")
    print(f"  Centeredness = {C:.4f}   is_above = {is_above}")
    print(f"  VB shift: {VB_start:.2f} -> {VB_end:.2f} ({decay_fraction:.0%})")
    VB_floor = RB / (Q - 1)
    print(f"  VB floor (denominator > 0): {VB_floor:.2f}")
    Z_start = compute_Z(VA, VB, P)
    VA_end_cr = compute_VA_from_VB(RA, RB, VB_end, Q)
    Z_end = compute_Z(VA_end_cr, VB_end, P)
    print(f"  Z_start = {Z_start:.4f}  Z_end = {Z_end:.4f}")
    print(f"  Approx 1-step loss ~ (DeltaZ)^2/(4X) = {(Z_end-Z_start)**2/(4*X):.2f}")
    print("=" * 72)

    # ── Experiment 1: Loss vs N ────────────────────────────────────────

    N_values = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    schedules = ["geometric", "linear_VB", "linear_Z"]
    results = {s: [] for s in schedules}

    for N in N_values:
        for sched in schedules:
            try:
                loss, _, _ = run_shift(
                    RA, RB, VA, VB_start, VB_end, Q, P, N, sched
                )
            except (ValueError, AssertionError) as e:
                loss = np.nan
            results[sched].append(loss)

    # Optimal 2-step (single point)
    try:
        loss_opt2, _, _ = run_shift_optimal_2step(
            RA, RB, VA, VB_start, VB_end, Q, P
        )
    except (ValueError, AssertionError):
        loss_opt2 = np.nan

    # Table
    loss_1 = results["geometric"][0]
    print(f"\n{'N':>5s}  {'Geo VB':>12s}  {'Lin VB':>12s}  {'Lin Z':>12s}"
          f"  {'Geo/1step':>9s}  {'LinZ/1step':>10s}  {'LinZ/Geo':>9s}")
    print("-" * 80)
    for j, N in enumerate(N_values):
        g = results["geometric"][j]
        lv = results["linear_VB"][j]
        lz = results["linear_Z"][j]
        print(f"{N:>5d}  {g:>12.6f}  {lv:>12.6f}  {lz:>12.6f}"
              f"  {g / loss_1:>9.4f}  {lz / loss_1:>10.4f}  {lz / g:>9.4f}")

    print(f"\n  Optimal 2-step loss:  {loss_opt2:.6f}")
    print(f"  Geometric N=2 loss:  {results['geometric'][1]:.6f}"
          f"  (opt/geo = {loss_opt2 / results['geometric'][1]:.4f})")
    print(f"  Linear Z  N=2 loss:  {results['linear_Z'][1]:.6f}"
          f"  (opt/linZ = {loss_opt2 / results['linear_Z'][1]:.4f})")

    # ── Experiment 2: Z and VB trajectories at N=8 ─────────────────────

    N_viz = 8
    traj_data = {}
    for sched in schedules:
        VB_traj, Z_traj, loss_traj = [VB_start], [], []
        VA_s = VA  # stale
        Z_traj.append(compute_Z(VA_s, VB_start, P))

        RA_c, RB_c = RA, RB
        if sched == "linear_Z":
            Z0 = Z_traj[0]
            VA_end_a = compute_VA_from_VB(RA, RB, VB_end, Q)
            Z_end_val = compute_Z(VA_end_a, VB_end, P)

        for i in range(1, N_viz + 1):
            frac = i / N_viz
            if sched == "geometric":
                VB_i = VB_start * (VB_end / VB_start) ** frac
            elif sched == "linear_VB":
                VB_i = VB_start + frac * (VB_end - VB_start)
            else:
                Z_i = Z0 + frac * (Z_end_val - Z0)
                VB_i = solve_VB_for_Z(RA_c, RB_c, Z_i, Q, P)

            try:
                VA_i = compute_VA_from_VB(RA_c, RB_c, VB_i, Q)
                VB_traj.append(VB_i)
                Z_traj.append(compute_Z(VA_i, VB_i, P))
                RA_c, RB_c, loss = micro_step(RA_c, RB_c, VA_i, VB_i, P)
                loss_traj.append(loss)
            except (ValueError, AssertionError):
                break

        traj_data[sched] = {
            "VB": np.array(VB_traj),
            "Z": np.array(Z_traj),
            "loss": np.array(loss_traj),
        }

    # ── Experiment 3: sweep shift size at N=2 ──────────────────────────

    decay_sweep = np.linspace(0.80, 0.99, 30)
    sweep = {s: [] for s in ["geometric", "linear_Z", "optimal_2step"]}
    for df in decay_sweep:
        VB_e = VB * df
        try:
            g, _, _ = run_shift(RA, RB, VA, VB_start, VB_e, Q, P, 2, "geometric")
            lz, _, _ = run_shift(RA, RB, VA, VB_start, VB_e, Q, P, 2, "linear_Z")
            o2, _, _ = run_shift_optimal_2step(RA, RB, VA, VB_start, VB_e, Q, P)
        except (AssertionError, ValueError):
            g = lz = o2 = np.nan
        sweep["geometric"].append(g)
        sweep["linear_Z"].append(lz)
        sweep["optimal_2step"].append(o2)

    # ── Plots ──────────────────────────────────────────────────────────

    colours = {"geometric": "C0", "linear_VB": "C1", "linear_Z": "C2"}
    labels = {
        "geometric": "Geometric VB (contract)",
        "linear_VB": "Linear VB",
        "linear_Z": "Linear Z (optimal)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) Loss vs N
    ax = axes[0, 0]
    for s in schedules:
        ax.plot(N_values, results[s], "o-", ms=4, color=colours[s], label=labels[s])
    ax.axhline(loss_opt2, color="C3", ls=":", label=f"Optimal 2-step = {loss_opt2:.4f}")
    ax.set_xlabel("Steps N")
    ax.set_ylabel("Total arb loss")
    ax.set_title("Arb loss vs interpolation steps")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,1) Ratio linear_Z / geometric
    ax = axes[0, 1]
    ratios = np.array(results["linear_Z"]) / np.array(results["geometric"])
    ax.plot(N_values, ratios, "o-", color="C2")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Steps N")
    ax.set_ylabel("Loss(Linear Z) / Loss(Geometric VB)")
    ax.set_title("Relative improvement of Z-optimal")
    ax.grid(True, alpha=0.3)

    # (1,0) Z trajectories at N=8
    ax = axes[1, 0]
    steps = np.arange(N_viz + 1)
    for s in schedules:
        ax.plot(steps, traj_data[s]["Z"], "o-", ms=4, color=colours[s], label=labels[s])
    ax.set_xlabel("Step")
    ax.set_ylabel("Z = sqrt(P)*VA - VB/sqrt(P)")
    ax.set_title(f"Z trajectory (N={N_viz})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) 2-step loss vs shift size
    ax = axes[1, 1]
    shift_pct = (1 - decay_sweep) * 100
    ax.plot(shift_pct, sweep["geometric"], color="C0", label="Geometric VB (N=2)")
    ax.plot(shift_pct, sweep["linear_Z"], color="C2", label="Linear Z (N=2)")
    ax.plot(shift_pct, sweep["optimal_2step"], ":", color="C3", label="Optimal 2-step")
    ax.set_xlabel("Shift size (% VB decay)")
    ax.set_ylabel("Arb loss")
    ax.set_title("2-step loss vs shift magnitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reclamm_interpolation_benchmark.png", dpi=150)
    print("\nSaved reclamm_interpolation_benchmark.png")

    # ── Per-step loss bar chart for N=8 ────────────────────────────────

    fig2, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, N_viz + 1)
    w = 0.25
    for i, s in enumerate(schedules):
        ax.bar(x + i * w, traj_data[s]["loss"], w, color=colours[s], label=labels[s])
    ax.set_xlabel("Step")
    ax.set_ylabel("Per-step arb loss")
    ax.set_title(f"Per-step loss distribution (N={N_viz})")
    ax.legend(fontsize=8)
    ax.set_xticks(x + w)
    plt.tight_layout()
    plt.savefig("reclamm_interpolation_perstep.png", dpi=150)
    print("Saved reclamm_interpolation_perstep.png")

    # ── Experiment 4: small-shift regime (paper's approximation valid) ───

    print("\n" + "=" * 72)
    print("Experiment 4: Optimal 2-step vs Geometric N=2 at small shifts")
    print("  (reserves nearly constant → paper's analysis should hold)")
    print("-" * 72)
    print(f"  {'Decay %':>8s}  {'Geo N=2':>12s}  {'LinZ N=2':>12s}  "
          f"{'Opt2':>12s}  {'Opt2/Geo':>9s}  {'Opt2/LinZ':>9s}")
    print("-" * 72)

    small_decays = [0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80]
    for df in small_decays:
        VB_e = VB * df
        try:
            g, _, _ = run_shift(RA, RB, VA, VB_start, VB_e, Q, P, 2, "geometric")
            lz, _, _ = run_shift(RA, RB, VA, VB_start, VB_e, Q, P, 2, "linear_Z")
            o2, _, _ = run_shift_optimal_2step(
                RA, RB, VA, VB_start, VB_e, Q, P
            )
        except (ValueError, AssertionError) as e:
            print(f"  {(1-df)*100:>7.1f}%  FAILED: {e}")
            continue
        print(f"  {(1-df)*100:>7.1f}%  {g:>12.6f}  {lz:>12.6f}  "
              f"{o2:>12.6f}  {o2/g:>9.6f}  {o2/lz:>9.6f}")

    print("=" * 72)

    # ── Experiment 5: brute-force JAX-optimised Z targets ────────────────

    print("\n" + "=" * 72)
    print("Experiment 5: Brute-force optimal Z targets (JAX + L-BFGS-B)")
    print("  Parameterisation: softplus gaps → sorted Z targets")
    print("  Initialised at linear Z (uniform gaps)")
    print("-" * 72)

    opt_N_values = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    opt_losses = {}
    opt_Z_trajs = {}

    for N in opt_N_values:
        loss_bf, Z_bf = optimise_z_targets(
            RA, RB, Q, P, Z_start, Z_end, N, verbose=True
        )
        opt_losses[N] = loss_bf
        opt_Z_trajs[N] = Z_bf

    # Comparison table
    print(f"\n  {'N':>5s}  {'Geometric':>12s}  {'Linear Z':>12s}  "
          f"{'BF Optimal':>12s}  {'BF/LinZ':>9s}  {'BF/Geo':>9s}")
    print("-" * 72)
    for N in opt_N_values:
        idx = N_values.index(N) if N in N_values else None
        g = results["geometric"][idx] if idx is not None else np.nan
        lz = results["linear_Z"][idx] if idx is not None else np.nan
        bf = opt_losses[N]
        print(f"  {N:>5d}  {g:>12.6f}  {lz:>12.6f}  "
              f"{bf:>12.6f}  {bf/lz:>9.6f}  {bf/g:>9.6f}")

    # ── Plot: overlay brute-force on the main loss-vs-N chart ────────────

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # (left) Loss vs N with brute-force overlay
    ax = axes3[0]
    for s in schedules:
        ax.plot(N_values, results[s], "o-", ms=4, color=colours[s],
                label=labels[s])
    bf_Ns = sorted(opt_losses.keys())
    bf_vals = [opt_losses[n] for n in bf_Ns]
    ax.plot(bf_Ns, bf_vals, "s--", ms=5, color="C3", label="BF Optimal (JAX)")
    ax.set_xlabel("Steps N")
    ax.set_ylabel("Total arb loss")
    ax.set_title("Arb loss vs interpolation steps (with BF optimal)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (right) Z trajectory comparison at N=8
    ax = axes3[1]
    N_cmp = 8
    steps_cmp = np.arange(N_cmp + 1)

    # Geometric: compute Z trajectory from VB
    z_geo = [Z_start]
    RA_t, RB_t = RA, RB
    for i in range(1, N_cmp + 1):
        frac = i / N_cmp
        VB_i = VB_start * (VB_end / VB_start) ** frac
        VA_i = compute_VA_from_VB(RA_t, RB_t, VB_i, Q)
        z_geo.append(compute_Z(VA_i, VB_i, P))
        RA_t, RB_t, _ = micro_step(RA_t, RB_t, VA_i, VB_i, P)

    # Linear Z
    z_linz = [Z_start]
    RA_t, RB_t = RA, RB
    for i in range(1, N_cmp + 1):
        frac = i / N_cmp
        Z_i = Z_start + frac * (Z_end - Z_start)
        VB_i = solve_VB_for_Z(RA_t, RB_t, Z_i, Q, P)
        VA_i = compute_VA_from_VB(RA_t, RB_t, VB_i, Q)
        z_linz.append(compute_Z(VA_i, VB_i, P))
        RA_t, RB_t, _ = micro_step(RA_t, RB_t, VA_i, VB_i, P)

    # BF optimal
    z_bf = [Z_start] + list(opt_Z_trajs[N_cmp])
    # Trace actual Z achieved after arb at each step
    z_bf_actual = [Z_start]
    RA_t, RB_t = RA, RB
    for i in range(N_cmp):
        VB_i = solve_VB_for_Z(RA_t, RB_t, opt_Z_trajs[N_cmp][i], Q, P)
        VA_i = compute_VA_from_VB(RA_t, RB_t, VB_i, Q)
        z_bf_actual.append(compute_Z(VA_i, VB_i, P))
        RA_t, RB_t, _ = micro_step(RA_t, RB_t, VA_i, VB_i, P)

    ax.plot(steps_cmp, z_geo, "o-", ms=4, color="C0", label="Geometric VB")
    ax.plot(steps_cmp, z_linz, "o-", ms=4, color="C2", label="Linear Z")
    ax.plot(steps_cmp, z_bf_actual, "s--", ms=5, color="C3",
            label="BF Optimal")
    ax.plot(steps_cmp, np.linspace(Z_start, Z_end, N_cmp + 1),
            ":", color="gray", alpha=0.5, label="Ideal linear Z")
    ax.set_xlabel("Step")
    ax.set_ylabel("Z = sqrt(P)*VA - VB/sqrt(P)")
    ax.set_title(f"Z trajectory comparison (N={N_cmp})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reclamm_interpolation_bruteforce.png", dpi=150)
    print("\nSaved reclamm_interpolation_bruteforce.png")


if __name__ == "__main__":
    main()
