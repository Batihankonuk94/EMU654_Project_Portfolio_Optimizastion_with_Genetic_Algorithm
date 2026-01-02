# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:13:56 2025

@author: batihan.konuk
"""

# -*- coding: utf-8 -*-
"""
FINAL ROBUST PORTFOLIO OPTIMIZER (ALL METRICS & LIT COMPARISON)
---------------------------------------------------------------
Methods:
   1. Hybrid GA (Numba Accelerated + Parallel) -> High Quality (Iter=1000*N)
   2. Exact MIQP (Gurobi) -> Benchmark
   3. UEF (Unconstrained Efficient Frontier) -> Reference

Features:
   - Saves individual result files per dataset (GA & MIQP).
   - Calculates robust Gap % between GA and MIQP.
   - Includes full literature comparison (Chang, Deng, ARO) with Times.
   - Reports MIQP deviation from UEF.
"""

import os
import math
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gurobipy as gp
from gurobipy import GRB
from numba import njit

# ------------------------------
# 0) CONFIG & PATHS
# ------------------------------
BASE_DIR = r"C:\Users\batihan.konuk\EMU654_Project"  # Kendi yolunuzu buraya yazın
PORT_FILES = ["port1.txt", "port2.txt", "port3.txt", "port4.txt"]

OUT_DIR = os.path.join(BASE_DIR, "results_final_robust")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# 1) PROBLEM PARAMETERS
# ------------------------------
K = 10
e_min = 0.01
d_max = 1.0

# Lambda grid (50 points)
E = 50
LAMS = np.linspace(0.0, 1.0, E)

# UEF grid (2000 points for precision)
UEF_POINTS = 2000
UEF_LAMS = np.linspace(0.0, 1.0, UEF_POINTS)

# ------------------------------
# 2) ALGORITHM SETTINGS
# ------------------------------
# GA Settings (High Quality)
POP_SIZE = 100
SEEDS = [42]  # Tek seed yeterli çünkü paralel lambda taraması yapıyoruz
ITER_MULTIPLIER = 1000  # Literatür standardı: 1000 * N iterasyon
N_JOBS_GA = -1  # Tüm çekirdekleri kullan

# MIQP Settings
TIMELIMIT_MIQP = 3600  # Her veri seti için toplam süre bütçesi (saniye) veya lambda başı
# Not: Burada lambda başına değil, veri seti başına havuz mantığı kullanacağız.
MIPGAP_MIQP = 0.0001
OUTPUTFLAG_GUROBI = 0
THREADS_GUROBI = 1

# ------------------------------
# 3) LITERATURE DATA
# ------------------------------
# Veriler makalelerden alınmıştır.
LIT_DATA = {
    "port1": {
        "Name": "Hang Seng", "N": 31,
        "Chang_Err": 1.0974, "Chang_Time": 172,
        "Deng_Err": 1.0953, "Deng_Time": 4.8,
        "ARO_Err": 1.4181, "ARO_Time": np.nan
    },
    "port2": {
        "Name": "DAX 100", "N": 85,
        "Chang_Err": 2.5424, "Chang_Time": 544,
        "Deng_Err": 2.5417, "Deng_Time": 26.8,
        "ARO_Err": 1.3190, "ARO_Time": np.nan
    },
    "port3": {
        "Name": "FTSE 100", "N": 89,
        "Chang_Err": 1.1076, "Chang_Time": 573,
        "Deng_Err": 1.0628, "Deng_Time": 31.4,
        "ARO_Err": 0.8151, "ARO_Time": np.nan
    },
    "port4": {
        "Name": "S&P 100", "N": 98,
        "Chang_Err": 1.9328, "Chang_Time": 638,
        "Deng_Err": 1.6890, "Deng_Time": 34.5,
        "ARO_Err": 1.4468, "ARO_Time": np.nan
    },
}

# ------------------------------
# 4) DATA LOADER
# ------------------------------
def read_orlib_portfolio(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f if ln.strip()]

    n = int(float(raw[0].split()[0]))
    pairs = [raw[i].split() for i in range(1, 1 + n)]
    mu = np.array([float(p[0]) for p in pairs], dtype=np.float64)
    stdev = np.array([float(p[1]) for p in pairs], dtype=np.float64)

    corr = np.eye(n, dtype=np.float64)
    for ln in raw[1 + n:]:
        parts = ln.split()
        if len(parts) != 3: continue
        i, j = int(parts[0]) - 1, int(parts[1]) - 1
        rho = float(parts[2])
        corr[i, j] = rho
        corr[j, i] = rho

    Sigma = corr * np.outer(stdev, stdev)
    return mu, np.ascontiguousarray(Sigma)

# ------------------------------
# 5) GA KERNEL (NUMBA ACCELERATED)
# ------------------------------
@njit(fastmath=True, cache=True)
def fast_evaluate(mu, Sigma, mask, s_vals, lam, e_min_val, d_max_val):
    idxs = np.where(mask)[0]
    n_sub = len(idxs)
    
    if n_sub == 0: return np.inf, np.zeros_like(mu), 0.0, 0.0, 0.0

    e_sub = np.full(n_sub, e_min_val)
    d_sub = np.full(n_sub, d_max_val)
    s_sub = s_vals[mask]
    
    sum_s = np.sum(s_sub)
    if sum_s <= 1e-12:
        s_sub = np.ones(n_sub) / n_sub
        sum_s = 1.0

    # Algorithm 1: Repair
    F = 1.0 - np.sum(e_sub)
    if F < -1e-9: return np.inf, np.zeros_like(mu), 0.0, 0.0, 0.0

    w_sub = e_sub + (s_sub / sum_s) * F
    fixed = np.zeros(n_sub, dtype=np.bool_)
    
    for _ in range(50):
        viol = (~fixed) & (w_sub > d_sub + 1e-9)
        if not np.any(viol): break
        fixed = fixed | viol
        
        if np.all(fixed):
            if abs(np.sum(d_sub) - 1.0) > 1e-5:
                return np.inf, np.zeros_like(mu), 0.0, 0.0, 0.0
            w_sub = d_sub.copy()
            break
        
        s_free_sum = 0.0
        e_free_sum = 0.0
        d_fixed_sum = 0.0
        
        for i in range(n_sub):
            if not fixed[i]:
                s_free_sum += s_sub[i]
                e_free_sum += e_sub[i]
            else:
                d_fixed_sum += d_sub[i]
        
        if s_free_sum <= 1e-12: return np.inf, np.zeros_like(mu), 0.0, 0.0, 0.0
        
        F_new = 1.0 - (e_free_sum + d_fixed_sum)
        for i in range(n_sub):
            if not fixed[i]:
                w_sub[i] = e_sub[i] + (s_sub[i] / s_free_sum) * F_new
            else:
                w_sub[i] = d_sub[i]

    w_full = np.zeros_like(mu)
    for k in range(n_sub): w_full[idxs[k]] = w_sub[k]

    risk = np.dot(w_full, np.dot(Sigma, w_full))
    ret = np.dot(mu, w_full)
    if risk < 0: risk = 0.0
    sigma = np.sqrt(risk)
    obj = lam * risk - (1.0 - lam) * ret
    
    return obj, w_full, risk, sigma, ret

def create_random_population(n, K, pop_size, rng):
    pop_mask = np.zeros((pop_size, n), dtype=bool)
    pop_s = np.random.rand(pop_size, n)
    for i in range(pop_size):
        indices = rng.choice(n, K, replace=False)
        pop_mask[i, indices] = True
    return pop_mask, pop_s

def repair_cardinality(mask, s_vals, K, rng):
    n = len(mask)
    idxs = np.where(mask)[0]
    current_k = len(idxs)
    if current_k == K: return mask, s_vals
    
    new_mask = mask.copy()
    new_s = s_vals.copy()
    
    if current_k > K:
        selected_s = new_s[idxs]
        sorted_args = np.argsort(selected_s)
        to_remove = idxs[sorted_args[:(current_k - K)]]
        new_mask[to_remove] = False
    elif current_k < K:
        not_selected = np.where(~new_mask)[0]
        to_add = rng.choice(not_selected, K - current_k, replace=False)
        new_mask[to_add] = True
        for idx in to_add: new_s[idx] = rng.random()
        
    return new_mask, new_s

def ga_crossover_mutate(mask1, s1, mask2, s2, K, rng):
    n = len(mask1)
    child_mask = np.zeros(n, dtype=bool)
    child_s = np.zeros(n)
    
    rand_vec = rng.random(n)
    from_p1 = rand_vec < 0.5
    child_mask[from_p1] = mask1[from_p1]
    child_mask[~from_p1] = mask2[~from_p1]
    child_s[from_p1] = s1[from_p1]
    child_s[~from_p1] = s2[~from_p1]
    
    # Mutation: Multiply s by 0.9 or 1.1 (Chang style)
    if rng.random() < 0.5:
        idxs = np.where(child_mask)[0]
        if len(idxs) > 0:
            target = rng.choice(idxs)
            child_s[target] *= (0.9 if rng.random() < 0.5 else 1.1)
            
    return repair_cardinality(child_mask, child_s, K, rng)

def solve_ga_single_seed(mu, Sigma, K, e_min, d_max, lam, pop_size, T_iter, seed):
    rng = np.random.default_rng(seed)
    n = len(mu)
    pop_mask, pop_s = create_random_population(n, K, pop_size, rng)
    fits = np.zeros(pop_size)
    cache_stats = np.zeros((pop_size, 3)) # var, sig, ret
    cache_w = np.zeros((pop_size, n))
    
    local_H = []
    
    for i in range(pop_size):
        obj, w, var, sig, ret = fast_evaluate(mu, Sigma, pop_mask[i], pop_s[i], lam, e_min, d_max)
        fits[i] = obj
        if w is not None:
            cache_w[i] = w
            cache_stats[i] = [var, sig, ret]
            local_H.append((sig, ret))
            
    best_idx = np.argmin(fits)
    best_val = fits[best_idx]
    best_sol = (pop_mask[best_idx].copy(), pop_s[best_idx].copy(), cache_w[best_idx].copy(), cache_stats[best_idx].copy())
    
    indices = np.arange(pop_size)
    for _ in range(T_iter):
        c1, c2 = rng.choice(indices, 2, replace=False)
        p1 = c1 if fits[c1] < fits[c2] else c2
        c3, c4 = rng.choice(indices, 2, replace=False)
        p2 = c3 if fits[c3] < fits[c4] else c4
        
        child_mask, child_s = ga_crossover_mutate(pop_mask[p1], pop_s[p1], pop_mask[p2], pop_s[p2], K, rng)
        f_c, w_c, var_c, sig_c, ret_c = fast_evaluate(mu, Sigma, child_mask, child_s, lam, e_min, d_max)
        
        if w_c is not None:
            local_H.append((sig_c, ret_c))
            worst = np.argmax(fits)
            if f_c < fits[worst]:
                pop_mask[worst], pop_s[worst] = child_mask, child_s
                fits[worst], cache_w[worst] = f_c, w_c
                cache_stats[worst] = [var_c, sig_c, ret_c]
                if f_c < best_val:
                    best_val = f_c
                    best_sol = (child_mask, child_s, w_c, [var_c, sig_c, ret_c])
                    
    _, _, final_w, final_stats = best_sol
    return best_val, final_w, final_stats, local_H

def run_ga_dataset_parallel(mu, Sigma, K, e_min, d_max, LAMS, POP, iter_per_asset, SEEDS):
    n = len(mu)
    T = int(iter_per_asset * n)
    t0 = time.perf_counter()
    
    # Run Parallel
    results = Parallel(n_jobs=N_JOBS_GA)(
        delayed(solve_ga_single_seed)(mu, Sigma, K, e_min, d_max, float(lam), POP, T, seed)
        for lam in LAMS for seed in SEEDS
    )
    
    # Aggregation
    rows = []
    H_global = []
    
    res_idx = 0
    lambda_bests = {lam: (np.inf, None, None) for lam in LAMS}
    
    for lam in LAMS:
        for seed in SEEDS:
            val, w, stats, loc_H = results[res_idx]
            H_global.extend(loc_H)
            
            if val < lambda_bests[lam][0]:
                lambda_bests[lam] = (val, w, stats)
            res_idx += 1
            
    for lam in LAMS:
        b_val, b_w, b_stats = lambda_bests[lam]
        if b_w is not None:
            sel_str = ",".join(map(str, np.where(b_w > 1e-6)[0] + 1))
            rows.append({
                "lambda": lam, "obj": b_val, 
                "variance": b_stats[0], "sigma": b_stats[1], "return": b_stats[2],
                "selected_assets": sel_str
            })
        else:
            rows.append({"lambda": lam, "obj": np.nan})
        
    df_ga = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    
    # H set unique & sort
    H_global = list(set(H_global))
    H_global.sort(key=lambda x: (x[0], -x[1]))
    df_H = pd.DataFrame(H_global, columns=["sigma", "return"])
    
    return df_ga, df_H, time.perf_counter() - t0

# ------------------------------
# 6) UEF & MIQP SOLVERS
# ------------------------------
def compute_uef_gurobi(mu, Sigma, UEF_LAMS):
    n = len(mu)
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    w = m.addMVar(n, lb=0.0, ub=1.0)
    m.addConstr(w.sum() == 1.0)
    risk_expr = w @ Sigma @ w
    ret_expr = mu @ w
    
    rows = []
    for lam in UEF_LAMS:
        m.setObjective(lam * risk_expr - (1.0 - lam) * ret_expr, GRB.MINIMIZE)
        m.optimize()
        if m.SolCount > 0:
            v = w.X
            rows.append({"lambda": lam, "sigma": math.sqrt(v @ Sigma @ v), "return": mu @ v})
    return pd.DataFrame(rows).sort_values("return").reset_index(drop=True)

def solve_miqp_dataset(mu, Sigma, K, e_min, d_max, LAMS):
    n = len(mu)
    m = gp.Model()
    m.Params.OutputFlag = OUTPUTFLAG_GUROBI
    m.Params.MIPGap = MIPGAP_MIQP
    m.Params.Threads = THREADS_GUROBI
    
    x = m.addMVar(n, lb=0.0, ub=1.0)
    z = m.addMVar(n, vtype=GRB.BINARY)
    m.addConstr(x.sum() == 1.0)
    m.addConstr(z.sum() == K)
    m.addConstr(x >= e_min * z)
    m.addConstr(x <= d_max * z)
    
    risk_expr = x @ Sigma @ x
    ret_expr = mu @ x
    
    rows = []
    t0_global = time.perf_counter()
    budget_left = TIMELIMIT_MIQP
    
    for lam in LAMS:
        if budget_left <= 0:
            rows.append({"lambda": lam, "obj": np.nan, "sigma": np.nan, "return": np.nan})
            continue
            
        m.Params.TimeLimit = budget_left
        m.setObjective(lam * risk_expr - (1.0 - lam) * ret_expr, GRB.MINIMIZE)
        
        t_start = time.time()
        m.optimize()
        elapsed = time.time() - t_start
        budget_left -= elapsed
        
        val, sig, ret = np.nan, np.nan, np.nan
        if m.SolCount > 0:
            xv = x.X
            val = m.ObjVal
            sig = math.sqrt(xv @ Sigma @ xv)
            ret = mu @ xv
        rows.append({"lambda": lam, "obj": val, "sigma": sig, "return": ret})
        
    return pd.DataFrame(rows), time.perf_counter() - t0_global

# ------------------------------
# 7) METRICS & CALCULATION
# ------------------------------
def interp_y(x, xs, ys):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    idx = np.searchsorted(xs, x)
    x0, x1 = xs[idx-1], xs[idx]
    y0, y1 = ys[idx-1], ys[idx]
    if abs(x1-x0) < 1e-12: return y0
    return y0 + (x - x0)/(x1 - x0) * (y1 - y0)

def calc_error(df_points, df_uef):
    uef_sig = df_uef["sigma"].values
    uef_ret = df_uef["return"].values
    
    idx_sig = np.argsort(uef_sig)
    uef_sig_sorted = uef_sig[idx_sig]
    uef_ret_sorted_by_sig = uef_ret[idx_sig]
    
    errs = []
    for _, row in df_points.iterrows():
        if pd.isna(row["sigma"]) or pd.isna(row["return"]): continue
        s_p, r_p = row["sigma"], row["return"]
        
        # Vertical error (fixed sigma)
        r_star = interp_y(s_p, uef_sig_sorted, uef_ret_sorted_by_sig)
        e1 = abs(r_p - r_star) / max(abs(r_star), 1e-9)
        
        # Horizontal error (fixed return)
        s_star = interp_y(r_p, uef_ret, uef_sig)
        e2 = abs(s_p - s_star) / max(abs(s_star), 1e-9)
        
        errs.append(100.0 * min(e1, e2))
    return np.mean(errs) if errs else np.nan

# ------------------------------
# 8) MAIN
# ------------------------------
if __name__ == "__main__":
    summary_data = []

    print(f"Starting Comprehensive Run (Iterations={ITER_MULTIPLIER}*N)...")

    for fname in PORT_FILES:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path): continue
        
        tag = os.path.splitext(fname)[0]
        print(f"\n=== Processing {tag} ===")
        
        # Load
        mu, Sigma = read_orlib_portfolio(path)
        lit = LIT_DATA.get(tag, {})
        
        # 1. UEF
        print("  1. Computing UEF...")
        df_uef = compute_uef_gurobi(mu, Sigma, UEF_LAMS)
        df_uef.to_csv(os.path.join(OUT_DIR, f"uef_{tag}.csv"), index=False)
        
        # 2. GA
        print(f"  2. Running GA (Parallel, {N_JOBS_GA} jobs)...")
        df_ga, df_H, ga_time = run_ga_dataset_parallel(
            mu, Sigma, K, e_min, d_max, LAMS, POP_SIZE, ITER_MULTIPLIER, SEEDS
        )
        df_ga.to_csv(os.path.join(OUT_DIR, f"ga_results_{tag}.csv"), index=False)
        ga_err = calc_error(df_ga, df_uef)
        
        # 3. MIQP
        print(f"  3. Running MIQP (Budget={TIMELIMIT_MIQP}s)...")
        df_miqp, miqp_time = solve_miqp_dataset(mu, Sigma, K, e_min, d_max, LAMS)
        df_miqp.to_csv(os.path.join(OUT_DIR, f"miqp_results_{tag}.csv"), index=False)
        miqp_err = calc_error(df_miqp, df_uef) 
        
        # 4. Metric Calculation (Robust Gap)
        # Round lambda to avoid float mismatch
        df_ga["lam_round"] = df_ga["lambda"].round(5)
        df_miqp["lam_round"] = df_miqp["lambda"].round(5)
        
        merged = pd.merge(df_ga, df_miqp, on="lam_round", suffixes=("_GA", "_MIQP"))
        
        # Gap Calculation (Robust)
        # Gap = |GA - MIQP| / (|MIQP| + eps) * 100
        # Only where MIQP found a solution
        valid_mask = merged["obj_MIQP"].notna() & merged["obj_GA"].notna()
        
        if valid_mask.any():
            objs_miqp = merged.loc[valid_mask, "obj_MIQP"]
            objs_ga = merged.loc[valid_mask, "obj_GA"]
            
            diff = (objs_ga - objs_miqp).abs()
            denom = objs_miqp.abs() + 1e-9 # Avoid division by zero
            gaps = (diff / denom) * 100.0
            mean_gap = gaps.mean()
        else:
            mean_gap = np.nan
            
        # Add to summary
        summary_data.append({
            "Dataset": tag,
            "N": len(mu),
            
            # --- OUR RESULTS ---
            "GA_Mean_Err%": ga_err,
            "GA_Time(s)": ga_time,
            "MIQP_Mean_Err%": miqp_err,
            "MIQP_Time(s)": miqp_time,
            "GA_vs_MIQP_Gap%": mean_gap,
            
            # --- LITERATURE ---
            "Chang_Err%": lit.get("Chang_Err", np.nan),
            "Chang_Time(s)": lit.get("Chang_Time", np.nan),
            "Deng_Err%": lit.get("Deng_Err", np.nan),
            "Deng_Time(s)": lit.get("Deng_Time", np.nan),
            "ARO_Err%": lit.get("ARO_Err", np.nan),
            "ARO_Time(s)": lit.get("ARO_Time", np.nan)
        })
        
        print(f"  -> {tag} Done. GA_Err: {ga_err:.4f}% | Gap: {mean_gap:.4f}%")

    # SAVE FINAL SUMMARY
    if summary_data:
        df_summ = pd.DataFrame(summary_data)
        
        # Organize columns nicely
        cols = [
            "Dataset", "N", 
            "GA_Mean_Err%", "GA_Time(s)", 
            "MIQP_Mean_Err%", "MIQP_Time(s)", "GA_vs_MIQP_Gap%",
            "Chang_Err%", "Chang_Time(s)", 
            "Deng_Err%", "Deng_Time(s)", 
            "ARO_Err%", "ARO_Time(s)"
        ]
        df_summ = df_summ[cols]
        
        print("\n=== FINAL COMPREHENSIVE SUMMARY ===")
        print(df_summ.to_string(index=False))
        
        csv_path = os.path.join(OUT_DIR, "final_summary_comprehensive.csv")
        df_summ.to_csv(csv_path, index=False)
        
        tex_path = os.path.join(OUT_DIR, "final_summary_comprehensive.tex")
        with open(tex_path, "w") as f:
            f.write(df_summ.to_latex(index=False, float_format="%.4f"))
            
        print(f"\nSaved to: {csv_path}")
        
        
        
# Yüklenen dosyalardan okuma
results = {}

# Mevcut dosyalar
files_map = {
    "port1": {"ga": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\ga_results_port1.csv", "miqp": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\miqp_results_port1.csv"},
    "port2": {"ga": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\ga_results_port2.csv", "miqp": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\miqp_results_port2.csv"},
    "port3": {"ga": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\ga_results_port3.csv", "miqp": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\miqp_results_port3.csv"},
    "port4": {"ga": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\ga_results_port4.csv", "miqp": r"C:\Users\batihan.konuk\EMU654_Project\results_final_robust\miqp_results_port4.csv"}
}

print(f"{'Dataset':<10} | {'GA Mean Obj':<20} | {'MIQP Mean Obj':<20}")
print("-" * 56)

for p, paths in files_map.items():
    ga_val = "Dosya Yüklenmedi"
    miqp_val = "Dosya Yüklenmedi"
    
    # GA
    if paths["ga"]:
        try:
            df = pd.read_csv(paths["ga"])
            ga_val = df['obj'].mean()
        except: pass
            
    # MIQP
    if paths["miqp"]:
        try:
            df = pd.read_csv(paths["miqp"])
            miqp_val = df['obj'].mean()
        except: pass
    
    # Formatlama
    g_str = f"{ga_val:.6f}" if isinstance(ga_val, float) else ga_val
    m_str = f"{miqp_val:.6f}" if isinstance(miqp_val, float) else miqp_val
    
    print(f"{p:<10} | {g_str:<20} | {m_str:<20}")