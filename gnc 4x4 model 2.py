# -*- coding: utf-8 -*-
"""
GNC MIMO 4x4 for TWO PCCs + CRASH RECOVERY + VECTOR FITTING
Model 2 (MMC_SCAN / 400kV / 800MVA)
Scope: 5-100 Hz (1 Hz step)

Adapted from GNC mimo.py (Model 1) with Model 2 attributes:
  - casefile, Vbase_LL, Sbase_3ph updated for Model 2
  - Signal paths updated with Substep|MMC_DCS1 prefix
  - RSCAD_PLOT_WINDOW_S added + settle_time minimum enforced
  - IO retry double-call bug fixed
  - LOW_CONF diagnostic printing added
  - SVG saving added
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import rtds.rscadfx

try:
    from rtds.comms import RSCADError
except Exception:
    RSCADError = Exception

# ==================================================
# 0) USER CONFIG
# ==================================================
casefile = r"C:\Users\muass\OneDrive\Documenten\RSCAD\RTDS_USER_FX\fileman\01 HVDC\Harmonic Scan (Measured)\MMC_SCAN.rtfx"

VALID_F_MIN = 5.0
VALID_F_MAX = 100.0
F_STEP_HZ   = 1.0

Vbase_LL  = 400e3
Sbase_3ph = 800e6
f0 = 50.0

SCR = 3.5
XR  = 15.0

A0    = 1.0
A_MIN = 0.2
A_MAX = 3.0
F_KNEE = 20.0
ALPHA  = 0.7

F_SLOW_MAX = 12.0

TIMEOUT_S   = 60.0
MIN_SAMPLES = 1400
POLL_S      = 0.7

SLOW_N_CYCLES = 20
FAST_N_CYCLES = 12
SLOW_MIN_S    = 1.8
FAST_MIN_S    = 0.8
SLOW_MAX_S    = 3.5
FAST_MAX_S    = 1.8

HARM_WARN_RATIO     = 0.08   # 8% => WARN
HARM_LOWCONF_RATIO  = 0.25   # 25% => LOW_CONF

DRIFT_WARN_DEG_HI = 30.0
DRIFT_WARN_DEG_LO = 80.0
DRIFT_LOWCONF_DEG_HI = 80.0
DRIFT_LOWCONF_DEG_LO = 140.0

SNR_MIN_MAG_V = 1e-4
SNR_MIN_MAG_I = 1e-5

F_SEARCH_ON_DRIFT = True
F_SEARCH_BW_HZ    = 0.8

# Robustness / recovery
RUN_RETRIES     = 7
RUN_BASE_WAIT   = 0.9
STOP_SETTLE     = 1.7

# IOException recovery
IO_RETRY_COUNT   = 6
IO_RETRY_WAIT_S  = 8.0

# RSCAD plot window duration (must match RSCAD's "Plot Update Window" setting).
# update_plots() fails if the simulation hasn't run for at least this long.
RSCAD_PLOT_WINDOW_S = 4.1   # RSCAD is configured to 4.0001 s; use 4.1 for margin

# Extra waits
POST_PULSE_WAIT = 0.18
POST_RUN_WAIT   = 0.35
AFTER_REOPEN_WAIT  = 1.2
AFTER_COMPILE_WAIT = 2.5

DET_THR     = 1e-12
COND_THR    = 1e6
PINV_RCOND  = 1e-10

ZOOM_XLIM = (-5.0, 3.0)
ZOOM_YLIM = (-5.0, 5.0)

STALE_RETRIES_PER_EXPERIMENT = 2

WINDING_MIN_DIST_WARN = 0.05

# Vector fitting
VF_ENABLE          = True
VF_TARGET_ERROR    = 0.01
VF_MAX_ORDER       = 80
VF_EVAL_DENSE      = False

# ==================================================
# 0a) TWO PCC DEFINITIONS  (Model 2 signal paths)
# ==================================================
PCC1 = {
    "name": "PCC1",
    "Vd_path": "Subsystem #1|CTLs|Vars|Vdpu11",
    "Vq_path": "Subsystem #1|CTLs|Vars|Vqpu11",
    "Id_path": "Subsystem #1|CTLs|Vars|Id",
    "Iq_path": "Subsystem #1|CTLs|Vars|Iq",
}
PCC2 = {
    "name": "PCC2",
    "Vd_path": "Subsystem #1|CTLs|Vars|Vdpu22",
    "Vq_path": "Subsystem #1|CTLs|Vars|Vqpu22",
    "Id_path": "Subsystem #1|CTLs|Vars|Id1",
    "Iq_path": "Subsystem #1|CTLs|Vars|Iq1",
}

INJECTIONS = [
    {"name": "inj1", "Finj_slider": "Finj",  "d_gain_slider": "d_gain",  "q_gain_slider": "q_gain",  "PB_d": "PB",  "PB_q": "PB1"},
    {"name": "inj2", "Finj_slider": "Finj1", "d_gain_slider": "d_gain1", "q_gain_slider": "q_gain1", "PB_d": "PB2", "PB_q": "PB3"},
]

MUTUAL_ENABLE = False
R12_pu = 0.0
X12_pu = 0.0

# ==================================================
# 1) BASE + GRID
# ==================================================
Zbase = (Vbase_LL ** 2) / Sbase_3ph
Ibase = Sbase_3ph / (np.sqrt(3.0) * Vbase_LL)
S_sc       = SCR * Sbase_3ph
Zmag_ohm   = (Vbase_LL**2) / S_sc
R_grid_ohm = Zmag_ohm / np.sqrt(1.0 + XR**2)
X_grid_ohm = XR * R_grid_ohm
R_grid_pu  = R_grid_ohm / Zbase
X_grid_pu  = X_grid_ohm / Zbase

print("====================================")
print("BASE + GRID CHECK (PU)")
print(f"Zbase={Zbase:.6f}, R_pu={R_grid_pu:.6e}, X_pu={X_grid_pu:.6e}")
print(f"Mutual={MUTUAL_ENABLE} | R12={R12_pu:.6e}, X12={X12_pu:.6e}")
print("====================================\n")

# ==================================================
# 2) HELPERS
# ==================================================
def is_slow_mode(f_hz): return float(f_hz) <= float(F_SLOW_MAX)

def inj_amplitude(f_hz):
    f_hz = float(f_hz)
    if f_hz <= 0: return float(np.clip(A0, A_MIN, A_MAX))
    scale = (f_hz / F_KNEE) ** ALPHA if f_hz > F_KNEE else 1.0
    return float(np.clip(A0 * scale, A_MIN, A_MAX))

def settle_time(f_hz):
    f_hz = float(f_hz)
    # Minimum must exceed RSCAD_PLOT_WINDOW_S so update_plots() finds a full window of data.
    if is_slow_mode(f_hz):
        return float(max(RSCAD_PLOT_WINDOW_S + 0.2, min(7.0, 14.0 / max(f_hz, 0.7))))
    return float(max(RSCAD_PLOT_WINDOW_S + 0.2, min(7.0, 8.0 / max(f_hz, 2.0))))

def analysis_window_params(f_hz):
    if is_slow_mode(f_hz): return SLOW_N_CYCLES, SLOW_MIN_S, SLOW_MAX_S
    return FAST_N_CYCLES, FAST_MIN_S, FAST_MAX_S

def lockin_phasor(x, t, f_ref):
    x = np.asarray(x, dtype=float); t = np.asarray(t, dtype=float)
    if len(x) < 32: raise RuntimeError("Too few samples for lock-in.")
    x_ac = x - np.mean(x); N = len(x_ac)
    w = np.hanning(N); ww = np.mean(w) if np.mean(w) != 0 else 1.0
    ref = np.exp(-1j * 2.0 * np.pi * float(f_ref) * t)
    return (2.0 / N) * np.sum((x_ac * w) * ref) / ww

def select_analysis_window(t, data_dict, f_hz):
    t = np.asarray(t)
    Ncy, Tmin, Tmax = analysis_window_params(float(f_hz))
    T_need = max(Tmin, min(Tmax, Ncy / max(float(f_hz), 0.2)))
    T_use = min(T_need, max(0.2, float(t[-1] - t[0])))
    idx = np.where(t >= t[-1] - T_use)[0]
    if len(idx) < 64: return t, data_dict
    i0 = idx[0]
    return t[i0:], {k: np.asarray(v)[i0:] for k, v in data_dict.items()}

def get_timeseries_consistent(case, sig_handles, timeout_s=30.0, min_samples=1000, poll_s=0.8):
    t_start = time.time(); last_len = -1; stable_count = 0; io_retries = 0
    while time.time() - t_start < timeout_s:
        try:
            case.update_plots()
        except Exception as e:
            if io_retries < IO_RETRY_COUNT:
                io_retries += 1
                print(f"[IO RETRY {io_retries}/{IO_RETRY_COUNT}] update_plots: {e} -> wait {IO_RETRY_WAIT_S}s")
                time.sleep(IO_RETRY_WAIT_S)
                # Do NOT call update_plots() here — the outer loop will retry it.
                continue
            else:
                raise
        time.sleep(poll_s)
        t_ref = None; data = {}; ok = True
        for name, sig in sig_handles.items():
            tt = np.asarray(sig.get_time_data()); xx = np.asarray(sig.get_data())
            if len(tt) < min_samples or len(tt) != len(xx): ok = False; break
            if t_ref is None: t_ref = tt
            elif len(tt) != len(t_ref): ok = False; break
            data[name] = xx
        if ok:
            if len(t_ref) == last_len: stable_count += 1
            else: stable_count = 0; last_len = len(t_ref)
            if stable_count >= 1: return t_ref, data
    raise RuntimeError("Could not get consistent time series (timeout).")

def estimate_fs(t):
    dt = np.diff(np.asarray(t, dtype=float))
    fs = 1.0 / np.mean(dt)
    return float(fs), float(np.std(dt)/np.mean(dt)) if np.mean(dt) > 0 else (float(fs), np.inf)

def spectrum_peak_near(x, t, f0_hz, bw_hz=0.8):
    x = np.asarray(x, dtype=float); t = np.asarray(t, dtype=float)
    fs, _ = estimate_fs(t); N = len(x)
    if N < 256: return float(f0_hz)
    w = np.hanning(N); X = np.fft.rfft((x - np.mean(x)) * w)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    idx = np.where((freqs >= max(0, f0_hz-bw_hz)) & (freqs <= f0_hz+bw_hz))[0]
    if len(idx) < 3: return float(f0_hz)
    return float(freqs[idx[np.argmax(np.abs(X[idx]))]])

def phase_drift_deg(x, t, f_ref):
    x = np.asarray(x, dtype=float); t = np.asarray(t, dtype=float); N = len(x)
    if N < 128: return 0.0
    mid = N // 2
    return float(np.degrees(np.angle(lockin_phasor(x[mid:], t[mid:], f_ref) / lockin_phasor(x[:mid], t[:mid], f_ref))))

def harmonic_ratios(x, t, f_ref):
    X1 = lockin_phasor(x, t, f_ref); eps = 1e-30
    X2 = lockin_phasor(x, t, 2.0*float(f_ref)); X3 = lockin_phasor(x, t, 3.0*float(f_ref))
    return float(np.abs(X2)/(np.abs(X1)+eps)), float(np.abs(X3)/(np.abs(X1)+eps)), X1

def safe_stop(case, settle_s=STOP_SETTLE):
    try: case.stop()
    except: pass
    time.sleep(settle_s)

def inv_or_pinv(A, det_thr=DET_THR, cond_thr=COND_THR, rcond=PINV_RCOND):
    detA = np.linalg.det(A); condA = np.linalg.cond(A); used_pinv = False
    if (abs(detA) < det_thr) or (condA > cond_thr):
        used_pinv = True; Ainv = np.linalg.pinv(A, rcond=rcond)
    else: Ainv = np.linalg.inv(A)
    return Ainv, detA, condA, used_pinv

def track_eig_branches(eig_results, freqs):
    e0 = np.array(eig_results[freqs[0]], dtype=complex); nL = len(e0)
    lam = np.zeros((nL, len(freqs)), dtype=complex); lam[:, 0] = e0
    for k in range(1, len(freqs)):
        prev = lam[:, k-1]; cur = list(np.array(eig_results[freqs[k]], dtype=complex))
        for i in range(nL):
            d = [abs(cur[j] - prev[i]) for j in range(len(cur))]
            jmin = int(np.argmin(d)); lam[i, k] = cur.pop(jmin)
    return [lam[i, :] for i in range(nL)]

def winding_number_of_complex_curve(z):
    z = np.asarray(z, dtype=complex); ang = np.unwrap(np.angle(z))
    dphi = float(ang[-1] - ang[0]); return int(np.round(dphi / (2.0 * np.pi))), dphi

def check_winding_number_reliability(z, freqs, threshold=WINDING_MIN_DIST_WARN):
    z = np.asarray(z, dtype=complex); freqs = np.asarray(freqs, dtype=float); warnings = []
    dists = np.abs(z); md = float(np.min(dists)); mi = int(np.argmin(dists))
    if md < threshold:
        warnings.append(f"det(I+L) near origin: |det|={md:.3e} at f={freqs[mi]:.2f} Hz")
    if len(freqs) >= 2:
        df = np.diff(freqs); med = float(np.median(df))
        for i, g in enumerate(df):
            if g > 3.0 * med:
                warnings.append(f"Gap: {freqs[i]:.1f}->{freqs[i+1]:.1f} Hz (df={g:.1f}, med={med:.1f})")
    return warnings

def dq_RL_block_pu(f_hz, R_pu, X_pu, f0_hz):
    w_pu = float(f_hz) / float(f0_hz)
    Zdd = float(R_pu) + 1j * w_pu * float(X_pu)
    return np.array([[Zdd, -float(X_pu)], [float(X_pu), Zdd]], dtype=complex)

def build_Zgrid_4x4_pu(f_hz, f0_hz, Rg1, Xg1, Rg2, Xg2, R12=0.0, X12=0.0):
    Z11 = dq_RL_block_pu(f_hz, Rg1, Xg1, f0_hz); Z22 = dq_RL_block_pu(f_hz, Rg2, Xg2, f0_hz)
    Z12 = dq_RL_block_pu(f_hz, R12, X12, f0_hz); Z21 = Z12.copy()
    return np.vstack([np.hstack([Z11, Z12]), np.hstack([Z21, Z22])])

# ==================================================
# 2b) CRASH RECOVERY
# ==================================================
def is_stale_object_error(e):
    s = str(e).lower()
    return ("invalid processing object" in s) or ("cannot be found" in s) or ("remote invocation error" in s)

def is_unable_to_start_case_error(e):
    s = str(e).lower()
    return ("unable to start case" in s) or ("unable to start" in s)

def is_io_exception(e):
    s = str(e).lower()
    return ("ioexception" in s) or ("io exception" in s) or ("plot" in s and "ack" in s)

def get_obj_by_name_anytype(case, name, pref, fallbacks):
    for t in [pref] + [x for x in fallbacks if x != pref]:
        try:
            obj = case.get_object_by_name(name, t)
            if obj is not None: return obj
        except: pass
    return None

def build_handles(case):
    SL = ["slider", "knob", "dial", "numeric"]; BT = ["button", "pushbutton", "toggle", "switch"]
    Hinj = {}
    for inj in INJECTIONS:
        nm = inj["name"]
        F = get_obj_by_name_anytype(case, inj["Finj_slider"], "slider", SL)
        d = get_obj_by_name_anytype(case, inj["d_gain_slider"], "slider", SL)
        q = get_obj_by_name_anytype(case, inj["q_gain_slider"], "slider", SL)
        pd = get_obj_by_name_anytype(case, inj["PB_d"], "button", BT)
        pq = get_obj_by_name_anytype(case, inj["PB_q"], "button", BT)
        miss = []
        if F is None: miss.append(inj["Finj_slider"])
        if d is None: miss.append(inj["d_gain_slider"])
        if q is None: miss.append(inj["q_gain_slider"])
        if pd is None: miss.append(inj["PB_d"])
        if pq is None: miss.append(inj["PB_q"])
        if miss: raise RuntimeError(f"Missing objects for {nm}: {miss}")
        Hinj[nm] = {"Finj": F, "d_gain": d, "q_gain": q, "PB_d": pd, "PB_q": pq}
    Hpcc = {}
    for key, P in [("pcc1", PCC1), ("pcc2", PCC2)]:
        Hpcc[key] = {
            "sigVd": case.get_signal(P["Vd_path"]), "sigVq": case.get_signal(P["Vq_path"]),
            "sigId": case.get_signal(P["Id_path"]), "sigIq": case.get_signal(P["Iq_path"]),
        }
    return Hinj, Hpcc

def rebuild_sig_handles(Hpcc):
    return {
        "Vd1": Hpcc["pcc1"]["sigVd"], "Vq1": Hpcc["pcc1"]["sigVq"],
        "Id1": Hpcc["pcc1"]["sigId"], "Iq1": Hpcc["pcc1"]["sigIq"],
        "Vd2": Hpcc["pcc2"]["sigVd"], "Vq2": Hpcc["pcc2"]["sigVq"],
        "Id2": Hpcc["pcc2"]["sigId"], "Iq2": Hpcc["pcc2"]["sigIq"],
    }

def pulse_only_active_pair(Hinj, inj_name, pulse_s=0.10):
    Hinj[inj_name]["PB_d"].position = 1; Hinj[inj_name]["PB_q"].position = 1
    time.sleep(pulse_s)
    Hinj[inj_name]["PB_d"].position = 0; Hinj[inj_name]["PB_q"].position = 0
    time.sleep(0.06)

def recover_by_reopen(app, case):
    print("[HARD RECOVERY] Reopening case...")
    try: safe_stop(case, settle_s=STOP_SETTLE)
    except: pass
    try: case.close()
    except: pass
    time.sleep(0.9)
    case = app.open_case(casefile); safe_stop(case, settle_s=1.2)
    case.compile(); time.sleep(AFTER_COMPILE_WAIT)
    Hinj, Hpcc = build_handles(case); time.sleep(AFTER_REOPEN_WAIT)
    print("[HARD RECOVERY] Complete.")
    return case, Hinj, Hpcc

def safe_run_with_recovery(app, case, Hinj, Hpcc, retries=RUN_RETRIES, base_wait=RUN_BASE_WAIT):
    last = None
    for k in range(1, retries + 1):
        try:
            time.sleep(0.10); case.run(); time.sleep(POST_RUN_WAIT); return case, Hinj, Hpcc
        except RSCADError as e:
            last = e
            if is_stale_object_error(e) or is_unable_to_start_case_error(e):
                print(f"[RUN FAIL {k}/{retries}] {e} -> HARD RECOVERY")
                case, Hinj, Hpcc = recover_by_reopen(app, case); continue
            if is_io_exception(e):
                print(f"[RUN FAIL {k}/{retries}] IOException -> flush, wait {IO_RETRY_WAIT_S}s")
                try: case.update_plots()
                except: pass
                time.sleep(IO_RETRY_WAIT_S); continue
            wait = base_wait * (1.5 ** (k-1)) + random.uniform(0.05, 0.45)
            print(f"[RUN FAIL {k}/{retries}] {e} -> wait {wait:.2f}s"); safe_stop(case, settle_s=wait)
    print(f"[RUN FAIL] All retries exhausted -> final HARD RECOVERY")
    case, Hinj, Hpcc = recover_by_reopen(app, case)
    return case, Hinj, Hpcc

# ==================================================
# 3) QUALITY CLASSIFICATION (4x4)
# ==================================================
def classify_quality_4x4(q, f_inj):
    warn = []; f_inj = float(f_inj); slow = is_slow_mode(f_inj)
    drift_warn = DRIFT_WARN_DEG_LO if slow else DRIFT_WARN_DEG_HI
    drift_lowc = DRIFT_LOWCONF_DEG_LO if slow else DRIFT_LOWCONF_DEG_HI
    drift_abs = float(q["drift_deg_max"])
    hmax = float(q["harm_max"])
    snr_flag = bool(q["snr_flag_any"]); inv_flag = bool(q["used_pinv"])

    if hmax > HARM_WARN_RATIO: warn.append("HARMONICS_WARN")
    if drift_abs > drift_warn: warn.append("DRIFT_WARN")
    if snr_flag: warn.append("SNR_WARN")
    if inv_flag: warn.append("INV_WARN")

    low_conf = snr_flag or (drift_abs > drift_lowc) or (hmax > HARM_LOWCONF_RATIO)
    if inv_flag and (snr_flag or (drift_abs > drift_lowc) or (hmax > HARM_LOWCONF_RATIO)):
        low_conf = True

    status = "LOW_CONF" if low_conf else ("WARN" if warn else "OK")
    valid_for_curves = (status in ("OK", "WARN"))
    valid_for_claims = (status in ("OK", "WARN"))
    return status, warn, bool(valid_for_curves), bool(valid_for_claims)

# ==================================================
# 4) CORE MEASUREMENT: Y_tot (4x4)
# ==================================================
ALL_V_SIGNALS = ["Vd1", "Vq1", "Vd2", "Vq2"]
ALL_I_SIGNALS = ["Id1", "Iq1", "Id2", "Iq2"]

def measure_Ytot_4x4_at(app, case, Hinj, Hpcc, f_inj, A_slider):
    f_inj = float(f_inj); A_slider = float(A_slider); Ts = settle_time(f_inj)
    experiments = [("inj1", A_slider, 0.0), ("inj1", 0.0, A_slider),
                   ("inj2", A_slider, 0.0), ("inj2", 0.0, A_slider)]
    V_ph = np.zeros((4, 4), dtype=complex); I_ph = np.zeros((4, 4), dtype=complex)
    all_drift, all_harm, all_vmag, all_imag, f_used_list = [], [], [], [], []
    fs_est, dt_rel_std = np.nan, np.nan

    for col, (inj_name, d_gain, q_gain) in enumerate(experiments):
        attempts = 0
        while True:
            try:
                safe_stop(case, settle_s=STOP_SETTLE); time.sleep(0.10)
                for nm in ("inj1", "inj2"):
                    Hinj[nm]["Finj"].value = float(f_inj)
                    Hinj[nm]["d_gain"].value = 0.0; Hinj[nm]["q_gain"].value = 0.0
                Hinj[inj_name]["d_gain"].value = float(d_gain)
                Hinj[inj_name]["q_gain"].value = float(q_gain)
                time.sleep(0.10)
                pulse_only_active_pair(Hinj, inj_name, pulse_s=0.10)
                time.sleep(POST_PULSE_WAIT)
                case, Hinj, Hpcc = safe_run_with_recovery(app, case, Hinj, Hpcc)
                sig_handles = rebuild_sig_handles(Hpcc)
                time.sleep(Ts)
                t, data = get_timeseries_consistent(case, sig_handles, timeout_s=TIMEOUT_S, min_samples=MIN_SAMPLES, poll_s=POLL_S)
                fs_est, dt_rel_std = estimate_fs(t)
                t2, data2 = select_analysis_window(t, data, f_inj)

                f_ref = float(f_inj)
                xVd_exc = data2["Vd1"] if inj_name == "inj1" else data2["Vd2"]
                dVd = phase_drift_deg(xVd_exc, t2, f_ref)
                drift_warn = DRIFT_WARN_DEG_LO if is_slow_mode(f_inj) else DRIFT_WARN_DEG_HI
                if F_SEARCH_ON_DRIFT and abs(dVd) > drift_warn:
                    fpk = spectrum_peak_near(xVd_exc, t2, f_ref, bw_hz=F_SEARCH_BW_HZ)
                    if abs(phase_drift_deg(xVd_exc, t2, fpk)) < abs(dVd): f_ref = float(fpk)
                f_used_list.append(float(f_ref))

                xVd = data2["Vd1"] if inj_name == "inj1" else data2["Vd2"]
                xId = data2["Id1"] if inj_name == "inj1" else data2["Id2"]
                all_drift.append(abs(phase_drift_deg(xVd, t2, f_ref)))

                H2v, H3v, Vd1_ph = harmonic_ratios(xVd, t2, f_ref)
                all_harm.append(max(H2v, H3v))
                all_vmag.append(float(np.abs(Vd1_ph)))
                _, _, Id1_ph = harmonic_ratios(xId, t2, f_ref)
                all_imag.append(float(np.abs(Id1_ph)))

                for idx_r, sig in enumerate(["Vd1","Vq1","Vd2","Vq2"]):
                    V_ph[idx_r, col] = lockin_phasor(data2[sig], t2, f_ref)
                for idx_r, sig in enumerate(["Id1","Iq1","Id2","Iq2"]):
                    I_ph[idx_r, col] = lockin_phasor(data2[sig], t2, f_ref)
                break

            except RSCADError as e:
                if (is_stale_object_error(e) or is_unable_to_start_case_error(e)) and attempts < STALE_RETRIES_PER_EXPERIMENT:
                    attempts += 1; print(f"[RECOVER] col={col} -> HARD RECOVERY ({attempts})")
                    case, Hinj, Hpcc = recover_by_reopen(app, case); continue
                raise
            except Exception as e:
                if attempts < 3:
                    attempts += 1; print(f"[RECOVER] col={col} attempt={attempts} {e} -> wait 10s then HARD RECOVERY")
                    time.sleep(10.0)
                    case, Hinj, Hpcc = recover_by_reopen(app, case); continue
                raise

    safe_stop(case, settle_s=STOP_SETTLE)
    for nm in ("inj1", "inj2"):
        try: Hinj[nm]["d_gain"].value = 0.0; Hinj[nm]["q_gain"].value = 0.0
        except: pass

    V_inv, detV, condV, used_pinv = inv_or_pinv(V_ph)
    Y_tot = I_ph @ V_inv
    q = {
        "A_slider": float(A_slider), "settle_s": float(Ts),
        "fs_est": float(fs_est), "dt_rel_std": float(dt_rel_std),
        "f_used_mean": float(np.mean(f_used_list)) if f_used_list else float(f_inj),
        "f_used_list": [float(x) for x in f_used_list],
        "drift_deg_max": float(max(all_drift)) if all_drift else 0.0,
        "harm_max": float(max(all_harm)) if all_harm else 0.0,
        "snr_flag_any": any(v < SNR_MIN_MAG_V for v in all_vmag) or any(i < SNR_MIN_MAG_I for i in all_imag),
        "detV": detV, "condV": float(condV), "used_pinv": bool(used_pinv),
    }
    return Y_tot, q, case, Hinj, Hpcc

# ==================================================
# 5) VECTOR FITTING (4x4, robust, no auto_fit)
# ==================================================
def vector_fit_Ytot_4x4(freqs_ok, Ytot_ok, freqs_eval, target_error=VF_TARGET_ERROR, max_order=VF_MAX_ORDER):
    import skrf as rf
    freqs_ok = np.asarray(freqs_ok, dtype=float)
    s_data = np.zeros((len(freqs_ok), 4, 4), dtype=complex)
    for i in range(len(freqs_ok)): s_data[i] = Ytot_ok[i]
    nw = rf.Network(frequency=rf.Frequency.from_f(freqs_ok, unit='Hz'), s=s_data)

    pole_configs = [(2, 4), (3, 6), (4, 8), (5, 10)]
    best_vf = None; best_rms = np.inf

    for n_real, n_cmplx in pole_configs:
        try:
            vf = rf.vectorFitting.VectorFitting(nw)
            print(f"[VF] Trying {n_real} real + {n_cmplx} complex poles...")
            vf.vector_fit(n_poles_real=n_real, n_poles_cmplx=n_cmplx)
            rms = vf.get_rms_error()
            print(f"[VF]   RMS = {rms:.4e}")
            if rms < best_rms: best_rms = rms; best_vf = vf
            if rms < target_error: print("[VF]   Target reached."); break
        except Exception as e:
            print(f"[VF]   Failed: {e}"); continue

    if best_vf is None:
        print("[VF ERROR] All configs failed."); return None, np.inf, None

    print(f"[VF] Best RMS = {best_rms:.4e}")
    Y_vf = {}; freqs_eval = np.asarray(freqs_eval, dtype=float)
    for i_r in range(4):
        for j_c in range(4):
            resp = best_vf.get_model_response(i_r, j_c, freqs=freqs_eval)
            for k, f in enumerate(freqs_eval):
                if f not in Y_vf: Y_vf[f] = np.zeros((4, 4), dtype=complex)
                Y_vf[f][i_r, j_c] = resp[k]
    return Y_vf, float(best_rms), best_vf

# ==================================================
# 6) MAIN SWEEP
# ==================================================
inject_freqs = list(np.arange(VALID_F_MIN, VALID_F_MAX + 1e-9, F_STEP_HZ, dtype=float))

eig_results = {}; det_results = {}; abs_det_results = {}; min_dist_results = {}
quality_log = {}; valid_for_curves_map = {}; valid_for_claims_map = {}
Ytot_raw = {}

with rtds.rscadfx.remote_connection() as app:
    case = app.open_case(casefile); print("Case opened.")
    safe_stop(case, settle_s=1.2); case.compile(); time.sleep(AFTER_COMPILE_WAIT)
    Hinj, Hpcc = build_handles(case)

    try:
        for f in inject_freqs:
            A = inj_amplitude(f); mode = "SLOW" if is_slow_mode(f) else "FAST"
            print(f"\n==== 4x4: f={f:6.2f} Hz | {mode} | A={A:.3f} | settle={settle_time(f):.2f}s ====")

            Y_tot, q, case, Hinj, Hpcc = measure_Ytot_4x4_at(app, case, Hinj, Hpcc, f, A)
            Ytot_raw[f] = Y_tot.copy()

            Rm = R12_pu if MUTUAL_ENABLE else 0.0; Xm = X12_pu if MUTUAL_ENABLE else 0.0
            Z_grid = build_Zgrid_4x4_pu(f, f0, R_grid_pu, X_grid_pu, R_grid_pu, X_grid_pu, Rm, Xm)
            Y_conv = Y_tot - np.linalg.inv(Z_grid); L = Z_grid @ Y_conv
            eigvals = np.linalg.eigvals(L); eig_results[f] = eigvals
            I4 = np.eye(4, dtype=complex); detIL = np.linalg.det(I4 + L)
            det_results[f] = detIL; abs_det_results[f] = float(np.abs(detIL))
            min_dist_results[f] = float(np.min(np.abs(1.0 + eigvals)))

            status, warn_list, v_curve, v_claim = classify_quality_4x4(q, f)
            q["status"] = status; q["warnings"] = warn_list
            q["valid_for_curves"] = bool(v_curve); q["valid_for_claims"] = bool(v_claim)
            quality_log[f] = q; valid_for_curves_map[f] = bool(v_curve)
            valid_for_claims_map[f] = bool(v_claim)

            print(f"[4x4] [{status}] {' '.join(warn_list) if warn_list else ''}")
            print(f"  |det|={abs_det_results[f]:.3e} | min|1+lam|={min_dist_results[f]:.3e} | curves={v_curve}")
            if status == "LOW_CONF":
                print(f"  [DIAG] drift_max={q['drift_deg_max']:.1f}° | "
                      f"harm_max={q['harm_max']:.3f} | snr_flag={q['snr_flag_any']}")

    finally:
        safe_stop(case, settle_s=1.0)
        try:
            for nm in ("inj1", "inj2"): Hinj[nm]["d_gain"].value = 0.0; Hinj[nm]["q_gain"].value = 0.0
        except: pass
        try: case.close()
        except: pass

# ==================================================
# 7) POST: RAW PLOTS
# ==================================================
SVG_DIR = "."   # folder to save SVGs; change to e.g. r"C:\results" if needed

def save_fig(name):
    import os
    plt.savefig(os.path.join(SVG_DIR, f"{name}.svg"), format="svg", bbox_inches="tight")
    print(f"Saved: {name}.svg")
    plt.close()

freqs_all = sorted(eig_results.keys())
freqs_curves = [f for f in freqs_all if valid_for_curves_map.get(f, False)]
freqs_claims = [f for f in freqs_all if valid_for_claims_map.get(f, False)]

if len(freqs_curves) < 2:
    raise RuntimeError("Too few curve-valid points.")

lam_branches = track_eig_branches({f: eig_results[f] for f in freqs_curves}, freqs_curves)

plt.figure()
for i, lam in enumerate(lam_branches):
    plt.plot(lam.real, lam.imag, "-o", markersize=2, label=f"lam{i+1}")
plt.scatter([-1], [0], color="red", zorder=5)
plt.title("4x4 GNC eigenloci [RAW, OK+WARN]"); plt.grid(True); plt.legend(); plt.tight_layout()
save_fig("gnc_4x4_eigenloci_raw")

plt.figure()
for i, lam in enumerate(lam_branches):
    plt.plot(lam.real, lam.imag, "-o", markersize=2, label=f"lam{i+1}")
plt.scatter([-1], [0], color="red", zorder=5)
plt.xlim(ZOOM_XLIM); plt.ylim(ZOOM_YLIM)
plt.title("4x4 GNC eigenloci zoom [RAW, OK+WARN]"); plt.grid(True); plt.legend(); plt.tight_layout()
save_fig("gnc_4x4_eigenloci_raw_zoom")

det_curve = np.array([det_results[f] for f in freqs_curves], dtype=complex)
plt.figure()
plt.plot(det_curve.real, det_curve.imag, "-o", markersize=2, label="det(I+L)")
plt.scatter([0], [0], color="red", zorder=5)
plt.title("4x4 Nyquist det(I+L) [RAW, OK+WARN]"); plt.grid(True); plt.legend(); plt.tight_layout()
save_fig("gnc_4x4_nyquist_raw")

if len(freqs_claims) >= 2:
    det_claims = np.array([det_results[f] for f in freqs_claims], dtype=complex)
    N_det, dphi_det = winding_number_of_complex_curve(det_claims)
    ww = check_winding_number_reliability(det_claims, np.array(freqs_claims))
else:
    N_det, dphi_det, ww = 0, 0.0, ["Too few claims points."]

print(f"\n==== 4x4 RAW RESULTS ====")
print(f"Curves: {len(freqs_curves)}/{len(freqs_all)} | Claims: {len(freqs_claims)}/{len(freqs_all)}")
print(f"Winding: N~{N_det}")
for w in ww: print(f"  WARNING: {w}")

if len(freqs_claims):
    min_dist_claims = np.array([min_dist_results[f] for f in freqs_claims])
    idx = int(np.argmin(min_dist_claims))
    print(f"Min RSMM = {min_dist_claims[idx]:.4e} at f = {freqs_claims[idx]:.2f} Hz")

# ==================================================
# 8) VECTOR FITTING (after RTDS disconnected)
# ==================================================
if VF_ENABLE:
    print("\n==== VECTOR FITTING (4x4) ====")
    freqs_vf_in = sorted([f for f in freqs_all if valid_for_curves_map.get(f, False)])
    Ytot_vf_in = np.array([Ytot_raw[f] for f in freqs_vf_in])

    if len(freqs_vf_in) < 5:
        print("[VF] Too few points. Skipping.")
    else:
        freqs_vf_eval = sorted(set(inject_freqs))
        if VF_EVAL_DENSE:
            freqs_vf_eval = sorted(set(freqs_vf_eval + list(np.arange(VALID_F_MIN, VALID_F_MAX+1e-9, 0.5))))
        freqs_vf_eval = np.array(freqs_vf_eval, dtype=float)

        Y_vf, vf_rms, vf_obj = vector_fit_Ytot_4x4(freqs_vf_in, Ytot_vf_in, freqs_vf_eval)

        if Y_vf is not None:
            eig_vf = {}; det_vf = {}; md_vf = {}
            Rm = R12_pu if MUTUAL_ENABLE else 0.0; Xm = X12_pu if MUTUAL_ENABLE else 0.0
            for f in freqs_vf_eval:
                Z_grid = build_Zgrid_4x4_pu(f, f0, R_grid_pu, X_grid_pu, R_grid_pu, X_grid_pu, Rm, Xm)
                L = Z_grid @ (Y_vf[f] - np.linalg.inv(Z_grid))
                eig_vf[f] = np.linalg.eigvals(L)
                det_vf[f] = np.linalg.det(np.eye(4, dtype=complex) + L)
                md_vf[f] = float(np.min(np.abs(1.0 + eig_vf[f])))

            fvs = sorted(eig_vf.keys())
            lam_vf = track_eig_branches(eig_vf, fvs)

            plt.figure()
            for i, lam in enumerate(lam_vf):
                plt.plot(lam.real, lam.imag, "-.", markersize=1, label=f"lam{i+1} VF")
            plt.scatter([-1], [0], color="red", zorder=5)
            plt.title("4x4 GNC eigenloci [VF]"); plt.grid(True); plt.legend(); plt.tight_layout()
            save_fig("gnc_4x4_eigenloci_vf")

            plt.figure()
            for i, lam in enumerate(lam_vf):
                plt.plot(lam.real, lam.imag, "-.", markersize=1, label=f"lam{i+1} VF")
            plt.scatter([-1], [0], color="red", zorder=5)
            plt.xlim(ZOOM_XLIM); plt.ylim(ZOOM_YLIM)
            plt.title("4x4 GNC eigenloci zoom [VF]"); plt.grid(True); plt.legend(); plt.tight_layout()
            save_fig("gnc_4x4_eigenloci_vf_zoom")

            det_vf_c = np.array([det_vf[f] for f in fvs], dtype=complex)
            plt.figure()
            plt.plot(det_vf_c.real, det_vf_c.imag, "-.", markersize=1, label="det(I+L) VF")
            plt.scatter([0], [0], color="red", zorder=5)
            plt.title("4x4 Nyquist det(I+L) [VF]"); plt.grid(True); plt.legend(); plt.tight_layout()
            save_fig("gnc_4x4_nyquist_vf")

            N_vf, dphi_vf = winding_number_of_complex_curve(det_vf_c)
            print(f"[VF] Winding: N~{N_vf}")

            print("[VF] Complete.")
        else:
            print("[VF] Failed - raw results only.")

# Save results
try:
    np.savez("gnc_4x4_model2_results.npz",
        freqs=np.array(freqs_all), eig_vals=np.array([eig_results[f] for f in freqs_all]),
        det_vals=np.array([det_results[f] for f in freqs_all]),
        min_dist=np.array([min_dist_results[f] for f in freqs_all]),
        curves=np.array([valid_for_curves_map[f] for f in freqs_all]),
        claims=np.array([valid_for_claims_map[f] for f in freqs_all]))
    print("Saved: gnc_4x4_model2_results.npz")
except Exception as e:
    print(f"Save failed: {e}")

print("\n==== 4x4 MODEL 2 COMPLETE ====")
