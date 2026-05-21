# -*- coding: utf-8 -*-
"""
GNC SISO (dq, 2-run D/Q) + robust DSP + pragmatic validity policy
+ CRASH RECOVERY (soft/hard/IOException from thesis Ch4)
+ VECTOR FITTING post-processing (runs after RTDS disconnected)
Scope: 5-100 Hz (1 Hz step)

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
# 0) USER CONFIG (identical to original)
# ==================================================
casefile = r"C:\Users\muass\OneDrive\Documenten\RSCAD\RTDS_USER_FX\fileman\01 HVDC\Harmonic Scan (Measured)\MMC_SCAN.rtfx"

VALID_F_MIN = 5.0
VALID_F_MAX = 100
F_STEP_HZ   = 1.0

Vbase_LL = 400e3
Sbase_3ph = 800e6
f0 = 50.0
omega0 = 2.0 * np.pi * f0

SCR = 3.5
XR  = 15.0

A0 = 1.0
A_MIN = 0.2
A_MAX = 3.0
F_KNEE = 20.0
ALPHA  = 0.7

F_SLOW_MAX = 12.0

TIMEOUT_S   = 30.0
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
STOP_SETTLE     = 1.0

# IOException recovery
IO_RETRY_COUNT   = 4
IO_RETRY_WAIT_S  = 3.0

# RSCAD plot window duration (set to match RSCAD's "Plot Update Window" setting).
# update_plots() fails if the simulation hasn't run for at least this long.
RSCAD_PLOT_WINDOW_S = 4.1   # RSCAD is configured to 4.0001 s; use 4.1 for margin

# Hard recovery waits
AFTER_REOPEN_WAIT  = 1.2
AFTER_COMPILE_WAIT = 2.5

DET_THR     = 1e-12
COND_THR    = 1e6
PINV_RCOND  = 1e-10

ZOOM_XLIM = (-5.0, 3.0)
ZOOM_YLIM = (-5.0, 5.0)

WINDING_MIN_DIST_WARN = 0.05

# Vector fitting
VF_ENABLE          = True
VF_TARGET_ERROR    = 0.01
VF_MAX_ORDER       = 60
VF_EVAL_DENSE      = False    # set True for smoother plots (slower)
VF_DENSE_STEP_HZ   = 0.5

pcc = {
    "name": "PCC1",
    "Vd_path": "Subsystem #1|CTLs|Vars|Vdpu11",
    "Vq_path": "Subsystem #1|CTLs|Vars|Vqpu11",
    "Id_path": "Subsystem #1|CTLs|Vars|Id",
    "Iq_path": "Subsystem #1|CTLs|Vars|Iq",
}

# LTI CHECK CONFIG
RUN_LTI_CHECK = False
LTI_EVERY_N_HZ = 10.0
LTI_INCLUDE_BAND_EDGES = True
LTI_INCLUDE_SPECIAL = [10.0, 50.0]
# LTI_TEST_FREQS = [10, 30, 50, 70, 90]  # uncomment for explicit list

LTI_K_TINY  = 0.20
LTI_K_SMALL = 0.35
LTI_K_LARGE = 0.55
LTI_THR_NORM = 0.15
LTI_THR_MAXE = 0.25
LTI_MONO_R_MIN = 0.30
LTI_MONO_R_MAX = 3.00
LTI_REPEATS_PER_AMP = 1
LTI_DROP_WORST = 1
LTI_INTER_RUN_COOLDOWN_S = 0.4
LTI_WARMUP_EXTRA_S_LOW = 8.0
LTI_WARMUP_EXTRA_S_HIGH = 1.5

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
print(f"Zbase = {Zbase:.6f} ohm, Ibase = {Ibase:.3f} A")
print(f"SCR={SCR}, X/R={XR} -> R_pu={R_grid_pu:.6e}, X_pu={X_grid_pu:.6e}")
print("====================================\n")

# ==================================================
# 2) HELPERS (original + crash recovery additions)
# ==================================================
def is_slow_mode(f_hz): return float(f_hz) <= float(F_SLOW_MAX)

def inj_amplitude(f_hz):
    f_hz = float(f_hz)
    if f_hz <= 0: return float(np.clip(A0, A_MIN, A_MAX))
    scale = (f_hz / F_KNEE) ** ALPHA if f_hz > F_KNEE else 1.0
    return float(np.clip(A0 * scale, A_MIN, A_MAX))

def settle_time(f_hz):
    f_hz = float(f_hz)
    # Minimum settle must exceed RSCAD_PLOT_WINDOW_S so that update_plots()
    # finds a full window of simulation data on the very first call.
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
                # A double back-to-back call was causing all retries to fail.
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

def build_Zgrid_pu(f_hz, R_pu, X_pu, f0_hz):
    w_pu = float(f_hz) / float(f0_hz)
    Zdd = R_pu + 1j * w_pu * X_pu
    return np.array([[Zdd, -X_pu], [+X_pu, Zdd]], dtype=complex)

def pulse_reset(pb, pb1, pulse_s=0.10):
    pb.position = 1; pb1.position = 1; time.sleep(pulse_s)
    pb.position = 0; pb1.position = 0; time.sleep(0.05)

def safe_stop(case, settle_s=STOP_SETTLE):
    try: case.stop()
    except: pass
    time.sleep(settle_s)

def is_stale_object_error(e):
    s = str(e).lower()
    return ("invalid processing object" in s) or ("cannot be found" in s) or ("remote invocation error" in s)

def is_unable_to_start_case_error(e):
    s = str(e).lower()
    return ("unable to start case" in s) or ("unable to start" in s)

def is_io_exception(e):
    s = str(e).lower()
    return ("ioexception" in s) or ("io exception" in s) or ("plot" in s and "ack" in s)

def rebuild_handles(case):
    H = {}
    H["Finj"]   = case.get_object_by_name("Finj", "slider")
    H["d_gain"] = case.get_object_by_name("d_gain", "slider")
    H["q_gain"] = case.get_object_by_name("q_gain", "slider")
    H["PB"]     = case.get_object_by_name("PB", "button")
    H["PB1"]    = case.get_object_by_name("PB1", "button")
    H["sigVd"]  = case.get_signal(pcc["Vd_path"])
    H["sigVq"]  = case.get_signal(pcc["Vq_path"])
    H["sigId"]  = case.get_signal(pcc["Id_path"])
    H["sigIq"]  = case.get_signal(pcc["Iq_path"])
    return H

def recover_by_reopen(app, case):
    print("[HARD RECOVERY] Reopening case and rebuilding handles...")
    try: safe_stop(case, settle_s=STOP_SETTLE)
    except: pass
    try: case.close()
    except: pass
    time.sleep(0.9)
    case = app.open_case(casefile)
    safe_stop(case, settle_s=1.2)
    case.compile()
    time.sleep(AFTER_COMPILE_WAIT)
    H = rebuild_handles(case)
    time.sleep(AFTER_REOPEN_WAIT)
    print("[HARD RECOVERY] Complete.")
    return case, H

def safe_run(app, case, H, retries=RUN_RETRIES, base_wait=RUN_BASE_WAIT):
    last = None
    for k in range(1, retries + 1):
        try:
            case.run(); time.sleep(0.25); return case, H
        except RSCADError as e:
            last = e
            if is_stale_object_error(e) or is_unable_to_start_case_error(e):
                print(f"[RUN FAIL {k}/{retries}] {e} -> HARD RECOVERY")
                case, H = recover_by_reopen(app, case); continue
            if is_io_exception(e):
                print(f"[RUN FAIL {k}/{retries}] IOException -> flush, wait {IO_RETRY_WAIT_S}s")
                try: case.update_plots()
                except: pass
                time.sleep(IO_RETRY_WAIT_S); continue
            wait = base_wait * (1.5 ** (k-1)) + random.uniform(0.0, 0.35)
            print(f"[RUN FAIL {k}/{retries}] {e} -> wait {wait:.2f}s, STOP, retry")
            safe_stop(case, settle_s=wait)
    print(f"[RUN FAIL] All {retries} retries exhausted -> final HARD RECOVERY")
    case, H = recover_by_reopen(app, case)
    return case, H

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

def cabs(z): return float(np.abs(z))

def dominance_metrics(Vd_d, Vq_d, Vd_q, Vq_q):
    eps = 1e-30
    return cabs(Vd_d)/(cabs(Vq_d)+eps), cabs(Vq_q)/(cabs(Vd_q)+eps)

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

# ==================================================
# 3) QUALITY CLASSIFICATION
# ==================================================
def classify_quality(q, f_inj):
    warn = []; f_inj = float(f_inj); slow = is_slow_mode(f_inj)
    drift_warn = DRIFT_WARN_DEG_LO if slow else DRIFT_WARN_DEG_HI
    drift_lowc = DRIFT_LOWCONF_DEG_LO if slow else DRIFT_LOWCONF_DEG_HI
    drift_abs = max(abs(q["drift_deg_D"]), abs(q["drift_deg_Q"]))
    hmax = max(q["H2v_D"], q["H2v_Q"], q["H3v_D"], q["H3v_Q"])
    snr_flag = (q["Vmag_D"] < SNR_MIN_MAG_V) or (q["Vmag_Q"] < SNR_MIN_MAG_V) or \
               (q["Imag_D"] < SNR_MIN_MAG_I) or (q["Imag_Q"] < SNR_MIN_MAG_I)
    inv_flag = bool(q["used_pinv"])

    if hmax > HARM_WARN_RATIO: warn.append("HARMONICS_WARN")
    if drift_abs > drift_warn: warn.append("DRIFT_WARN")
    if snr_flag: warn.append("SNR_WARN")
    if inv_flag: warn.append("INV_WARN")

    low_conf = snr_flag or (drift_abs > drift_lowc) or (hmax > HARM_LOWCONF_RATIO)
    if inv_flag and (snr_flag or (drift_abs > drift_lowc) or (hmax > HARM_LOWCONF_RATIO)):
        low_conf = True

    status = "LOW_CONF" if low_conf else ("WARN" if warn else "OK")
    valid_for_curves = (status in ("OK", "WARN"))
    valid_for_claims = (status in ("OK", "WARN")) and (status != "LOW_CONF")
    return status, warn, bool(valid_for_curves), bool(valid_for_claims)

# ==================================================
# 4) CORE MEASUREMENT (2-run dq, with recovery)
# ==================================================
def measure_Ytot_at(app, case, H, f_inj, A_slider, force_f_ref=None, disable_f_search=False):
    f_inj = float(f_inj); A_slider = float(A_slider); Ts = settle_time(f_inj)
    sig_handles = {"Vd": H["sigVd"], "Vq": H["sigVq"], "Id": H["sigId"], "Iq": H["sigIq"]}

    Vd_ph = np.zeros(2, dtype=complex); Vq_ph = np.zeros(2, dtype=complex)
    Id_ph = np.zeros(2, dtype=complex); Iq_ph = np.zeros(2, dtype=complex)
    drift_deg = {"D": None, "Q": None}; h2v = {"D": None, "Q": None}
    h3v = {"D": None, "Q": None}; vmag = {"D": None, "Q": None}
    imag = {"D": None, "Q": None}; f_used = {"D": float(f_inj), "Q": float(f_inj)}
    fs_est = None; dt_rel_std = None

    for mode_idx, (d_gain, q_gain) in enumerate([(A_slider, 0.0), (0.0, A_slider)]):
        tag = "D" if mode_idx == 0 else "Q"
        safe_stop(case, settle_s=STOP_SETTLE); time.sleep(0.08)
        H["Finj"].value = float(f_inj)
        H["d_gain"].value = float(d_gain); H["q_gain"].value = float(q_gain)
        time.sleep(0.08)
        pulse_reset(H["PB"], H["PB1"], pulse_s=0.10)
        case, H = safe_run(app, case, H)
        sig_handles = {"Vd": H["sigVd"], "Vq": H["sigVq"], "Id": H["sigId"], "Iq": H["sigIq"]}
        time.sleep(Ts)
        t, data = get_timeseries_consistent(case, sig_handles, timeout_s=TIMEOUT_S, min_samples=MIN_SAMPLES, poll_s=POLL_S)
        fs_est, dt_rel_std = estimate_fs(t)
        t2, data2 = select_analysis_window(t, data, f_inj)

        if force_f_ref is not None:
            f_ref = float(force_f_ref)
        else:
            f_ref = float(f_inj)
            dVd = phase_drift_deg(data2["Vd"], t2, f_ref)
            drift_warn = DRIFT_WARN_DEG_LO if is_slow_mode(f_inj) else DRIFT_WARN_DEG_HI
            if (not disable_f_search) and F_SEARCH_ON_DRIFT and abs(dVd) > drift_warn:
                fpk = spectrum_peak_near(data2["Vd"], t2, f_ref, bw_hz=F_SEARCH_BW_HZ)
                if abs(phase_drift_deg(data2["Vd"], t2, fpk)) < abs(dVd):
                    f_ref = float(fpk)

        f_used[tag] = float(f_ref)
        drift_deg[tag] = float(phase_drift_deg(data2["Vd"], t2, f_ref))
        H2v, H3v, Vd1 = harmonic_ratios(data2["Vd"], t2, f_ref)
        _, _, Id1 = harmonic_ratios(data2["Id"], t2, f_ref)
        h2v[tag], h3v[tag] = H2v, H3v
        vmag[tag] = float(np.abs(Vd1)); imag[tag] = float(np.abs(Id1))
        Vd_ph[mode_idx] = lockin_phasor(data2["Vd"], t2, f_ref)
        Vq_ph[mode_idx] = lockin_phasor(data2["Vq"], t2, f_ref)
        Id_ph[mode_idx] = lockin_phasor(data2["Id"], t2, f_ref)
        Iq_ph[mode_idx] = lockin_phasor(data2["Iq"], t2, f_ref)

    safe_stop(case, settle_s=STOP_SETTLE)
    H["d_gain"].value = 0.0; H["q_gain"].value = 0.0

    V_exc = np.array([[Vd_ph[0], Vd_ph[1]], [Vq_ph[0], Vq_ph[1]]], dtype=complex)
    I_exc = np.array([[Id_ph[0], Id_ph[1]], [Iq_ph[0], Iq_ph[1]]], dtype=complex)
    d_dom, q_dom = dominance_metrics(Vd_ph[0], Vq_ph[0], Vd_ph[1], Vq_ph[1])
    V_inv, detV, condV, used_pinv = inv_or_pinv(V_exc)
    Y_tot = I_exc @ V_inv

    q = {"A_slider": float(A_slider), "settle_s": float(Ts),
         "fs_est": float(fs_est) if fs_est else np.nan,
         "dt_rel_std": float(dt_rel_std) if dt_rel_std else np.nan,
         "f_used_D": float(f_used["D"]), "f_used_Q": float(f_used["Q"]),
         "drift_deg_D": float(drift_deg["D"]), "drift_deg_Q": float(drift_deg["Q"]),
         "H2v_D": float(h2v["D"]), "H3v_D": float(h3v["D"]),
         "H2v_Q": float(h2v["Q"]), "H3v_Q": float(h3v["Q"]),
         "Vmag_D": float(vmag["D"]), "Vmag_Q": float(vmag["Q"]),
         "Imag_D": float(imag["D"]), "Imag_Q": float(imag["Q"]),
         "detV": detV, "condV": float(condV),
         "d_dom": float(d_dom), "q_dom": float(q_dom), "used_pinv": bool(used_pinv)}
    return Y_tot, q, case, H

# ==================================================
# 5) LTI CHECK
# ==================================================
def robust_avg_Y(Y_list):
    if len(Y_list) == 1: return Y_list[0]
    Ys = np.array(Y_list); Y_med = np.median(Ys, axis=0)
    d = [np.linalg.norm(Y - Y_med) for Y in Y_list]
    idx = np.argsort(d); keep = idx[:max(1, len(Y_list) - LTI_DROP_WORST)]
    return np.mean([Y_list[i] for i in keep], axis=0)

def lti_check_at_freq(app, case, H, ft):
    ft = float(ft); A_base = inj_amplitude(ft)
    A_tiny  = float(np.clip(LTI_K_TINY  * A_base, A_MIN, A_MAX))
    A_small = float(np.clip(LTI_K_SMALL * A_base, A_MIN, A_MAX))
    A_large = float(np.clip(LTI_K_LARGE * A_base, A_MIN, A_MAX))
    warm_extra = LTI_WARMUP_EXTRA_S_LOW if is_slow_mode(ft) else LTI_WARMUP_EXTRA_S_HIGH

    def measure_repeat(A):
        nonlocal case, H
        Ys, qs = [], []
        for _ in range(LTI_REPEATS_PER_AMP):
            Y, q, case, H = measure_Ytot_at(app, case, H, ft, A, force_f_ref=ft, disable_f_search=True)
            Ys.append(Y); qs.append(q); time.sleep(LTI_INTER_RUN_COOLDOWN_S)
        Yavg = robust_avg_Y(Ys)
        drifts = np.array([abs(qq["drift_deg_D"]) + abs(qq["drift_deg_Q"]) for qq in qs])
        return Yavg, qs[int(np.argsort(drifts)[len(drifts)//2])]

    safe_stop(case, settle_s=STOP_SETTLE)
    H["Finj"].value = float(ft); H["d_gain"].value = float(A_small); H["q_gain"].value = 0.0
    pulse_reset(H["PB"], H["PB1"], pulse_s=0.10)
    case, H = safe_run(app, case, H)
    time.sleep(settle_time(ft) + warm_extra)
    safe_stop(case, settle_s=STOP_SETTLE)
    H["d_gain"].value = 0.0; H["q_gain"].value = 0.0; time.sleep(0.2)

    YT, qT = measure_repeat(A_tiny)
    YS, qS = measure_repeat(A_small)
    YL, qL = measure_repeat(A_large)

    nS = np.linalg.norm(YS) + 1e-30
    dST_norm = float(np.linalg.norm(YS - YT) / nS)
    dLS_norm = float(np.linalg.norm(YL - YS) / nS)
    dST_max = float(np.max(np.abs(YS - YT) / (np.abs(YS) + 1e-30)))
    dLS_max = float(np.max(np.abs(YL - YS) / (np.abs(YS) + 1e-30)))
    r = float((np.linalg.norm(YL - YS) + 1e-30) / (np.linalg.norm(YS - YT) + 1e-30))

    PASS = bool((dST_norm <= LTI_THR_NORM) and (dLS_norm <= LTI_THR_NORM) and
                (dST_max <= LTI_THR_MAXE) and (dLS_max <= LTI_THR_MAXE) and
                (LTI_MONO_R_MIN <= r <= LTI_MONO_R_MAX))

    return {"f": ft, "PASS": PASS, "ST_norm": dST_norm, "ST_max": dST_max,
            "LS_norm": dLS_norm, "LS_max": dLS_max, "mono_r": r}, case, H

def build_lti_freq_list():
    if "LTI_TEST_FREQS" in globals():
        freqs = [float(x) for x in globals()["LTI_TEST_FREQS"]]
    else:
        freqs = list(np.arange(VALID_F_MIN, VALID_F_MAX + 1e-9, LTI_EVERY_N_HZ))
        if LTI_INCLUDE_BAND_EDGES: freqs += [VALID_F_MIN, VALID_F_MAX]
        freqs += [float(x) for x in LTI_INCLUDE_SPECIAL]
    freqs = [f for f in freqs if VALID_F_MIN - 1e-9 <= f <= VALID_F_MAX + 1e-9]
    return sorted(set([round(float(f), 6) for f in freqs]))

# ==================================================
# 6) VECTOR FITTING (runs after RTDS disconnected)
# ==================================================
def vector_fit_Ytot_2x2(freqs_ok, Ytot_ok, freqs_eval, target_error=VF_TARGET_ERROR, max_order=VF_MAX_ORDER):
    """
    Robust vector fitting: tries direct vector_fit() with increasing pole counts.
    Never uses auto_fit (which can hang indefinitely).
    Picks the best fit from the tried pole counts.
    """
    import skrf as rf
    freqs_ok = np.asarray(freqs_ok, dtype=float)
    s_data = np.zeros((len(freqs_ok), 2, 2), dtype=complex)
    for i in range(len(freqs_ok)): s_data[i] = Ytot_ok[i]
    nw = rf.Network(frequency=rf.Frequency.from_f(freqs_ok, unit='Hz'), s=s_data)

    # Try increasing pole counts, pick the best
    pole_configs = [
        (2, 4),   # 2 real + 4 complex = 10 poles total
        (3, 6),   # 3 real + 6 complex = 15 poles total
        (4, 8),   # 4 real + 8 complex = 20 poles total
        (5, 10),  # 5 real + 10 complex = 25 poles total
    ]

    best_vf = None
    best_rms = np.inf

    for n_real, n_cmplx in pole_configs:
        try:
            vf = rf.vectorFitting.VectorFitting(nw)
            print(f"[VF] Trying {n_real} real + {n_cmplx} complex poles...")
            vf.vector_fit(n_poles_real=n_real, n_poles_cmplx=n_cmplx)
            rms = vf.get_rms_error()
            print(f"[VF]   RMS = {rms:.4e}")
            if rms < best_rms:
                best_rms = rms
                best_vf = vf
            if rms < target_error:
                print(f"[VF]   Target reached, stopping.")
                break
        except Exception as e:
            print(f"[VF]   Failed: {e}")
            continue

    if best_vf is None:
        print("[VF ERROR] All pole configs failed.")
        return None, np.inf, None

    print(f"[VF] Best RMS = {best_rms:.4e}")

    # Evaluate at requested frequencies
    Y_vf = {}
    freqs_eval = np.asarray(freqs_eval, dtype=float)
    # Batch evaluate for speed
    for i_row in range(2):
        for j_col in range(2):
            resp = best_vf.get_model_response(i_row, j_col, freqs=freqs_eval)
            for k, f in enumerate(freqs_eval):
                if f not in Y_vf:
                    Y_vf[f] = np.zeros((2, 2), dtype=complex)
                Y_vf[f][i_row, j_col] = resp[k]

    return Y_vf, float(best_rms), best_vf

# ==================================================
# 7) MAIN SWEEP
# ==================================================
inject_freqs = list(np.arange(VALID_F_MIN, VALID_F_MAX + 1e-9, F_STEP_HZ, dtype=float))
lti_freqs = build_lti_freq_list()

eig_results = {}; det_results = {}; abs_det_results = {}; min_dist_results = {}
quality_log = {}; valid_for_curves_map = {}; valid_for_claims_map = {}
Ytot_raw = {}; lti_summary = []

with rtds.rscadfx.remote_connection() as app:
    case = app.open_case(casefile)
    print("Case opened.")
    safe_stop(case, settle_s=1.2)
    case.compile()
    H = rebuild_handles(case)

    try:
        if RUN_LTI_CHECK:
            print("\n==== LTI SANITY CHECK ====")
            print(f"Frequencies: {lti_freqs}")
            for ft in lti_freqs:
                res, case, H = lti_check_at_freq(app, case, H, ft)
                lti_summary.append(res)
                print(f"[LTI] f={res['f']:6.1f} | {'PASS' if res['PASS'] else 'FAIL'} | mono r={res['mono_r']:.2f}")
            print("====\n")

        for f in inject_freqs:
            A = inj_amplitude(f); mode = "SLOW" if is_slow_mode(f) else "FAST"
            print(f"\n==== f={f:6.2f} Hz | {mode} | A={A:.3f} | settle={settle_time(f):.2f}s ====")

            Y_tot, q, case, H = measure_Ytot_at(app, case, H, f, A)
            Ytot_raw[f] = Y_tot.copy()

            Z_grid = build_Zgrid_pu(f, R_grid_pu, X_grid_pu, f0)
            Y_conv = Y_tot - np.linalg.inv(Z_grid)
            L = Z_grid @ Y_conv
            eigvals = np.linalg.eigvals(L)
            eig_results[f] = eigvals
            I2 = np.eye(2, dtype=complex)
            detIL = np.linalg.det(I2 + L)
            det_results[f] = detIL
            abs_det_results[f] = float(np.abs(detIL))
            min_dist_results[f] = float(np.min(np.abs(1.0 + eigvals)))

            status, warn_list, v_curve, v_claim = classify_quality(q, f)
            q["status"] = status; q["warnings"] = warn_list
            q["valid_for_curves"] = bool(v_curve); q["valid_for_claims"] = bool(v_claim)
            quality_log[f] = q
            valid_for_curves_map[f] = bool(v_curve)
            valid_for_claims_map[f] = bool(v_claim)

            print(f"[{pcc['name']}] [{status}] {' '.join(warn_list) if warn_list else ''}")
            print(f"  |det|={abs_det_results[f]:.3e} | min|1+lam|={min_dist_results[f]:.3e} | curves={v_curve}")
            if status == "LOW_CONF":
                print(f"  [DIAG] drift_D={q['drift_deg_D']:+.1f}° drift_Q={q['drift_deg_Q']:+.1f}° | "
                      f"H2={max(q['H2v_D'],q['H2v_Q']):.3f} H3={max(q['H3v_D'],q['H3v_Q']):.3f} | "
                      f"Vmag={min(q['Vmag_D'],q['Vmag_Q']):.2e} Imag={min(q['Imag_D'],q['Imag_Q']):.2e}")

    finally:
        safe_stop(case, settle_s=1.0)
        try: H["d_gain"].value = 0.0; H["q_gain"].value = 0.0
        except: pass
        try: case.close()
        except: pass

# ==================================================
# 8) POST: RAW PLOTS (after RTDS disconnected)
# ==================================================
SVG_DIR = "."   # folder to save SVGs; change to e.g. r"C:\results" if needed

def save_fig(name):
    import os
    plt.savefig(os.path.join(SVG_DIR, f"{name}.svg"), format="svg", bbox_inches="tight")
    print(f"Saved: {name}.svg")

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
plt.title("GNC eigenloci [RAW, OK+WARN]"); plt.grid(True); plt.legend(); plt.tight_layout(); save_fig("gnc_eigenloci_raw"); plt.show()

plt.figure()
for i, lam in enumerate(lam_branches):
    plt.plot(lam.real, lam.imag, "-o", markersize=2, label=f"lam{i+1}")
plt.scatter([-1], [0], color="red", zorder=5)
plt.xlim(ZOOM_XLIM); plt.ylim(ZOOM_YLIM)
plt.title("GNC eigenloci zoom [RAW, OK+WARN]"); plt.grid(True); plt.legend(); plt.tight_layout(); save_fig("gnc_eigenloci_raw_zoom"); plt.show()

det_curve = np.array([det_results[f] for f in freqs_curves], dtype=complex)
plt.figure()
plt.plot(det_curve.real, det_curve.imag, "-o", markersize=2, label="det(I+L)")
plt.scatter([0], [0], color="red", zorder=5)
plt.title("Nyquist det(I+L) [RAW, OK+WARN]"); plt.grid(True); plt.legend(); plt.tight_layout(); save_fig("gnc_nyquist_raw"); plt.show()

if len(freqs_claims) >= 2:
    det_claims = np.array([det_results[f] for f in freqs_claims], dtype=complex)
    N_det, dphi_det = winding_number_of_complex_curve(det_claims)
    ww = check_winding_number_reliability(det_claims, np.array(freqs_claims))
else:
    N_det, dphi_det, ww = 0, 0.0, ["Too few claims points."]

print(f"\n==== GNC RESULTS ====")
print(f"Curves: {len(freqs_curves)}/{len(freqs_all)} | Claims: {len(freqs_claims)}/{len(freqs_all)}")
print(f"Winding: N~{N_det}")
for w in ww: print(f"  WARNING: {w}")

if len(freqs_claims):
    min_dist_claims = np.array([min_dist_results[f] for f in freqs_claims])
    idx = int(np.argmin(min_dist_claims))
    print(f"Min RSMM = {min_dist_claims[idx]:.4e} at f = {freqs_claims[idx]:.2f} Hz")

print(f"\n==== LTI SUMMARY ====")
if RUN_LTI_CHECK and lti_summary:
    print(f"{sum(1 for r in lti_summary if r['PASS'])}/{len(lti_summary)} PASS")
    for r in lti_summary:
        print(f"  f={r['f']:6.1f} | {'PASS' if r['PASS'] else 'FAIL'} | r={r['mono_r']:.2f}")

# ==================================================
# 9) VECTOR FITTING (after RTDS disconnected - safe)
# ==================================================
if VF_ENABLE:
    print("\n==== VECTOR FITTING ====")
    freqs_vf_in = sorted([f for f in freqs_all if valid_for_curves_map.get(f, False)])
    Ytot_vf_in = np.array([Ytot_raw[f] for f in freqs_vf_in])

    if len(freqs_vf_in) < 5:
        print("[VF] Too few points. Skipping.")
    else:
        freqs_vf_eval = sorted(set(inject_freqs))
        if VF_EVAL_DENSE:
            freqs_vf_eval = sorted(set(freqs_vf_eval + list(np.arange(VALID_F_MIN, VALID_F_MAX+1e-9, VF_DENSE_STEP_HZ))))
        freqs_vf_eval = np.array(freqs_vf_eval, dtype=float)

        Y_vf, vf_rms, vf_obj = vector_fit_Ytot_2x2(freqs_vf_in, Ytot_vf_in, freqs_vf_eval)

        if Y_vf is not None:
            eig_vf = {}; det_vf = {}; md_vf = {}
            for f in freqs_vf_eval:
                Z_grid = build_Zgrid_pu(f, R_grid_pu, X_grid_pu, f0)
                L = Z_grid @ (Y_vf[f] - np.linalg.inv(Z_grid))
                eig_vf[f] = np.linalg.eigvals(L)
                det_vf[f] = np.linalg.det(np.eye(2, dtype=complex) + L)
                md_vf[f] = float(np.min(np.abs(1.0 + eig_vf[f])))

            fvs = sorted(eig_vf.keys())
            lam_vf = track_eig_branches(eig_vf, fvs)

            plt.figure()
            for i, lam in enumerate(lam_vf):
                plt.plot(lam.real, lam.imag, "-.", markersize=1, label=f"lam{i+1} VF")
            plt.scatter([-1], [0], color="red", zorder=5)
            plt.title("GNC eigenloci [VF]"); plt.grid(True); plt.legend(); plt.tight_layout(); save_fig("gnc_eigenloci_vf"); plt.show()

            plt.figure()
            for i, lam in enumerate(lam_vf):
                plt.plot(lam.real, lam.imag, "-.", markersize=1, label=f"lam{i+1} VF")
            plt.scatter([-1], [0], color="red", zorder=5)
            plt.xlim(ZOOM_XLIM); plt.ylim(ZOOM_YLIM)
            plt.title("GNC eigenloci zoom [VF]"); plt.grid(True); plt.legend(); plt.tight_layout(); save_fig("gnc_eigenloci_vf_zoom"); plt.show()

            det_vf_c = np.array([det_vf[f] for f in fvs], dtype=complex)
            plt.figure()
            plt.plot(det_vf_c.real, det_vf_c.imag, "-.", markersize=1, label="det(I+L) VF")
            plt.scatter([0], [0], color="red", zorder=5)
            plt.title("Nyquist det(I+L) [VF]"); plt.grid(True); plt.legend(); plt.tight_layout(); save_fig("gnc_nyquist_vf"); plt.show()

            N_vf, dphi_vf = winding_number_of_complex_curve(det_vf_c)
            print(f"[VF] Winding: N~{N_vf}")

            print("[VF] Complete.")
        else:
            print("[VF] Failed - raw results only.")

# Save after everything (no RTDS connection)
try:
    np.savez("gnc_siso_results.npz",
        freqs=np.array(freqs_all), eig_vals=np.array([eig_results[f] for f in freqs_all]),
        det_vals=np.array([det_results[f] for f in freqs_all]),
        min_dist=np.array([min_dist_results[f] for f in freqs_all]),
        curves=np.array([valid_for_curves_map[f] for f in freqs_all]),
        claims=np.array([valid_for_claims_map[f] for f in freqs_all]))
    print("Saved: gnc_siso_results.npz")
except Exception as e:
    print(f"Save failed: {e}")

print("\n==== COMPLETE ====")