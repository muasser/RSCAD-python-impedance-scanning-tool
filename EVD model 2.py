# -*- coding: utf-8 -*-
"""
EVD pipeline for TWO PCCs (two injections) -> 2n x 2n with n=2 => 4x4 dq-domain
Scope: 5–100 Hz (1 Hz step)

This script reuses EXACTLY the same RTDS measuring pipeline as your 4x4 GNC script:
- 4 experiments per frequency (inj1 d/q, inj2 d/q)
- builds V_exc (4x4) and I_exc (4x4)
- computes measured Y_tot(4x4) = I_exc * inv(V_exc)

Then it performs KU Leuven-style EVD diagnostics on the CLOSED-LOOP IMPEDANCE:
    Z_cl(jw) = (Y_tot(jw))^{-1}

Outputs:
✅ eigenloci of eigenvalues of Z_cl(jw)
✅ critical frequency identification via |lambda_max(jw)|
✅ bus participation factors at f* using left+right eigenvectors

Quality gating:
- same metrics as your 4x4 code (harmonics, drift, SNR, inversion-flag)
- curves (eigenloci + |lambda_max|) exclude LOW_CONF by default
- participation factors computed at f* (only if that point is curve-valid)

NOTE:
- No Z_grid / Y_grid / Y_conv / L = Zg*Yc here (as requested).
- You can switch to EVD on Y_tot directly, but Z-tool-like workflow uses Z_cl.

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
# 0) USER CONFIG (same structure as your 4x4 GNC code)
# ==================================================
casefile = r"C:\Users\muass\OneDrive\Documenten\RSCAD\RTDS_USER_FX\fileman\01 HVDC\Harmonic Scan (Measured)\MMC_SCAN.rtfx"

# Sweep band
VALID_F_MIN = 5.0
VALID_F_MAX = 100
F_STEP_HZ   = 1.0

# Base (kept, even if not used directly)
Vbase_LL  = 525e3
Sbase_3ph = 1000e6
f0 = 50.0

# Injection amplitude scheduling (slider scaling; your model converts to A SI)
A0    = 1.0
A_MIN = 0.2
A_MAX = 3.0
F_KNEE = 20.0
ALPHA  = 0.7

# Low-frequency boundary for slow-mode
F_SLOW_MAX = 12.0

# Plot buffer polling
TIMEOUT_S   = 60.0
MIN_SAMPLES = 1400
POLL_S      = 0.7

# DSP: analysis windows (tail)
SLOW_N_CYCLES = 20
FAST_N_CYCLES = 12
SLOW_MIN_S    = 1.8
FAST_MIN_S    = 0.8
SLOW_MAX_S    = 3.5
FAST_MAX_S    = 1.8

# Quality thresholds
HARM_WARN_RATIO     = 0.08   # 2% => WARN
HARM_LOWCONF_RATIO  = 0.25   # 10% => LOW_CONF

DRIFT_WARN_DEG_HI = 30.0
DRIFT_WARN_DEG_LO = 80.0
DRIFT_LOWCONF_DEG_HI = 80.0
DRIFT_LOWCONF_DEG_LO = 140.0

SNR_MIN_MAG_V = 1e-4
SNR_MIN_MAG_I = 1e-5

# Peak-search settings
F_SEARCH_ON_DRIFT = True
F_SEARCH_BW_HZ    = 0.8

# Robustness
RUN_RETRIES     = 7
RUN_BASE_WAIT   = 0.9
STOP_SETTLE     = 1.7

# IOException recovery
IO_RETRY_COUNT  = 6
IO_RETRY_WAIT_S = 8.0

# RSCAD plot window — update_plots() needs sim to have run this long first
RSCAD_PLOT_WINDOW_S = 4.1

# Extra waits that reduce "Unable to start case" in mid band
POST_PULSE_WAIT = 0.18
POST_RUN_WAIT   = 0.35
AFTER_REOPEN_WAIT  = 1.2
AFTER_COMPILE_WAIT = 2.5

# Numerics / inversion robustness (same knobs)
DET_THR     = 1e-12
COND_THR    = 1e6
PINV_RCOND  = 1e-10

# EVD settings
EIG_ON_ZCL = True          # True: eig(Z_cl=inv(Y_tot)), False: eig(Y_tot)
PLOT_EIGENLOCI = True
PLOT_LAMMAX    = True
PLOT_PARTICIPATION_BAR = True

# Branch tracking
TRACK_BRANCHES = True

# If a stale-handle error occurs, how many times to refresh+retry that experiment
STALE_RETRIES_PER_EXPERIMENT = 2


# ==================================================
# 0a) TWO PCC DEFINITIONS (signals)
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

# ==================================================
# 0b) TWO INJECTIONS (sliders + buttons)
# ==================================================
INJECTIONS = [
    {
        "name": "inj1",
        "Finj_slider": "Finj",
        "d_gain_slider": "d_gain",
        "q_gain_slider": "q_gain",
        "PB_d": "PB",
        "PB_q": "PB1",
    },
    {
        "name": "inj2",
        "Finj_slider": "Finj1",
        "d_gain_slider": "d_gain1",
        "q_gain_slider": "q_gain1",
        "PB_d": "PB2",
        "PB_q": "PB3",
    },
]


# ==================================================
# 1) HELPERS (copied/adapted from your 4x4 script)
# ==================================================
def is_slow_mode(f_hz: float) -> bool:
    return float(f_hz) <= float(F_SLOW_MAX)

def inj_amplitude(f_hz):
    f_hz = float(f_hz)
    if f_hz <= 0:
        return float(np.clip(A0, A_MIN, A_MAX))
    scale = 1.0
    if f_hz > F_KNEE:
        scale = (f_hz / F_KNEE) ** ALPHA
    A = A0 * scale
    return float(np.clip(A, A_MIN, A_MAX))

def settle_time(f_hz):
    f_hz = float(f_hz)
    if is_slow_mode(f_hz):
        Ts = max(RSCAD_PLOT_WINDOW_S + 0.2, min(7.0, 14.0 / max(f_hz, 0.7)))
    else:
        Ts = max(RSCAD_PLOT_WINDOW_S + 0.2, min(7.0, 8.0 / max(f_hz, 2.0)))
    return float(Ts)

def analysis_window_params(f_hz):
    if is_slow_mode(f_hz):
        return SLOW_N_CYCLES, SLOW_MIN_S, SLOW_MAX_S
    else:
        return FAST_N_CYCLES, FAST_MIN_S, FAST_MAX_S

def lockin_phasor(x, t, f_ref):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(x) < 32:
        raise RuntimeError("Too few samples for lock-in.")
    x_ac = x - np.mean(x)
    N = len(x_ac)
    w = np.hanning(N)
    ww = np.mean(w) if np.mean(w) != 0 else 1.0
    ref = np.exp(-1j * 2.0 * np.pi * float(f_ref) * t)
    ph = (2.0 / N) * np.sum((x_ac * w) * ref) / ww
    return ph

def select_analysis_window(t, data_dict, f_hz):
    t = np.asarray(t)
    T_avail = float(t[-1] - t[0])

    Ncy, Tmin, Tmax = analysis_window_params(float(f_hz))
    T_need = Ncy / max(float(f_hz), 0.2)
    T_need = max(Tmin, min(Tmax, T_need))

    T_use = min(T_need, max(0.2, T_avail))
    t_end = t[-1]
    t_start = t_end - T_use
    idx = np.where(t >= t_start)[0]
    if len(idx) < 64:
        return t, data_dict
    i0 = idx[0]
    t2 = t[i0:]
    out = {k: np.asarray(v)[i0:] for k, v in data_dict.items()}
    return t2, out

def estimate_fs(t):
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    fs = 1.0 / np.mean(dt)
    dt_rel_std = np.std(dt) / np.mean(dt) if np.mean(dt) > 0 else np.inf
    return float(fs), float(dt_rel_std)

def spectrum_peak_near(x, t, f0_hz, bw_hz=0.8):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    fs, _ = estimate_fs(t)
    N = len(x)
    if N < 256:
        return float(f0_hz)

    x_ac = x - np.mean(x)
    w = np.hanning(N)
    X = np.fft.rfft(x_ac * w)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    fmin = max(0.0, float(f0_hz) - float(bw_hz))
    fmax = float(f0_hz) + float(bw_hz)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(idx) < 3:
        return float(f0_hz)

    mag = np.abs(X[idx])
    k = int(idx[np.argmax(mag)])
    return float(freqs[k])

def phase_drift_deg(x, t, f_ref):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    N = len(x)
    if N < 128:
        return 0.0
    mid = N // 2
    ph1 = lockin_phasor(x[:mid], t[:mid], f_ref)
    ph2 = lockin_phasor(x[mid:], t[mid:], f_ref)
    d = np.angle(ph2 / ph1)
    return float(np.degrees(d))

def harmonic_ratios(x, t, f_ref):
    X1 = lockin_phasor(x, t, f_ref)
    X2 = lockin_phasor(x, t, 2.0*float(f_ref))
    X3 = lockin_phasor(x, t, 3.0*float(f_ref))
    eps = 1e-30
    h2 = np.abs(X2) / (np.abs(X1) + eps)
    h3 = np.abs(X3) / (np.abs(X1) + eps)
    return float(h2), float(h3), X1

def get_timeseries_consistent(case, sig_handles, timeout_s=30.0, min_samples=1000, poll_s=0.8):
    t_start = time.time()
    last_len = -1
    stable_count = 0
    io_retries = 0
    while time.time() - t_start < timeout_s:
        try:
            case.update_plots()
        except Exception as e:
            if io_retries < IO_RETRY_COUNT:
                io_retries += 1
                print(f"[IO RETRY {io_retries}/{IO_RETRY_COUNT}] update_plots: {e} -> wait {IO_RETRY_WAIT_S}s")
                time.sleep(IO_RETRY_WAIT_S)
                continue
            else:
                raise
        time.sleep(poll_s)
        t_ref = None
        data = {}
        ok = True
        for name, sig in sig_handles.items():
            tt = np.asarray(sig.get_time_data())
            xx = np.asarray(sig.get_data())
            if len(tt) < min_samples or len(tt) != len(xx):
                ok = False
                break
            if t_ref is None:
                t_ref = tt
            else:
                if len(tt) != len(t_ref):
                    ok = False
                    break
            data[name] = xx
        if ok:
            if len(t_ref) == last_len:
                stable_count += 1
            else:
                stable_count = 0
                last_len = len(t_ref)
            if stable_count >= 1:
                return t_ref, data
    raise RuntimeError("Could not get consistent time series (timeout).")

def safe_stop(case, settle_s=STOP_SETTLE):
    try:
        case.stop()
    except Exception:
        pass
    time.sleep(settle_s)

def inv_or_pinv(A, det_thr=DET_THR, cond_thr=COND_THR, rcond=PINV_RCOND):
    detA = np.linalg.det(A)
    condA = np.linalg.cond(A)
    used_pinv = False
    if (abs(detA) < det_thr) or (condA > cond_thr):
        used_pinv = True
        Ainv = np.linalg.pinv(A, rcond=rcond)
    else:
        Ainv = np.linalg.inv(A)
    return Ainv, detA, condA, used_pinv

def track_eig_branches(eig_results, freqs):
    e0 = np.array(eig_results[freqs[0]], dtype=complex)
    nL = len(e0)
    lam = np.zeros((nL, len(freqs)), dtype=complex)
    lam[:, 0] = e0
    for k in range(1, len(freqs)):
        prev = lam[:, k-1]
        cur = list(np.array(eig_results[freqs[k]], dtype=complex))
        for i in range(nL):
            d = [abs(cur[j] - prev[i]) for j in range(len(cur))]
            jmin = int(np.argmin(d))
            lam[i, k] = cur.pop(jmin)
    return [lam[i, :] for i in range(nL)]

def is_stale_object_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("invalid processing object" in s) or ("cannot be found" in s) or ("remote invocation error" in s)

def is_unable_to_start_case_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("unable to start case" in s) or ("unable to start" in s)

def get_obj_by_name_anytype(case, name, preferred_type, fallback_types):
    types_to_try = [preferred_type] + [t for t in fallback_types if t != preferred_type]
    for t in types_to_try:
        try:
            obj = case.get_object_by_name(name, t)
            if obj is not None:
                return obj
        except Exception:
            pass
    return None

def build_handles(case):
    SLIDER_FALLBACKS = ["slider", "knob", "dial", "numeric"]
    BUTTON_FALLBACKS = ["button", "pushbutton", "toggle", "switch"]

    Hinj = {}
    for inj in INJECTIONS:
        nm = inj["name"]

        Finj = get_obj_by_name_anytype(case, inj["Finj_slider"], "slider", SLIDER_FALLBACKS)
        d_g  = get_obj_by_name_anytype(case, inj["d_gain_slider"], "slider", SLIDER_FALLBACKS)
        q_g  = get_obj_by_name_anytype(case, inj["q_gain_slider"], "slider", SLIDER_FALLBACKS)

        PBd  = get_obj_by_name_anytype(case, inj["PB_d"], "button", BUTTON_FALLBACKS)
        PBq  = get_obj_by_name_anytype(case, inj["PB_q"], "button", BUTTON_FALLBACKS)

        missing = []
        if Finj is None: missing.append(f"{inj['Finj_slider']} (slider)")
        if d_g  is None: missing.append(f"{inj['d_gain_slider']} (slider)")
        if q_g  is None: missing.append(f"{inj['q_gain_slider']} (slider)")
        if PBd  is None: missing.append(f"{inj['PB_d']} (button/toggle/switch)")
        if PBq  is None: missing.append(f"{inj['PB_q']} (button/toggle/switch)")

        if missing:
            raise RuntimeError(
                "RSCAD object(s) not found or wrong type for injection '{}':\n  - {}\n"
                "Fix: verify exact names in RSCADFX and whether they are buttons/toggles/switches.\n"
                .format(nm, "\n  - ".join(missing))
            )

        Hinj[nm] = {"Finj": Finj, "d_gain": d_g, "q_gain": q_g, "PB_d": PBd, "PB_q": PBq}

    Hpcc = {
        "pcc1": {
            "sigVd": case.get_signal(PCC1["Vd_path"]),
            "sigVq": case.get_signal(PCC1["Vq_path"]),
            "sigId": case.get_signal(PCC1["Id_path"]),
            "sigIq": case.get_signal(PCC1["Iq_path"]),
        },
        "pcc2": {
            "sigVd": case.get_signal(PCC2["Vd_path"]),
            "sigVq": case.get_signal(PCC2["Vq_path"]),
            "sigId": case.get_signal(PCC2["Id_path"]),
            "sigIq": case.get_signal(PCC2["Iq_path"]),
        },
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
    pb_d = Hinj[inj_name]["PB_d"]
    pb_q = Hinj[inj_name]["PB_q"]
    pb_d.position = 1
    pb_q.position = 1
    time.sleep(pulse_s)
    pb_d.position = 0
    pb_q.position = 0
    time.sleep(0.06)

def recover_by_reopen(app, case):
    try:
        safe_stop(case, settle_s=STOP_SETTLE)
    except Exception:
        pass
    try:
        case.close()
    except Exception:
        pass

    time.sleep(0.9)

    case = app.open_case(casefile)
    safe_stop(case, settle_s=1.2)

    case.compile()
    time.sleep(AFTER_COMPILE_WAIT)

    Hinj, Hpcc = build_handles(case)
    time.sleep(AFTER_REOPEN_WAIT)
    return case, Hinj, Hpcc

def safe_run_with_recovery(app, case, Hinj, Hpcc, retries=RUN_RETRIES, base_wait=RUN_BASE_WAIT):
    last = None
    for k in range(1, retries + 1):
        try:
            time.sleep(0.10)
            case.run()
            time.sleep(POST_RUN_WAIT)
            return case, Hinj, Hpcc
        except RSCADError as e:
            last = e

            if is_unable_to_start_case_error(e):
                print(f"[RUN FAIL {k}/{retries}] Unable to start case -> HARD RECOVERY (reopen+compile).")
                case, Hinj, Hpcc = recover_by_reopen(app, case)
                continue

            if is_stale_object_error(e):
                print(f"[RUN FAIL {k}/{retries}] Stale-ish run error -> HARD RECOVERY (reopen+compile).")
                case, Hinj, Hpcc = recover_by_reopen(app, case)
                continue

            wait = base_wait * (1.5 ** (k - 1)) + random.uniform(0.05, 0.45)
            print(f"[RUN FAIL {k}/{retries}] {e} -> STOP, wait {wait:.2f}s, retry")
            safe_stop(case, settle_s=wait)

    raise last


# ==================================================
# 2) QUALITY CLASSIFICATION (same policy as your 4x4)
# ==================================================
def classify_quality_4x4(q, f_inj):
    warn = []
    f_inj = float(f_inj)
    slow = is_slow_mode(f_inj)

    drift_warn = DRIFT_WARN_DEG_LO if slow else DRIFT_WARN_DEG_HI
    drift_lowc = DRIFT_LOWCONF_DEG_LO if slow else DRIFT_LOWCONF_DEG_HI

    drift_abs = float(q["drift_deg_max"])
    drift_flag_warn = drift_abs > drift_warn
    drift_flag_lowc = drift_abs > drift_lowc

    hmax = float(q["harm_max"])
    harm_flag_warn = hmax > HARM_WARN_RATIO
    harm_flag_lowc = hmax > HARM_LOWCONF_RATIO

    snr_flag = bool(q["snr_flag_any"])
    inv_flag = bool(q["used_pinv"])

    if harm_flag_warn:  warn.append("HARMONICS_WARN")
    if drift_flag_warn: warn.append("DRIFT_WARN")
    if snr_flag:        warn.append("SNR_WARN")
    if inv_flag:        warn.append("INV_WARN")

    low_conf = snr_flag or drift_flag_lowc or harm_flag_lowc
    if inv_flag:
       warn.append("INV_WARN")
       # Only escalate to LOW_CONF if another pillar also fails
       if snr_flag or drift_flag_lowc or harm_flag_lowc:
        low_conf = True

    if low_conf:
        status = "LOW_CONF"
    elif len(warn) > 0:
        status = "WARN"
    else:
        status = "OK"

    valid_for_curves = (status in ("OK", "WARN"))
    return status, warn, bool(valid_for_curves)


# ==================================================
# 3) CORE MEASUREMENT: Y_tot (4x4) from TWO PORTS (unchanged pipeline)
# ==================================================
def measure_Ytot_4x4_at(app, case, Hinj, Hpcc, f_inj, A_slider):
    """
    Runs 4 experiments:
      col0: inj1 d
      col1: inj1 q
      col2: inj2 d
      col3: inj2 q

    Returns: Y_tot, q, case, Hinj, Hpcc
    (case/handles may be replaced by recovery)
    """
    f_inj = float(f_inj)
    A_slider = float(A_slider)
    Ts = settle_time(f_inj)

    experiments = [
        ("inj1", A_slider, 0.0),
        ("inj1", 0.0, A_slider),
        ("inj2", A_slider, 0.0),
        ("inj2", 0.0, A_slider),
    ]

    V_ph = np.zeros((4, 4), dtype=complex)
    I_ph = np.zeros((4, 4), dtype=complex)

    drift_list, harm_list, vmag_list, imag_list, f_used_list = [], [], [], [], []
    fs_est, dt_rel_std = np.nan, np.nan

    for col, (inj_name, d_gain, q_gain) in enumerate(experiments):
        attempts = 0
        while True:
            try:
                safe_stop(case, settle_s=STOP_SETTLE)
                time.sleep(0.10)

                # set frequency on BOTH injections, zero all gains first
                for nm in ("inj1", "inj2"):
                    Hinj[nm]["Finj"].value = float(f_inj)
                    Hinj[nm]["d_gain"].value = 0.0
                    Hinj[nm]["q_gain"].value = 0.0

                # set only active gains
                Hinj[inj_name]["d_gain"].value = float(d_gain)
                Hinj[inj_name]["q_gain"].value = float(q_gain)
                time.sleep(0.10)

                # pulse ONLY active injection pair
                pulse_only_active_pair(Hinj, inj_name, pulse_s=0.10)
                time.sleep(POST_PULSE_WAIT)

                # RUN with recovery
                case, Hinj, Hpcc = safe_run_with_recovery(app, case, Hinj, Hpcc)
                sig_handles = rebuild_sig_handles(Hpcc)

                time.sleep(Ts)

                # collect time series
                t, data = get_timeseries_consistent(
                    case, sig_handles,
                    timeout_s=TIMEOUT_S, min_samples=MIN_SAMPLES, poll_s=POLL_S
                )
                fs_est, dt_rel_std = estimate_fs(t)

                t2, data2 = select_analysis_window(t, data, f_inj)

                # reference frequency selection (same method as your 4x4)
                f_ref = float(f_inj)
                xVd_ref = data2["Vd1"] if inj_name == "inj1" else data2["Vd2"]
                dVd = phase_drift_deg(xVd_ref, t2, f_ref)

                drift_warn = DRIFT_WARN_DEG_LO if is_slow_mode(f_inj) else DRIFT_WARN_DEG_HI
                if F_SEARCH_ON_DRIFT and abs(dVd) > drift_warn:
                    fpk = spectrum_peak_near(xVd_ref, t2, f_ref, bw_hz=F_SEARCH_BW_HZ)
                    dVd2 = phase_drift_deg(xVd_ref, t2, fpk)
                    if abs(dVd2) < abs(dVd):
                        f_ref = float(fpk)

                f_used_list.append(float(f_ref))

                # drift + harmonics on excited port Vd
                xVd = data2["Vd1"] if inj_name == "inj1" else data2["Vd2"]
                xId = data2["Id1"] if inj_name == "inj1" else data2["Id2"]
                drift_list.append(abs(phase_drift_deg(xVd, t2, f_ref)))

                H2v, H3v, Vd1_ph = harmonic_ratios(xVd, t2, f_ref)
                harm_list.append(max(H2v, H3v))

                vmag_list.append(float(np.abs(Vd1_ph)))
                _, _, Id1_ph = harmonic_ratios(xId, t2, f_ref)
                imag_list.append(float(np.abs(Id1_ph)))

                # 4 output voltage phasors
                Vd1 = lockin_phasor(data2["Vd1"], t2, f_ref)
                Vq1 = lockin_phasor(data2["Vq1"], t2, f_ref)
                Vd2 = lockin_phasor(data2["Vd2"], t2, f_ref)
                Vq2 = lockin_phasor(data2["Vq2"], t2, f_ref)
                V_ph[:, col] = np.array([Vd1, Vq1, Vd2, Vq2], dtype=complex)

                # 4 output current phasors
                Id1p = lockin_phasor(data2["Id1"], t2, f_ref)
                Iq1p = lockin_phasor(data2["Iq1"], t2, f_ref)
                Id2p = lockin_phasor(data2["Id2"], t2, f_ref)
                Iq2p = lockin_phasor(data2["Iq2"], t2, f_ref)
                I_ph[:, col] = np.array([Id1p, Iq1p, Id2p, Iq2p], dtype=complex)

                break

            except RSCADError as e:
                if (is_stale_object_error(e) or is_unable_to_start_case_error(e)) and attempts < STALE_RETRIES_PER_EXPERIMENT:
                    attempts += 1
                    print(f"[RECOVER] col={col} {e} -> HARD RECOVERY (reopen+compile) ({attempts}/{STALE_RETRIES_PER_EXPERIMENT})")
                    case, Hinj, Hpcc = recover_by_reopen(app, case)
                    continue
                raise

            except Exception as e:
                if attempts < 1:
                    attempts += 1
                    print(f"[RECOVER] col={col} {e} -> HARD RECOVERY (reopen+compile) (1/1)")
                    case, Hinj, Hpcc = recover_by_reopen(app, case)
                    continue
                raise

    # stop + zero gains
    safe_stop(case, settle_s=STOP_SETTLE)
    for nm in ("inj1", "inj2"):
        try:
            Hinj[nm]["d_gain"].value = 0.0
            Hinj[nm]["q_gain"].value = 0.0
        except Exception:
            pass

    V_exc = V_ph
    I_exc = I_ph

    V_inv, detV, condV, used_pinv = inv_or_pinv(V_exc)
    Y_tot = I_exc @ V_inv

    drift_deg_max = float(np.max(drift_list)) if len(drift_list) else 0.0
    harm_max      = float(np.max(harm_list)) if len(harm_list) else 0.0
    snr_flag_any  = any(vm < SNR_MIN_MAG_V for vm in vmag_list) or any(im < SNR_MIN_MAG_I for im in imag_list)

    q = {
        "A_slider": float(A_slider),
        "settle_s": float(Ts),
        "fs_est": float(fs_est),
        "dt_rel_std": float(dt_rel_std),
        "f_used_mean": float(np.mean(f_used_list)) if len(f_used_list) else float(f_inj),
        "f_used_list": [float(x) for x in f_used_list],
        "drift_deg_max": float(drift_deg_max),
        "harm_max": float(harm_max),
        "snr_flag_any": bool(snr_flag_any),
        "detV": detV,
        "condV": float(condV),
        "used_pinv": bool(used_pinv),
    }
    return Y_tot, q, case, Hinj, Hpcc


# ==================================================
# 4) EVD: left/right eigenvectors + participation
# ==================================================
def eig_left_right(A: np.ndarray):
    """
    Returns eigenvalues lam, right eigenvectors V, left eigenvectors U (columns),
    scaled such that u_k^H v_k = 1 for each mode k.
    """
    lam, V = np.linalg.eig(A)
    lamH, UH = np.linalg.eig(A.conj().T)

    U = np.zeros_like(V, dtype=complex)
    used = set()
    for k in range(len(lam)):
        target = np.conj(lam[k])
        d = np.array([abs(lamH[j] - target) if j not in used else np.inf for j in range(len(lamH))])
        jmin = int(np.argmin(d))
        used.add(jmin)
        U[:, k] = UH[:, jmin]

    # bi-orthonormal scaling: u^H v = 1
    for k in range(len(lam)):
        denom = np.vdot(U[:, k], V[:, k])
        if abs(denom) > 1e-18:
            U[:, k] = U[:, k] / denom

    return lam, V, U

def participation_factors(u_col: np.ndarray, v_col: np.ndarray):
    """
    Element-wise p_i = |u_i^* v_i|, normalized to sum=1.
    """
    p = np.abs(np.conj(u_col) * v_col).real.astype(float)
    s = float(np.sum(p)) + 1e-30
    return p / s


# ==================================================
# 5) MAIN SWEEP (EVD on Z_cl or on Y_tot)
# ==================================================
inject_freqs = list(np.arange(VALID_F_MIN, VALID_F_MAX + 1e-9, F_STEP_HZ, dtype=float))

eigvals_by_f = {}        # f -> eigenvalues of object (Z_cl or Y_tot)
lammax_by_f  = {}        # f -> max |lambda|
quality_log  = {}        # f -> q dict incl status etc.
valid_for_curves_map = {}  # f -> bool
eigvecs_store = {}       # f -> store lam,V,U for participation use

OBJ_LABEL = "Z_cl = inv(Y_tot)" if EIG_ON_ZCL else "Y_tot"

with rtds.rscadfx.remote_connection() as app:
    case = app.open_case(casefile)
    print("Case opened.")
    safe_stop(case, settle_s=1.2)
    case.compile()
    time.sleep(AFTER_COMPILE_WAIT)

    Hinj, Hpcc = build_handles(case)

    try:
        for f in inject_freqs:
            A = inj_amplitude(f)
            mode = "SLOW" if is_slow_mode(f) else "FAST"
            print("\n====================================")
            print(f"EVD sweep: f = {f:6.2f} Hz | mode={mode} | A_inj(slider) = {A:.3f} | settle = {settle_time(f):.2f}s")
            print("====================================")

            Y_tot, q, case, Hinj, Hpcc = measure_Ytot_4x4_at(app, case, Hinj, Hpcc, f, A)

            # Quality classification (same policy)
            status, warn_list, v_curve = classify_quality_4x4(q, f)
            q["status"] = status
            q["warnings"] = warn_list
            q["valid_for_curves"] = bool(v_curve)
            quality_log[f] = q
            valid_for_curves_map[f] = bool(v_curve)

            # Object for EVD
            if EIG_ON_ZCL:
                Z_cl, detY, condY, used_pinvY = inv_or_pinv(Y_tot)
                A_evd = Z_cl
                q["detYtot"] = detY
                q["condYtot"] = float(condY)
                q["used_pinvYtot"] = bool(used_pinvY)
            else:
                A_evd = Y_tot

            lam, V, U = eig_left_right(A_evd)
            eigvals_by_f[f] = lam
            lammax_by_f[f] = float(np.max(np.abs(lam)))

            # Store vectors only for curve-valid points
            if v_curve:
                eigvecs_store[f] = {"lam": lam, "V": V, "U": U}

            print(f"[4x4 EVD] @ f={f:6.2f} Hz  [{status}]  {(' '.join(warn_list)) if warn_list else ''}")
            print(f"  f_used_list={q['f_used_list']} | fs≈{q['fs_est']:.1f} Hz | dt_rel_std={q['dt_rel_std']:.2e}")
            print(f"  det(V_exc) = {q['detV'].real:+.3e}{q['detV'].imag:+.3e}j | cond(V_exc)= {q['condV']:.3e} | used_pinv={q['used_pinv']}")
            print(f"  drift_deg_max={q['drift_deg_max']:.1f} deg | harm_max={q['harm_max']:.3f} | snr_flag_any={q['snr_flag_any']}")
            print(f"  EVD object: {OBJ_LABEL} | |lambda_max|={lammax_by_f[f]:.4e} | curves={v_curve}")

    finally:
        safe_stop(case, settle_s=1.0)
        try:
            for nm in ("inj1", "inj2"):
                Hinj[nm]["d_gain"].value = 0.0
                Hinj[nm]["q_gain"].value = 0.0
        except Exception:
            pass
        try:
            case.close()
        except Exception:
            pass


# ==================================================
# 6) POST: eigenloci + lambda_max + participation @ f*
# ==================================================
freqs_all = sorted(eigvals_by_f.keys())
freqs_curves = [f for f in freqs_all if valid_for_curves_map.get(f, False)]

if len(freqs_curves) < 2:
    raise RuntimeError("Too few curve-valid points for EVD plots. Improve SNR / reduce harmonics / relax thresholds.")

# --- Eigenloci (tracked branches)
if PLOT_EIGENLOCI:
    if TRACK_BRANCHES:
        lam_branches = track_eig_branches({f: eigvals_by_f[f] for f in freqs_curves}, freqs_curves)
        plt.figure()
        for i, lam in enumerate(lam_branches):
            plt.plot(lam.real, lam.imag, "-", marker="o", markersize=2, label=f"λ{i+1}(jω)")
    else:
        plt.figure()
        for f in freqs_curves:
            lam = np.array(eigvals_by_f[f], dtype=complex)
            plt.plot(lam.real, lam.imag, "o", markersize=2)

    plt.axhline(0, linewidth=0.6)
    plt.axvline(0, linewidth=0.6)
    plt.xlabel("Re{λ(jω)}")
    plt.ylabel("Im{λ(jω)}")
    plt.title(f"EVD eigenloci of {OBJ_LABEL} (curves = OK+WARN, {VALID_F_MIN:.0f}–{VALID_F_MAX:.0f} Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("evd_eigenloci.svg", format="svg", bbox_inches="tight"); print("Saved: evd_eigenloci.svg")
    plt.show()

# --- |lambda_max(ω)| tracking + f*
if PLOT_LAMMAX:
    lammax_curve = np.array([lammax_by_f[f] for f in freqs_curves], dtype=float)
    f_star = float(freqs_curves[int(np.argmax(lammax_curve))])
    lammax_star = float(np.max(lammax_curve))

    plt.figure()
    plt.plot(freqs_curves, lammax_curve, "-o", markersize=2)
    plt.xlabel("f [Hz]")
    plt.ylabel(r"$|\lambda_{\max}(j\omega)|$")
    plt.title(r"Critical-frequency indicator for " + OBJ_LABEL)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evd_lammax.svg", format="svg", bbox_inches="tight"); print("Saved: evd_lammax.svg")
    plt.show()

    print("\n====================================")
    print("EVD: critical-frequency identification")
    print("====================================")
    print(f"Curve-valid points: {len(freqs_curves)}/{len(freqs_all)}")
    print(f"f* = {f_star:.2f} Hz where |lambda_max| is maximized: {lammax_star:.4e}")
    print("====================================\n")
else:
    f_star = None

# --- Participation factors at f*
if (f_star is not None) and (f_star in eigvecs_store):
    lam = eigvecs_store[f_star]["lam"]
    V   = eigvecs_store[f_star]["V"]
    U   = eigvecs_store[f_star]["U"]

    # dominant mode at f*: index of max |λ|
    k_star = int(np.argmax(np.abs(lam)))
    lam_star = lam[k_star]

    # participation over 4 channels: [Vd1,Vq1,Vd2,Vq2] dimension
    p = participation_factors(U[:, k_star], V[:, k_star])
    labels = [f"{PCC1['name']}-d", f"{PCC1['name']}-q", f"{PCC2['name']}-d", f"{PCC2['name']}-q"]

    print("====================================")
    print("Participation factors at f* (dominant mode)")
    print("====================================")
    print(f"At f* = {f_star:.2f} Hz: dominant eigenvalue λ* = {lam_star.real:+.4e}{lam_star.imag:+.4e}j  |λ*|={abs(lam_star):.4e}")
    for i, lab in enumerate(labels):
        print(f"  p[{lab}] = {p[i]:.4f}")
    print("  (p normalized to sum=1)")
    print("====================================\n")

    if PLOT_PARTICIPATION_BAR:
        plt.figure()
        plt.bar(labels, p)
        plt.ylabel("Participation (normalized)")
        plt.title(f"Participation factors at f* = {f_star:.2f} Hz (dominant mode) | {OBJ_LABEL}")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig("evd_participation.svg", format="svg", bbox_inches="tight"); print("Saved: evd_participation.svg")
        plt.show()
else:
    print("[WARN] Could not compute participation factors (no stored eigvecs at f*).")
