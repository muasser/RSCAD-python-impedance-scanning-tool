# -*- coding: utf-8 -*-
"""
GNC MIMO Stability Tool v23 — RSMM at critical frequency f*
==============================================================================
TWO PCCs (two injections) -> 4x4 dq-domain

PARAMETER MAPPING (draft slider name -> physical parameter):
  GF  slider  ->  St2Gu1KpId  (inner current loop Kp) — swept
  GF1 slider  ->  St2Gu1KpVdc (outer DC voltage loop Kp) — fixed at 5.0

FOUR SWEEP MODES (set via SWEEP_MODE):
  "GF"         — KpId sweep at one fixed SCR case (ACTIVE_CASE).
  "SCR"        — SCR sensitivity at fixed KpId.
  "XR"         — X/R sensitivity at fixed KpId and SCR.
  "GF_PER_SCR" — KpId sweep per SCR case.

ACTIVE_CASE:
  "A" -> SCR=5.0, R=3.664 Ohm, L=0.175 H
  "B" -> SCR=3.5, R=5.238 Ohm, L=0.250 H  <- baseline
  "C" -> SCR=2.5, R=7.334 Ohm, L=0.350 H
  "D" -> SCR=2.0, R=9.167 Ohm, L=0.438 H
  "ALL" -> only with GF_PER_SCR
"""

import os as _os
import time
import random
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as dsp
import rtds.rscadfx

try:
    from rtds.comms import RSCADError
except Exception:
    RSCADError = Exception

# Custom exception for when the SSL/RSCAD connection is fully dead
class ConnectionDeadError(RuntimeError):
    pass

def is_connection_dead_error(e):
    """Detects whether the RSCAD SSL connection is fully broken."""
    s = str(type(e).__name__).lower() + str(e).lower()
    return any(k in s for k in (
        "ssleofError".lower(),
        "eof occurred",
        "rscad fx is not connected",
        "connection has been terminated",
        "communicationerror",
        "ssl",
    ))


OUT_DIR = _os.path.dirname(_os.path.abspath(__file__))

# ==================================================
# 0) USER CONFIG
# ==================================================
casefile = r"C:\Users\muass\OneDrive\Documenten\RSCAD\RTDS_USER_FX\fileman\01 HVDC\Offshore Windfarm\MMC_HVDC_OffshoreWindfarm.rtfx"

SWEEP_MODE  = "GF_PER_SCR"
ACTIVE_CASE = "ALL"

GRID_CASES = {
    "A": {"SCR": 5.0, "XR": 15.0, "R": 3.664, "L": 0.175},
    "B": {"SCR": 3.5, "XR": 15.0, "R": 5.238, "L": 0.250},
    "C": {"SCR": 2.5, "XR": 15.0, "R": 7.334, "L": 0.350},
    "D": {"SCR": 2.0, "XR": 15.0, "R": 9.167, "L": 0.438},
}

VALID_F_MIN = 5
VALID_F_MAX = 12
F_STEP_HZ   = 1.0

RSMM_TARGET = 0.79

# Placeholder; actual value is determined after the sweep.
F_CRIT_FIXED = None

# --------------------------------------------------
# CRITICAL FREQUENCY — auto-determined as the frequency
# with the largest RSMM variation across all sweep runs
# (highest std dev over KpPLL sweep).
# --------------------------------------------------

# GF slider -> St2Gu1KpPLL (sweep), nominal = 5.0
GF_SWEEP_VALUES = [1,5,10,15]

# Only needed when SWEEP_MODE = "XR"
XR_SWEEP_VALUES = [5.0, 7.5, 10.0, 15.0, 20.0]

# GF1 slider -> St2Gu1TiPLL fixed at nominal 0.01
KPVDC_FIXED = 0.01

# Baseline KpPLL for SCR/XR modes
GF_BASELINE  = 5.0
SCR_BASELINE = 3.5
XR_BASELINE  = 15.0

GRID_SETTLE_S = 3.0
GF_SETTLE_S   = 2.0

RSMM_PROXIMITY_THR = 5.0

Vbase_LL  = 525e3
Sbase_3ph = 1000e6
f0        = 50.0

A0     = 1.0
A_MIN  = 0.2
A_MAX  = 3.0
F_KNEE = 20.0
ALPHA  = 0.7

F_SLOW_MAX = 4.0

TIMEOUT_S   = 30.0
MIN_SAMPLES = 1400
POLL_S      = 1.0

POST_PULSE_WAIT    = 0.18
POST_RUN_WAIT      = 1.5
STOP_SETTLE        = 2.5
AFTER_REOPEN_WAIT  = 1.2
AFTER_COMPILE_WAIT = 2.5

PLOT_ACK_RETRIES = 4
PLOT_ACK_WAIT_S  = 3.0

PLOT_UPDATE_TIMEOUT_S = 10.0

SLOW_N_CYCLES = 20
FAST_N_CYCLES = 12
SLOW_MIN_S    = 1.8
FAST_MIN_S    = 0.8
SLOW_MAX_S    = 3.5
FAST_MAX_S    = 1.8

HARM_WARN_RATIO    = 0.08
HARM_LOWCONF_RATIO = 0.25

DRIFT_WARN_DEG_HI    = 30.0
DRIFT_WARN_DEG_LO    = 80.0
DRIFT_LOWCONF_DEG_HI = 80.0
DRIFT_LOWCONF_DEG_LO = 140.0

SNR_MIN_MAG_V = 1e-4
SNR_MIN_MAG_I = 1e-5

F_SEARCH_ON_DRIFT = True
F_SEARCH_BW_HZ    = 0.8

BPF_ENABLE  = True
BPF_BW_HZ   = 1.5
BPF_NUMTAPS = 255

NOTCH_ENABLE = True
NOTCH_Q      = 30.0

RUN_RETRIES    = 7
RUN_BASE_WAIT  = 0.9

DET_THR    = 1e-12
COND_THR   = 1e6
PINV_RCOND = 1e-10

ZOOM_XLIM = (-5.0, 3.0)
ZOOM_YLIM = (-5.0, 5.0)

STALE_RETRIES_PER_EXPERIMENT = 2

BETWEEN_RUNS_SETTLE_S = 8.0


# ==================================================
# 0a) TWO PCC DEFINITIONS — same as v18
# ==================================================
PCC1 = {
    "name":    "PCC1",
    "Vd_path": "Subsystem #1|CTLs|Vars|Vdpu11",
    "Vq_path": "Subsystem #1|CTLs|Vars|Vqpu11",
    "Id_path": "Subsystem #1|CTLs|Vars|Id",
    "Iq_path": "Subsystem #1|CTLs|Vars|Iq",
}

PCC2 = {
    "name":    "PCC2",
    "Vd_path": "Subsystem #1|CTLs|Vars|Vdpu22",
    "Vq_path": "Subsystem #1|CTLs|Vars|Vqpu22",
    "Id_path": "Subsystem #1|CTLs|Vars|Id1",
    "Iq_path": "Subsystem #1|CTLs|Vars|Iq1",
}

# ==================================================
# 0b) INJECTIONS — same as v18
# ==================================================
INJECTIONS = [
    {
        "name":          "inj1",
        "Finj_slider":   "Finj",
        "d_gain_slider": "d_gain",
        "q_gain_slider": "q_gain",
        "GF_slider":     "GF",
        "GF1_slider":    "GF1",
        "R_slider":      "R",
        "L_slider":      "L",
        "PB_d":          "PB",
        "PB_q":          "PB1",
    },
    {
        "name":          "inj2",
        "Finj_slider":   "Finj1",
        "d_gain_slider": "d_gain1",
        "q_gain_slider": "q_gain1",
        "GF_slider":     "GF1",
        "GF1_slider":    None,
        "R_slider":      None,
        "L_slider":      None,
        "PB_d":          "PB2",
        "PB_q":          "PB3",
    },
]

MUTUAL_ENABLE = False
R12_pu = 0.0
X12_pu = 0.0


# ==================================================
# 1) BASE
# ==================================================
Zbase = (Vbase_LL ** 2) / Sbase_3ph
Ibase = Sbase_3ph / (np.sqrt(3.0) * Vbase_LL)

print("====================================")
print("GNC MIMO Stability Tool v23")
print("====================================")
print(f"SWEEP_MODE  : {SWEEP_MODE}")
print(f"ACTIVE_CASE : {ACTIVE_CASE}")
print(f"Freq window : {VALID_F_MIN}-{VALID_F_MAX} Hz  (step={F_STEP_HZ} Hz)")
print(f"GF  slider  -> St2Gu1KpPLL sweep  : {GF_SWEEP_VALUES}")
print(f"GF1 slider  -> St2Gu1TiPLL vast   : {KPVDC_FIXED}")
print("f*          : auto-determined after sweep (highest RSMM std over runs)")
print(f"GF_BASELINE : {GF_BASELINE}  (nominale KpPLL)")
print(f"RSMM_TARGET : {RSMM_TARGET}")
print("====================================")
print("GRID CASES:")
for k, v in GRID_CASES.items():
    marker = " <-- ACTIVE" if k == ACTIVE_CASE else ""
    print(f"  Case {k}: SCR={v['SCR']:.1f}  X/R={v['XR']:.0f}  "
          f"R={v['R']:.3f} Ohm  L={v['L']:.3f} H{marker}")
print("====================================\n")


# ==================================================
# 2) DSP HELPERS — same as v18
# ==================================================
def is_slow_mode(f_hz):
    return float(f_hz) <= float(F_SLOW_MAX)

def inj_amplitude(f_hz):
    f_hz  = float(f_hz)
    scale = (f_hz / F_KNEE) ** ALPHA if f_hz > F_KNEE else 1.0
    return float(np.clip(A0 * scale, A_MIN, A_MAX))

def settle_time(f_hz):
    f_hz = float(f_hz)
    if is_slow_mode(f_hz):
        return float(max(3.0, min(7.0, 14.0 / max(f_hz, 0.7))))
    return float(max(0.8, min(2.2, 8.0 / max(f_hz, 2.0))))

def analysis_window_params(f_hz):
    if is_slow_mode(f_hz):
        return SLOW_N_CYCLES, SLOW_MIN_S, SLOW_MAX_S
    return FAST_N_CYCLES, FAST_MIN_S, FAST_MAX_S

def estimate_fs(t):
    t  = np.asarray(t, dtype=float)
    dt = np.diff(t)
    fs = 1.0 / np.mean(dt)
    return float(fs), float(np.std(dt) / np.mean(dt) if np.mean(dt) > 0 else np.inf)

def select_analysis_window(t, data_dict, f_hz):
    t               = np.asarray(t)
    Ncy, Tmin, Tmax = analysis_window_params(float(f_hz))
    T_need          = max(Tmin, min(Tmax, Ncy / max(float(f_hz), 0.2)))
    T_use           = min(T_need, max(0.2, float(t[-1] - t[0])))
    idx             = np.where(t >= t[-1] - T_use)[0]
    if len(idx) < 64:
        return t, data_dict
    i0 = idx[0]
    return t[i0:], {k: np.asarray(v)[i0:] for k, v in data_dict.items()}

def spectrum_peak_near(x, t, f0_hz, bw_hz=0.8):
    x     = np.asarray(x, dtype=float)
    t     = np.asarray(t, dtype=float)
    fs, _ = estimate_fs(t)
    N     = len(x)
    if N < 256:
        return float(f0_hz)
    X     = np.fft.rfft((x - np.mean(x)) * np.hanning(N))
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    idx   = np.where((freqs >= max(0, f0_hz - bw_hz)) &
                     (freqs <= f0_hz + bw_hz))[0]
    if len(idx) < 3:
        return float(f0_hz)
    return float(freqs[idx[np.argmax(np.abs(X[idx]))]])

def bandpass_filter(x, fs, f_ref, bw_hz=BPF_BW_HZ, numtaps=BPF_NUMTAPS):
    if not BPF_ENABLE:
        return x
    nyq  = fs / 2.0
    f_lo = max(0.5, float(f_ref) - float(bw_hz))
    f_hi = min(nyq * 0.98, float(f_ref) + float(bw_hz))
    if f_lo >= f_hi or f_hi >= nyq:
        return x
    try:
        taps = dsp.firwin(int(numtaps) | 1,
                          [f_lo / nyq, f_hi / nyq],
                          pass_zero=False)
        return dsp.filtfilt(taps, [1.0], x)
    except Exception:
        return x

def notch_at(x, fs, f_notch, Q=NOTCH_Q):
    nyq = fs / 2.0
    if float(f_notch) <= 0 or float(f_notch) >= nyq:
        return x
    try:
        b, a = dsp.iirnotch(float(f_notch) / nyq, float(Q))
        return dsp.filtfilt(b, a, x)
    except Exception:
        return x

def apply_harmonic_notches(x, fs, f_ref):
    if not NOTCH_ENABLE:
        return x
    x = notch_at(x, fs, 2.0 * float(f_ref))
    x = notch_at(x, fs, 3.0 * float(f_ref))
    return x

def lockin_phasor(x, t, f_ref):
    x     = np.asarray(x, dtype=float)
    t     = np.asarray(t, dtype=float)
    if len(x) < 32:
        raise RuntimeError("Too few samples for lock-in.")
    fs, _ = estimate_fs(t)
    x_f   = bandpass_filter(x, fs, f_ref)
    x_ac  = x_f - np.mean(x_f)
    N     = len(x_ac)
    w     = np.hanning(N)
    ww    = np.mean(w) if np.mean(w) != 0 else 1.0
    ref   = np.exp(-1j * 2.0 * np.pi * float(f_ref) * t)
    return (2.0 / N) * np.sum((x_ac * w) * ref) / ww

def phase_drift_deg(x, t, f_ref):
    x   = np.asarray(x, dtype=float)
    t   = np.asarray(t, dtype=float)
    if len(x) < 128:
        return 0.0
    mid = len(x) // 2
    ph1 = lockin_phasor(x[:mid], t[:mid], f_ref)
    ph2 = lockin_phasor(x[mid:], t[mid:], f_ref)
    return float(np.degrees(np.angle(ph2 / ph1)))

def harmonic_ratios(x, t, f_ref):
    x     = np.asarray(x, dtype=float)
    t     = np.asarray(t, dtype=float)
    fs, _ = estimate_fs(t)
    X1    = lockin_phasor(x, t, f_ref)
    x_n   = apply_harmonic_notches(x, fs, f_ref)
    X2    = lockin_phasor(x_n, t, 2.0 * float(f_ref))
    X3    = lockin_phasor(x_n, t, 3.0 * float(f_ref))
    eps   = 1e-30
    return (float(np.abs(X2) / (np.abs(X1) + eps)),
            float(np.abs(X3) / (np.abs(X1) + eps)),
            X1)


# ==================================================
# 3) RSCAD INTERFACE HELPERS — same as v18
# ==================================================
def is_recoverable_plot_error(e):
    s = str(e).lower()
    return ("read timed out" in s or
            "ioexception"   in s or
            "plot deadlock" in s or
            "update_plots() hung" in s)

def update_plots_with_timeout(case, timeout_s=None):
    if timeout_s is None:
        timeout_s = PLOT_UPDATE_TIMEOUT_S
    result_exc = [None]

    def _target():
        try:
            case.update_plots()
        except Exception as e:
            result_exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=float(timeout_s))

    if t.is_alive():
        raise RuntimeError(
            f"update_plots() hung for >{timeout_s:.0f}s — "
            f"RSCAD plot deadlock on rack"
        )
    if result_exc[0] is not None:
        raise result_exc[0]

def get_timeseries_consistent(case, sig_handles, timeout_s=30.0,
                               min_samples=1000, poll_s=1.0):
    t_start      = time.time()
    last_len     = -1
    stable_count = 0
    while time.time() - t_start < timeout_s:
        update_plots_with_timeout(case)
        time.sleep(poll_s)
        t_ref = None
        data  = {}
        ok    = True
        for name, sig in sig_handles.items():
            tt = np.asarray(sig.get_time_data())
            xx = np.asarray(sig.get_data())
            if len(tt) < min_samples or len(tt) != len(xx):
                ok = False
                break
            if t_ref is None:
                t_ref = tt
            elif len(tt) != len(t_ref):
                ok = False
                break
            data[name] = xx
        if ok:
            if len(t_ref) == last_len:
                stable_count += 1
            else:
                stable_count = 0
                last_len     = len(t_ref)
            if stable_count >= 1:
                return t_ref, data
    raise RuntimeError("Could not get consistent time series (timeout).")

def safe_stop(case, settle_s=STOP_SETTLE):
    try:
        case.stop()
    except Exception:
        pass
    time.sleep(settle_s)

def inv_or_pinv(A):
    detA  = np.linalg.det(A)
    condA = np.linalg.cond(A)
    if abs(detA) < DET_THR or condA > COND_THR:
        return np.linalg.pinv(A, rcond=PINV_RCOND), detA, float(condA), True
    return np.linalg.inv(A), detA, float(condA), False

def track_eig_branches(eig_results, freqs):
    e0  = np.array(eig_results[freqs[0]], dtype=complex)
    nL  = len(e0)
    lam = np.zeros((nL, len(freqs)), dtype=complex)
    lam[:, 0] = e0
    for k in range(1, len(freqs)):
        prev = lam[:, k-1]
        cur  = list(np.array(eig_results[freqs[k]], dtype=complex))
        for i in range(nL):
            jmin      = int(np.argmin([abs(cur[j] - prev[i]) for j in range(len(cur))]))
            lam[i, k] = cur.pop(jmin)
    return [lam[i, :] for i in range(nL)]

def winding_number_of_complex_curve(z):
    z = np.asarray(z, dtype=complex)
    if len(z) > 1:
        diffs = np.abs(np.diff(z))
        mags  = np.abs(z[:-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(mags > 1e-12, diffs / mags, 0.0)
        if np.any(rel > 0.5):
            print(f"  [WARN] Det curve large jumps (max={np.max(rel):.2f})"
                  f" — winding number may be unreliable.")
    ang  = np.unwrap(np.angle(z))
    dphi = float(ang[-1] - ang[0])
    return int(np.round(dphi / (2.0 * np.pi))), dphi

def dq_RL_block_pu(f_hz, R_pu, X_pu, f0_hz):
    w   = float(f_hz) / float(f0_hz)
    Zdd = float(R_pu) + 1j * w * float(X_pu)
    return np.array([[Zdd, -float(X_pu)],
                     [float(X_pu), Zdd]], dtype=complex)

def build_Zgrid_4x4_pu(f_hz, f0_hz, Rg1, Xg1, Rg2, Xg2, R12=0.0, X12=0.0):
    Z11 = dq_RL_block_pu(f_hz, Rg1, Xg1, f0_hz)
    Z22 = dq_RL_block_pu(f_hz, Rg2, Xg2, f0_hz)
    Z12 = dq_RL_block_pu(f_hz, R12, X12, f0_hz)
    return np.vstack([np.hstack([Z11, Z12]),
                      np.hstack([Z12, Z22])])

def is_stale_object_error(e):
    s = str(e).lower()
    return any(k in s for k in ("invalid processing object",
                                 "cannot be found",
                                 "remote invocation error"))

def is_unable_to_start_case_error(e):
    s = str(e).lower()
    return any(k in s for k in ("unable to start case", "unable to start"))

def get_obj_by_name_anytype(case, name, preferred_type, fallback_types):
    for t in [preferred_type] + [x for x in fallback_types if x != preferred_type]:
        try:
            obj = case.get_object_by_name(name, t)
            if obj is not None:
                return obj
        except Exception:
            pass
    return None


# ==================================================
# 4) BUILD HANDLES — same as v18
# ==================================================
def build_handles(case):
    SLIDER_FB = ["slider", "knob", "dial", "numeric"]
    BUTTON_FB = ["button", "pushbutton", "toggle", "switch"]

    Hinj = {}
    for inj in INJECTIONS:
        nm   = inj["name"]
        Finj = get_obj_by_name_anytype(case, inj["Finj_slider"],   "slider", SLIDER_FB)
        d_g  = get_obj_by_name_anytype(case, inj["d_gain_slider"], "slider", SLIDER_FB)
        q_g  = get_obj_by_name_anytype(case, inj["q_gain_slider"], "slider", SLIDER_FB)
        gf   = get_obj_by_name_anytype(case, inj["GF_slider"],     "slider", SLIDER_FB)
        PBd  = get_obj_by_name_anytype(case, inj["PB_d"], "button", BUTTON_FB)
        PBq  = get_obj_by_name_anytype(case, inj["PB_q"], "button", BUTTON_FB)

        gf1 = None
        if inj.get("GF1_slider") is not None:
            gf1 = get_obj_by_name_anytype(case, inj["GF1_slider"], "slider", SLIDER_FB)

        H_R = None
        H_L = None
        if inj.get("R_slider") is not None:
            H_R = get_obj_by_name_anytype(case, inj["R_slider"], "slider", SLIDER_FB)
        if inj.get("L_slider") is not None:
            H_L = get_obj_by_name_anytype(case, inj["L_slider"], "slider", SLIDER_FB)

        missing = []
        if Finj is None: missing.append(f"{inj['Finj_slider']} (slider)")
        if d_g  is None: missing.append(f"{inj['d_gain_slider']} (slider)")
        if q_g  is None: missing.append(f"{inj['q_gain_slider']} (slider)")
        if gf   is None: missing.append(f"{inj['GF_slider']} (slider)")
        if PBd  is None: missing.append(f"{inj['PB_d']} (button)")
        if PBq  is None: missing.append(f"{inj['PB_q']} (button)")
        if inj.get("GF1_slider") is not None and gf1 is None:
            missing.append(f"{inj['GF1_slider']} (slider — GF1 PCC2 EPLL)")
        if inj.get("R_slider") is not None and H_R is None:
            missing.append(f"{inj['R_slider']} (slider — grid R)")
        if inj.get("L_slider") is not None and H_L is None:
            missing.append(f"{inj['L_slider']} (slider — grid L)")

        if missing:
            raise RuntimeError(
                f"RSCAD objects not found for injection '{nm}':\n"
                + "\n".join(f"  - {m}" for m in missing)
            )

        Hinj[nm] = {
            "Finj":   Finj,
            "d_gain": d_g,
            "q_gain": q_g,
            "GF":     gf,
            "GF1":    gf1,
            "R":      H_R,
            "L":      H_L,
            "PB_d":   PBd,
            "PB_q":   PBq,
        }

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


# ==================================================
# 5) GRID IMPEDANCE — same as v18
# ==================================================
def set_grid_impedance(Hinj, case_key):
    if case_key not in GRID_CASES:
        raise ValueError(f"Unknown case '{case_key}'.")
    params = GRID_CASES[case_key]
    R_val  = float(params["R"])
    L_val  = float(params["L"])
    print(f"  [GRID] Case {case_key}: SCR={params['SCR']:.1f}  "
          f"X/R={params['XR']:.0f}  R={R_val:.3f} Ohm  L={L_val:.3f} H")
    Hinj["inj1"]["R"].value = R_val
    time.sleep(0.05)
    Hinj["inj1"]["L"].value = L_val
    time.sleep(0.05)
    print(f"  [GRID] Waiting {GRID_SETTLE_S}s for grid model to settle...")
    time.sleep(GRID_SETTLE_S)
    print(f"  [GRID] Settled on Case {case_key}.")

def set_grid_impedance_xr(Hinj, scr_val, xr_val):
    S_sc_ = float(scr_val) * Sbase_3ph
    Zm_   = (Vbase_LL ** 2) / S_sc_
    R_ohm = Zm_ / np.sqrt(1.0 + float(xr_val) ** 2)
    L_h   = R_ohm * float(xr_val) / (2.0 * np.pi * f0)
    print(f"  [GRID] SCR={scr_val:.2f}  X/R={xr_val:.1f}  "
          f"R={R_ohm:.3f} Ohm  L={L_h:.3f} H")
    Hinj["inj1"]["R"].value = R_ohm
    time.sleep(0.05)
    Hinj["inj1"]["L"].value = L_h
    time.sleep(0.05)
    print(f"  [GRID] Waiting {GRID_SETTLE_S}s for grid model to settle...")
    time.sleep(GRID_SETTLE_S)
    print(f"  [GRID] Settled.")
    return R_ohm, L_h


# ==================================================
# 6) GF CONTROL
# ==================================================
def set_gf_slider(Hinj, gf_val):
    gf_val = float(gf_val)

    # GF -> St2Gu1KpPLL: being swept
    Hinj["inj1"]["GF"].value = gf_val

    # GF1 -> St2Gu1TiPLL: always fixed at nominal 0.01
    if Hinj["inj1"]["GF1"] is not None:
        Hinj["inj1"]["GF1"].value = KPVDC_FIXED
        print(f"  [GF] St2Gu1KpPLL (GF)   = {gf_val:.4f}  [SWEEP]")
        print(f"  [GF] St2Gu1TiPLL (GF1)  = {KPVDC_FIXED:.4f}  [FIXED]")
        print(f"  [GF] Waiting {GF_SETTLE_S}s for PLL to settle...")
    else:
        print(f"  [GF] WARNING: GF1 handle is None — "
              f"St2Gu1TiPLL not updated!")

    time.sleep(GF_SETTLE_S)
    print(f"  [GF] Settled.")


# ==================================================
# 7) RECOVERY — same as v18
# ==================================================
def pulse_only_active_pair(Hinj, inj_name, pulse_s=0.10):
    Hinj[inj_name]["PB_d"].position = 1
    Hinj[inj_name]["PB_q"].position = 1
    time.sleep(pulse_s)
    Hinj[inj_name]["PB_d"].position = 0
    Hinj[inj_name]["PB_q"].position = 0
    time.sleep(0.06)

def recover_by_reopen(app, case):
    for fn in [lambda: safe_stop(case, settle_s=STOP_SETTLE),
               lambda: case.close()]:
        try:
            fn()
        except Exception:
            pass
    time.sleep(0.9)
    try:
        case = app.open_case(casefile)
    except Exception as e:
        if is_connection_dead_error(e):
            raise ConnectionDeadError(
                f"RSCAD connection fully lost — cannot recover: {e}"
            )
        raise
    safe_stop(case, settle_s=1.2)
    case.compile()
    time.sleep(AFTER_COMPILE_WAIT)
    Hinj, Hpcc = build_handles(case)
    time.sleep(AFTER_REOPEN_WAIT)
    return case, Hinj, Hpcc

def safe_run_with_recovery(app, case, Hinj, Hpcc,
                            retries=RUN_RETRIES, base_wait=RUN_BASE_WAIT):
    last = None
    for k in range(1, retries + 1):
        try:
            time.sleep(0.10)
            case.run()
            for ack_attempt in range(1, PLOT_ACK_RETRIES + 1):
                try:
                    update_plots_with_timeout(case)
                    break
                except Exception as ack_err:
                    if is_recoverable_plot_error(ack_err):
                        print(f"  [PLOT] Recoverable error attempt "
                              f"{ack_attempt}/{PLOT_ACK_RETRIES}: {ack_err}")
                        if ack_attempt < PLOT_ACK_RETRIES:
                            time.sleep(PLOT_ACK_WAIT_S)
                        else:
                            raise
                    else:
                        raise
            return case, Hinj, Hpcc
        except ConnectionDeadError:
            raise
        except Exception as e:
            if is_connection_dead_error(e):
                raise ConnectionDeadError(f"Connection dead in safe_run: {e}")
            last = e
            if (is_recoverable_plot_error(e) or is_stale_object_error(e) or
                    is_unable_to_start_case_error(e)):
                print(f"[RUN FAIL {k}/{retries}] {type(e).__name__}: {e}"
                      f" -> HARD RECOVERY")
                case, Hinj, Hpcc = recover_by_reopen(app, case)
            else:
                wait = base_wait * (1.5 ** (k - 1)) + random.uniform(0.05, 0.45)
                print(f"[RUN FAIL {k}/{retries}] {e} -> wait {wait:.2f}s")
                safe_stop(case, settle_s=wait)
    raise last


# ==================================================
# 8) QUALITY CLASSIFICATION — REMOVED IN V20
# ==================================================


# ==================================================
# 9) RSMM — same as v18
# ==================================================
def compute_rsmm(L_matrix, eigvals=None, thr=RSMM_PROXIMITY_THR):
    L_matrix = np.asarray(L_matrix, dtype=complex)
    n        = L_matrix.shape[0]
    IplusL   = np.eye(n, dtype=complex) + L_matrix
    sv       = np.linalg.svd(IplusL, compute_uv=False)
    rsmm     = float(np.min(sv))
    n_near   = 0
    n_far    = 0
    if eigvals is not None:
        eigvals   = np.asarray(eigvals, dtype=complex)
        dists     = np.abs(1.0 + eigvals)
        near_mask = dists < float(thr)
        n_near    = int(np.sum(near_mask))
        n_far     = int(np.sum(~near_mask))
    return rsmm, n_near, n_far, False


# ==================================================
# 10) CORE MEASUREMENT — same as v18
# ==================================================
def measure_Ytot_4x4_at(app, case, Hinj, Hpcc, f_inj, A_slider):
    f_inj    = float(f_inj)
    A_slider = float(A_slider)
    Ts       = settle_time(f_inj)

    experiments = [
        ("inj1", A_slider, 0.0),
        ("inj1", 0.0,      A_slider),
        ("inj2", A_slider, 0.0),
        ("inj2", 0.0,      A_slider),
    ]

    V_ph = np.zeros((4, 4), dtype=complex)
    I_ph = np.zeros((4, 4), dtype=complex)
    drift_list, harm_list, vmag_list, imag_list, f_used_list = [], [], [], [], []
    fs_est = np.nan

    for col, (inj_name, d_gain, q_gain) in enumerate(experiments):
        attempts = 0
        while True:
            try:
                safe_stop(case, settle_s=STOP_SETTLE)
                time.sleep(0.10)
                for nm in ("inj1", "inj2"):
                    Hinj[nm]["Finj"].value   = float(f_inj)
                    Hinj[nm]["d_gain"].value = 0.0
                    Hinj[nm]["q_gain"].value = 0.0
                Hinj[inj_name]["d_gain"].value = float(d_gain)
                Hinj[inj_name]["q_gain"].value = float(q_gain)
                time.sleep(0.10)
                pulse_only_active_pair(Hinj, inj_name, pulse_s=0.10)
                time.sleep(POST_PULSE_WAIT)
                case, Hinj, Hpcc = safe_run_with_recovery(app, case, Hinj, Hpcc)
                sig_handles = rebuild_sig_handles(Hpcc)
                time.sleep(Ts)
                t, data = get_timeseries_consistent(
                    case, sig_handles,
                    timeout_s=TIMEOUT_S, min_samples=MIN_SAMPLES, poll_s=POLL_S)
                fs_est, _ = estimate_fs(t)
                t2, data2 = select_analysis_window(t, data, f_inj)
                f_ref = float(f_inj)
                xVd   = data2["Vd1"] if inj_name == "inj1" else data2["Vd2"]
                dVd   = phase_drift_deg(xVd, t2, f_ref)
                drift_thr = DRIFT_WARN_DEG_LO if is_slow_mode(f_inj) else DRIFT_WARN_DEG_HI
                if F_SEARCH_ON_DRIFT and abs(dVd) > drift_thr:
                    fpk  = spectrum_peak_near(xVd, t2, f_ref, bw_hz=F_SEARCH_BW_HZ)
                    dVd2 = phase_drift_deg(xVd, t2, fpk)
                    if abs(dVd2) < abs(dVd):
                        f_ref = float(fpk)
                f_used_list.append(float(f_ref))
                xVd = data2["Vd1"] if inj_name == "inj1" else data2["Vd2"]
                xId = data2["Id1"] if inj_name == "inj1" else data2["Id2"]
                drift_list.append(abs(phase_drift_deg(xVd, t2, f_ref)))
                H2v, H3v, Vd1_ph = harmonic_ratios(xVd, t2, f_ref)
                harm_list.append(max(H2v, H3v))
                vmag_list.append(float(np.abs(Vd1_ph)))
                _, _, Id1_ph = harmonic_ratios(xId, t2, f_ref)
                imag_list.append(float(np.abs(Id1_ph)))
                V_ph[:, col] = np.array([
                    lockin_phasor(data2["Vd1"], t2, f_ref),
                    lockin_phasor(data2["Vq1"], t2, f_ref),
                    lockin_phasor(data2["Vd2"], t2, f_ref),
                    lockin_phasor(data2["Vq2"], t2, f_ref),
                ], dtype=complex)
                I_ph[:, col] = np.array([
                    lockin_phasor(data2["Id1"], t2, f_ref),
                    lockin_phasor(data2["Iq1"], t2, f_ref),
                    lockin_phasor(data2["Id2"], t2, f_ref),
                    lockin_phasor(data2["Iq2"], t2, f_ref),
                ], dtype=complex)
                break
            except ConnectionDeadError:
                raise
            except Exception as e:
                if is_connection_dead_error(e):
                    raise ConnectionDeadError(f"Connection dead in measure: {e}")
                if (is_recoverable_plot_error(e) or is_stale_object_error(e) or
                        is_unable_to_start_case_error(e)) and \
                        attempts < STALE_RETRIES_PER_EXPERIMENT:
                    attempts += 1
                    print(f"[RECOVER] col={col} -> HARD RECOVERY "
                          f"({attempts}/{STALE_RETRIES_PER_EXPERIMENT}): {e}")
                    case, Hinj, Hpcc = recover_by_reopen(app, case)
                    continue
                raise

    safe_stop(case, settle_s=STOP_SETTLE)
    for nm in ("inj1", "inj2"):
        try:
            Hinj[nm]["d_gain"].value = 0.0
            Hinj[nm]["q_gain"].value = 0.0
        except Exception:
            pass

    V_inv, detV, condV, used_pinv = inv_or_pinv(V_ph)
    Y_tot = I_ph @ V_inv

    q = {
        "A_slider":      float(A_slider),
        "settle_s":      float(Ts),
        "fs_est":        float(fs_est),
        "f_used_mean":   float(np.mean(f_used_list)) if f_used_list else float(f_inj),
        "f_used_list":   [float(x) for x in f_used_list],
        "drift_deg_max": float(np.max(drift_list)) if drift_list else 0.0,
        "harm_max":      float(np.max(harm_list))  if harm_list  else 0.0,
        "snr_flag_any":  bool(any(v < SNR_MIN_MAG_V for v in vmag_list) or
                              any(i < SNR_MIN_MAG_I for i in imag_list)),
        "detV":          detV,
        "condV":         float(condV),
        "used_pinv":     bool(used_pinv),
    }
    return Y_tot, q, case, Hinj, Hpcc


# ==================================================
# 11) SINGLE OPERATING POINT — same as v18
# ==================================================
def run_single_point(app, case, Hinj, Hpcc, case_key, inject_freqs):
    params  = GRID_CASES[case_key]
    scr_val = float(params["SCR"])
    xr_val  = float(params["XR"])
    S_sc_   = scr_val * Sbase_3ph
    Zm_     = (Vbase_LL ** 2) / S_sc_
    Rg_ohm  = Zm_ / np.sqrt(1.0 + xr_val ** 2)
    Xg_ohm  = xr_val * Rg_ohm
    Rg_pu   = Rg_ohm / Zbase
    Xg_pu   = Xg_ohm / Zbase

    eig_results_      = {}
    min_dist_results_ = {}

    for f in inject_freqs:
        A    = inj_amplitude(f)
        mode = "SLOW" if is_slow_mode(f) else "FAST"
        print(f"  [f={f:.1f} Hz | mode={mode} | A={A:.3f}]", end=" ", flush=True)

        try:
            Y_tot, q, case, Hinj, Hpcc = measure_Ytot_4x4_at(
                app, case, Hinj, Hpcc, f, A)
        except ConnectionDeadError as e:
            print(f"\n  [DEAD] Connection lost at f={f:.1f} Hz — "
                  f"frequency skipped. {e}")
            break

        Z_grid = build_Zgrid_4x4_pu(f, f0, Rg_pu, Xg_pu, Rg_pu, Xg_pu)
        L      = Z_grid @ (Y_tot - np.linalg.inv(Z_grid))
        eigvals = np.linalg.eigvals(L)
        eig_results_[f] = eigvals

        rsmm, n_near, n_far, used_fallback = compute_rsmm(L, eigvals=eigvals)
        min_dist_results_[f] = rsmm

        prox_info = f"near={n_near}/far={n_far}"
        print(f"RSMM={rsmm:.4f}  {prox_info}")

    all_freqs_measured = sorted(min_dist_results_.keys())
    print(f"  RSMM profile: {{ {', '.join(f'{f}: {round(min_dist_results_[f],3)}' for f in all_freqs_measured)} }}")

    all_freqs = sorted(eig_results_.keys())
    winding = 0
    if len(all_freqs) >= 2:
        I4       = np.eye(4, dtype=complex)
        det_vals = [np.linalg.det(I4 + np.diag(eig_results_[f]))
                    for f in all_freqs]
        winding, _ = winding_number_of_complex_curve(np.array(det_vals))

    verdict = "STABLE" if winding == 0 else "UNSTABLE"

    return (min_dist_results_, eig_results_,
            verdict, winding, case, Hinj, Hpcc)


# ==================================================
# 12) PLOTS
# ==================================================
_FONT_SIZE = 11
plt.rcParams.update({
    'font.size':        _FONT_SIZE,
    'axes.labelsize':   _FONT_SIZE,
    'xtick.labelsize':  _FONT_SIZE,
    'ytick.labelsize':  _FONT_SIZE,
    'legend.fontsize':  _FONT_SIZE,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
})
_SAVE = dict(format='svg', bbox_inches='tight', pad_inches=0.02)

def plot_eigenloci(eig_results_, freqs, title_suffix, out_path=None):
    all_freqs = sorted(eig_results_.keys())
    if len(all_freqs) < 2:
        print("  [PLOT] Insufficient points for eigenloci plot.")
        return
    lam_branches = track_eig_branches(eig_results_, all_freqs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Eigenloci — {title_suffix}", fontsize=11)
    for ax, zoom, ttl in zip(axes, [False, True],
                              ["Full (all 4 eigenvalues)", "Zoom around (-1, 0)"]):
        for i, lam in enumerate(lam_branches):
            ax.scatter(lam.real, lam.imag, s=10, label=f"λ{i+1}")
        ax.scatter([-1], [0], color="red", zorder=5, s=60, label="(-1,0)")
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(-1 + RSMM_PROXIMITY_THR * np.cos(theta),
                RSMM_PROXIMITY_THR * np.sin(theta),
                "--", color="orange", linewidth=0.8,
                label=f"RSMM zone (r={RSMM_PROXIMITY_THR})")
        ax.axhline(0, lw=0.5, color="gray")
        ax.axvline(0, lw=0.5, color="gray")
        ax.set_xlabel("Re{λ(jω)}")
        ax.set_ylabel("Im{λ(jω)}")
        ax.set_title(ttl)
        ax.legend(fontsize=7)
        ax.grid(True)
        if zoom:
            ax.set_xlim(ZOOM_XLIM)
            ax.set_ylim(ZOOM_YLIM)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, **_SAVE)
        print(f"Saved: {_os.path.basename(out_path)}")
    plt.show()

def plot_rsmm_profile(freqs, min_dist_results_, title_suffix, out_path=None):
    if not min_dist_results_:
        return
    all_f  = sorted(min_dist_results_.keys())
    rsmm_v = [min_dist_results_[f] for f in all_f]
    plt.figure(figsize=(10, 4))
    plt.scatter(all_f, rsmm_v, color="steelblue", s=60, zorder=5)
    plt.axhline(RSMM_TARGET, color="red", linestyle="--",
                label=f"Target RSMM = {RSMM_TARGET}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("RSMM = σ_min(I+L)")
    plt.title(f"RSMM profile — {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, **_SAVE)
        print(f"Saved: {_os.path.basename(out_path)}")
    plt.show()


def plot_single_sweep_summary(sweep_results, f_star, xlabel, title, out_path=None):
    param_vals    = [r["param_val"] for r in sweep_results]
    rsmm_fcrit    = [r["rsmm_profile"].get(f_star, float('nan')) for r in sweep_results]

    plt.figure(figsize=(9, 5))
    plt.scatter(param_vals, rsmm_fcrit, color="steelblue", s=80, zorder=5)
    plt.axhline(RSMM_TARGET, color="red", linestyle="--", linewidth=1.5,
                label=f"Target RSMM = {RSMM_TARGET}")
    plt.axvline(GF_BASELINE, color="green", linestyle=":", linewidth=1.5,
                label=f"Nominal KpPLL = {GF_BASELINE}")
    for pv, rv in zip(param_vals, rsmm_fcrit):
        if not (rv != rv):
            plt.annotate(f"{rv:.3f}", (pv, rv),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(f"RSMM(f*={f_star:.1f} Hz) = sigma_min(I+L)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, **_SAVE)
        print(f"Saved: {_os.path.basename(out_path)}")
    plt.show()


def plot_gf_per_scr_summary(gf_per_scr_results, f_star, out_path=None):
    colors = {"A": "#2196F3", "B": "#4CAF50", "C": "#FF9800", "D": "#F44336"}
    plt.figure(figsize=(10, 6))
    for case_key, points in gf_per_scr_results.items():
        kppll_vals = [p[0] for p in points]
        rsmm_vals  = [p[1] for p in points]
        scr        = GRID_CASES[case_key]["SCR"]
        color      = colors.get(case_key, "gray")
        plt.scatter(kppll_vals, rsmm_vals,
                    color=color, s=80, zorder=5,
                    label=f"Case {case_key} (SCR={scr:.1f})")
    plt.axhline(RSMM_TARGET, color="black", linestyle="--", linewidth=1.2,
                label=f"Target RSMM = {RSMM_TARGET}")
    plt.axvline(GF_BASELINE, color="green", linestyle=":", linewidth=1.2,
                label=f"Nominal KpPLL = {GF_BASELINE}")
    plt.xlabel("St2Gu1KpPLL — SRF-PLL proportional gain")
    plt.ylabel(f"RSMM(f*={f_star:.1f} Hz) = sigma_min(I+L)")
    plt.title(f"KpPLL sweep per SCR — RSMM(f*={f_star:.1f} Hz) vs St2Gu1KpPLL\n"
              f"St2Gu1TiPLL (GF1) = {KPVDC_FIXED} (fixed)  |  X/R = {XR_BASELINE:.0f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, **_SAVE)
        print(f"Saved: {_os.path.basename(out_path)}")
    plt.show()


def print_gf_per_scr_table(gf_per_scr_results, f_star):
    print(f"\n{'='*65}")
    print(f"  RESULTS TABLE — KpPLL vs RSMM(f*={f_star:.1f} Hz) per SCR")
    print(f"  (St2Gu1TiPLL fixed at {KPVDC_FIXED})")
    print(f"{'='*65}")
    print(f"  {'Case':>5}  {'SCR':>5}  {'KpPLL*':>8}  {'RSMM(f*)':>10}  {'Verdict':>10}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*10}")
    for case_key, points in gf_per_scr_results.items():
        scr = GRID_CASES[case_key]["SCR"]
        if not points:
            print(f"  {case_key:>5}  {scr:>5.1f}  {'--':>8}  {'--':>10}  {'--':>10}")
            continue
        best    = max(points, key=lambda p: p[1])
        verdict = "STABLE" if best[1] >= RSMM_TARGET else "MARGINAL"
        flag    = " v" if best[1] >= RSMM_TARGET else ""
        print(f"  {case_key:>5}  {scr:>5.1f}  {best[0]:>8.4f}  {best[1]:>10.4f}  {verdict:>10}{flag}")
    print(f"{'='*65}\n")


# ==================================================
# 13) SWEEP PLAN — FIXED
# ==================================================
def build_sweep_plan():
    n_pts = int((VALID_F_MAX - VALID_F_MIN) / F_STEP_HZ) + 1

    if SWEEP_MODE == "GF":
        if ACTIVE_CASE not in GRID_CASES:
            raise ValueError(f"ACTIVE_CASE='{ACTIVE_CASE}' unknown.")
        params = GRID_CASES[ACTIVE_CASE]
        print(f"\n{'='*60}")
        print(f"  MODE: KpPLL sweep at fixed grid condition")
        print(f"  Case   : {ACTIVE_CASE}  (SCR={params['SCR']:.1f}, X/R={params['XR']:.0f})")
        print(f"  R={params['R']:.3f} Ohm  L={params['L']:.3f} H")
        print(f"  KpPLL sweep : {GF_SWEEP_VALUES}")
        print(f"  TiPLL fixed : {KPVDC_FIXED}")
        print("  f* critical : auto-determined after sweep")
        print(f"  Freq   : {VALID_F_MIN}-{VALID_F_MAX} Hz ({n_pts} points)")
        print(f"  Auto-stop at RSMM > {RSMM_TARGET}")
        print(f"  Total runs (max): {len(GF_SWEEP_VALUES)*n_pts*4}")
        print(f"{'='*60}\n")
        return [("GF", gf, ACTIVE_CASE) for gf in GF_SWEEP_VALUES]

    elif SWEEP_MODE == "SCR":
        print(f"\n{'='*60}")
        print(f"  MODE: SCR sensitivity at fixed KpId={GF_BASELINE}")
        print(f"  KpVdc fixed: {KPVDC_FIXED}")
        print(f"  Cases  : A -> B -> C -> D")
        print(f"  Freq   : {VALID_F_MIN}-{VALID_F_MAX} Hz ({n_pts} points)")
        print(f"  Total runs: {len(GRID_CASES)*n_pts*4}")
        print(f"{'='*60}\n")
        return [("SCR", None, k) for k in GRID_CASES.keys()]

    elif SWEEP_MODE == "XR":
        if ACTIVE_CASE not in GRID_CASES:
            raise ValueError(f"ACTIVE_CASE='{ACTIVE_CASE}' unknown.")
        base_scr = GRID_CASES[ACTIVE_CASE]["SCR"]
        print(f"\n{'='*60}")
        print(f"  MODE: X/R sensitivity (SCR={base_scr:.1f} fixed, KpId={GF_BASELINE})")
        print(f"  KpVdc fixed: {KPVDC_FIXED}")
        print(f"  X/R sweep: {XR_SWEEP_VALUES}")
        print(f"  Freq: {VALID_F_MIN}-{VALID_F_MAX} Hz ({n_pts} points)")
        print(f"  Total runs: {len(XR_SWEEP_VALUES)*n_pts*4}")
        print(f"{'='*60}\n")
        return [("XR", xr, ACTIVE_CASE) for xr in XR_SWEEP_VALUES]

    elif SWEEP_MODE == "GF_PER_SCR":
        if ACTIVE_CASE == "ALL":
            cases = list(GRID_CASES.keys())
        elif ACTIVE_CASE in GRID_CASES:
            cases = [ACTIVE_CASE]
        else:
            raise ValueError(f"ACTIVE_CASE='{ACTIVE_CASE}' unknown.")
        n_runs = len(cases) * len(GF_SWEEP_VALUES) * n_pts * 4
        print(f"\n{'='*60}")
        print(f"  MODE: KpPLL sweep per SCR case")
        print(f"  Cases      : {cases}")
        print(f"  KpPLL sweep: {GF_SWEEP_VALUES}")
        print(f"  TiPLL fixed : {KPVDC_FIXED}")
        print("  f* critical : auto-determined after sweep")
        print(f"  Freq   : {VALID_F_MIN}-{VALID_F_MAX} Hz ({n_pts} points)")
        print(f"  Total runs: {n_runs}  (~{n_runs*17/3600:.1f} h)")
        print(f"{'='*60}\n")
        return [("GF_PER_SCR", gf, ck)
                for ck in cases
                for gf in GF_SWEEP_VALUES]

    else:
        raise ValueError(f"Unknown SWEEP_MODE: {SWEEP_MODE!r}. "
                         f"Choose: 'GF', 'SCR', 'XR', 'GF_PER_SCR'")


# ==================================================
# 14) MAIN — same as v18
# ==================================================
inject_freqs       = list(np.arange(VALID_F_MIN, VALID_F_MAX + 1e-9,
                                     F_STEP_HZ, dtype=float))
sweep_plan         = build_sweep_plan()
sweep_results      = []
gf_per_scr_results = {}
all_rsmm_profiles  = {}

with rtds.rscadfx.remote_connection() as app:
    case = app.open_case(casefile)
    print("Case opened.")
    safe_stop(case, settle_s=1.2)
    case.compile()
    time.sleep(AFTER_COMPILE_WAIT)
    Hinj, Hpcc = build_handles(case)
    print("Handles built. Starting sweep...\n")

    try:
        current_case_key = None

        for run_idx, (param_label, param_val, case_key) in enumerate(sweep_plan):

            if run_idx > 0:
                print(f"\n  [SETTLE] {BETWEEN_RUNS_SETTLE_S}s between runs...")
                time.sleep(BETWEEN_RUNS_SETTLE_S)

            if case_key != current_case_key:
                print(f"\n  [GRID] Switching to Case {case_key}...")
                set_grid_impedance(Hinj, case_key)
                current_case_key = case_key

            if SWEEP_MODE in ("GF", "GF_PER_SCR"):
                set_gf_slider(Hinj, param_val)
            elif SWEEP_MODE == "XR":
                set_grid_impedance_xr(Hinj,
                                      GRID_CASES[case_key]["SCR"],
                                      param_val)
                set_gf_slider(Hinj, GF_BASELINE)
            else:
                set_gf_slider(Hinj, GF_BASELINE)

            params = GRID_CASES[case_key]
            print(f"\n{'='*60}")
            print(f"  RUN {run_idx+1}/{len(sweep_plan)}")
            print(f"  Mode={SWEEP_MODE}  Case={case_key}  SCR={params['SCR']:.1f}  X/R={params['XR']:.0f}")
            if SWEEP_MODE in ("GF", "GF_PER_SCR"):
                print(f"  St2Gu1KpPLL (GF)  = {param_val:.4f}  [SWEEP]")
                print(f"  St2Gu1TiPLL (GF1) = {KPVDC_FIXED:.4f}  [FIXED]")
            print(f"{'='*60}")

            try:
                (min_dist_results_, eig_results_,
                 verdict, winding, case, Hinj, Hpcc) = run_single_point(
                    app, case, Hinj, Hpcc, case_key, inject_freqs
                )
            except ConnectionDeadError as e:
                print(f"\n  [DEAD] Connection lost during run {run_idx+1} — run skipped, sweep continues.")
                print(f"  Error: {e}")
                print(f"  Restart RSCAD manually and re-run if needed.")
                break

            profile_key = (param_val, case_key)
            all_rsmm_profiles[profile_key] = dict(min_dist_results_)

            result = {
                "param_label":  param_label,
                "param_val":    param_val,
                "case_key":     case_key,
                "scr":          params["SCR"],
                "xr":           params["XR"],
                "rsmm_profile": dict(min_dist_results_),
                "verdict":      verdict,
                "winding":      winding,
            }

            if SWEEP_MODE == "GF_PER_SCR":
                if case_key not in gf_per_scr_results:
                    gf_per_scr_results[case_key] = []
                gf_per_scr_results[case_key].append(
                    (param_val, dict(min_dist_results_)))
            else:
                sweep_results.append(result)

            print(f"\n  RESULT:")
            print(f"  Case={case_key}  SCR={params['SCR']:.1f}  "
                  + (f"KpPLL={param_val:.4f}" if SWEEP_MODE in
                     ("GF", "GF_PER_SCR") else ""))
            print(f"  GNC verdict : {verdict}  (winding={winding})")
            print(f"  RSMM profile: "
                  f"{{ {', '.join(f'{f}: {round(v,3)}' for f,v in sorted(min_dist_results_.items()))} }}")

            title_sfx = (f"Case {case_key} SCR={params['SCR']:.1f}  "
                         + (f"KpPLL={param_val:.4f}  TiPLL={KPVDC_FIXED}"
                            if SWEEP_MODE in ("GF", "GF_PER_SCR")
                            else f"KpPLL={GF_BASELINE}  TiPLL={KPVDC_FIXED}"))
            plot_eigenloci(eig_results_, inject_freqs, title_sfx,
                           out_path=_os.path.join(OUT_DIR, f"addon_eigenloci_run{run_idx+1:02d}.svg"))
            plot_rsmm_profile(inject_freqs, min_dist_results_, title_sfx,
                              out_path=_os.path.join(OUT_DIR, f"addon_rsmm_run{run_idx+1:02d}.svg"))

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
# 15) FINAL RESULTS — FIXED
# ==================================================
print(f"\n{'='*60}")
print(f"  SWEEP COMPLETE — mode: {SWEEP_MODE}")
print(f"  St2Gu1TiPLL (GF1) fixed at: {KPVDC_FIXED}")
print(f"{'='*60}")


def determine_fcrit(all_rsmm_profiles):
    if not all_rsmm_profiles:
        return None, {}
    all_freqs = set()
    for profile in all_rsmm_profiles.values():
        all_freqs.update(profile.keys())
    all_freqs = sorted(all_freqs)
    std_per_freq = {}
    for f in all_freqs:
        vals = [profile[f] for profile in all_rsmm_profiles.values()
                if f in profile]
        if len(vals) >= 2:
            std_per_freq[f] = float(np.std(vals))
    if not std_per_freq:
        return all_freqs[0] if all_freqs else None, std_per_freq
    f_star = max(std_per_freq, key=std_per_freq.get)
    print(f"\n  AUTO f* DETERMINATION:")
    print(f"  Std dev of RSMM per frequency across all runs:")
    for f in sorted(std_per_freq):
        marker = " <-- f* (highest variation)" if f == f_star else ""
        print(f"    f={f:.1f} Hz  std={std_per_freq[f]:.4f}{marker}")
    print(f"  => f* = {f_star:.1f} Hz")
    return f_star, std_per_freq


def plot_fcrit_motivation(std_per_freq, f_star, out_path=None):
    if not std_per_freq:
        return
    freqs_std = sorted(std_per_freq.keys())
    stds      = [std_per_freq[f] for f in freqs_std]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(freqs_std, stds, color="steelblue", alpha=0.7, width=0.6)
    for bar, f in zip(bars, freqs_std):
        if f == f_star:
            bar.set_color("tomato")
            bar.set_alpha(1.0)
    plt.axvline(f_star, color="red", linewidth=1.5, linestyle="--",
                label=f"f* = {f_star:.1f} Hz (highest variation)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Std dev of RSMM over KpPLL sweep")
    plt.title("Basis for f* selection — RSMM variation per frequency\n"
              "f* is the frequency most sensitive to KpPLL changes")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, **_SAVE)
        print(f"Saved: {_os.path.basename(out_path)}")
    plt.show()


f_star, std_per_freq = determine_fcrit(all_rsmm_profiles)
F_CRIT_FIXED = f_star

if f_star is None:
    print("  No data available for f* determination.")
else:
    plot_fcrit_motivation(std_per_freq, f_star,
                          out_path=_os.path.join(OUT_DIR, "addon_fcrit_motivation.svg"))

    if SWEEP_MODE == "GF_PER_SCR":
        gf_per_scr_fcrit = {}
        for case_key, points in gf_per_scr_results.items():
            gf_per_scr_fcrit[case_key] = []
            for param_val, profile in points:
                rsmm_fcrit = profile.get(f_star, float('nan'))
                gf_per_scr_fcrit[case_key].append((param_val, rsmm_fcrit))

        print_gf_per_scr_table(gf_per_scr_fcrit, f_star)
        plot_gf_per_scr_summary(gf_per_scr_fcrit, f_star,
                                out_path=_os.path.join(OUT_DIR, "addon_gf_per_scr_summary.svg"))

    else:
        param_vals = [r["param_val"] for r in sweep_results]
        rsmm_fcrit = [r["rsmm_profile"].get(f_star, float('nan'))
                      for r in sweep_results]
        verdicts   = [r["verdict"] for r in sweep_results]

        print(f"\n  {'KpPLL':>12}  {'RSMM(f*)':>10}  {'Verdict':>10}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
        for pv, rv, vd in zip(param_vals, rsmm_fcrit, verdicts):
            flag = " v" if rv >= RSMM_TARGET else ""
            print(f"  {pv:>12.4f}  {rv:>10.4f}  {vd:>10}{flag}")
        print(f"{'='*60}")

        if len(sweep_results) >= 2:
            plot_single_sweep_summary(
                sweep_results,
                f_star,
                xlabel="St2Gu1KpPLL — SRF-PLL proportional gain",
                title=(f"KpPLL sweep — Case {ACTIVE_CASE}  "
                       f"(SCR={GRID_CASES[ACTIVE_CASE]['SCR']:.1f}, "
                       f"X/R={GRID_CASES[ACTIVE_CASE]['XR']:.0f})  "
                       f"f*={f_star:.1f} Hz (auto-determined)"),
                out_path=_os.path.join(OUT_DIR, "addon_gf_summary.svg"),
            )