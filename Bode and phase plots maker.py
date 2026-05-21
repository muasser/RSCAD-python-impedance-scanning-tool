# -*- coding: utf-8 -*-
"""
Impedantie-extractie + GNC via eigenwaarden van L(jw)
+ FSAT-style Bode plots (Yconv, Zconv=inv(Yconv), Zgrid) en eigenvalue bode.

ROUTE A (RSCAD plotbuffer time-series via get_signal) + 2-run D/Q injectie

- Injectie: D-run en Q-run (zelfde frequentie), zodat V_exc 2x2 invertible is
- Reset: PB & PB1 via .position (case STOP)
- Lock-in phasor (synchronous detection)
- Bouw Y_conv(jw), Z_grid(jw), L(jw)=Y*Z, eigenwaarden
- Plots:
    * Bode: Ydd/Ydq/Yqd/Yqq
    * Bode: Zconv = inv(Yconv)  (handig als FSAT Z toont)
    * Bode: Zgrid
    * Eigenvalue Bode (|λ|, ∠λ) van L(jw)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import rtds.rscadfx

# --------------------------------------------------
# 0. Basisinstellingen
# --------------------------------------------------
casefile = r"C:\Users\muass\OneDrive\Documenten\RSCAD\RTDS_USER_FX\fileman\testcase\testcase.rtfx"

# Sweep: kies passend voor validatie (FSAT-vergelijk)
F_START = 1
F_END   = 100   # of 61
F_STEP  = 1
inject_freqs = list(range(F_START, F_END + 1, F_STEP))

A_inj = 0.05
SETTLE_S = 1.2
TIMEOUT_S = 25.0
MIN_SAMPLES = 1000
DET_THR = 1e-12
COND_THR = 1e6

# Grid params (jouw voorbeeld)
R_g = 0.1468
L_g = 0.003878
omega0 = 2.0 * np.pi * 60.0

# Save results (handig om later FSAT vergelijking offline te doen)
SAVE_RESULTS = True
OUT_NPZ = "scan_results_fsatstyle_noNorm.npz"

# --------------------------------------------------
# 1. Helpers
# --------------------------------------------------
def lockin_phasor(x, t, f_inj):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(x) < 8:
        raise RuntimeError("Te weinig samples voor lock-in.")
    x_ac = x - np.mean(x)
    N = len(x_ac)
    w = np.hanning(N)
    ww = np.mean(w) if np.mean(w) != 0 else 1.0
    ref = np.exp(-1j * 2.0 * np.pi * f_inj * t)
    ph = (2.0 / N) * np.sum((x_ac * w) * ref) / ww
    return ph

def get_timeseries_consistent(case, sig_handles, timeout_s=30.0, min_samples=1000):
    t_start = time.time()
    last_len = -1
    stable_count = 0

    while time.time() - t_start < timeout_s:
        case.update_plots()
        time.sleep(0.2)

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

            if stable_count >= 2:
                return t_ref, data

    raise RuntimeError("Kon geen consistente time series ophalen (timeout).")

def build_Zgrid_blockdiag(num_pcc, f, R_g, L_g, omega0):
    size = 2 * num_pcc
    s = 1j * 2.0 * np.pi * f
    Zdd = R_g + s * L_g
    Zdq = -omega0 * L_g
    Zqd =  omega0 * L_g

    Z_2x2 = np.array([[Zdd, Zdq],
                      [Zqd, Zdd]], dtype=complex)

    Z = np.zeros((size, size), dtype=complex)
    for p in range(num_pcc):
        i0 = 2 * p
        Z[i0:i0+2, i0:i0+2] = Z_2x2
    return Z

def pulse_reset(pb, pb1, pulse_s=0.08):
    pb.position = 1
    pb1.position = 1
    time.sleep(pulse_s)
    pb.position = 0
    pb1.position = 0
    time.sleep(0.05)

def cabs(z): return float(np.abs(z))

def print_excitation_report(f, pcc_name, Vd_d, Vq_d, Vd_q, Vq_q, detV, condV):
    eps = 1e-30
    d_dom = cabs(Vd_d) / (cabs(Vq_d) + eps)
    q_dom = cabs(Vq_q) / (cabs(Vd_q) + eps)
    print(f"\n[{pcc_name}] @ f={f:.2f} Hz")
    print(f"  det(V_exc) = {detV.real:+.3e}{detV.imag:+.3e}j")
    print(f"  cond(V_exc)= {condV:.3e}")
    print(f"  |Vd_d|={cabs(Vd_d):.6e}, |Vq_d|={cabs(Vq_d):.6e}  -> d-dom={d_dom:.3e}")
    print(f"  |Vd_q|={cabs(Vd_q):.6e}, |Vq_q|={cabs(Vq_q):.6e}  -> q-dom={q_dom:.3e}")

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

# ---------------- Plot helpers ----------------
def bode_mag_phase(H_vec):
    H_vec = np.asarray(H_vec, dtype=complex)
    mag = np.abs(H_vec)
    ph  = np.angle(H_vec, deg=True)
    ph = np.rad2deg(np.unwrap(np.deg2rad(ph)))
    return mag, ph

def plot_matrix_bode(freqs, M_of_f, title_prefix, pcc_name="PCC1"):
    freqs_sorted = np.array(sorted(M_of_f.keys()), dtype=float)

    elems = {
        "11": np.array([M_of_f[f][0,0] for f in freqs_sorted], dtype=complex),
        "12": np.array([M_of_f[f][0,1] for f in freqs_sorted], dtype=complex),
        "21": np.array([M_of_f[f][1,0] for f in freqs_sorted], dtype=complex),
        "22": np.array([M_of_f[f][1,1] for f in freqs_sorted], dtype=complex),
    }

    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} — {pcc_name}", fontsize=14)

    order = [("11","11"), ("12","12"), ("21","21"), ("22","22")]
    for row, (k, lab) in enumerate(order):
        mag, ph = bode_mag_phase(elems[k])

        axm = axes[row, 0]
        axp = axes[row, 1]

        axm.plot(freqs_sorted, mag, "-")
        axm.set_ylabel("Magnitude")
        axm.set_title(f"{title_prefix.split()[0]}{lab}")
        axm.grid(True)
        axm.set_xlabel("f (Hz)")

        axp.plot(freqs_sorted, ph, "-")
        axp.set_ylabel("Phase (deg)")
        axp.set_xlabel("f (Hz)")
        axp.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_eigenvalue_bode(freqs, lam_branches):
    freqs = np.asarray(freqs, dtype=float)

    plt.figure(figsize=(12, 5))
    for i, lam in enumerate(lam_branches):
        mag, _ = bode_mag_phase(lam)
        plt.plot(freqs, mag, "-", label=f"|λ{i+1}|")
    plt.xlabel("f (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Eigenvalue Bode — Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, lam in enumerate(lam_branches):
        _, ph = bode_mag_phase(lam)
        plt.plot(freqs, ph, "-", label=f"∠λ{i+1}")
    plt.xlabel("f (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title("Eigenvalue Bode — Phase")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 2. PCC configuratie
# --------------------------------------------------
pcc_list = [
    {
        "name": "PCC1",
        "Vd_path": "Subsystem #1|CTLs|Vars|Vdpu11",
        "Vq_path": "Subsystem #1|CTLs|Vars|Vqpu11",
        "Id_path": "Subsystem #1|CTLs|Vars|Id",
        "Iq_path": "Subsystem #1|CTLs|Vars|Iq",
    },
]
num_pcc = len(pcc_list)
size = 2 * num_pcc

# --------------------------------------------------
# 3. Result containers
# --------------------------------------------------
Y_results = {}
Zgrid_results = {}
L_results = {}
eig_results = {}

Y_pcc = {}       # f -> 2x2
Zconv_pcc = {}   # f -> 2x2 = inv(Y_pcc)
Zgrid_pcc = {}   # f -> 2x2

bad_freqs = []

# --------------------------------------------------
# 4. RSCAD sweep (D/Q runs)
# --------------------------------------------------
with rtds.rscadfx.remote_connection() as app:
    case = app.open_case(casefile)
    print("Case opened.")
    case.stop()
    case.compile()

    Finj_var   = case.get_object_by_name("Finj", "slider")
    d_gain_var = case.get_object_by_name("d_gain", "slider")
    q_gain_var = case.get_object_by_name("q_gain", "slider")
    PB_var  = case.get_object_by_name("PB", "button")
    PB1_var = case.get_object_by_name("PB1", "button")

    sigV = {}
    sigI = {}
    for pcc in pcc_list:
        sigV[f"{pcc['name']}_Vd"] = case.get_signal(pcc["Vd_path"])
        sigV[f"{pcc['name']}_Vq"] = case.get_signal(pcc["Vq_path"])
        sigI[f"{pcc['name']}_Id"] = case.get_signal(pcc["Id_path"])
        sigI[f"{pcc['name']}_Iq"] = case.get_signal(pcc["Iq_path"])

    try:
        for f in inject_freqs:
            print("\n====================================")
            print(f"Injectiefrequentie: f = {f:.2f} Hz")
            print("====================================")

            Vd_ph_modes = np.zeros((2, num_pcc), dtype=complex)
            Vq_ph_modes = np.zeros((2, num_pcc), dtype=complex)
            Id_ph_modes = np.zeros((2, num_pcc), dtype=complex)
            Iq_ph_modes = np.zeros((2, num_pcc), dtype=complex)

            for mode_idx, (d_gain, q_gain) in enumerate([(A_inj, 0.0), (0.0, A_inj)]):
                mode_name = "D-RUN" if mode_idx == 0 else "Q-RUN"
                print(f"  --> Run {mode_idx+1}: {mode_name} (d_gain={d_gain}, q_gain={q_gain})")

                case.stop()
                Finj_var.value = float(f)
                d_gain_var.value = float(d_gain)
                q_gain_var.value = float(q_gain)

                pulse_reset(PB_var, PB1_var, pulse_s=0.08)

                case.run()
                time.sleep(SETTLE_S)

                for _ in range(5):
                    case.update_plots()
                    time.sleep(0.15)

                sig_handles = {}
                sig_handles.update(sigV)
                sig_handles.update(sigI)
                t, data = get_timeseries_consistent(case, sig_handles, timeout_s=TIMEOUT_S, min_samples=MIN_SAMPLES)

                for p_idx, pcc in enumerate(pcc_list):
                    keyVd = f"{pcc['name']}_Vd"
                    keyVq = f"{pcc['name']}_Vq"
                    keyId = f"{pcc['name']}_Id"
                    keyIq = f"{pcc['name']}_Iq"

                    Vd_ph_modes[mode_idx, p_idx] = lockin_phasor(data[keyVd], t, f)
                    Vq_ph_modes[mode_idx, p_idx] = lockin_phasor(data[keyVq], t, f)
                    Id_ph_modes[mode_idx, p_idx] = lockin_phasor(data[keyId], t, f)
                    Iq_ph_modes[mode_idx, p_idx] = lockin_phasor(data[keyIq], t, f)

            case.stop()
            d_gain_var.value = 0.0
            q_gain_var.value = 0.0

            Y_big = np.zeros((size, size), dtype=complex)
            freq_ok = True

            for p_idx, pcc in enumerate(pcc_list):
                r0 = 2 * p_idx
                r1 = r0 + 1

                Vd_d = Vd_ph_modes[0, p_idx]; Vq_d = Vq_ph_modes[0, p_idx]
                Id_d = Id_ph_modes[0, p_idx]; Iq_d = Iq_ph_modes[0, p_idx]

                Vd_q = Vd_ph_modes[1, p_idx]; Vq_q = Vq_ph_modes[1, p_idx]
                Id_q = Id_ph_modes[1, p_idx]; Iq_q = Iq_ph_modes[1, p_idx]

                V_exc = np.array([[Vd_d, Vd_q],
                                  [Vq_d, Vq_q]], dtype=complex)
                I_meas = np.array([[Id_d, Id_q],
                                   [Iq_d, Iq_q]], dtype=complex)

                detV = np.linalg.det(V_exc)
                condV = np.linalg.cond(V_exc)

                print_excitation_report(f, pcc["name"], Vd_d, Vq_d, Vd_q, Vq_q, detV, condV)

                if abs(detV) < DET_THR or condV > COND_THR:
                    print(f"  [SKIP] V_exc slecht conditioneerd bij f={f} Hz, {pcc['name']}")
                    freq_ok = False
                    break

                Y_2x2 = I_meas @ np.linalg.inv(V_exc)
                Y_big[r0:r1+1, r0:r1+1] = Y_2x2

            if not freq_ok:
                bad_freqs.append(f)
                continue

            Y_results[f] = Y_big
            Y_pcc[f] = Y_big[0:2, 0:2]

            Z_grid = build_Zgrid_blockdiag(num_pcc, f, R_g, L_g, omega0)
            Zgrid_results[f] = Z_grid
            Zgrid_pcc[f] = Z_grid[0:2, 0:2]

            try:
                Zconv_pcc[f] = np.linalg.inv(Y_pcc[f])
            except np.linalg.LinAlgError:
                bad_freqs.append(f)
                continue

            # tekenconventie check later: evt. -Y_big @ Z_grid
            L = Y_big @ Z_grid
            L_results[f] = L
            eig_results[f] = np.linalg.eigvals(L)

    finally:
        try:
            case.stop()
        except Exception:
            pass
        try:
            d_gain_var.value = 0.0
            q_gain_var.value = 0.0
        except Exception:
            pass
        try:
            case.close()
        except Exception:
            pass

# --------------------------------------------------
# 5. Plots (FSAT-style, in Hz)
# --------------------------------------------------
freqs = sorted(Y_pcc.keys())
if len(freqs) == 0:
    raise RuntimeError("Geen geldige resultaten. Check cond/det, signals, window size.")

print(f"\nOK freq points: {len(freqs)}  |  Skipped: {len(bad_freqs)}")
if len(bad_freqs) > 0:
    print("Skipped freqs:", bad_freqs)

plot_matrix_bode(freqs, Zgrid_pcc,  title_prefix="Z_grid", pcc_name="PCC1")
plot_matrix_bode(freqs, Y_pcc,      title_prefix="Y_conv", pcc_name="PCC1")
plot_matrix_bode(freqs, Zconv_pcc,  title_prefix="Z_conv (=inv(Y_conv))", pcc_name="PCC1")

lam_branches = track_eig_branches(eig_results, freqs)
plot_eigenvalue_bode(freqs, lam_branches)

# --------------------------------------------------
# 6. Save results
# --------------------------------------------------
if SAVE_RESULTS:
    freqs_arr = np.array(freqs, dtype=float)
    Y_arr = np.array([Y_pcc[f] for f in freqs], dtype=complex)
    Zg_arr = np.array([Zgrid_pcc[f] for f in freqs], dtype=complex)
    Zc_arr = np.array([Zconv_pcc[f] for f in freqs], dtype=complex)
    L_eigs = np.array([eig_results[f] for f in freqs], dtype=complex)

    np.savez(
        OUT_NPZ,
        freqs=freqs_arr,
        Y_pcc=Y_arr,
        Zgrid_pcc=Zg_arr,
        Zconv_pcc=Zc_arr,
        eigvals=L_eigs,
        skipped=np.array(bad_freqs, dtype=float),
    )
    print(f"\nSaved: {OUT_NPZ}")
