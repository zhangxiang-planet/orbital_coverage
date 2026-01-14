#!/usr/bin/env python3
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from astropy.time import Time
from astropy import units as u
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from scipy.optimize import newton


# ---------- Orbital phase helpers (copied + simplified from your general code) ----------
def kepler_equation(E, M, e):
    return E - e * np.sin(E) - M

def solve_eccentric_anomaly(M, e):
    return newton(kepler_equation, M, args=(M, e))

def true_anomaly(M, e):
    E = solve_eccentric_anomaly(M, e)
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

def calculate_orbital_phase_array(times_jd, jd0, p_days, eccen):
    """
    Phase in [0,1). For e >= 0.1, use true anomaly / 2pi (as in your general script).
    For e < 0.1, use mean anomaly / 2pi.
    """
    M = 2 * np.pi * (times_jd - jd0) / p_days
    if eccen >= 0.1:
        nu = np.array([true_anomaly(m, eccen) for m in M])  # radians
        phase = (nu % (2 * np.pi)) / (2 * np.pi)
    else:
        phase = (M % (2 * np.pi)) / (2 * np.pi)
    return np.mod(phase, 1.0)


# ---------- NenuFAR imaging directory parsing ----------
def parse_directory_name(dirname):
    # expects: YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS_...
    start_date, start_time, end_date, end_time = dirname.split("_")[:4]
    return start_date, start_time, end_date, end_time

def to_time(date_str, time_str):
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return Time(dt, scale="utc")

def split_segments(t0: Time, t1: Time, step_min=10):
    """Return Time array sampled from t0 to < t1 with cadence step_min minutes."""
    step = step_min * u.min
    n = int(np.floor(((t1 - t0).to(u.min) / step).value))
    if n <= 0:
        return Time([], format="jd", scale="utc")
    return t0 + np.arange(n) * step


def main():
    # ====== User-specific knobs ======
    PLANET_NAME = "XO-3 b"   # NASA archive name
    FIELD_TOKEN = "XO-3"      # <-- CHANGE if your dirname contains XO_3 or XO-3 etc.
    CADENCE_MIN = 10         # for both background and segment splitting

    # Fixed month window (UTC)
    t_month_start = Time("2023-12-01T00:00:00", format="isot", scale="utc")
    t_month_end   = Time("2024-01-01T00:00:00", format="isot", scale="utc")

    # NenuFAR imaging base paths (Dec 2023 only)
    base_paths = [
        "/databf/nenufar-nri/LT02/2023/12/",
        "/databf/nenufar-nri/ES02/2023/12/",
    ]

    # ====== Get orbital parameters (AUTO, simplified) ======
    table = NasaExoplanetArchive.query_object(PLANET_NAME)
    # Keep published confirmed if possible, otherwise fall back to whatever exists
    confirmed_mask = table['soltype'] == 'Published Confirmed'
    confirmed_table = table[confirmed_mask]

    # Calculate the average eccentricity (excluding NaN values)
    eccentricity_values = confirmed_table['pl_orbeccen']
    average_eccentricity = np.nanmean(eccentricity_values)

    # Determine if the orbit is eccentric
    is_eccentric = average_eccentricity >= 0.1

    # Apply further filtering based on orbit type
    if is_eccentric:
        # For eccentric orbits, ensure epoch of periastron is available
        mask = ~np.isnan(confirmed_table['pl_orbtper'])
    else:
        # For circular orbits, proceed with non-zero transit midpoint
        mask = confirmed_table['pl_tranmid'] > 0

    if np.sum(mask) == 0:
        mask = ~np.isnan(confirmed_table['pl_orbper'])

    # Apply the mask and find the reference with the smallest error in orbital period
    if np.any(mask):
        filtered_table = confirmed_table[mask]
        small_err = np.nanmin(filtered_table['pl_orbpererr1'] - filtered_table['pl_orbpererr2'])
        best_reference_mask = (filtered_table['pl_orbpererr1'] - filtered_table['pl_orbpererr2']) == small_err
        best_reference = filtered_table[best_reference_mask]
        print("Best reference found.")
    else:
        print("No exoplanets match the criteria.")

    reference = best_reference

    if len(reference) > 1:
        print("Multiple references found. Using the first one.")
        reference = reference[0]
        
    p_e = float(reference['pl_orbper'].value)
    p_e_error = float(small_err.value / 2)
    p_days = p_e  # Orbital period in days

    if is_eccentric:
        jd0 = float(reference['pl_orbtper'].value)  # Epoch of periastron for eccentric orbits
        eccen = float(reference['pl_orbeccen'].value)
    else:
        jd0 = float(reference['pl_tranmid'].value)  # Transit midpoint for circular orbits
        eccen = 0


        # Will lose phase coherence (10 percent of one orbit) by what time?
    esti_cut = float((p_e/p_e_error)*(0.1*p_e) + jd0)
    # print(esti_cut)
    esti_cut_utc = Time(esti_cut, format='jd', scale='utc').iso

    # print("Reference Name:", extract_reference_name(reference['pl_refname']))  # Replace with actual column name if different
    # print("Publication Date:", reference['pl_pubdate'])
    print("Orbital Period:", p_e)  # Replace with actual column name if different
    print("Error in Orbital Period:", p_e_error)  # Replace with actual column name if different
    jd0_description = "Time of Periastron:" if is_eccentric else "Time of Conjunction (Transit Midpoint):"
    print(jd0_description, jd0)


    print(f"Using {PLANET_NAME}: P={p_days:.8f} d, e={eccen:.3f}, jd0={jd0:.6f}")

    # ====== Background phases_all for the whole month ======
    n_steps = int(np.floor(((t_month_end - t_month_start).to(u.min) / (CADENCE_MIN * u.min)).value))
    times_all = t_month_start + np.arange(n_steps) * (CADENCE_MIN * u.min)
    phases_all = calculate_orbital_phase_array(times_all.jd, jd0, p_days, eccen)

    # ====== Find Dec 2023 observations and build collected segments ======
    seg_times = []
    seg_phases = []

    obs_starts = []
    obs_ends = []

    for bp in base_paths:
        if not os.path.isdir(bp):
            continue
        for dirname in os.listdir(bp):
            if FIELD_TOKEN not in dirname:
                continue
            if "CALIBRATOR" in dirname:
                continue

            try:
                sd, st, ed, et = parse_directory_name(dirname)
            except Exception:
                continue

            t0 = to_time(sd, st)
            t1 = to_time(ed, et)

            # Keep only observations overlapping Dec 2023
            if t1 <= t_month_start or t0 >= t_month_end:
                continue

            obs_starts.append(t0.jd)
            obs_ends.append(t1.jd)

            ts = split_segments(t0, t1, step_min=CADENCE_MIN)
            if len(ts) == 0:
                continue
            ph = calculate_orbital_phase_array(ts.jd, jd0, p_days, eccen)
            seg_times.append(ts.to_datetime())
            seg_phases.append(ph)

    if len(seg_times) == 0:
        raise RuntimeError("No matching observations found. Check FIELD_TOKEN and paths.")

    seg_times = np.concatenate(seg_times)
    seg_phases = np.concatenate(seg_phases)

    obs_starts = np.array(obs_starts)
    obs_ends = np.array(obs_ends)

    # ====== Phase-coverage histogram from observation start/end ======
    num_bins = 360
    bin_edges = np.linspace(0, 1, num_bins + 1)
    phase_coverage = np.zeros(num_bins)

    ph_start = calculate_orbital_phase_array(obs_starts, jd0, p_days, eccen)
    ph_end   = calculate_orbital_phase_array(obs_ends,   jd0, p_days, eccen)

    for ps, pe in zip(ph_start, ph_end):
        start_bin = int(ps * num_bins)
        end_bin = int(pe * num_bins)
        if end_bin < start_bin:  # wrap
            phase_coverage[start_bin:] += 1
            phase_coverage[: end_bin + 1] += 1
        else:
            phase_coverage[start_bin : end_bin + 1] += 1

    # ====== Combined figure ======
    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.02}
    )

    # Background (all phases in the month)
    ax1.scatter(
        phases_all,
        times_all.to_datetime(),
        c="gray", s=3, alpha=0.2, marker="o", zorder=1, label="Orbital phases of XO-3 b"
    )

    # Collected segments
    ax1.scatter(
        seg_phases,
        seg_times,
        s=15, alpha=1, marker="o", zorder=3, label="Observations of XO-3 b",
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylabel("Time (UTC)")
    ax1.set_title("XO-3 b orbital-phase coverage")
    ax1.legend()
    ax1.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Histogram
    ax2.bar(bin_edges[:-1], phase_coverage, width=1/num_bins, align="edge")
    ax2.set_xlabel("Orbital phase")
    ax2.set_ylabel("N obs")
    ax2.set_xticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    out = "XO3_NENUFAR_IMAGING_2023-12_combined.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="w")
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()