"""Is the *measured* wake angular speed right? Four checks, only one of which is self-referential."""
from pathlib import Path

import numpy as np

from decode import head_direction as hdmod
from decode import loader

CRCNS = Path("/home/sabrina/Documents/experiments/datasets/th1")

def circ_corr(a, b):
    return float(np.abs(np.mean(np.exp(1j * (a - b)))))

def report(mouse, session):
    sid = f"Mouse{mouse}-{session}"
    print(f"\n===== {sid} =====")
    data = loader.load_session(mouse=mouse, session=session)
    hd = hdmod.head_direction(data=data)
    t = np.asarray(hd.index, float); a = np.asarray(hd.values, float)

    # --- A. round trip: signed velocity integrated back to the angle it came from
    wake = loader.longest_bout(epochs=loader.load_state_epochs(data=data, state="Awake"))
    seg = hd.restrict(wake)
    ts, as_ = np.asarray(seg.index, float), np.asarray(seg.values, float)
    v = hdmod.angular_velocity(angles=as_, times=ts)
    recon = hdmod.integrate_velocity(velocity=v, times=ts, initial_angle=float(as_[0]))
    err = np.abs((recon - as_ + np.pi) % (2 * np.pi) - np.pi)
    print(f"A. round trip on {len(ts)} samples: max err {err.max():.2e} rad, mean {err.mean():.2e}")

    # --- B. independent implementation: Neuroscope's own .ang for the same session
    ang_path = CRCNS / sid / f"{sid}.ang"
    if ang_path.exists():
        ref = np.fromfile(str(ang_path), sep="\n").astype(float)
        ref[ref < -0.5] = np.nan
        ref = ref % (2 * np.pi)
        n = min(len(ref), len(a))
        if len(ref) == len(a):
            ok = np.isfinite(ref[:n]) & np.isfinite(a[:n])
            d = (ref[:n][ok] - a[:n][ok] + np.pi) % (2 * np.pi) - np.pi
            print(f"B. vs CRCNS .ang on {ok.sum()} shared samples: circ corr {circ_corr(ref[:n][ok], a[:n][ok]):.4f}, "
                  f"median |diff| {np.median(np.abs(d)):.4f} rad, 95th {np.percentile(np.abs(d), 95):.4f}, "
                  f"constant offset {np.angle(np.mean(np.exp(1j*d))):.4f}")
            # and the speed each implementation implies, same code, same epochs
            for label, series in (("ours", a), ("crcns", ref[:n])):
                good = np.isfinite(series[:n])
                sp = hdmod.angular_speed(angles=series[:n][good], times=t[:n][good], smooth_s=0.1)
                g = np.diff(t[:n][good])
                print(f"   path speed from {label:5s}: {np.mean(sp[g < 5*np.median(g)]):.3f} rad/s")
        else:
            print(f"B. .ang length {len(ref)} != LED length {len(a)}; not sample-aligned")
    else:
        print("B. no CRCNS .ang for this session")

    # --- C. how much of the 'measured' speed is tracking jitter?
    print("C. bandwidth sensitivity (longest wake bout):")
    print(f"   {'decim':>6} {'eff Hz':>7} {'path (smooth 0)':>16} {'path (smooth .1)':>17} {'net@1s':>8}")
    for k in (1, 2, 4, 8, 16):
        tk, ak = ts[::k], as_[::k]
        row = [f"   {k:>6d} {1/np.median(np.diff(tk)):>7.1f}"]
        for sm in (0.0, 0.1):
            sp = hdmod.angular_speed(angles=ak, times=tk, smooth_s=sm)
            g = np.diff(tk)
            row.append(f"{np.mean(sp[g < 5*np.median(g)]):>16.3f}" if sm == 0 else f"{np.mean(sp[g < 5*np.median(g)]):>17.3f}")
        row.append(f"{hdmod.net_speed(angles=ak, times=tk, tau_s=1.0):>8.3f}")
        print("".join(row))

    # --- D. physical cross-check: head direction vs direction of travel while running
    keys = set(data.keys())
    for red_key, blue_key in hdmod.LED_KEYS:
        if red_key in keys and blue_key in keys:
            red, blue = data[red_key], data[blue_key]
            break
    pos = (np.asarray(red.values, float) + np.asarray(blue.values, float)) / 2.0
    tp = np.asarray(red.index, float)
    inside = (tp >= float(np.asarray(wake.start)[0])) & (tp <= float(np.asarray(wake.end)[0]))
    pos, tp = pos[inside], tp[inside]
    step = 8  # ~5 Hz, so translation between samples exceeds tracking noise
    dp = pos[step:] - pos[:-step]
    dist = np.hypot(dp[:, 0], dp[:, 1])
    travel = np.arctan2(dp[:, 1], dp[:, 0]) % (2 * np.pi)
    hd_here = a[inside][: len(travel)]
    fast = np.isfinite(dist) & np.isfinite(hd_here) & (dist > np.nanpercentile(dist, 80))
    d = (travel[fast] - hd_here[fast] + np.pi) % (2 * np.pi) - np.pi
    print(f"D. head direction vs direction of travel, fastest 20% ({fast.sum()} samples): "
          f"circ corr {circ_corr(travel[fast], hd_here[fast]):.4f}, "
          f"offset {np.angle(np.mean(np.exp(1j*d))):.3f} rad, "
          f"median |resid| {np.median(np.abs(d - np.angle(np.mean(np.exp(1j*d))))):.3f}")

if __name__ == "__main__":
    for m, s in ((25, 140130), (28, 140313)):
        report(m, s)
