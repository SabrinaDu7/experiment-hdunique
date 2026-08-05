# FYI — quirks that bite more than once

Small facts about the data and the tooling that were each found while chasing one bug, and each turn
out to matter somewhere else. A quirk belongs here when **knowing it would change how you write
unrelated code**. The full story of any one of them lives in the results document of the question
that found it; this file is the index, not the record.

Newest first.

---

## DANDI has no head-direction field — the LEDs are the only source

*Found 2026-08-05 in `angular1`. Affects: anything wanting a measured head angle.*

`dandiset 000056` carries `acquisition/RedLED` and `acquisition/BlueLED` (duplicated under
`processing/behavior/SubjectPosition/`) and nothing else behavioural. There is no head-direction
column. `stimulus/presentation/hdD` looks like one and is not: it holds the strings `"HD UP ON"` /
`"HD DOWN ON"`, i.e. digital event markers.

So head direction must be computed as `arctan2` of the red-minus-blue vector. The CRCNS `.ang` files
contain the original computed angle, but they are **not** part of DANDI, and this repo does not
depend on them — they are useful only as an independent check, and one has now been done (below).

## `-1` in the LED positions means "detection failed", and it is finite

*Found 2026-08-05 in `angular1`. Affects: head direction, angular speed, tuning curves, and anything
else derived from position.*

The NWB says so in its own description: *"Raw sensor data. Values of -1 indicate that LED detection
failed."* The trap is that `-1` passes `np.isfinite`, so the obvious guard keeps it and treats
`(-1, -1)` as a position the animal was actually at.

The failure rate is very uneven across animals, which is what makes this dangerous — it corrupts some
animals far more than others and so **manufactures between-animal differences**, which is the exact
quantity this repo exists to measure:

| session | samples with a failed detection |
|---|---|
| Mouse25-140130 | 0.4% |
| Mouse28-140313 | 1.8% |
| Mouse17-130130 | 5.5% |
| Mouse12-120806 | 14.8% |
| Mouse12-120808 | **20.5%** |

Keeping them inflated measured net angular speed by 1.48× for Mouse12-120808 and 1.02× for
Mouse25-140130 — in rank order of the failure rate. Dropping them brings agreement with the
independently computed `.ang` to **1.000 for every animal tested**.

Guard with `decode.head_direction.FAILED_DETECTION`, not with `isfinite`.

## The LED-to-head-axis convention differs per animal by a constant rotation

*Found 2026-08-05 in `angular1`. Affects: any absolute head direction. Harmless to speed.*

Compared against `.ang` over the 19 sessions with CRCNS data, our LED angle differs by a rotation
that is tight within each animal and different between them:

| animal | rotation |
|---|---|
| Mouse25 | ≈ 0 |
| Mouse12, Mouse17 | ≈ +π/2 |
| Mouse28 | ≈ ±π |

Presumably the LEDs sit differently on each animal's headstage, and the original pipeline corrected
for it per animal. After the `-1` fix the circular correlation is 0.9998–1.0000, so **the rotation is
the entire remaining difference**.

It cancels in everything currently computed — differencing (angular speed), tuning-curve correlations
ρ(θ), and the decode RMSE, which fits an offset and a flip anyway. It would matter the moment a
preferred direction, or any comparison of absolute heading between animals, is reported.

## A state label in DANDI is not a behavioural session

*Found 2026-08-05 in `angular1`. Affects: every question that reads `load_state_epochs`.*

The `states` table scores a whole day-long recording, so `"Awake"` is a union: several
multi-thousand-second bouts separated by sleep, plus dozens of brief arousals. Mouse28-140313 has 67
"Awake" intervals totalling 17045 s, median 76 s, longest 3875 s.

Pooling them and fitting one ring to the concatenation decodes at 0.97–1.48 rad against measured head
direction, where chance is π/√3 = 1.81. The same code on a single contiguous bout gives 0.39–0.43.
Head speed also differs about two-fold between a session's longest wake bout and the rest of its
wake, so the union is not one behaviour in any sense.

Choose bouts explicitly (`loader.contiguous_bouts`) rather than inheriting the union. **The REM path
still pools**, and whether that costs anything is not yet established.

The corollary for wake: most wake bouts do not decode at all. Only 2 of 14 bouts of at least 300 s
across two sessions clear the 0.5 rad bar — the long behavioural ones. The rest are rest-box wake,
where the animal barely turns and the ring is never traversed.

## Path-length angular speed is not a property of the head

*Found 2026-08-05 in `angular1`. Affects: any comparison of two angular signals.*

Mean |dθ/dt| accumulates tracking jitter, so it depends on how finely you sample. On the same wake
bout, decimating from 39 Hz to 2.4 Hz drops it from 1.675 to 0.688 rad/s. Net displacement over a
fixed window does not: 0.467 → 0.419 over the same range.

Two signals at different effective bandwidths — a 39 Hz LED trace and a 10 Hz decode — therefore
cannot be compared on path length; the difference in sampling reads as a difference in speed. Use
`net_speed` at a stated τ for anything being graded, and report path length only alongside it.

## A figure token must start with `FIG_`

*Found 2026-08-05 in `angular1`. Affects: any new experiment producing figures.*

`@FIGURES@` collects tokens by that prefix, and `render`'s unreferenced-figure error keys off it too.
A figure recorded under any other name is invisible to both: it renders as nothing and raises
nothing. Two real figures went missing from a results document this way.

`values.figure()` now forces the prefix, so this should not recur — but the same class of trap
applies to any convention enforced by prefix rather than by type.
