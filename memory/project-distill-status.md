---
name: project-distill-status
description: Where the supercombo distillation stands (as of 2026-10-31 22:40) and where the full record lives
metadata:
  node_type: memory
  type: project
  originSessionId: fdabf76d-b147-4353-b3fb-ed51f739931a
  modified: 2026-10-31T11:58:10.440Z
---

**2026-11-01:**
- The road test of lrg_calL FAILED. Laneless driving drifted left: the plan aimed 20 cm left of the lane centre. It was also weak at intersections without lane lines, and the path looked squashed on hills.
- The offline plan error at 3-4 s is worse than the shipped model's; the 2 s metric hid it.
- Training moved to the Linux PC. The handover was sent to the Remote Control session "Supercombo network 학습(Linux)" at 04:20. Details are in HANDOVER.md §00000000000000.
- The Mac does no more training. Its disk is nearly full.

**Eval-metric goal met 2026-10-31 (not on the road).** The final model is **lrg_calL**: ckpt/mac_lrg/ema_calL.pt, built as outputs/quant_lrg_calL_ext/m.axmodel. It was verified at 22:38 on the NPU (p2sim) and the board, on 5 routes pooled:

| | shipped | lrg_calL |
|---|---|---|
| Latency (ms) | 14.0 | 11.21 |
| plan | 9.94 | 8.93 |
| Lanes, slope-fixed (cm) | 14.86 | 10.61 |
| Width, slope-fixed (cm) | 16.46 | 14.31 |
| lead_x | 5.83 | 4.30 |
| False / missed lead (%) | 5.06 / 10.59 | 5.04 / 9.47 |

The false/missed lead cross-validated (LORO) estimate is 4.79 / 9.74.

How it was built:
- LRF is LRC retrained with slope-fixed lane labels.
- LRG adds Quality Focal Loss on lead_prob (train.py --lead-prob-focal 2).
- A logistic far-lead calibration is applied on top (lead_cal.py --logistic).

lrf_calL (20:56, LRF + the same calibration) also beat the shipped model on every metric; lrg_calL is better on most.

**Road test prep (2026-11-01):**
- The edgepilot branch `student-cpu-heads` (uncommitted, in ~/Documents/ax630/edgepilot) adds a CPU-heads contract: the axmodel outputs hx_* and modeld reads a .heads file.
- The package is supercombo_distill/deploy/lrg_calL/. Run scripts/deploy_student.py in this order: install, check, select [--shadow], rollback [--params].
- The board install/check is not done yet: the board was offline at 03:00.

**Weak spots:**
- The daytime route 2026-10-05--16-28 (tunnels, ~40 km/h; earlier mislabelled "night highway") loses on plan and on false leads.
- Lead AUC is 0.9764 vs 0.9772.
- The next lever is more daytime driving data: only 36 training units are daytime.

**Earlier mistakes that were corrected:**
- Route names are in KST. train.py --night-weight's luma < 150 test picks mostly daytime units (auto exposure), so it has effectively been a day weight. Behaviour kept, docs fixed.
- The 2026-10-30 "goal met" claim (R18q) relied on the S8 sim.
- LRC's lane win came from LD lane labels projected onto a flat road. The fix is scripts/lane_geom.py slope_fix and train.py --ld-slope-fix; judge lanes on lanes_ldfix_cm / width_ldfix_cm.
- The grid-fit lead calibration overfit (LORO lost on missed leads). Use lead_cal.py --logistic.

**Why:** Judge only on NPU-equivalent numbers, check that the labels are right, and cross-validate anything fitted on the eval routes. The full history is in HANDOVER.md section 1 and reports/experiments.html.

**How to apply:** Read those before continuing rather than re-deriving. Related: [[user-working-style]].
