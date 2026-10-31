# supercombo_distill (Mac → Linux 인수인계, 2026-11-01)

openpilot big 주행 모델(teacher)을 MaixCAM2(AX630C NPU)용 student로 증류한 작업이다.
Mac에서 한 작업을 Linux 학습 서버로 넘기려고 올린 저장소다.
먼저 `HANDOVER.md` 1절 맨 위 네 절과 `reports/experiments.html`을 읽는다.

## 저장소에 있는 것

| 폴더 | 내용 |
| --- | --- |
| `scripts/`, `configs/` | 학습(`train.py`), 평가(`eval_student.py`, `p2sim.py`), 라벨링(`ep_label.py`), 양자화 빌드(`quant_build.py`), 보드(`board_eval.py`, `deploy_student.py`), 진단(`diag_*.py`) |
| `deploy/` | 실차 시험 패키지 `lrg_calL/`, edgepilot modeld CPU 헤드 패치(`edgepilot_student_cpu_heads.patch`, edgepilot beccd3d 기준) |
| `reports/` | 시도와 성공/실패 기록(`experiments.json` → `scripts/report.py` → `experiments.html`) |
| `logs/` | 평가·진단 결과 텍스트 |
| `memory/` | Claude Code 메모리 요약 |
| `HANDOVER.md` | 전체 인수인계 문서 |

## Release `handover-1101`에 있는 것 (큰 파일)

| 파일 | 내용 |
| --- | --- |
| `ckpt_lrc_lrd_lrf_lrg.tar` | `ckpt/mac_{lrc,lrd,lrf,lrg}`. 실차 시험한 모델은 `mac_lrg/ema_calL.pt` |
| `quant_lrg_calL_ext.tar` | lrg_calL Pulsar2 빌드(`outputs/quant_lrg_calL_ext`, `m.axmodel`, calibration) |
| `outputs_diag_lane_viz_compare.tar` | 차선 경사 진단 그림, 비교 영상 |
| `drive_2026-11-01--03-24-42-199.tar` | 실차 시험 원본(19분, lrg_calL이 CAN 송신으로 조향) |
| `drive_2026-10-30--00-08-39-565_events.tar` | 비교용 출시 모델 주행(events, params만) |
| `drive_2026-10-30--19-05-02-089.tar.part-*` | 새 주행 원본(야간). 2 GB 제한 때문에 나눴다 |
| `drive_2026-10-31--05-17-39-046.tar.part-*` | 새 주행 원본(새벽). 2 GB 제한 때문에 나눴다 |
| `SHA256SUMS` | 위 파일의 체크섬 |

받기와 풀기:

```sh
gh release download handover-1101 -R cwal1220/supercombo_distill -D dl
cd dl && sha256sum -c SHA256SUMS
for f in *.tar; do tar -xf "$f"; done
cat drive_2026-10-30--19-05-02-089.tar.part-* | tar -xf -
cat drive_2026-10-31--05-17-39-046.tar.part-* | tar -xf -
```

주행 원본은 edgepilot 녹화 형식 v14다(`segments/NNN/road.h264`, `events/NNN.bin`). `scripts/ep_label.py`로 teacher 라벨을 만든다.
라벨된 데이터(`data/ep_label`, 247 GB)는 올리지 않았다.
