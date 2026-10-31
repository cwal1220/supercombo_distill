# supercombo_distill 인수인계 (2026-10-27 19:40 기준)

openpilot의 **big 주행 모델(teacher)**을 증류해서, edgepilot이 MaixCAM2(**AX630C NPU**)에서 돌리는 supercombo보다
성능이 좋고 NPU에 최적화된 **student 모델**을 만드는 프로젝트다. 입출력 계약은 기존 모델과 같아서 그대로 교체할 수 있게 한다.

- 폴더 이름: 2026-10-27에 `sc_distill`에서 `supercombo_distill`로 바꿨다.
  - PC에는 이전 경로 호환용 상대 symlink `work/sc_distill -> supercombo_distill`이 남아 있다. 옛 로그와 Pulsar2 docker 경로 호환용이고 지워도 된다.
  - 경로가 들어간 곳은 모두 고쳤다: 스크립트, Pulsar2 설정, 두 기기의 venv 실행 파일.
- **Mac 폴더 이동 (2026-10-27 19:30):** `~/supercombo_distill` → `~/Documents/supercombo_distill`. 옛 경로 symlink는 없다.
  - venv는 절대경로가 박혀 있어서 tensorboard, pip 같은 실행 파일이 깨졌다. 그래서 `.venv`(py3.9)와 `.venv314`를 지우고 `.venv` 하나로 새로 만들었다 (12.2).
  - Mac에서 `scripts/bench_mac_sweep.sh`와 `scripts/sync_mac.py`(`--dst` 기본값 `Documents/supercombo_distill`)를 고쳤다.
    **PC 쪽 같은 파일도 똑같이 고칠 것.** sync_mac.py는 PC의 scripts/와 configs/로 Mac을 덮어쓰고, PC 쪽 기본값이 옛 경로이면 Mac 홈에 새 폴더를 만든다.
  - `~/Documents`는 macOS 보호 폴더다. PC에서 ssh로 이 폴더에 접근할 때 "원격 로그인"의 전체 디스크 접근 허용이 필요할 수 있다 (아직 확인하지 않음).
- 이 문서 위치: PC `/home/chan/pulsar2/work/supercombo_distill/HANDOVER.md`, Mac `~/Documents/supercombo_distill/HANDOVER.md` (Mac 쪽이 폴더 이동과 venv 재구성 내용을 더 담고 있다)
- 모든 스크립트는 맨 위 docstring에 상세 설명과 사용법이 있다. 먼저 읽을 것.
- 사용자: chan.park@stradvision.com. 한국어로 대화한다. 지시가 짧고 결정이 빠르다.

---

## 1. 지금 상태와 바로 할 일

### 00000000000000. 실차 시험 결과와 분석 (2026-11-01 03:14~04:10), Linux로 이전
- **사용자 지시(04:10)**: "mac에서 더이상 학습을 안할거야. gpu 자원이 풍부한 리눅스에서 작업할거야." 이후 작업은 Linux 세션
  "Supercombo network 학습(Linux)"이 이어받는다. Mac은 디스크가 거의 차 있다(여유 2 GB, swap 9 GB).
- **시험**: 03:14 lrg_calL을 CAN 송신 모드로 select → 주행 → 03:44 사용자가 `rollback --params`.
  - 보드 상태: modeld는 새 바이너리(student-cpu-heads)이고 출시 모델을 돌린다. `params/`는 시험 전으로 되돌렸다.
    시험 중에 학습한 params는 `/root/edgepilot/params.student/`에 있다.
  - 주행 기록 `2026-11-01--03-24-42-199`(19분, 22,299프레임, `params/driving_model.json` = lrg_calL): 보드 recordings,
    외장 디스크 `/Volumes/Backup/handover_linux_1101/raw_drives/`. 비교용 출시 모델 주행 `2026-10-30--00-08-39-565`(events만)도 같은 곳에 있다.
- **사용자가 본 문제**:
  1. 기존 모델보다 laneless 주행 능력이 현저히 떨어진다. 특히 차선 왼쪽으로 붙어 차선을 이탈하는 경우가 많다.
  2. 차선이 없는 교차로에서 주행 능력이 현저히 떨어진다.
  3. 웹 BEV에서 언덕을 만나면 주행 경로가 찌그러져 보인다.
- **분석(04:10까지)**:
  - `scripts/diag_drive_lateral.py`(녹화의 ModelState/ControlState만 쓴다, y는 오른쪽 +):
    - laneless 결합 3,991프레임: 차 위치 평균 17 cm 왼쪽(p90 52 cm). plan − 자기 차선 중앙은 10/20/30 m에서 −20/−19/−19 cm(왼쪽)이다.
    - 차선 모드에서는 ~0이다. 출시 모델 주행(10-30)은 laneless −2 cm다. 학생 plan이 중앙으로 돌아가지 않고 왼쪽을 겨눈다.
  - `scripts/diag_plan_bias.py`(평가 8 unit, S8 오프라인, `logs/diag_plan_bias_lrg_calL.txt`; 출력 머리말의 "+ = left"는 잘못이고 **+ = 오른쪽**이다):
    - 오프라인에서는 왼쪽 치우침이 없다(+0.7/+1.8/+2.9/+3.9 cm, 1~4 s).
    - **긴 horizon plan 오차가 출시 모델보다 크다.** 3/4 s에서 19.3/34.1 vs 27.3(4 s), 차선 없음 37/66 vs 30/50, 회전 45/83 vs 37/63 cm.
      선발 지표 plan_y2s(2 s만)가 이것을 가렸다.
    - 통행 방향 입력을 바꾸면 plan이 4 s에서 ~12 cm 움직인다. 입력을 쓰고 있으므로 반전 학습 가설은 약하다.
  - 언덕(`scripts/diag_drive_hill.py`와 즉석 분석): 웹 BEV는 ModelState plan의 (x, y)만 평면에 그린다(bev_data.js pointsAt).
    - plan의 굽음은 출시 모델보다 크지 않다.
    - 4 s 위치/(4·v)는 평지 0.97, 오르막 0.94다(출시 ~1.00). 학생 경로가 짧다.
    - 차선 폭 비 30/10 m는 내리막 1.085, 오르막 0.95다(출시 1.01/0.97).
    - 사용자에게 어떤 찌그러짐인지 물었으나 아직 답이 없다.
  - pose 회전 잡음은 출시의 ~2.5배다(`logs/diag_aux_heads_lrg_calL.txt`).
  - 학습 데이터에 가로 이동 augmentation이 있었다(cal1: `--rand-rpy-deg 1 1 2 --rand-offset 0.3`, cal1 51 + cal2 10 unit).
- **가설**:
  - (a) 중앙에서 벗어났을 때 복귀하는 plan이 약하다(covariate shift, BC 문제).
  - (b) 실차 ISP 프레임과 녹화 H.264 디코딩 프레임의 차이.
  - (c) 긴 horizon plan 약점.
- **다음 할 일(Linux)**:
  1. 학생 주행을 teacher로 라벨링한다. Mac에서 `ep_label.py ... --zstd`를 돌리다 디스크가 차서 멈췄다.
     - 출력의 `rec_*` = 학생 실차 출력, `big` = teacher, `small` = 출시 fp32다. 같은 프레임에서 다음을 비교한다.
       - 차가 중앙을 벗어났을 때 teacher와 학생의 plan
       - 차선 없는 교차로의 plan
       - 언덕 구간의 plan과 차선
     - 학생을 디코딩 프레임에 오프라인으로 돌려 `rec_*`와 비교한다(실차 프레임 차이 확인).
  2. 평가 unit에 `ep_label --offset ±0.4`(+yaw)를 적용한다. 같은 위치 이동에 teacher와 학생 plan이 어떻게 반응하는지 비교한다(복귀 능력).
  3. 선발 지표에 plan y 3/4/6 s와 차선 없음·회전 구간을 넣는다(eval_student/p2sim).
  4. 재학습 후보:
     - 긴 horizon 가중
     - 더 큰 가로 이동·yaw augmentation(±0.5~0.8 m)
     - 이 주행처럼 학생이 몰아 벗어난 장면을 teacher로 라벨해 넣는 DAgger
     - 주간 데이터 추가
  5. 실차 재시험 전에 보드 rehearsal(`deploy_student.py check`)로 laneless 동작을 먼저 본다.
- **Mac에서 옮긴 것**: `outputs/diag_y2s`와 `outputs/coreml_cache`는 `/Volumes/Backup/supercombo_offload_20261031/outputs_2200/`에 있고 원래 자리에 symlink를 남겼다.

### 0000000000000. 실차 시험 준비 (2026-11-01 03:00, 사용자 지시: "실차 테스트를 해볼거야. edgepilot 코드 & 바이너리 준비해줘")
- **edgepilot**: `~/Documents/ax630/edgepilot`의 브랜치 `student-cpu-heads`(HEAD beccd3d 기준, 아직 커밋하지 않음).
  - modeld가 CPU 헤드 코어를 받는다. 출력 `out_hidden` + `hx_<head>` 12개이고, 옆의 `<name>.heads`로 헤드 선형층을 CPU fp32로 계산한다.
  - 새 파일 `src/model/cpu_heads.*`, 바뀐 곳 `supercombo_model.*`, `model_output_assembly.h`(kCpuHeadParts), 테스트 `tests/gtest_cpu_heads.cc`, 문서 3개.
  - 출시 모델(분할 출력)도 그대로 돈다. 호스트 테스트는 보드 컴파일러(docker)로 378개가 모두 통과했다.
  - 보드 빌드 버전은 `v0.0.9-1-gbeccd3d+student-cpu-heads`다.
- **패키지 `deploy/lrg_calL/`**: axmodel, `.heads`, modeld, SHA256SUMS, README(설치·시험 순서·약점).
  - `scripts/export_cpu_heads.py`: ckpt → `.heads`. 보드 특징 1,497프레임에서 C++ 결과가 torch와 6e-5 차이다.
  - `scripts/deploy_student.py`: `package` / `install` / `check` / `select [--shadow]` / `rollback [--params] [--binary]` / `status`.
    - 모델 선택은 systemd drop-in의 `EDGEPILOT_MODEL`로 한다. `params/driving_model.json`이 route마다 녹화된다.
    - `select`는 처음 한 번 `params.before_student/`를 백업한다.
- **보조 출력 점검(`scripts/diag_aux_heads.py`, `logs/diag_aux_heads_lrg_calL.txt`)**:
  - pose 회전 잡음은 출시 fp32의 약 2.5배이고, 보고하는 표준편차는 teacher 수준이다. 조향은 CAN 요레이트를 쓰므로 직접 영향은 없다.
  - 도로 높이, desire, 도로 경계는 출시 모델과 같거나 낫다.
- **아직 안 함**: 보드 install/check(rehearsal). 03:00에 보드가 ping에 응답하지 않았다.

### 000000000000. 최종 모델 lrg_calL (2026-10-31 22:38, NPU·보드 검증): lrf_calL보다 대부분 지표가 낫다
- 빌드 `outputs/quant_lrg_calL_ext/m.axmodel`, 체크포인트 `ckpt/mac_lrg/ema_calL.pt`
  - LRG = LRF + `--lead-prob-focal 2 --lead-prob-weight 2.0`
  - logistic 보정: x0 80, b0 −0.08, b1 0.52, a 1.080

  | 지표(5경로, NPU = 보드) | 출시 | lrg_calL | lrf_calL |
  |---|---|---|---|
  | 보드 latency | 14.0 ms | 11.21 ms | 11.22 ms |
  | plan_y2s cm | 9.94 | 8.93 | 9.07 |
  | lat_acc | 0.0691 | 0.0461 | 0.0468 |
  | 차선/폭 경사 보정 cm | 14.86 / 16.46 | 10.61 / 14.31 | 10.62 / 14.30 |
  | 차선/폭 공식 cm | 18.58 / 25.85 | 14.56 / 23.11 | 14.48 / 22.79 |
  | lead_x m | 5.83 | 4.30 | 4.43 |
  | gas2 | 0.236 | 0.126 | 0.131 |
  | 헛/놓친 lead % | 5.06 / 10.59 | 5.04 / 9.47 | 4.98 / 9.93 |
  | 교차 검증 % | – | 4.79 / 9.74 | 4.81 / 10.13 |
  | AUC | 0.9772 | 0.9767 | 0.9764 |

- 보드: `logs/board_eval_lrg_calL_ext.txt`(출력 = S8). 보드 lead_prob는 시뮬레이션과 최대 0.0012 차이이고, 0.5 판단이 달라진 프레임은 1,007개 중 1개다.
- 약점: 주간 경로 10-05--16-28은 여전히 진다(헛 lead 13.6 vs 6.0%, __015 plan 18.2 vs 12.4).
  다음 개선은 주간 주행 데이터다(학습 데이터 중 주간은 36 unit).
- 공간(22:50, 여유 2.1 GB → 7.3 GB): 다음을 `/Volumes/Backup/supercombo_offload_20261031/outputs_2200/`로 옮겼다(삭제 아님).
  - 중간 빌드 8개: quant_lrb/lrc_cal/lrc_calE/lrd/lrd_cal/lrf/lrf_calE/lrg_ext. APFS clone이라 옮겨도 공간은 0.4 GB만 늘었다.
  - `outputs/coreml_cache`(4.8 GB): 라벨링할 때는 되돌린다.

### 00000000000. (이전) 최종 모델 lrf_calL (2026-10-31 20:56, NPU·보드 검증): 평가 5경로 합산의 모든 지표에서 출시 모델보다 좋다
- 빌드 `outputs/quant_lrf_calL_ext/m.axmodel`, 체크포인트 `ckpt/mac_lrf/ema_calL.pt`(LRF + logistic 먼 앞차 보정). 입출력 형식은 출시 모델과 같다.

  | 지표(5경로, NPU p2sim = 보드) | 출시 | lrf_calL |
  |---|---|---|
  | 보드 latency | 14.0 ms | 11.22 ms |
  | plan_y2s cm | 9.94 | 9.07 |
  | lat_acc | 0.0691 | 0.0468 |
  | 차선/폭 경사 보정 LD 기준 cm | 14.86 / 16.46 | 10.62 / 14.30 |
  | 차선/폭 공식(평지 LD) cm | 18.58 / 25.85 | 14.48 / 22.79 |
  | lead_x m | 5.83 | 4.43 |
  | gas2 | 0.236 | 0.131 |
  | 헛/놓친 lead % | 5.06 / 10.59 | 4.98 / 9.93 (교차 검증 4.81 / 10.13) |
  | lead AUC | 0.9772 | 0.9764 |

- 보드: `logs/board_eval_lrf_calL_ext.txt`(출력 = S8, 소수 넷째 자리). 보드 출력으로 계산한 lead_prob도 시뮬레이션과 최대 0.004 차이이고, 0.5 판단이 바뀐 프레임은 0개다.
- 약점: 주간 경로 10-05--16-28(터널 포함, 시속 40 km 안팎; 그동안 "야간 고속도로"라고 잘못 적었다)에서 헛 lead 12.8 vs 6.0%, plan도 두 unit 모두 진다. 10-25는 앞차가 비슷하다. AUC가 조금 낮다.
- 진행 중: LRG(`chain_lrg2.sh`, Quality Focal Loss)의 S8 평가 → 빌드 → NPU → logistic 보정 → 보드.
- **경로 성격(프레임으로 확인, 이름 시각은 KST)**: 09-27, 10-22, 10-25 야간 시내 / 10-05--00-07 야간 고속도로 / 10-05--16-28 주간(터널).
- **`--night-weight`의 휘도 판정(<150)은 거의 반대다**: 자동 노출로 야간 시내가 밝다. 휘도 '야간' 35 unit 중 33개가 주간이고, 실제 야간 143 unit 중 141개는 빠진다.
  학습 데이터의 75%가 야간이라 `--night-weight 2.5`는 사실상 주간 가중이었다(주간·야간 비율이 비슷해짐). 동작은 그대로 두고 설명만 고쳤다.
- **LRG(21:48, NPU `lrg_npu`)**: plan 8.93, lead_x 4.30, AUC 0.9747. logistic 보정 표본 내 5.04/9.45%, 교차 검증 4.79/9.74%(lrf_calL은 4.81/10.13%).
  주간 경로의 헛 lead는 그대로다(13.5%). `ckpt/mac_lrg/ema_calL.pt` → `quant_lrg_calL_ext` 빌드·NPU·보드는 chain_lrg2가 진행한다. 낫다면 최종 모델을 바꾼다.

### 0000000000. 추가 개선 진행 중 (2026-10-30 22:10~, 사용자 지시: "더 개선해.")
- **R18q의 남은 약점(S8, 5경로 프레임별 분석, `outputs/diag_y2s/dump/r18q_ema_s8`)**:
  - plan: 큰 회전에서 진다. |teacher y@2s| 1~3 m 구간에서 21.6 vs 17.4 cm, 3 m 이상에서 61 vs 53 cm다(출시 모델).
    이 구간은 프레임의 12%지만 오차의 36%다. 시간 지연은 0.1 s(0.6 cm)뿐이고 대부분 흩어짐이다.
    원인으로 보이는 것: plan loss는 teacher σ로 나눈 L1이라 σ가 큰 회전 구간이 덜 학습된다(지표는 절대 cm).
  - 헛 lead(`scripts/diag_falselead.py`, `scripts/lead_quality.py`): student는 헛 lead 8.8% / 놓친 lead 7.6%, 출시 모델은 5.1% / 10.6%다.
    - 검출 AUC는 0.974 vs 0.977로 거의 같다. 60 m 안의 헛 lead는 student가 더 적고(1.8% vs 2.0%), 차이는 60 m 밖에서 난다(7.1% vs 3.0%).
    - 평가 경로를 하나씩 빼고 임계점을 정하는 보정(leave-one-route-out)으로는 5.15% / 11.05%다(출시 5.06% / 10.59%). 근소하게 진다.
    - 학습 주행으로 임계점을 정하면 오히려 나빠진다(학습 주행에서는 student가 둘 다 이긴다). 일반화 문제라 학습으로 줄여야 한다.
  - `train.py --lead-prob-weight w`(기본 0.5)를 추가했다.
- **low-rank FC (`scripts/lowrank.py`, arch `hist_rank`, `vis_rank`)**:
  - temporal.0(6.6M)과 vis_fc(4.2M)는 매 프레임 가중치를 DDR에서 읽는 데 시간이 든다. 학습 프레임의 activation으로 데이터 기반 분해를 했다.
    hidden state는 상위 96개 주성분이 에너지의 99.93%라서 사실상 저차원이다.
  - rank 128/256이면 Pulsar2 max_cycle이 6.73M에서 5.53M으로 줄어든다(보드 약 10.3~10.8 ms로 추정). fine-tune 없이도 S8 성능이 R18q와 같거나 낫다
    (plan 9.52, lead 4.77, lanes 10.04; `logs/eval_lowrank_init.txt`, `ckpt/mac_lr/init_h128_v256.pt`).
- **layer4 (`scripts/mimic.py`, arch `r18_layers` 4)**: 아낀 cycle로 ImageNet layer4를 RepConv down 대신 넣는다(max_cycle 6.81M, R18q 수준).
  layer4와 새 neck이 R18q의 neck 출력을 재현하도록 hint 학습했다(held-out 상대 MSE 0.13). 이어서 low-rank로 분해했다(`ckpt/mac_l4/init_lr.pt`).
  추가 학습 전 S8은 plan 10.55, lead 5.17이다.
- **LR 결과 (2026-10-31 02:13, `logs/eval_mac_lr.txt`, `ckpt/mac_lr/ema.pt`)**: 6개 지표 모두 출시 모델을 이긴다. latency 후보 1순위.

  | 지표(S8, 5경로) | 출시 | R18q EMA | LR EMA |
  |---|---|---|---|
  | plan_y2s cm | 9.94 | 9.62 | 9.76 |
  | lead_x m | 5.83 | 4.87 | 4.93 |
  | lat_acc | 0.0691 | 0.0461 | 0.0447 |
  | lanes cm | 18.58 | 9.85 | 9.44 |
  | width cm | 25.85 | 12.33 | 11.73 |
  | gas2 | 0.236 | 0.135 | 0.130 |
  | max_cycle | (14.0 ms) | 6.73M (12.49 ms) | 5.53M (추정 10.3~10.8 ms) |
  | lead AUC | 0.9772 | 0.9741 | 0.9755 |
  | 헛/놓친 lead @0.5 | 5.06 / 10.59 | 8.84 / 7.58 | 8.91 / 7.29 |
  | 경로 하나 빼고 보정 | — | 5.15 / 11.05 | 5.06 / 10.45 |

  - 경로별 lead AUC는 5개 중 4개에서 student가 높다. 주간 경로 10-05--16-28만 크게 진다(0.984 vs 0.994, 헛 lead 17% vs 6%).
    그 헛 lead의 80%는 student 기준 60 m 밖이고(중앙값 80 m, teacher 기준 101 m), 길게 이어진다(최장 83 프레임).
    프레임을 따라가 보면 history 탓보다는 거리 추정이 원인이다. teacher가 100~125 m로 보는 차를 student는 70~90 m로 짧게 보고,
    그 거리면 진짜 lead일 확률이 높으니 0.9를 낸다. 옆 차로 차(teacher y 6 m)를 y 3~4 m로 안쪽에 보는 경우도 있다.
    곧 야간의 먼 차에서 위치 회귀가 평균 쪽으로 끌린다. 먼 거리 lead에 가중치를 주는 loss나 더 해상도 높은 feature가 후보다.
- **LRA 결과 (2026-10-31 04:13, `logs/eval_mac_lra.txt`, `ckpt/mac_lra/last.pt`) = 현재 최고**:
  plan 9.08(LR 9.76, R18q 9.62, 출시 9.94), lanes 9.43, width 11.75, lat 0.0447, lead 4.91, gas2 0.130. max_cycle 5.53M.
  plan-abs loss가 크게 들었다. lead_prob 가중치 2배는 거의 효과가 없었다(AUC 0.9757, 헛 lead 8.80%).
  chain_next2는 이 결과로 L4에도 `--plan-abs-weight 0.5 --lead-prob-weight 1.0`을 켰고, 빌드 lrbest는 `ckpt/mac_lra/last.pt`가 된다.
- **헛 lead 원인 확정**: 헛 lead의 83%는 teacher x 60 m 밖이다. 거기서 teacher의 x σ는 17~25 m라 σ로 나눈 lead loss에서는
  7~9 m 오차가 0.4~0.7σ로 작게 보인다(30 m 안쪽 σ는 0.7~2.7 m). plan 회전 구간과 같은 구조다.
  → `train.py --lead-abs-weight w`를 추가했다. 현재 lead(3 horizon)의 x, y를 3 m / 0.3 m 단위로 더하는 loss이고, lead_prob와 무관하게 전 프레임에 건다.
- **`scripts/chain_next3.sh` (LRB)**: DONE_chain_next2를 기다렸다가 LRA와 L4 중 plan이 나은 쪽에서 시작한다.
  LRA 옵션에 `--lead-abs-weight 0.5`를 더해 QAT 2 epoch(lr 1e-4)를 돈다. 결과는 `logs/eval_mac_lrb.txt`, `logs/lead_lrb.txt`, `outputs/quant_lrb`.
  2026-10-31 04:55에 nohup으로 걸었다(PID 578, `logs/chain_next3.log`). lead_abs loss는 가짜 배치로 값과 gradient 위치만 확인했다
  (실데이터 CPU smoke는 1200 s 안에 첫 step까지 못 가서 끊겼다).
  예상 시각: L4 학습 07:30, L4 평가·빌드 08:20, LRB 10:15, LRB 분석·빌드 10:50.
- **중요: S8 시뮬레이션과 실제 NPU(Pulsar2) 차이 (2026-10-31 05:00~06:30)**
  - LRA 빌드(`outputs/quant_lra`)를 Pulsar2 Reference로 돌리니 unit 2026-10-25--22-42-26-175__015에서 plan 17.13 cm(S8 시뮬레이션 15.44)였다.
    R18q 최종 빌드도 18.40(S8 17.42)이다. 그동안의 S8 수치는 실제 NPU보다 낙관적이었다.
  - 원인 1, 가중치 격자: Pulsar2의 S8 가중치는 scale max|w|/127.5, 코드 [-128, 127]이다(`quant_axmodel.onnx`의 int8 코드와 scale로 확인, 모든 층 0% 차이).
    `student.fake_quant_s8`은 /127, [-127, 127]이어서 코드의 3~15%가 한 칸씩 달랐다. QAT가 엉뚱한 격자에 맞춰진 셈이다.
    → `fake_quant_s8`을 /127.5, [-128, 127]로 고쳤다(백업: scratchpad `student.py.grid127`). 이후 새로 뜨는 학습과 평가는 모두 맞는 격자를 쓴다.
    L4 학습 프로세스는 옛 코드를 이미 읽어서 옛 격자로 끝까지 돈다. LRB(chain_next3)는 맞는 격자로 돈다.
  - 원인 2, U16 activation 양자화: 격자를 고쳐도 S8 시뮬레이션 15.54 vs Pulsar2 17.13이다. 가중치는 같으니 나머지는 activation이다.
    → `scripts/p2sim.py`: S8 가중치 ONNX에 Pulsar2 빌드의 U16 scale·zero point를 그대로 fake-quant로 넣고 onnxruntime으로 closed loop를 돈다.
      Pulsar2 Reference를 정확히 재현한다(17.1325 vs 17.1326). unit당 1분(docker `pulsar2 run`은 15분). `--drop`으로 양자화기를 빼 보며 원인을 찾는다.
  - 평가 버그: `quant_eval_p2.py`는 temporal head(lead, lead_prob)를 history 없는 1단계 출력에서 가져왔다. 2단계에서 가져오도록 고쳤다.
    그 전의 Pulsar2 lead 지표(헛 lead 25.5% 등)는 틀렸다. `board_run.py`는 한 번에 도는 실제 런타임 방식이라 문제없다.
  - **보드 실측(06:09~06:15, unit 2026-10-25--22-42-26-175__015)**: R18q 최종 12.49 ms, LRA 11.22 ms(출시 14.0 ms).
    보드 출력은 Pulsar2 Reference와 rel err 1e-4로 같다. 그래서 `p2sim.py`는 보드와 같다고 봐도 된다.
    이 unit의 plan: 출시 18.08, R18q 18.40, LRA 17.13.
  - **U16 오차의 정체**: plan 증가분 1.60 cm 중 1.52 cm가 프레임 8.9%(저속 급회전, teacher |y2| 중앙값 3.8 m)에서 나온다.
    그 90 프레임 모두에서 NPU 쪽이 덜 꺾는다(평균 |y2| 4.08 → 3.68 m). 원인은 calibration 범위 밖 activation 클리핑이다
    (scratchpad `p2_clip.py`). desire_enc 출력이 프레임의 23%에서 잘리고(최댓값 1.40, 범위 0.65), hx_lanes 10%(40.6 / 16.8),
    vision_trunk 7%, hx_road_edges 6%, desire_state·plan.1 2~4%다. 무작위 256 프레임 calibration에 desire와 급회전이 거의 없었다.
    → `quant_build.py --extremes`: 학습 주행 후보(unit마다 무작위 per_unit + desire·급회전 per_unit)에서 텐서마다 최댓값·최솟값을 내는 프레임을
      먼저 넣고 나머지를 무작위로 채운다(샘플 수는 그대로, 디스크도 같다). 첫 빌드는 `outputs/quant_lra_ext`이다.
  - **`--extremes` 효과(07:25)**: LRA를 새 calibration으로 빌드하니(`outputs/quant_lra_ext`, 후보 1,537개 중 256개)
    unit __015에서 NPU(p2sim) 결과가 S8과 같아졌다. plan 17.13 → 15.54, lanes 14.88 → 13.94, lat 0.0629 → 0.0584.
    desire_enc 범위는 0.65 → 1.19, hx_lanes 16.8 → 41.9다. NPU 오차는 calibration 문제였고 해결됐다.
    5경로 전체 p2sim: `logs/p2sim_lra_ext_all.txt`(07:25 시작). 보드 측정: `logs/board_eval_lra_ext.txt`.
  - **`scripts/chain_final.sh`(PID 7372, 08:58에 멈췄고 아래 chain_final2.sh로 대체)**: DONE_chain_next3를 기다린다. 이어서 다음을 한다.
    1. L4와 LRB의 나은 체크포인트를 `--extremes`로 빌드한다(`quant_l4_ext`, `quant_lrb_ext`).
    2. 무작위 calibration 빌드의 calib·dataset은 Backup으로 옮긴다.
    3. p2sim 5경로 전체(`logs/p2sim_{lrb,l4}_ext_all.txt`)를 돌린다.
    4. 보드 측정을 한다.

    이전 board_queue(6263)는 무작위 calibration 빌드를 재려던 것이라 멈췄다.
  - **5경로 8 unit NPU 연산 결과(p2sim = 보드, 08:30)**: `logs/p2sim_{lra_ext,lra,r18q}_all.txt`

    | 지표(NPU, 5경로) | 출시 | R18q 최종(무작위 calib) | LRA(무작위 calib) | **LRA ext(새 calib)** |
    |---|---|---|---|---|
    | plan_y2s cm | 9.94 | 10.24 (짐) | 9.90 | **9.15** |
    | lead_x m | 5.83 | 4.83 | 4.73 | 4.73 |
    | lat_acc | 0.0691 | 0.0493 | 0.0547 | 0.0504 |
    | lanes cm | 18.58 | 10.60 | 10.69 | 9.70 |
    | width cm | 25.85 | 12.49 | 12.19 | 11.96 |
    | gas2 | 0.236 | 0.133 | 0.130 | 0.129 |
    | 보드 latency | 14.0 ms | 12.49 ms | 11.22 ms | 11.21 ms |
    | 헛 lead % | 5.06 | 8.52 | 11.77 | 11.99 |

    - **정정: 10-30 R18q '목표 달성'은 S8 기준이었다. 실제 NPU에서는 plan 10.24로 진다.** LRA ext는 공식 6개 지표와 latency를 모두 이긴다.
    - 남은 문제는 헛 lead다. LRA는 예전 S8 평가 8.83%에서 NPU 12.0%로 늘었다(8 unit 모두 2~4%p). R18q는 8.70 → 8.52다.
    - **원인 확정(08:55): 가중치 격자다. activation은 원인이 아니다.**
      - LRA는 고치기 전 격자(/127, [-127, 127])로 QAT를 했고, 예전 S8 평가도 그 격자로 했다.
      - 같은 가중치를 Pulsar2 격자로 양자화하면 헛 lead가 09-27__015에서 3.34 → 5.41%, 10-25__015에서 15.74 → 16.60%다. NPU 수치와 정확히 같다.
      - U16 activation이 lead_prob logit에 주는 오차는 표준편차 0.002, 평균 0.000이다(10-05--00-07 800 프레임). Concat 양자화를 빼도 같다.
      - 그래서 처음 의심한 Concat 대책(arch `split_temporal`, 그룹별 MatMul 합)은 되돌렸다. 코드는 scratchpad `student.py.split`에 있다.
      - CPU head bias 보정(`bias_correct.py`, 학습 unit에서 NPU−S8 평균 이동을 bias에 반영)도 이동이 1e-4라 쓰지 않는다. 코드는 scratchpad에 있다.
      - 해결: 올바른 격자로 다시 QAT한다(LRB, LRC). 새 calibration을 쓰면 NPU와 S8(새 격자)은 사실상 같다.
        이제 `eval_student.py --s8`과 `diag_y2s_dump.py --s8`이 NPU 대용으로 믿을 만하다.
    - `p2sim.py --dump <name>`: S8+U16 행의 lead 출력을 프레임별로 `outputs/diag_y2s/dump/<name>`에 저장한다(lead_quality.py와 같은 형식).
  - **`scripts/chain_final2.sh`(PID 9355, 08:58, SSHPASS는 env로만 전달)**: chain_final.sh를 대체한다. L4 빌드는 하지 않는다(옛 격자 QAT).
    1. DONE_mac_lrb가 생기면 LRC를 GPU에서 시작한다. LRC는 `ckpt/mac_lra/last.pt`를 LRB와 같은 옵션으로 QAT 2 epoch 학습한다(고친 격자).
       평가는 `logs/eval_mac_lrc.txt`에 남는다.
    2. chain_next3의 무작위 calibration 빌드(`--name lrb`)는 뜨자마자 PID로 멈춘다.
    3. LRB의 더 나은 체크포인트로 다음을 한다: `--extremes` 빌드 `quant_lrb_ext`, p2sim 5경로(`logs/p2sim_lrb_ext_all.txt`, dump `lrb_npu`), `logs/lead_lrb_npu.txt`, 보드 측정.
    4. LRC도 같은 순서로 한다(`quant_lrc_ext`, `logs/p2sim_lrc_ext_all.txt`, `logs/lead_lrc_npu.txt`).
    - 기록은 `logs/chain_final2.log`에 남는다.
  - **LRB 결과(10:44, S8 Pulsar2 격자, `logs/eval_mac_lrb.txt`)**: last는 plan 9.17, lead_x 4.54, lat 0.0468, lanes 9.48, width 11.83, gas2 0.133, 헛 lead 9.1%다.
    EMA는 plan 9.12, 헛 lead 9.3%다. lead-abs로 lead_x가 크게 좋아졌다(L4 5.11). 헛 lead는 LRA NPU(12.0)보다 적지만 출시(5.1)보다 많다.
  - **LRA ext lead 품질(NPU dump `lra_ext_npu`, `logs/lead_lra_ext_npu.txt`)**: AUC 0.9753(출시 0.9772), 헛/놓친 lead 11.99/6.14%(출시 5.06/10.59).
    경로 하나 빼고 보정하면 5.25/10.61%다. 남은 차이는 60 m 밖(10.2% vs 3.0%)에 있다.
  - **LRB ext NPU·보드(12:30~13:00, `ckpt/mac_lrb/ema.pt` → `outputs/quant_lrb_ext`) = 현재 NPU 기준 최고 성능**:

    | 지표(NPU, 5경로) | 출시 | LRA ext | **LRB ext** | LRC EMA (S8 고친 격자, NPU 대기) |
    |---|---|---|---|---|
    | plan_y2s cm | 9.94 | 9.15 | **9.07** | 9.11 |
    | lead_x m | 5.83 | 4.73 | **4.54** | 4.63 |
    | lat_acc | 0.0691 | 0.0504 | 0.0477 | 0.0454 |
    | lanes cm | 18.58 | 9.70 | 9.47 | 9.53 |
    | width cm | 25.85 | 11.96 | 11.80 | 11.82 |
    | gas2 | 0.236 | 0.129 | 0.132 | 0.129 |
    | 보드 latency | 14.0 ms | **11.21 ms** | 13.91 ms | (LR 구조 11.2 ms) |
    | 헛 lead % | 5.06 | 11.99 | 9.31 | 9.00 |

    - LRB ext 보드 출력은 S8 시뮬레이션과 같다(unit 10-25__015 plan 16.3545 vs 16.3534). L4 구조라 latency는 13.91 ms로, 예상(12.5 ms)보다 느리다.
    - 격자를 고친 뒤로 S8 평가와 NPU는 헛 lead까지 일치한다(LRB S8 9.32 vs NPU 9.31). LRC의 S8 평가에서 LRA 행도 11.97%로 NPU(11.99)와 같다.
    - lead 품질(`logs/lead_lrb_npu.txt`): AUC 0.9747, 헛/놓친 lead 9.31/7.40%다. 경로 하나 빼고 보정하면 4.97/10.83%다. 60 m 밖은 7.50% vs 3.04%다.
  - **LRC 결과(12:30, S8 고친 격자, `logs/eval_mac_lrc.txt`)**: EMA는 plan 9.11, lead_x 4.63, 헛 lead 9.0%, vx0 68.6이다.
    last는 plan 9.14, 헛 lead 8.7%다. 같은 구조의 LRA(12.0%)보다 헛 lead가 3%p 적다.
  - **LRC ext NPU·보드(14:03~14:08, `outputs/quant_lrc_ext`) = 현재 최고(NPU·보드 검증)**: plan 9.05, lead_x 4.63, lat 0.0458, lanes 9.53, width 11.82,
    gas2 0.129, 헛 lead 9.00%, 보드 11.21 ms. 보드 출력은 S8과 같다(plan 15.8198 vs 15.8197). 6개 지표와 latency를 모두 이기고 LRA ext보다 모든 지표가 낫다.
    60 m 안쪽 헛 lead는 1.41%로 출시(2.02%)보다 적다. 60 m 밖은 7.60% vs 3.04%다.
  - **먼 앞차 보정 `scripts/lead_cal.py`(student arch `lead_cal`)**: logit += b0 − b1·relu(x − x0)/20(x는 학생 자신의 lead 거리, t=0)을 모델 안에서 한다.
    - lead_prob feature 중 calibration 256 샘플에서 항상 0인 채널(LRB 5, 12, 15)을 골라, NPU 그래프에 64→16 linear를 더해 그 채널이 relu((x−x0)/20)를 내게 한다.
      CPU head에는 그 채널 가중치 −b1, bias +b0를 넣는다. 출력 형식은 그대로다. fp32 차이 1e-6, ONNX export를 확인했다.
    - 보정값은 학습·평가에 없는 새 주행에서 정한다(LRB S8 dump `lrb_s8_new`): x0 60, b0 +0.15, b1 0.25.
      평가 5경로에서 LRB 9.31/7.40 → 7.63/7.85%, LRC 9.00/7.69 → 6.96/8.30%다(헛/놓친, 출시 5.06/10.59).
    - 새 주행에서는 teacher 기준으로 출시 모델도 헛 lead 21.4%이고, 검출 AUC는 LRB 0.985로 출시 모델(0.982)보다 높다.
    - LRC 자체로 맞춘 보정값을 정하려고 `lrc_s8_new` dump를 만드는 중이다(14:05~).
    - `lead_cal.py --params x0 b0 b1`: 새 주행으로 fine-tune한 LRD처럼 보정값을 직접 맞출 수 없는 모델은 시작점(LRC)의 값을 쓴다.
  - **`scripts/chain_final3.sh`(PID 18071, 14:20, SSHPASS env)**:
    1. `lrc_s8_new`가 끝나면 LRC 보정값을 맞추고 `ckpt/mac_lrc/ema_cal.pt`를 만든다(`logs/lead_cal_lrc.txt`).
    2. chain_data의 LRD 빌드가 끝나면 `quant_lrc_cal_ext`를 빌드하고 p2sim을 돌린다(`logs/p2sim_lrc_cal_ext_all.txt`, `logs/lead_lrc_cal_npu.txt`).
    3. LRD p2sim이 끝나면 LRC 값으로 `<lrd>_cal.pt`를 만들고 같은 순서로 `lrd_cal`을 처리한다(`logs/lead_cal_lrd.txt`).
    4. DONE_chain_data 뒤에 두 빌드를 보드에서 잰다.
    - 기록은 `logs/chain_final3.log`에 남는다.
    - LRC 자체 보정값(14:29, `logs/lead_cal_lrc.txt`)은 x0 40, b0 +0.75, b1 0.50이다. 평가 5경로 예상치는 9.00/7.69 → 6.06/8.75%다(출시 5.06/10.59).
  - **평가 경로로 맞춘 보정 + 교차 검증(14:40)**: `lead_cal.py lrc_npu lrc_npu --loro`는 x0 선택까지 포함한 맞춤 절차를 경로 하나씩 빼고 검증한다.
    LRC는 4.81/10.15%로 출시 모델(5.06/10.59)보다 헛·놓친 lead 모두 낮다. LRB는 4.62/11.15%로 놓친 lead에서 진다.
    5경로 전체로 맞춘 값은 x0 80, b0 +0.20, b1 1.75이고(표본 내 5.04/9.76), `ckpt/mac_lrc/ema_calE.pt`로 저장했다(`logs/lead_cal_lrcE.txt`).
    - `scripts/chain_final4.sh`(PID 19202): chain_final3의 lrc_cal 빌드 뒤에 `quant_lrc_calE_ext`를 빌드하고, p2sim(`logs/p2sim_lrc_calE_ext_all.txt`,
      `logs/lead_lrc_calE_npu.txt`)을 돌린다. DONE_chain_data 뒤 다른 보드 측정이 없을 때 보드에서 잰다. 기록은 `logs/chain_final4.log`에 남는다.
  - **차선 재검토: 오르막·내리막(15:00, 사용자 지시 "lead보다 차선 위주로 먼저")**
    - **원인**: LD 차선 라벨은 카메라 높이의 평평한 도로에 투영돼 있다(`ld_teacher.to_road`). 앞 도로가 오르면 차선이 멀수록 벌어지고(0.6 m 이상 오름: 30 m/10 m 폭 1.35),
      내려가면 좁아진다(0.38). 학생 차선 y는 LD로, z는 teacher로 배워서 y와 z가 맞지 않고, 경사에 따라 차선이 흔들린다.
      LRC 폭 비율은 0.93~1.33, 출시 모델은 0.98~1.01이다. teacher 대비 오차는 LRC 9 → 47 cm, 출시 모델은 8~12 cm다(`scripts/diag_lane_grade.py`).
    - **공식 차선 지표 주의**: 평지 LD 기준이라 왜곡을 따라 한 학생에게 유리하다. 8 unit 전체에서 평지 LD 기준으로는 LRC 9.3, 출시 16.0 cm다.
      경사 보정 LD 기준으로는 LRC 13.4, 출시 13.1, teacher 12.9 cm로 LRC가 조금 진다(scratchpad `ldfix_metric.py`, `logs/diag_lane_grade_lrc.txt`).
      앞 도로 높이 ±0.1 m를 넘는 프레임이 42%다. 30 m 차선 흔들림은 모든 구간에서 LRC가 크다(평지 3.9 vs 3.3 cm).
    - **보정 `scripts/lane_geom.py` `slope_fix()`**: 평지 점 (X, y)는 광선 (X, y+offset, h)이다. teacher 차선 높이 z(x)와 교차시켜 다시 투영한다.
      보정 LD는 폭 비율 0.94~0.97로 일정하고, teacher 대비 오차는 46 → 7 cm다. 라벨 파일은 그대로 두고, 학습·평가 때 계산한다.
      - `train.py --ld-slope-fix`: 차선과 도로 경계에 적용한다. 백업은 scratchpad `train.py.pre_ldfix`다.
      - `eval_student.py` 새 열: `lanes_ldfix_cm`, `width_ldfix_cm`. 기존 열은 그대로다. 백업은 scratchpad `eval_student.py.pre_ldfix`다.
    - **`scripts/chain_lrf.sh`(PID 19462)**: DONE_mac_lrd 뒤에 LRC/LRD 중 S8 plan이 나은 쪽에서 `--ld-slope-fix`로 QAT 2 epoch를 학습한다(`ckpt/mac_lrf`).
      이어서 다음을 한다: `logs/eval_mac_lrf.txt`, `logs/diag_lane_grade_lrf.txt`, 빌드 `quant_lrf_ext`, p2sim, 보드 측정. 기록은 `logs/chain_lrf.log`에 남는다.
    - 예시 프레임: `outputs/diag_lane/clip3_t234_downhill.png`(LRC 차선이 20~40 m에서 안쪽으로 꺾인다).
  - **LRF 결과(17:28, S8, `logs/eval_mac_lrf.txt`, `ckpt/mac_lrf/ema.pt`)**: 경사 보정 기준 차선 10.62, 폭 14.30 cm로 출시(14.86/16.46)와
    LRC(13.72/20.74)를 모두 이긴다(eval_student 정의: 경로별 프레임 가중 평균). plan 9.10, lead_x 4.43, lat 0.0467, gas2 0.131은 LRC 수준이고,
    헛 lead는 9.00 → 8.08%다. 공식 lanes(평지 LD 기준)는 9.53 → 14.48 cm로 커졌다(왜곡을 더 이상 따라 하지 않아서, 출시 18.58보다는 낫다).
    - `p2sim.py`와 `diag_lane_grade.py`에도 경사 보정 LD 기준 열을 넣었다(백업 scratchpad `p2sim.py.pre_ldfix`, `diag_lane_grade.py.pre_ldfix`).
      diag의 마지막 줄 `all moving`이 8 unit 프레임 합산값이다(LRC 13.4, 출시 13.1, teacher 12.9; `logs/diag_lane_grade_lrc_fix.txt`).
    - LRF 높이별(`logs/diag_lane_grade_lrf_fix.txt`): 경사 보정 LD 대비 10.4 cm(출시 13.1)이고 모든 높이 구간에서 출시 모델보다 작다.
      폭 비율 0.98~1.08(LRC 0.88~1.35), 30 m 흔들림 3.1 cm(출시 3.8, LRC 4.6)다. teacher 대비 오차는 여전히 출시 모델이 조금 작다(평지 13.2 vs 11.1).
    - 영상: `outputs/viz_compare/clip3_lrc_vs_lrf.mp4`(영상 3번 구간, 위 LRC, 아래 LRF, 오른쪽이 학생). 정지 화면은 `outputs/diag_lane/clip3_t234_lrc_vs_lrf.png`다.
  - **먼 앞차 보정 NPU·보드 결과(17:06~17:21)**: lrc_cal(새 주행 맞춤) 6.05/8.76%, lrc_calE(평가 경로 맞춤) **5.03/9.77%**(출시 5.06/10.59),
    AUC 0.9757. lrc_calE 보드 11.21 ms, 출력 = S8 시뮬레이션. LRD NPU는 plan 9.20, 헛/놓친 8.01/7.74%, 보드 11.22 ms다.
  - **LRF NPU(18:50, `logs/p2sim_lrf_ext_all.txt`)**: S8과 같다(plan 9.07, ldfix 10.62/14.30, 헛/놓친 8.09/7.83%, AUC 0.9740).
    보정(`logs/lead_cal_lrfE.txt`, x0 80, b0 +0.10, b1 1.50): 표본 내 4.98/9.75%, 교차 검증 4.52/11.06%(놓친 lead에서 0.47%p 짐).
    보정 기준을 바꿔 봐도 교차 검증은 출시 근처다(scratchpad `loro_crit.py`). 보정은 ROC 위 지점만 옮기므로 AUC를 올려야 확실히 이긴다.
  - **logistic 보정 `lead_cal.py --logistic`(19:30)**: logit' = a·s + b0 − b1·relu(x−80)/20에서 a, b1을 teacher 확률에 대한 logistic regression으로 맞추고
    b0만 출시 헛 lead율에 맞춘다(a는 CPU head 전체에 곱함). LRF: a 0.791, b1 0.567, b0 −0.105, 표본 내 4.97/9.94%, **교차 검증 4.81/10.13%**
    (출시 5.06/10.59, `logs/lead_cal_lrfL.txt`). fold별 값이 안정적이다. `ckpt/mac_lrf/ema_calL.pt`(식과 2e-6). 백업 scratchpad `lead_cal.py.pre_logistic`.
    - 시험한 다른 특징(scratchpad `lead_lr.py`): |y| 항을 넣으면 교차 검증 4.89/9.68%지만 좌우 가중치가 비대칭이라 넣지 않았다.
    - `scripts/chain_lrf_calL.sh`(PID 29255): 빌드 `quant_lrf_calL_ext` → p2sim(`lrf_calL_npu`) → 보드(DONE_chain_lrf_cal 뒤). **최종 후보.**
    - chain_lrg.sh는 19:30에 셸만 멈췄다(학습 PID 27852는 그대로). 뒷부분은 `scripts/chain_lrg2.sh 27852`(PID 29256)가 logistic 보정으로 이어서 한다
      (`lead_cal_lrgL`, `quant_lrg_calL_ext`, 보드는 DONE_chain_lrf_calL 뒤).
  - **LRG `scripts/chain_lrg.sh`(PID 27844, 19:05)**: LRF EMA에서 `--lead-prob-focal 2 --lead-prob-weight 2.0`(Quality Focal Loss, 새 옵션,
    백업 scratchpad `train.py.pre_focal`)로 QAT 2 epoch → S8 평가 → 빌드 `quant_lrg_ext` → p2sim dump `lrg_npu` → 보정 `lead_cal_lrgE`
    → 빌드 `quant_lrg_calE_ext` → p2sim → 보드(DONE_chain_lrf_cal 뒤). 기록은 `logs/chain_lrg.log`에 남는다.
  - **`scripts/chain_lrf_cal.sh`(PID 25747, 17:35)**: chain_lrf의 p2sim(`lrf_npu`)이 끝나면 LRC와 같은 절차로 LRF 보정값을 맞추고(`logs/lead_cal_lrfE.txt`,
    교차 검증 포함), `ckpt/mac_lrf/ema_calE.pt` → 빌드 `quant_lrf_calE_ext` → p2sim(`lrf_calE_npu`) → 보드(DONE_chain_lrf 뒤). 최종 후보다.
  - **비교 영상 `scripts/viz_compare.py`(14:35)**: 모델 입력 프레임 위에 출시 모델(보드 기록)과 학생(S8 = NPU)의 경로, 차선, 도로 경계, 앞차를 나란히 그린다.
    아래에는 위에서 본 모습(teacher 참고)과 앞차 확률 그래프가 있다. 결과는 `outputs/viz_compare/lrc_vs_shipped.mp4`(71 s)이고, 세 구간을 이어 붙였다.
    1. 주간 곡선 + 앞차. teacher 라벨로만 골랐고 경로 오차는 출시 5.9, LRC 7.8 cm다.
    2. LRC 헛 lead가 가장 많은 야간 고속도로 구간. 헛 lead 프레임은 LRC 131, 출시 35다.
    3. 출시 모델 헛 lead가 가장 많은 야간 시내 구간. 헛 lead 프레임은 출시 71, LRC 33이다.
  - **새 데이터 라벨링 완료(12:58)**: 야간 5 unit(74,400 프레임), 새벽 3 unit(49,831 프레임). 모두 teacher와 100% 맞췄다.
    LRD는 13:00에 `ckpt/mac_lrc/ema.pt`(LR 구조)에서 시작했다(`logs/mac_lrd.log`).
  - **LRD 결과(15:03, S8, `logs/eval_mac_lrd.txt`)**: EMA plan 9.24, lat 0.0488, lanes 9.44, width 11.64, lead_x 4.62, 헛 lead 8.02%다.
    LRC EMA(9.11, 0.0454, 9.53, 11.82, 4.63, 9.00%)와 비교하면 헛 lead는 1%p 줄었지만 plan과 lat_acc는 나빠졌다. chain_data가 lrd_ext 빌드, p2sim, 보드 측정을 계속한다.
    LRF는 plan 기준으로 LRC EMA에서 시작했다(15:03).
  - **새 데이터(11:06~)**: 보드 `/root/edgepilot/recordings`에 학습에 안 쓴 주행 2개가 있었다.
    2026-10-30--19-05-02-089(야간, 61 segment, 4.5 GB)와 2026-10-31--05-17-39-046(새벽, 3.0 GB)이고, USB `edgepilot_drives/experiments`로 복사했다.
    - 공간: `~/Documents/ax630/experiments`의 원본 로그 6개(31 GB)를 USB 사본과 파일 수·크기가 같은지 확인한 뒤
      `/Volumes/Backup/ax630_experiments_offload_20261031/`로 옮겼다. 원래 자리에는 symlink를 남겼다. 여유가 11 GB에서 40 GB가 됐다.
      CoreML 캐시(`outputs/coreml_cache/CPUAndNeuralEngine`)는 Backup에서 되돌렸다.
    - **`scripts/chain_data.sh`(PID 12860, SSHPASS env)**:
      1. 두 주행을 라벨링한다(`logs/label/batch_mac_new1031.log`).
      2. DONE_mac_lrc를 기다린 뒤, LRB/LRC 중 S8 plan이 나은 쪽에서 LRD를 학습한다(같은 옵션, 2 epoch).
      3. `logs/eval_mac_lrd.txt`, 빌드 `quant_lrd_ext`, p2sim, dump `lrd_npu`, 보드 측정(chain_final2가 끝난 뒤).
      - 기록은 `logs/chain_data.log`에 남는다.
  - **L4 결과(08:10, S8, `logs/eval_mac_l4.txt`)**: last plan 9.02, lead 5.11, lat 0.0539, lanes 9.48, width 11.74, gas2 0.130, 헛 lead 7.8%.
    plan과 헛 lead는 LRA보다 조금 낫지만 lead_x, lat_acc가 나쁘고 latency 이득(6.81M)이 없다.
  - **chain_next2의 무작위 calibration 빌드(lrbest, l4)는 멈췄다(08:34)**. 쓸모없는 빌드라 DONE_chain_next2를 직접 썼다. 부분 빌드는 Backup으로 옮겼다.
    LRB는 08:34에 `ckpt/mac_l4/last.pt`에서 시작했다(`logs/chain_next3.log`). 10:40쯤 끝나고 chain_final이 이어받는다.
  - 디스크가 가득 차서(ENOSPC, 06:20) 다음을 `/Volumes/Backup/supercombo_offload_20261031/`로 옮겼다(삭제 아님):
    실패한 calibration 2,048개 빌드(quant_lra_c2k), 중단한 평가 임시 폴더, quant_lra_pa(정밀도 분석 빌드),
    quant_r18final과 quant_lra의 `calib/`, `quant/dataset/`, `outputs/coreml_cache/CPUAndNeuralEngine`(라벨링 때 다시 필요하면 되돌린다).
    여유 12 GB. 빌드 하나가 약 1.6 GB(calibration 2,048개면 3.2 GB)다.
- **`train.py --plan-abs-weight w`**: plan y(0~4 s)의 |오차|를 10 cm 단위로 더하는 loss다. 회전 구간이 지표만큼 학습되게 한다.
- **실행 중 (GPU)**:
  - `scripts/chain_lr.sh`는 LR 학습(`ckpt/mac_lr`, init_h128_v256, QAT 4 epoch, lr 1e-4)만 남기고 꼬리는 `scripts/chain_next2.sh`로 바꿨다.
  - chain_next.sh는 lead_prob 가중치를 넣기 전 판이라 멈췄다.
  - chain_next2 순서:
    1. LR 평가(`logs/eval_mac_lr.txt`)
    2. LRA: LR의 더 나은 체크포인트에서 `--plan-abs-weight 0.5 --lead-prob-weight 1.0`로 QAT 2 epoch(`logs/eval_mac_lra.txt`)
    3. L4 QAT 4 epoch(LRA가 plan에서 나으면 같은 loss 옵션 포함)
    4. Pulsar2 빌드(`outputs/quant_lrbest`, `outputs/quant_l4`)
  - 기록은 `logs/chain_next2.log`에 남는다.
- 처음 건 full-rank 이어 학습(R18r, `scripts/chain_r18r.sh`)은 low-rank가 더 낫다는 것을 보고 50분 뒤 멈췄다.
- 보드는 여전히 꺼져 있다. `board_when_up.sh`가 R18q 최종 빌드를 재려고 기다리고 있다.

### 000000000. 목표 달성 (2026-10-30 21:00): 최종 모델 R18q가 6개 지표 모두 출시 모델을 이겼다
- **최종 모델**: `ckpt/mac_r18q/ema.pt`
  - 구조: ImageNet ResNet-18 backbone(layer3까지 + RepConv), temporal lead head, lane_policy, policy_tokens 25, QAT.
  - arch: `{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25}`
  - Pulsar2 빌드: `outputs/quant_r18final/m.axmodel`(AX620E U16/S8, max_cycle 6.73M, `logs/quant_r18final.log`)
- **S8(배포 기준), 평가 경로 5개(`logs/eval_mac_r18q.txt`)**:

  | 지표 | 출시 모델 | R18q EMA |
  |---|---|---|
  | lat_acc | 0.0691 | **0.0461** |
  | plan_y2s_cm | 9.94 | **9.62** |
  | lanes_vs_ld_cm | 18.58 | **9.85** |
  | width_vs_ld_cm | 25.85 | **12.33** |
  | lead_x_m | 5.83 | **4.87** (lead 프레임 기준 3.38 vs 4.28) |
  | gas2 | 0.236 | **0.135** |
  | latency | 14.0 ms | **12.49 ms** (같은 구조의 r18e3 보드 실측) |

  - 공식 지표 밖이지만 헛 lead는 8.8%로 출시 모델(5.1%)보다 많다.
  - long_acc는 0.256이다(출시 모델 기록에는 없다).
- **검증**:
  - R18 구조의 보드 출력은 S8 시뮬레이션과 일치한다(`logs/board_eval_r18e3.txt`).
  - 최종 빌드의 보드 실측은 21:00에 보드가 꺼져 있어(ping 무응답) 하지 못했다.
    `scripts/board_when_up.sh`가 보드가 켜지면 자동으로 잰다(`logs/board_eval_r18final.txt`, DONE_board_r18final; 하루 동안 기다린다).
- **학습 경로**: qat3 → gap3b → Run A → R18(3 ep, fp32) → R18b(4 ep, fp32) → R18q(QAT 2 ep).
  - 마지막 세 단계는 `scripts/chain_r18*.sh`다. 각 단계의 수치는 아래 절과 `reports/experiments.html`에 있다.
- **시도와 성공/실패 기록**: `reports/experiments.html`(원본은 `reports/experiments.json`, 생성은 `scripts/report.py`)
- **남은 일**:
  1. 최종 빌드를 보드에서 확인한다(자동 대기 중).
  2. edgepilot modeld 탑재: CPU head(fp32 마지막 Linear)를 modeld에서 계산하도록 고쳐야 한다. 출력은 hx_* feature다.
  3. 헛 lead 줄이기.
  4. cal2 라벨링(10-10--11-16, 10-10--17-34)이 돌고 있다. 끝나면 다음 학습에 쓸 수 있다.

### 00000000. 진행 기록 (2026-10-30 01:25 ~ 21:00): /goal "출시 모델보다 좋은 모델" (모든 수단, 시도와 결과는 HTML로)
사용자 지시(/goal): 데이터 정제/추가(공간이 부족하면 USB나 외장하드로 옮긴다, open data 참조), 구조와 학습 전략 다양화(논문 참조),
그 밖의 모든 방법을 쓴다. 시도와 성공/실패를 HTML로 남긴다. 성공 기준은 출시 모델보다 좋은 성능이다. 필요하면 서브에이전트를 쓴다.
추가 지시: "보드도 켜뒀어. 새로운 주행 기록이 있는지 확인하고 라벨로 쓸 수 있으면 써".
- **보고서**: `reports/experiments.json`(시도 목록) → `scripts/report.py` → `reports/experiments.html`. 실험이 끝날 때마다 JSON에 추가하고 다시 만든다.
- **현재 최고**: swa_qat3(qat3 epoch 2·3 가중치 평균) plan 12.22, lead_x 6.76; qat3 plan 12.42, lead_x 6.65. 출시 모델은 9.32, 5.94다. 나머지 지표는 이미 이긴다.
- **Mac에서 라벨링(PC 없이)**:
  - teacher와 small ONNX를 HF LFS 미러에서 받았다(openpilot 커밋 64f9b47b6e의 포인터, sha256 일치). `models/`에 있다.
  - LD는 `~/Documents/LD/lane_detection.onnx`(PC의 FVRFBGRLEDT0001819와 같은 I/O)를 쓴다. prior y 표는 `ld_teacher.PRIOR_Y`에 넣었다.
  - 원본 주행: USB `/Volumes/Untitled/edgepilot_drives/experiments` (exFAT, `._*` 파일은 reader가 건너뛴다).
  - 검증: 이미 라벨된 세그먼트를 다시 라벨해 PC 라벨과 비교했다. teacher 상대 오차는 0.7% 이하, LD 차선은 0.5 mm이고 warp한 영상은 바이트 단위로 같다.
  - 속도: ORT CPU는 368 ms/프레임이다(다른 작업과 같이 돌면 1.2 s).
    **CoreML(Neural Engine)은 teacher의 음수 인덱스 Gather 5개를 틀리게 계산해 plan/lead가 100% 넘게 틀렸다.**
    인덱스를 양수로 바꾼 `models/big_driving_supercombo_coreml.onnx`는 CPU 대비 1% 안이고 80 ms/프레임이다.
    `ORT_COREML=CPUAndNeuralEngine`이면 `ep_label.py`가 이것을 쓴다. 첫 컴파일은 7~8분 걸리고 결과는 `outputs/coreml_cache`(약 5 GB)에 캐시된다.
  - `ep_label.py --no-small`(small은 평가 행에만 쓰여서 NaN으로 둔다), `scripts/label_queue.sh <목록> <원본> 2`, `scripts/label_one.sh`.
  - 대상: 보드의 새 주행 2개(10-29--03-10, 10-30--00-08, 각 약 20분; USB로 복사했다)와 PC에서 라벨링하지 않은 14개 경로(69 세그먼트).
    목록은 `configs/mac_label_queue.txt`이고 평가 경로 10-22--23-22는 뺐다. 학습 drive는 19개에서 약 34개가 된다.
- **디스크**: Mac 여유는 약 25 GB다.
  - 외장 HDD `/Volumes/Backup/supercombo_distill_offload/`로 옮긴 것: quant_flip8, diag dump(aug10/base5/nohist), w48d·5ep·scale ckpt, aug10/flip8 epN.pt.
  - **`scripts/zframes.py`**: 프레임을 한 장씩 zstd(3)로 압축한다(`frames.zst` + `frames_off.npy`). 1.5~1.7배 작아지고, 해제는 0.4 ms/프레임이다.
    모든 읽기 코드가 `load_frames()`를 쓴다(npy가 있으면 그것, 없으면 zst).
    새로 라벨된 unit은 label_one.sh가 바로 압축한다. **기존 학습 unit은 gap3b 학습이 끝난 뒤 변환할 것**(지금 run의 DataLoader worker는 예전 코드라 npy가 없으면 죽는다).
    평가 unit은 보드 측정(board_eval)이 frames.npy를 올리므로 그대로 둔다.
- **실험 (자세한 것은 HTML 보고서)**:
  - gap3 v1(plan t=0만 학습): 움직임은 읽게 됐지만 long_acc가 0.25에서 1.12로 망가져 1 epoch 뒤 중단했다.
  - **gap3b(v2) 진행 중**: 간격 샘플에서 plan 전체를 시간축 변환해 학습한다(x(kt), 속도 ×k, 가속 ×k²). pose, lead 위치, 차선류도 학습하고 EMA는 0.999다.
    02:33에 시작해 05:40쯤 끝난다. 결과는 `logs/eval_mac_base_gap3b.txt`(qat3, last, ema 비교), `logs/diag_gap_mac_base_gap3b.txt`, `logs/DONE_mac_base_gap3b`.
  - SWA(qat3 ep2+3): plan 12.22, lat 0.0498(좋아짐), lead 6.76(조금 나빠짐).
  - R18 backbone(ImageNet ResNet-18, `Student(backbone="r18")`): 2.34 GMAC이고 보드 latency는 11.77 ms다. 아직 학습하지 않았다.
- **새 학습 옵션 (train.py / student.py)**:
  - `--hist-noise r`, `--hist-drop p`: comma의 noise bottleneck과 ChauffeurNet의 past dropout
  - `--ema d`: `ema.pt`로 저장한다
  - arch `policy_tokens` 25, `temporal_heads` ["lead", "lead_prob"]: `expand_state()`가 새 입력을 0으로 두어 함수를 그대로 유지한다. `--init`으로 이어 학습할 수 있다
  - arch `backbone` "r18"
- **논문 조사(서브에이전트)**: 1) 재생 속도 augmentation, 2) 증강 입력을 teacher로 다시 라벨링하는 일관된 증류(Beyer et al.), 3) 데이터 추가, 4) 사전학습 backbone,
  5) flow 보조 head, 6) history noise/dropout, 7) EMA/SWA, 8) 헛 lead 대책. 목록은 HTML 보고서의 "참고한 논문" 표에 있다.
- 다음 계획:
  - gap3b 결과를 본다.
  - 기존 학습 unit을 압축한다.
  - 새 데이터와 최적 조합으로 학습한다(+ temporal lead head, hist 정규화).
  - ANE로 calibration을 흔든 replay를 teacher로 라벨링한다(일관된 증류).
  - 필요하면 R18을 학습한다.
- **04:55 기준 진행 상황 (자동으로 이어진다)**:
  - gap3b ep2(S8): plan 12.08, lead_x 6.57, lanes 10.10, vx0 146으로 qat3보다 좋다. 다만 커브에서 덜 꺾는 것(기울기 0.89)과 야간 경로 16-28은 그대로다.
  - 앙상블(swa_qat3 + gap3b ep2): plan 11.92다. 이득이 1.3%로 작다.
  - **`scripts/chain_runA.sh`가 대기 중이다.** gap3b train.py가 끝나면 학습 unit을 압축하고(`logs/zframes_convert.log`), gap3b 최종 평가를 기다린다.
    이어 Run A를 시작한다(`ckpt/mac_base_runA`, 4 epoch). init은 gap3b의 last와 ema 중 plan이 좋은 쪽이다.
    - 구성: 새 경로 데이터, `temporal_heads` lead, `lane_policy`, gap-aug 0.2, hist-noise 0.2, EMA. replay unit은 넣지 않는다(`--tags` 없음).
    - 결과: `logs/eval_mac_base_runA.txt`, `logs/DONE_mac_base_runA`.
  - **`lane_policy`(student.py)**: 차선과 도로 경계 head feature(64+32)를 temporal MLP 입력 끝에 붙인다.
    student의 차선은 LD 대비 10 cm로 출시 모델(18.5)보다 정확한데, plan은 커브에서 그 차선을 따라가지 않기 때문이다.
  - train.py: 마지막 epoch 뒤에는 bank를 다시 계산하지 않는다(run마다 23분 절약). `--refresh-every N`, `--tags`(replay unit 선택; 'all'이면 전부).
  - eval_student.py `--ensemble`: student들의 raw 출력 평균을 한 행으로 추가한다.
  - 라벨링 대기열(Neural Engine, 한 프로세스):
    1. `configs/mac_label_rest.txt`(새 데이터)
    2. `configs/mac_label_eval5.txt`(5번째 평가 경로 10-22--23-22)
    3. `configs/mac_replay_cal1.txt`(`--tag cal1 --rand-rpy-deg 1 1 2 --rand-offset 0.3`, 학습 경로의 절반)
    4. `configs/mac_replay_r2.txt`(`--tag r2 --rate 2`, 나머지 절반). replay는 `--zstd`로 청크마다 압축한다.
    - 로그: `logs/label/batch_*.log`, 끝나면 `logs/label/DONE_*`
  - `ep_label.py` 새 옵션: `--also`(한 프로세스에서 여러 경로를 처리해 모델을 한 번만 로드한다), `--tag`, `--rpy-deg`, `--offset`, `--rate`, `--rand-rpy-deg`, `--rand-offset`, `--zstd`.
  - **메모리 주의**: 16 GB Mac에서 CoreML teacher 프로세스 두 개와 학습을 같이 돌리면 스왑이 24 GB까지 차서 멈춘다. 라벨링은 한 프로세스만 돌린다.
- **07:15 기준**:
  - 학습 unit 58개를 모두 zstd로 압축했다(여유 6 → 61 GB). 새 unit은 라벨링 직후 압축된다.
  - 평가 경로 10-22--23-22도 라벨링했다. **이제 eval_student 기본값은 평가 경로 5개(8 unit)다.**
    - 5경로 기준 출시 모델: plan 9.94, lead 5.83, lat 0.0691, lanes 18.58, width 25.85, gas2 0.236
    - gap3b EMA: plan 12.45, lead 6.26, lat 0.0527, lanes 10.16, width 11.54, gas2 0.139
    - `logs/eval_5routes_baselines.txt`
  - Run A: 06:00에 시작했고 10:30쯤 끝난다. ep1(5경로)은 plan 13.30, lead 6.33, lanes 10.05였다.
  - 대기 중인 체인(모두 PID가 아니라 DONE 파일을 기다린다). **09:20에 순서를 바꿨다**:
    Run A ep3이 gap3b보다 조금밖에 좋아지지 않아서 큰 변화(R18)를 먼저 하고, replay가 다 모인 뒤 Run B를 한다.
    1. `scripts/chain_r18.sh`: Run A가 끝나면(11시쯤) 시작한다.
       - R18 backbone(ImageNet)을 `--init-partial`로 Run A의 head 아래에 넣는다.
       - `--tags r2,cal1`(그때까지 라벨된 것), `policy_tokens` 25, fp32 4 epoch, lr 1e-3, bank는 2 epoch마다(약 6 h).
       - 끝나면 QAT가 필요하다. 평가는 fp32로 한다(`logs/eval_mac_r18.txt`).
    2. `scripts/chain_runB.sh`: R18이 끝나면 시작한다. Run A의 더 나은 체크포인트에서 `--tags r2,cal1`, `policy_tokens` 25로 QAT 3 epoch를 돈다(약 4.5 h).
  - Run A ep3(5경로): plan 12.40, lead 6.24, lead_x_lf 4.63, 헛 lead 12.7%, lanes 9.99.
  - **Run A 최종(10:43, EMA, 5경로)**: plan 12.25, lead 6.13, lead_x_lf 4.54, 헛 lead 12.6%, vx0 124, lanes 9.97, lat 0.0505. 모든 지표가 gap3b보다 좋다.
    결과는 `logs/eval_mac_base_runA.txt`에 있다. train.py가 마지막 bank를 건너뛰어 10:04에 끝났다.
  - **R18 run 진행 중 (10:43~, `ckpt/mac_r18`, `logs/mac_r18.log`)**:
    - 139 unit(새 주행, r2, cal1 일부), 57.7만 샘플 중 epoch마다 40만을 뽑는다.
    - 처음 bank는 21.5분, 속도는 130 samples/s라 epoch는 약 51분이다. 14시쯤 끝난다.
    - 시작 loss가 높다(plan 3.45). backbone이 바뀌어 head가 적응하는 중이다.
    - 끝나면 `chain_runB.sh`가 Run B를 시작한다(~19시).
  - `--epoch-samples N`: epoch마다 N개를 새로 뽑는다(curve-sampling이면 가중 샘플링).
  - `--plan-lat-weight w`: plan loss 평균에서 횡방향 열(y, vy, ay, yaw, yaw rate)에 w배 가중치를 준다(가중 평균이라 전체 크기는 비슷하다).
    `--lead-weight`는 lead loss 가중치다(기본 0.5).
  - Run B(chain_runB.sh)는 11:40에 다시 걸었다: `--plan-lat-weight 3 --lead-weight 1.0 --epoch-samples 450000`, `--tags r2,cal1`, `policy_tokens` 25.
  - **13:15: R18 epoch 2(fp32, 5경로)에서 lead_x 5.65로 처음 출시 모델(5.83)을 이겼다.** lead_x_lf는 4.03(출시 4.28)이다.
    plan은 13.12(ep1 15.37), lanes는 12.9, lat_acc는 0.093(출시 0.069), 헛 lead는 9.7%다. 아직 빠르게 좋아지고 있다.
    - 그래서 **Run B 체인을 멈추고 `scripts/chain_r18b.sh`를 걸었다**:
      1. R18이 끝나면 더 나은 체크포인트(fp32 plan 기준)에서 `ckpt/mac_r18b`로 fp32 4 epoch를 더 돈다(lr 5e-4, `--plan-lat-weight 3`, 40만 샘플/epoch).
      2. 이어 `ckpt/mac_r18q`로 QAT 2 epoch(lr 2e-4)를 돌고 S8로 평가한다(`logs/eval_mac_r18q.txt`, Run A EMA와 비교). 21시쯤 끝난다.
      3. 그다음 Pulsar2 빌드와 보드 실측을 한다(R18 probe는 11.77 ms였다).
    - R18 QAT를 확인했다: BN을 conv에 접은 뒤 반올림한다. QAT forward와 S8 평가가 정확히 같고(차이 0), 저장, 불러오기, export가 된다.
      `R18Backbone.is_fused()`는 이제 RepConv까지 합쳐졌을 때만 True다.
  - **14:10 R18 3 epoch 끝 (fp32, 5경로, `logs/eval_mac_r18.txt`)**:
    - EMA 기준 plan 10.63, lead 5.29, lead_x_lf 3.81, lat 0.0472, lanes 11.75, width 14.52, gas2 0.131, 헛 lead 8.4%다.
    - **plan_y2s(출시 9.94)만 빼고 모든 지표가 출시 모델을 이긴다.**
  - R18b(`ckpt/mac_r18b`)는 14:15에 시작했다. 161 unit, 68.8만 샘플 중 epoch마다 40만을 뽑는다. 18:50쯤 끝나고, 이어 R18q(QAT)가 돈다.
  - **R18(ep3) PTQ 보드 실측(`outputs/quant_r18e3`, `logs/board_eval_r18e3.txt`)**: latency 12.49 ms, max_cycle 6.73M.
    unit 015에서 plan이 fp32 18.62, S8 18.70, 보드 18.55로 PTQ 손실이 거의 없다(ResNet은 8bit에 강하다). 그래서 QAT가 꼭 필요하지는 않을 수 있다.
    5경로 S8 평가는 `logs/eval_r18_ema_s8.txt`에 있다.
  - quant_build.py 보정 데이터는 이제 replay unit(`__tag`)을 뺀다. r18e3 빌드는 이 수정 전이라 r2 unit이 섞였다.
  - **R18 EMA S8(5경로, `logs/eval_r18_ema_s8.txt`)**: plan 10.68, lead 5.39, lat 0.0488, lanes 11.84, width 14.45, gas2 0.129, lead_x_lf 3.93, 헛 lead 8.5%.
    배포 기준으로도 plan_y2s만 출시 모델(9.94)에 진다.
  - R18 경로별 plan(`outputs/diag_y2s/dump/r18_ema`):
    - 8 unit 중 5개에서 출시 모델을 이긴다. 커브 기울기는 1.006으로 덜 꺾는 문제가 없어졌다. desire 구간과 3 m 이상 회전에서도 이긴다.
    - 남은 차이 +0.74 cm는 **야간 경로 16-28 두 unit(+0.90)**과 00-07(+0.23)에서 나온다.
  - 그래서:
    - `--night-weight w`(curve-sampling에서 야간 unit, 즉 road 영상 평균 Y < 150인 unit을 w배로 뽑는다; 2.5면 야간 비율이 약 37%)를 추가했다.
    - 야간 경로 4개를 calibration을 다시 흔들어 replay한다(`--tag cal2`, `configs/mac_replay_cal2_night.txt`, 15:10에 시작).
  - **체인 교체(15:10)**: chain_r18b.sh의 부모 셸을 멈췄다. R18b 학습(PID 55996)과 snap 평가는 계속 돈다.
    `scripts/chain_r18q.sh 55996`이 R18b가 끝나기를 기다린 뒤 다음을 한다:
    1. R18b를 평가한다(`logs/eval_mac_r18b.txt`, DONE_mac_r18b).
    2. R18q로 QAT 2 epoch를 돈다(`--night-weight 2.5 --tags r2,cal1,cal2 --plan-lat-weight 3`).
    3. S8로 평가한다(`logs/eval_mac_r18q.txt`, Run A EMA, R18 EMA와 비교).
  - cal1 replay는 끝났다. 디스크 부족으로 마지막 몇 경로는 빠졌다. 여유 공간은 13 GB이고, Pulsar2 빌드 중간 산출물은 외장 HDD로 옮겼다.
  - R18b ep1(fp32, lr 최고 구간): plan 12.79, lat 0.122(흔들림), lead 5.04, vx0 97, 헛 lead 17.4%.
  - chain_r18q.sh를 고쳤다(15:55, PID 55996을 기다린다).
    - R18 EMA, R18b last, R18b EMA를 한 번에 fp32로 평가한다.
    - plan이 가장 좋은 것에서 QAT를 시작한다. R18b가 R18 ep3보다 나빠도 손해가 없다.
  - **R18b ep3(fp32, 5경로)**: plan 10.30(출시 9.94와 3.6% 차이), lead 4.89, lanes 10.29, width 12.82, lat 0.0479, 헛 lead 7.1%, vx0 78.
    plan을 빼고 모두 이긴다. ep2는 plan 10.92, lat 0.094였다.
  - 디스크(16:10에 6.4 GB까지 떨어졌다):
    - cal2 라벨링을 멈췄다(10-04--12-32의 3 청크만 완성).
    - 예전 ckpt(epN.pt, gap3 v1, swa, r18_probe), diag dump, Pulsar2 quant 폴더를 외장 HDD로 옮겼다.
    - CoreML 캐시를 지웠다(다시 만들면 8분).
    - 평가 unit 7개도 zstd로 압축했다(보드 측정용 unit 015는 이미 보드에 올라가 있다). 여유는 17 GB다.
  - 17:39에 남은 야간 경로 3개의 cal2 replay를 다시 시작했다(`configs/mac_replay_cal2_night_b.txt`).
  - plan 출력을 시간축으로 평활하는 후처리는 오히려 나빠진다(10.68 → 10.75~11.75). 쓰지 않는다.
  - **R18b 최종(18:44, fp32, 5경로, `logs/eval_mac_r18b.txt`)**: EMA 기준 plan 10.04(출시 9.94와 1% 차이), lat 0.0448, lead 4.90, lead_x_lf 3.39, lanes 9.99, width 12.61, gas2 0.129, 헛 lead 7.5%, vx0 76.
    R18b EMA를 그대로 양자화한 S8 평가는 `logs/eval_r18b_ema_s8.txt`에 있다.
  - R18q(QAT, R18b EMA에서 시작, 166 unit, 야간 2.5배)는 18:45에 시작했다. 21시쯤 끝난다.
  - R18b EMA를 QAT 없이 양자화(S8, 5경로, `logs/eval_r18b_ema_s8.txt`): plan 10.08, lead 4.96, lat 0.0493, lanes 10.13, width 12.60, gas2 0.127.
    plan(9.94)만 1.4% 차이로 진다.
  - R18q ep1(S8): plan 10.54, lat 0.058. 가장 높은 lr에서 흔들려 R18b PTQ보다 나쁘다. R18은 PTQ 손실이 작아 QAT의 이득이 적다.
  - **20:05 chain_r18r.sh를 멈추고 `scripts/chain_r18c.sh`로 바꿨다**(실행 중, SSHPASS는 환경변수로만 넘겼다). DONE_mac_r18q와 DONE_final을 기다린 뒤:
    1. R18q last/EMA와 R18b EMA(PTQ) 중 S8 plan이 가장 좋은 것이 6개 지표(lat, plan, lanes, width, lead, gas2)를 모두 이기면 빌드와 보드 측정을 한다(r18win).
    2. 아니면 R18b EMA에서 fp32 4 epoch를 이어서 학습한다(R18c: lr 3e-4, 야간 3배, `--tags r2,cal1,cal2`, `--plan-lat-weight 3`).
       이어서 S8 평가, 최선 후보 빌드(`outputs/quant_r18c`), 보드 측정(`logs/board_eval_r18c.txt`)을 한다.
  - **`scripts/chain_final.sh`**(실행 중, `SSHPASS`는 실행 환경변수로만 넘겼다)가 DONE_mac_r18q를 기다린다. 이어서 다음을 한다:
    1. R18q의 더 나은 체크포인트(S8 plan)를 Pulsar2로 빌드한다(`outputs/quant_r18final`).
    2. 보드에서 잰다(`logs/board_eval_r18final.txt`). `logs/final.log`, DONE_final에 남긴다.
  - `ep_label.py --desire-inject S`: 가상 desire replay다. 약 S초마다 직진 구간(v > 8 m/s)에 차선 변경 desire를 3초 넣는다. 아직 돌려보지 않았다.
    평가 데이터에는 교차로 회전 desire가 학습 데이터보다 5배 많다(1.71% vs 0.35%).
  - replay 라벨링: r2(17 경로)가 진행 중이다. cal1은 디스크 부족으로 처음에 대부분 건너뛰어서, r2 다음에 다시 돈다(`configs/mac_replay_cal1b.txt`).
  - refresh_bank는 이제 batch마다 프레임을 한 번씩만 풀고, 3 스레드로 미리 읽는다(기존 코드와 결과 동일).
    zstd 해제기는 스레드마다 따로 둔다(하나를 같이 쓰면 데이터가 깨진다).
  - colima(docker VM, Pulsar2용)는 켜져 있다. 멈추는 것은 권한 정책이 막았다.

### 0000000. 끝난 것 (2026-10-30 01:19 ~ 02:30, 1 epoch 뒤 중단): 프레임 간격 augmentation v1 (`ckpt/mac_base_gap3`)
사용자 지시: "a 시작" (000000절의 개선안 A).
- **바꾼 것 (train.py; 기본값이면 예전과 같다. 이전 파일은 scratchpad `train.py.qat3`)**
  - `--gap-aug p`: 샘플의 p만큼 첫 프레임을 kept 프레임 j-g(g = 0, 1, 3, 4)에서 가져온다(정상은 g = 2, 즉 t-4). 간격 비율은 g/2다.
    - 이 샘플은 `gap_loss()`만 학습한다: pose 평균 6개와 plan t=0의 vx, yaw rate를 teacher 값 × 비율에 맞춘다. 단위는 teacher σ이고 L1에 capped()를 쓴다. 가중치는 0.5다.
    - 다른 loss는 이 샘플을 뺀다. history는 0으로 준다(주행 시작과 같은 상태).
    - j-g < 0이면 정상 샘플로 돌린다. flip과 함께 써도 된다(정답 배율은 선형이라 반전 뒤에 곱해도 같다).
  - 시험: 같은 프레임(g = 0)이면 두 프레임이 같다. 간격을 바꾼 샘플의 history는 0이다. teacher × 비율을 그대로 넣으면 gap_loss는 0이다.
    `--gap-aug 0`이면 샘플에 gap 필드가 없고 loss도 예전과 같다. tinygap(unit 하나)에서 QAT 학습과 저장이 된다.
  - qat3의 시작 gap loss는 4.47이다(정상 샘플의 pose loss는 0.66). 움직임을 못 읽는다는 진단과 맞는 값이다.
- **eval_student.py에 열 3개를 추가했다(기존 열은 그대로다):**
  - `lead_x_lf`: lead_x를 lead 프레임 수로 가중한 값
  - `false_lead_pct`: teacher에 lead가 없는 이동 프레임 중 모델이 lead를 낸 비율(%)
  - `vx0_cms`: plan의 현재 속도 오차(cm/s)
- 명령:
  `nohup caffeinate -i sh -c "(sleep 60; scripts/snap_epochs.sh ckpt/mac_base_gap3 3 eval-s8 > logs/snap_mac_base_gap3.log 2>&1) & .venv/bin/python scripts/train.py --out ckpt/mac_base_gap3 --epochs 3 --batch 64 --workers 4 --compile-refresh --aug --wd 0.01 --flip 0.3 --curve-sampling --qat --gap-aug 0.25 --lr 2e-4 --init ckpt/mac_base_qat3/last.pt && .venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_base_qat3/last.pt ckpt/mac_base_gap3/last.pt > logs/eval_mac_base_gap3.txt 2>&1; for u in 2026-10-05--16-28-44-410__000 2026-10-10--17-34-10-953__000; do .venv/bin/python scripts/diag_gap.py \$u ckpt/mac_base_gap3/last.pt; done > logs/diag_gap_gap3.txt 2>&1; echo DONE > logs/DONE_mac_base_gap3" > logs/mac_base_gap3.log 2>&1 < /dev/null &`
  - 예상 시간은 qat3와 같다: 처음 bank 23분에 epoch당 약 42분이라 04:00 전후에 끝나고, 평가와 간격 시험까지 04:15쯤 끝난다.
- 판단 기준(qat3 S8 기준값): plan_y2s 12.42, lead_x 6.65(lead 프레임 기준 약 4.9), 헛 lead 14.3%, 평가 경로 vx0 161 cm/s.
  간격 시험에서 pose vx가 0 / 0.5 / 1 / 2배에 가까워지는지도 본다(qat3: 0.63~0.72 / 0.89~0.95 / 1 / 1.01~1.08).
- 확인할 파일: `logs/mac_base_gap3.log`, `logs/snap_mac_base_gap3.log`, `logs/eval_mac_base_gap3_snaps.txt`, `logs/eval_mac_base_gap3.txt`, `logs/diag_gap_gap3.txt`, `logs/DONE_mac_base_gap3`

### 000000. 진단: qat3의 plan_y2s와 lead_x가 출시 모델보다 나쁜 이유 (2026-10-30 00:45 ~ 01:15)
사용자 지시: "1번 진행해" (원인을 보고 개선 학습을 계획).
- dump: `diag_y2s_dump.py --s8`로 qat3 S8의 plan과 lead를 평가 unit 7개와 학습 unit 3개에서 저장했다(`outputs/diag_y2s/dump/qat3_s8`). lead 필드를 추가했다.
- 분석: `diag_y2s.py qat3_s8 aug10`(`logs/diag_y2s_qat3.txt`), 새 `diag_lead.py qat3_s8`(`logs/diag_lead_qat3.txt`).

**결론: 둘 다 처음 보는 경로에서만 나쁘다(일반화). 핵심 원인은 student가 속도를 움직임에서 읽지 않고 장면에서 외운다는 것이다.**

1. **plan_y2s** (평가 경로 12.42, small 9.62, 학습 경로에서는 6.26으로 출시 모델 9.24보다 좋다)
   - 차이 2.80 cm 중 2.36 cm가 주간 경로 하나(10-05--16-28, unit 2개; 야간이 아니다)에서 나온다.
     __015 unit은 27.6 대 12.5인데 대부분 desire가 걸린 큰 회전에서 나온다.
   - 직선(|y2s| < 0.1 m)은 student가 더 좋다(4.50 대 5.21). 커브에서 덜 꺾는다(기울기 0.886, small 1.024, 학습 경로 0.983).
   - 밝기 문제는 아니다: 그 경로 영상의 Y를 1.35배로 낮처럼 밝혀도 오차는 그대로다(9.42 → 9.58, 27.6 → 28.4).
   - 현재 속도 vx0 오차는 평가 경로 161 cm/s(small 65), 학습 경로 30 cm/s다. yaw rate r0 오차와 y2s 오차의 상관은 0.52다.
2. **lead_x** (lead 프레임 기준: student 4.91, 출시 4.15, small 3.56 m; 학습 경로에서는 1.13으로 출시 2.31보다 좋다)
   - 모든 거리에서 조금씩 나쁘다(0~20 m에서도 2.07 대 1.16). 놓친 lead(9.7%)가 평균 1.25 m를 차지한다.
   - **헛 lead가 많다.** teacher는 lead가 없다는데 student가 있다고 하는 비율이 평가 경로 이동 프레임의 14.3%다(출시 4.9%, small 7.3%, 학습 경로 2.6%). 거리는 중앙값 90 m로 먼 것이 대부분이다.
   - `eval_student.py`의 lead_x는 unit별 평균을 lead 프레임이 아닌 이동 프레임 수로 가중한다. 그래서 lead가 드문 10-05--00-07 unit(오차 10.8 m)이 커져 6.65 / 5.94로 나온다. lead 프레임 기준으로는 4.91 / 4.15이고, 순위는 같다.
3. **원인 확인: 속도를 장면에서 외운다** (`scripts/diag_gap.py <unit> [ckpt]`)
   - history를 끄면 학습 경로에서도 plan vx0 오차가 25에서 267 cm/s가 된다(평가 경로 187 → 533).
   - pose head(현재 프레임 두 장만 본다)의 vx 오차는 평가 unit 214, 학습 unit 47 cm/s다.
   - 프레임 간격을 바꿔 pose를 봤다(정상 간격 = 1x):

     | 간격 | 0x (같은 프레임) | 0.5x | 1x | 2x |
     |---|---|---|---|---|
     | pose vx / teacher (학습 unit) | **0.63** | 0.89 | 0.96 | **1.01** |
     | pose vx / teacher (평가 unit) | **0.72** | 0.95 | 1.03 | **1.08** |
     | yaw rate / teacher (학습 unit) | 0.07 | 0.81 | 0.98 | 1.01 |

     움직임이 0이어도 속도를 63~72%로 내고, 움직임이 두 배여도 1~8%만 커진다. 속도는 거의 장면에서 나온다.
     yaw rate는 움직임에 반응하지만 비례하지는 않는다. 학습 drive가 19개뿐이라 "이 장소는 이 속도"를 외울 수 있다.

**개선 계획**
- A. **프레임 간격 augmentation (`--gap-aug p`, Mac에서 바로 가능, 제안).**
  - 샘플의 p만큼 첫 프레임을 다른 간격(0, 0.5, 1.5, 2배)의 프레임으로 바꾼다. 이때 정답은 물리적으로 정해진다:
    pose의 속도와 회전, plan t=0의 vx와 yaw rate는 teacher 값 × 간격 비율이다.
  - 이 샘플은 이 항목만 학습하고 나머지 loss는 뺀다. history는 0으로 준다(주행 시작과 같은 상태라 현재 프레임에서 읽게 된다).
  - 같은 장면에서 간격만 다르면 답이 달라지므로, 장면으로 속도를 외우는 지름길이 막힌다.
  - 실행 계획: qat3에서 QAT 3 epoch(약 2.5시간), 나머지 설정은 qat3와 같고 p = 0.25다.
  - 판단 기준: 평가 경로의 vx0와 r0 오차, 간격 시험에서 pose가 0 / 0.5 / 1 / 2배에 가까워지는지, plan_y2s(12.42), lead_x, 헛 lead 비율.
- B. **데이터: drive 다양성 (사용자 결정이 필요하다).**
  - PC에 라벨링하지 않은 21청크가 남아 있다. 프레임은 +10%뿐이지만 새 drive가 13개라 학습 drive가 19개에서 32개가 된다.
    PC에서 라벨링 30분, 전송 20GB가 필요하다. Mac 여유 공간은 34GB다.
  - 새로 녹화한다면 야간 시내의 회전과 차선 변경이 가장 부족하다.
- C. 평가 보완: lead 프레임 기준 lead_x, 헛 lead 비율, vx0/r0 오차를 `eval_student.py`에 추가한다(기존 열은 비교를 위해 그대로 둔다).

### 00000. 끝난 것 (2026-10-29 19:17 ~ 22:05): QAT 3 epoch (`ckpt/mac_base_qat3`)와 보드 실측
사용자 지시: "QAT 진행해. 이후 보드에서 측정해."

**결론: 지금 배포 후보는 `outputs/quant_qat3/m.axmodel`이다.**
- 보드 NPU latency는 5.32 ms로 출시 모델(14.0 ms)의 38%다.
- 보드 출력은 Pulsar2와 같다(상대 오차 0.0003). 정확도는 평가 경로 전체(S8 시뮬레이션)에서 flip8 fp32와 같거나 낫고, 보드에서 잰 unit 015에서도 거의 같다.
- 출시 모델보다 lat_acc, lanes, width, gas2는 크게 좋다. plan_y2s와 lead_x는 아직 나쁘다.
- edgepilot에 실제로 넣어 주행하는 것은 아직 하지 않았다.

- QAT: 0000절의 제안 명령 그대로다(flip8에서 시작, `--qat --lr 2e-4`, 3 epoch).
  - epoch마다 `snap_epochs.sh ... eval-s8`이 S8로 평가했다(`logs/eval_mac_base_qat3_snaps.txt`).
  - `logs/eval_mac_base_qat3.txt`에 flip8과 qat3의 S8 비교가 있다.
- **21:27에 끝났다.** 처음 bank 1397 s, epoch당 학습 약 18분(335 samples/s, flip8보다 빠르다)과 bank refresh 23분이다.
- **QAT 결과 (평가 경로 전체, S8 시뮬레이션, `logs/eval_mac_base_qat3_snaps.txt`, `logs/eval_mac_base_qat3.txt`)**

  | 모델 | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
  |---|---|---|---|---|---|---|---|
  | flip8 fp32 (참고) | 0.0522 | 0.251 | 12.70 | 10.29 | 11.81 | 6.69 | 0.133 |
  | flip8 S8 (PTQ) | 0.0555 | 0.259 | 13.48 | 11.74 | 13.58 | 7.26 | 0.146 |
  | qat3 ep1 S8 | 0.0571 | 0.258 | 12.71 | 10.34 | 11.88 | 6.65 | 0.129 |
  | qat3 ep2 S8 | 0.0555 | 0.262 | 12.26 | 10.37 | 11.94 | 6.84 | 0.132 |
  | **qat3 ep3 S8 (last.pt)** | **0.0532** | **0.253** | **12.42** | **10.29** | **11.83** | **6.65** | **0.132** |
  | board NPU (출시) | 0.0691 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |

  - QAT가 PTQ로 잃은 것을 모두 되찾았다. S8인데도 flip8 fp32와 같거나 낫다(plan 12.42 vs 12.70, lead 6.65 vs 6.69).
  - 출시 모델보다 여전히 나쁜 것은 plan_y2s(12.4 vs 9.3)와 lead_x(6.65 vs 5.94)다.
- **qat3 빌드와 보드 실측 (22:05, `logs/quant_build_qat3.out`, `logs/quant_eval_p2_qat3.txt`, `logs/board_eval_qat3.txt`)**
  - 빌드 명령은 `quant_build.py --ckpt ckpt/mac_base_qat3/last.pt --name qat3`다. 62 s 걸렸고 max_cycle은 3,252,119로 flip8과 같다.
  - 결과: `outputs/quant_qat3/m.axmodel`, `configs/quant_qat3.json`, `logs/quant_qat3.log`
  - 보드 latency(ax_run_model 500회)는 **5.32 ms**다(min 5.317, max 5.434).
  - 보드 출력과 Pulsar2 Reference의 상대 오차는 0.0002~0.0003이다.
  - 다시 재는 명령: `SSHPASS=<보드 비밀번호> .venv/bin/python scripts/board_eval.py --ckpt ckpt/mac_base_qat3/last.pt --build outputs/quant_qat3 --name qat3 --units 2026-10-25--22-42-26-175__015`

  | 모델 (unit 015, 1497프레임) | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
  |---|---|---|---|---|---|---|---|
  | flip8 fp32 | 0.0735 | 0.317 | 21.29 | 14.09 | 14.98 | 5.62 | 0.175 |
  | flip8 board NPU (PTQ) | 0.0887 | 0.320 | 23.48 | 17.94 | 16.48 | 4.69 | 0.164 |
  | qat3 S8 시뮬레이션 | 0.0768 | 0.324 | 21.24 | 14.21 | 15.30 | 5.54 | 0.175 |
  | qat3 Pulsar2 (Reference) | 0.0744 | 0.325 | 21.13 | 14.44 | 15.63 | 5.48 | 0.175 |
  | **qat3 board NPU** | **0.0744** | **0.325** | **21.13** | **14.44** | **15.63** | **5.48** | **0.175** |
  | qat3 fp32 (의미 없음) | 0.1129 | 0.341 | 20.28 | 18.41 | 18.92 | 8.09 | 0.188 |

  - 보드 NPU에서 qat3는 flip8 fp32와 거의 같다. PTQ에서 lanes가 3.9 cm 나빠졌던 것이 0.35 cm로 줄었다.
  - QAT 뒤에는 S8 시뮬레이션과 Pulsar2의 차이가 작다(lat는 시뮬레이션 쪽이 조금 나쁘고, lanes는 조금 좋다).
    그래서 평가 경로 전체의 `--s8` 수치(위 표)를 보드 수치로 봐도 된다.
  - **QAT 체크포인트는 fp32로 평가하면 안 된다.** 가중치가 S8 반올림에 맞춰져서 fp32 출력은 오히려 나쁘다. 항상 `--s8`로 평가한다.

**보드 실측: flip8 PTQ (2026-10-29 19:25, `scripts/board_eval.py`, `logs/board_eval_flip8.txt`)**
- Mac에는 sshpass가 없다. 대신 `SSH_ASKPASS` helper가 `$SSHPASS`를 echo하게 했다(임시 파일이고 비밀번호는 담지 않는다).
- 보드의 `/root/distill`에 모델, `board_run.py`, 평가 unit(frames.npy, pulse.npy)을 올린다. unit은 한 번만 올린다(590 MB, 약 2분).
- edgepilot(modeld)이 NPU를 잡고 있어서 pyaxengine이 `AX_ENGINE_Init`에서 실패한다. 측정하는 동안 서비스를 멈추고, 끝나면(실패해도) 다시 켠다.
- **latency (ax_run_model, warmup 20, 500회): 5.32 ms** (min 5.317, max 5.371). 출시 모델은 14.0 ms다.
  - pyaxengine `run()`은 Python 입력 복사를 포함해 9.7 ms다(중앙값).
- **보드 출력은 Pulsar2 Reference와 같다.** out_hidden과 policy 출력의 상대 오차가 0.0008이고, 지표는 소수 셋째 자리까지 같다.
  그러니 `quant_eval_p2.py`(Mac에서 프레임당 0.1 s)를 보드 결과로 봐도 된다.

| 모델 (unit 015, 1497프레임) | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
|---|---|---|---|---|---|---|---|
| fp32 | 0.0735 | 0.317 | 21.29 | 14.09 | 14.98 | 5.62 | 0.175 |
| S8 시뮬레이션 | 0.0882 | 0.320 | 22.96 | 17.20 | 16.30 | 4.76 | 0.164 |
| Pulsar2 (Reference) | 0.0887 | 0.320 | 23.47 | 17.95 | 16.48 | 4.69 | 0.164 |
| **board NPU** | 0.0887 | 0.320 | 23.48 | 17.94 | 16.48 | 4.69 | 0.164 |

### 0000. 양자화 확인 (2026-10-29 16:20 ~ 17:05): flip8을 Pulsar2로 빌드
사용자 지시: "양자화 먼저 해보자". 출시 모델 수치는 NPU에서 양자화된 상태로 잰 값이라, student도 양자화한 뒤 비교해야 공정하다.

**한 것**
- Mac에서 Pulsar2를 돌린다. `pulsar2:6.0-lite`(amd64) 이미지가 colima(aarch64 VM, Rosetta와 binfmt 켜짐)에 이미 있었다.
  - 이미지 entrypoint가 `/bin/bash`라서 `--entrypoint /bin/bash ... -c "pulsar2 ..."`로 실행해야 한다.
  - 프로젝트는 `/data/supercombo_distill`로 마운트한다(PC와 같은 경로).
- `scripts/quant_build.py`:
  1. ONNX export(fused, CPU heads 분리). PyTorch와 출력 차이가 1e-5 이하다. Mac `.venv`에 onnx와 onnxruntime을 설치했다.
  2. 보정 데이터: 학습 unit의 실제 프레임 256개. features_buffer는 그 체크포인트 자신의 hidden으로 채운다.
  3. Pulsar2 빌드(AX620E, NPU1, U16, MinMax, 출시 모델과 같은 설정). 빌드는 59 s 걸렸고 max_cycle은 3.25M이다(base 실측 6.0 ms 수준).
  - 결과: `outputs/quant_flip8/m.axmodel`, `configs/quant_flip8.json`, `logs/quant_flip8.log`
- `pulsar2 run` 모드 비교(보정 샘플 16개):
  - QuantAxModel `--mode Reference`는 컴파일된 m.axmodel 시뮬레이션과 출력이 사실상 같다(fp32 대비 상대 오차 0.0702 / 0.0702). 속도는 프레임당 약 0.1 s다.
  - 컴파일된 m.axmodel 시뮬레이션은 Rosetta에서 프레임당 약 20 s 걸린다.
  - `--mode NPUBackend`는 엉뚱한 값을 낸다(입력 처리 문제로 보인다). 쓰지 않는다.
- **오차 원인: 가중치 S8.** PyTorch에서 NPU 층 가중치만 채널별 대칭 S8(max|w|/127)로 반올림하면 Pulsar2 출력이 재현된다(fp32 대비 9.6%인 차이가 1.2%로 준다).
  - 대부분 backbone의 fused RepVGG conv에서 나온다. RepVGG를 합친 가중치가 INT8 PTQ에 약하다는 알려진 문제다.
  - U16 activation의 영향은 작다.
  - `student.fake_quant_s8()`, `quant_sim()`, `eval_student.py --s8`로 만들었다.

**결과 1: 평가 경로 전체, PyTorch S8 시뮬레이션 (`logs/eval_mac_base_flip8_s8.txt`)**

| 모델 | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
|---|---|---|---|---|---|---|---|
| flip8 fp32 | 0.0522 | 0.251 | 12.70 | 10.29 | 11.81 | 6.69 | 0.133 |
| **flip8 S8 (PTQ)** | 0.0555 | 0.259 | 13.48 | 11.74 | 13.58 | 7.26 | 0.146 |
| board NPU (출시) | 0.0691 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |

**결과 2: 평가 unit 하나(`2026-10-25--22-42-26-175__015`, 1497프레임), Pulsar2와 시뮬레이션 비교 (`scripts/quant_eval_p2.py`, `logs/quant_eval_p2_flip8.txt`)**
- 이 unit은 closed loop를 두 번 돌려서 쟀다. hidden은 영상에만 의존하므로 1차에서 hidden을 얻고, 2차에서 그 hidden으로 policy를 돌렸다.

| 모델 | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
|---|---|---|---|---|---|---|---|
| fp32 | 0.0735 | 0.317 | 21.29 | 14.09 | 14.98 | 5.62 | 0.175 |
| S8 시뮬레이션 | 0.0882 | 0.320 | 22.96 | 17.20 | 16.30 | 4.76 | 0.164 |
| Pulsar2 (Reference) | 0.0887 | 0.320 | 23.47 | 17.95 | 16.48 | 4.69 | 0.164 |

- 시뮬레이션은 Pulsar2가 만든 성능 저하의 약 80%를 재현한다(lat_acc는 거의 전부, lanes와 plan은 76~80%). 그러니 `--s8` 수치는 실제보다 조금 낙관적이다.
- 결론: PTQ만 해도 출시 모델보다 lat_acc, lanes, width, gas2는 여전히 좋다. 다만 fp32에서 얻은 이득의 일부를 잃는다(plan +0.8 cm, lanes +1.5 cm, lead +0.6 m).

**QAT 준비 (19:17에 시작했다, 00000절)**
- `train.py --qat`: NPU 층을 S8로 반올림한 가중치로 학습한다(straight-through gradient).
  - RepConv는 forward마다 가지를 합쳐(BN은 running 통계로 고정) 합친 가중치를 반올림한다. 학습되는 변수는 RepVGG 가지와 BN affine 그대로다.
  - 처음에는 fused 가중치를 직접 학습했는데, 첫 conv(in_scale 1/64가 접혀 있음)가 30 step 만에 58% 변했다. 실질 학습률이 층마다 크게 달라져서 이 방식으로 바꿨다.
  - 확인: QAT forward와 `--s8` 평가 경로의 출력이 정확히 같다(차이 0). 가지, BN gamma, Linear에 gradient가 흐른다.
  - tinydata2로 smoke test를 했다: 140 step에 합친 가중치 변화는 중앙값 0.6%, 최대 2.4%였다. export도 된다.
- 체크포인트는 float(unfused) 가중치를 담고 `"qat": True`가 붙는다. 평가는 `--s8`로 한다.
- `snap_epochs.sh`의 세 번째 인자 `eval-s8`은 epoch마다 `--s8`로 평가한다.
- 제안 명령(약 3시간: 처음 bank 23분과 epoch당 53분 x 3):
  `nohup caffeinate -i sh -c "(sleep 60; scripts/snap_epochs.sh ckpt/mac_base_qat3 3 eval-s8 > logs/snap_mac_base_qat3.log 2>&1) & .venv/bin/python scripts/train.py --out ckpt/mac_base_qat3 --epochs 3 --batch 64 --workers 4 --compile-refresh --aug --wd 0.01 --flip 0.3 --curve-sampling --qat --lr 2e-4 --init ckpt/mac_base_flip8/last.pt && .venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_base_flip8/last.pt ckpt/mac_base_qat3/last.pt > logs/eval_mac_base_qat3.txt 2>&1; echo DONE > logs/DONE_mac_base_qat3" > logs/mac_base_qat3.log 2>&1 < /dev/null &`
- QAT가 끝나면:
  1. `quant_build.py --ckpt ckpt/mac_base_qat3/last.pt --name qat3`
  2. `quant_eval_p2.py`로 Pulsar2 결과를 확인한다.
  3. 보드에서 latency와 출력을 잰다(MaixCAM2 비밀번호는 사용자에게 받는다).

### 000. 끝난 것 (2026-10-29 04:39 ~ 12:21): 좌우 반전과 커브 oversampling 8 epoch (`ckpt/mac_base_flip8`)
사용자 지시: "현재 데이터와 현재 시스템의 디스크 공간을 고려하여 더 개선이 가능할지 검토하고 추가 학습 진행하자. 데이터가 더 필요하면 다운받아".

**결과: 지금까지 가장 좋은 student다 (`ckpt/mac_base_flip8/last.pt` = ep8).** aug10보다 lead_x와 gas2는 같고 나머지 지표는 모두 좋다.
- plan_y2s는 13.65에서 12.70으로 7% 줄었다. lanes는 11.40에서 10.29로 10% 줄었다.
- lat_acc는 0.052로 small fp32(0.0515) 수준이 됐다.
- 출시 모델(board NPU)보다 lat_acc, lanes, width, gas2는 좋다. plan_y2s(12.70 vs 9.32)와 lead_x(6.69 vs 5.94)는 아직 나쁘다.
- lat_acc는 lr이 높은 ep1~5 동안 0.055~0.106으로 흔들렸지만, lr이 내려간 ep7~8에 안정됐다.
- 같은 데이터로 얻는 개선은 줄어들고 있다. plan은 ep5~8 동안 12.5~12.9에서 멈췄고 lanes도 ep7~8에서 거의 그대로다.
  남은 plan 차이는 00.1의 진단대로 일반화 문제이고, 데이터를 늘리는 것이 다음 수단이다.

| 모델 (평가 경로 4개, v > 5) | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
|---|---|---|---|---|---|---|---|
| flip8 ep1 | 0.0856 | 0.266 | 13.05 | 11.81 | 13.23 | 6.86 | 0.133 |
| flip8 ep2 | 0.1061 | 0.288 | 13.67 | 11.22 | 12.62 | 6.81 | 0.131 |
| flip8 ep3 | 0.0641 | 0.252 | 12.65 | 10.97 | 13.09 | 6.54 | 0.126 |
| flip8 ep4 | 0.0545 | 0.305 | 13.26 | 10.79 | 12.28 | 6.59 | 0.129 |
| flip8 ep5 | 0.0695 | 0.265 | 12.49 | 10.66 | 12.16 | 6.73 | 0.131 |
| flip8 ep6 | 0.0579 | 0.257 | 12.88 | 10.45 | 11.88 | 6.75 | 0.131 |
| flip8 ep7 | 0.0543 | 0.256 | 12.55 | 10.28 | 11.78 | 6.64 | 0.133 |
| **flip8 ep8 (최종)** | **0.0522** | **0.251** | **12.70** | **10.29** | **11.81** | 6.69 | 0.133 |
| aug10 최종 | 0.0547 | 0.260 | 13.65 | 11.40 | 12.72 | 6.67 | 0.131 |
| board NPU (출시) | 0.0691 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |
| small fp32 | 0.0515 | 0.206 | 9.62 | 16.35 | 23.03 | 5.43 | 0.090 |

- 걸린 시간: epoch당 약 53분(학습 약 30분과 refresh 약 23.5분)이다. 학습은 11:46에 끝났고, 최종 평가는 12:21에 끝났다.
- 아래는 실행 당시 기록이다.

**데이터를 받지 않은 이유 (Mac 여유 디스크 41GB, 16GB RAM)**
- comma2k19(33h)는 짝수 프레임만 저장해도 약 470GB다(프레임당 393KB). 여유 공간에는 약 2시간 분량만 들어간다.
- 카메라가 EON이고 wide 카메라가 없다. 미국 고속도로 위주라 약점인 국내 시내 커브와도 거리가 멀다.
- teacher ONNX, LD 모델, onnxruntime이 Mac에 없다(라벨링은 PC 전용이다).
- 남은 edgepilot 21청크(+10%)는 PC에 있다.
- 그래서 지금 데이터를 더 다양하게 쓰는 쪽을 택했다. 00.1의 진단상 문제는 일반화다(학습 경로 7.2cm, 평가 경로 13.7cm).

**바꾼 것 (train.py; 기본값이면 예전과 같다. 이전 파일은 scratchpad `train.py.aug10`)**
- `--flip p`: 샘플의 p만큼 좌우 반전한다.
  - 영상: YUV6의 Y 위상 채널을 [2,3,0,1,4,5]로 바꾸고 열을 뒤집는다. 전체 해상도 Y를 좌우로 뒤집은 것과 정확히 같다(확인함).
  - teacher 출력은 `mirror_big()`의 index와 부호로 바꾼다.
    - plan의 y, vy, ay, roll, yaw, droll, dyaw와 lead의 y, pose와 road_transform의 1, 3, 5, wide_from_device의 0, 2는 부호를 바꾼다.
    - lanes, road_edges, lane_prob는 좌우 순서를 뒤집는다.
    - meta의 깜빡이는 33::4와 34::4를 바꾸고, desire는 L/R을 바꾼다.
  - LD 배열, desire 입력, history도 반전한다. history는 반전 프레임으로 따로 계산한 두 번째 bank(`Bank.hf`)에서 가져온다.
  - traffic_convention은 [0, 1]로 준다. 반전된 우측통행 도로는 좌측통행 도로이기 때문이다. 평가와 배포는 그대로 [1, 0]이다.
  - 검증: 반전이 자기 역함수인지 확인했다. aug10에 반전 영상을 넣은 출력과 원래 출력에 mirror_big을 적용한 값을 비교했더니(맞은 매핑 / 매핑 없이):
    - lanes y는 0.59 / 2.15 m
    - road_edges는 0.99 / 2.75 m
    - plan y 성분은 0.048 / 0.147
    - 매핑이 맞다는 뜻이다. lead y는 aug10이 반전에 반응하지 않아(prior처럼 -0.6을 낸다) 판단 근거가 되지 않는다.
- `--curve-sampling`: 샘플 가중치를 1 + [teacher |y2s| > 0.3 m] + [> 1 m] + 2[desire]로 주고, 정지(v < 1 m/s)는 절반으로 준다.
  - 커브와 desire 샘플은 29%인데 뽑히는 비율은 54%가 된다. 정지 샘플은 15%에서 6%로 준다.
  - 학습 샘플의 15%가 정지 상태였다(평가는 v > 5만 본다).
- bank refresh가 두 번(원래와 반전)이라 epoch당 약 22분이다. epoch는 학습 약 32분에 refresh 22분으로 약 54분이다.

**명령** (Mac `~/Documents/supercombo_distill`):
`nohup caffeinate -i sh -c "(sleep 60; scripts/snap_epochs.sh ckpt/mac_base_flip8 8 eval > logs/snap_mac_base_flip8.log 2>&1) & .venv/bin/python scripts/train.py --out ckpt/mac_base_flip8 --epochs 8 --batch 64 --workers 4 --compile-refresh --aug --wd 0.01 --flip 0.3 --curve-sampling --lr 1e-3 --init ckpt/mac_base_aug10/last.pt && .venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_base_aug10/last.pt ckpt/mac_base_flip8/last.pt > logs/eval_mac_base_flip8.txt 2>&1; echo DONE > logs/DONE_mac_base_flip8" > logs/mac_base_flip8.log 2>&1 < /dev/null &`
- aug10 최종(누적 16 epoch)에서 새 OneCycle(최대 lr 1e-3)로 8 epoch 학습한다. 예상 종료는 **13시 전후**다.
- 실측:
  - 처음 bank(원래와 반전) 계산은 1412 s였다.
  - 학습 속도는 202 samples/s로 예전과 같다. epoch당 학습 약 31분에 refresh 약 23.5분이라 **12:15쯤 학습이 끝나고 12:30쯤 최종 평가가 나온다**.
  - 메모리 여유 35%, 스왑 1.6GB다.
  - 시작 loss는 plan 0.80, lanes 0.72다(aug10 끝은 0.44, 0.42). 커브 위주로 뽑고 반전 장면이 새로워서 높은 것으로, 예상했던 일이다.
- 판단 기준은 aug10 최종(plan_y2s 13.65, lanes 11.40, lat_acc 0.055)이다. ep1의 평가가 나빠도 lr이 높은 구간이니 ep4 이후를 볼 것.
- 확인할 파일:
  - `logs/mac_base_flip8.log`
  - `logs/snap_mac_base_flip8.log`
  - `logs/eval_mac_base_flip8_snaps.txt`
  - `logs/eval_mac_base_flip8.txt`(aug10과 한 표)
  - `logs/DONE_mac_base_flip8`

### 00. 끝난 것 (2026-10-28 19:58 ~ 10-29 03:07): base 개선판 10 epoch (`ckpt/mac_base_aug10`, 지금은 flip8이 더 좋다)
사용자 지시: "base 모델을 개선해서 더 학습시키자".

**결과: 지금까지 가장 좋은 student다 (`ckpt/mac_base_aug10/last.pt` = ep10).** 평가 경로에서 base 최종보다 모든 지표가 좋다.
lanes, width, lead는 w48d와 같은 수준인데 latency는 6.0 ms다(w48d 25.6 ms). 출시 모델에는 lanes, width, lat_acc, gas2에서 이기고
plan_y2s(13.65 대 9.32)와 lead_x(6.67 대 5.94)에서 진다. plan_y2s는 ep6부터 13.5~14.2에 멈춰 있어서 학습을 더 돌려도 줄지 않는다.
최종 표: `logs/eval_mac_base_aug10.txt`.

**진단 (학습 전에 한 것)**
1. 학습 경로 3개에서 평가했다 (`logs/eval_trainroutes_base_w48d.txt`; 경로는 2026-10-04--12-32-26-070__000, 2026-10-10--11-16-58-871__000, 2026-10-10--17-34-10-953__000).
   **base는 학습 경로에서는 출시 모델보다 모든 지표가 좋다. 평가 경로에서만 나쁘다. 즉 일반화(과적합) 문제다.**

   | model (학습 경로) | lat_acc | plan_y2s_cm | lanes_vs_ld | lead_x_m |
   |---|---|---|---|---|
   | 출시 모델 | 0.045 | 9.24 | 16.25 | 2.82 |
   | base 최종 | **0.038** | **8.29** | 9.95 | 1.63 |
   | w48d 최종 | 0.096 | 15.23 | **8.87** | **1.40** |

   - base의 plan_y2s는 학습 경로 8.3cm, 평가 경로 14.1cm다. 출시 모델은 두 경로 모두 9.3cm 안팎이다.
   - w48d는 학습 경로에서도 plan이 15.2cm다. plan을 학습 데이터에서조차 맞추지 못한 것(underfit)이고, base와는 다른 문제다.
2. 평가 경로 하나(2026-10-05--00-07-29-810)에서 plan 오차를 성분별로 봤다. 단위는 teacher σ이고 스크립트는 `scripts/diag_plan.py <ckpt> <unit dir> <frames>`다.
   - 큰 것은 종방향이다. x 4.1σ(2초 지점 4.7σ), vx 3.4σ(t=0에서 6.2σ), 즉 현재 속도 추정 오차다.
   - 횡방향 y는 1.1σ다.
   - z와 roll의 t=0은 3%가 상한 20σ에 걸린다.
3. **loss 버그**: `mdn_loss`의 `clamp(max=20)` 때문에, 20σ를 넘는 원소는 gradient가 0이었다.
   - road_transform의 첫 두 평균값은 항상 20σ를 넘어서 한 번도 학습되지 않았다. base의 road_transform loss가 7.0에 머문 이유가 이것이다 ((20+20+0.4+1.2+0.6+1.1)/6).

**바꾼 것** (train.py, 기본값은 예전과 같고 loss 상한만 모든 run에 적용된다)
- `mdn_loss`: 20σ를 넘는 오차는 `20 + 20·ln(err/20)`으로 계산한다. 로그로 커지므로 gradient가 20/err로 남는다.
  **이제 road_transform loss는 예전 run과 숫자 크기가 다르다** (첫 배치 약 19.8).
- `--aug`: GPU에서 사진 성질만 바꾼다. 샘플의 80%에 적용하고, 한 샘플의 두 프레임에는 같은 값, 두 카메라에는 따로 뽑은 값을 쓴다.
  - Y는 gain 0.7~1.3에 offset ±25를 준다.
  - UV는 채도 0.7~1.3에 shift ±8을 준다.
  - 픽셀 노이즈는 σ 0~4다.
  - 기하 변형은 하지 않는다 (target이 보정된 시점에 묶여 있다).
  - 배치당 약 9ms (MPS, 64×12×128×256 두 번)다.
- `--wd`: AdamW weight decay 옵션. 기본 1e-4, 이번 run은 0.01이다.
- `scripts/snap_epochs.sh`: 세 번째 인자로 `eval`을 주면, 스냅샷마다 CPU에서 평가를 백그라운드로 돌린다.
  - 결과는 `logs/eval_<run>_snaps.txt`와 run의 TensorBoard에 남는다.
- 시험: 작은 유닛 하나로 `--init --aug --wd 0.01 --compile-refresh` 전체 경로를 돌려 통과했다.
  - 한 배치에서 road_transform CPU head의 첫 두 행에 gradient가 생기는 것을 확인했다.

**명령**
`nohup caffeinate -i sh -c "(sleep 60; scripts/snap_epochs.sh ckpt/mac_base_aug10 10 eval > logs/snap_mac_base_aug10.log 2>&1) & .venv/bin/python scripts/train.py --out ckpt/mac_base_aug10 --epochs 10 --batch 64 --workers 4 --compile-refresh --aug --wd 0.01 --init ckpt/mac_base_5ep/last.pt && .venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_base_5ep/last.pt ckpt/mac_base_aug10/last.pt > logs/eval_mac_base_aug10.txt 2>&1; echo DONE > logs/DONE_mac_base_aug10" > logs/mac_base_aug10.log 2>&1 < /dev/null &`
- base 최종(누적 6 epoch)에서 새 OneCycle로 10 epoch 이어 학습한다. 끝나면 누적 16 epoch이다.
- 예상 시간: epoch당 약 40분이고 처음 bank 계산이 약 11분이라, 03시 전후에 끝난다.
- 실측:
  - 처음 bank 계산 668 s.
  - step 400에서 196 samples/s. 예전 base run의 같은 시점은 206이었으니 augmentation 때문에 약 5% 느리다.
  - epoch당 학습 약 31분에 refresh 약 11분이라 **03:10쯤 끝난다**.
  - road_transform loss는 13.8에서 11.6으로 내려가고 있다. 막혀 있던 원소가 학습되고 있다는 뜻이다.
  - 메모리 여유 32%, 스왑 1.7GB.
- epoch별 평가 (epoch 하나에 약 41분, 평가는 스냅샷마다 CPU로 약 5.5분, 마지막 epoch는 03:07에 끝났다):

  | | lat_acc | long_acc | plan_y2s | lanes | width | lead_x | gas2 |
  |---|---|---|---|---|---|---|---|
  | ep1 | 0.238 | 0.357 | 24.10 | 16.14 | 17.58 | 9.19 | 0.128 |
  | ep2 | 0.134 | 0.330 | 25.71 | 14.36 | 15.84 | 7.44 | 0.137 |
  | ep3 | 0.076 | 0.324 | 16.12 | 13.23 | 14.61 | 7.11 | 0.132 |
  | ep4 | 0.148 | 0.312 | 15.14 | 12.55 | 14.29 | 7.14 | 0.127 |
  | ep5 | 0.063 | 0.299 | 15.08 | 12.41 | 13.91 | 7.05 | 0.130 |
  | ep6 | 0.059 | 0.276 | 14.17 | 11.76 | 13.16 | 6.90 | 0.127 |
  | ep7 | 0.060 | 0.265 | 14.09 | 11.56 | 13.00 | 6.94 | 0.125 |
  | ep8 | 0.060 | 0.273 | **13.55** | 11.51 | 12.93 | 6.59 | 0.130 |
  | ep9 | 0.057 | 0.266 | 13.61 | 11.47 | 12.83 | **6.59** | 0.131 |
  | **ep10 (최종)** | **0.055** | **0.260** | 13.65 | **11.40** | **12.72** | 6.67 | 0.131 |
  | (base 최종) | 0.063 | 0.278 | 14.12 | 13.49 | 14.55 | 7.45 | 0.133 |
  | (w48d 최종) | 0.096 | 0.319 | 16.80 | 11.32 | 12.66 | 6.64 | 0.126 |
  | (출시 모델) | 0.069 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |

  - ep3에서 lr은 아직 1.7e-3이다. 그런데도 lanes(13.23)와 lead(7.11)는 이미 base 최종(13.49, 7.45)보다 좋다.
  - ep9에서 모든 지표가 base 최종보다 좋다. lanes, width, lead는 w48d 최종과 비슷하다(latency는 6.0 ms 대 25.6 ms).
  - plan_y2s는 ep6부터 14.2 → 14.1 → 13.5 → 13.6으로 거의 멈췄다. 출시 모델(9.3)과의 차이는 lr 감쇠만으로는 줄지 않는다.
  - ep4의 lat_acc 0.148은 튀는 값이다(ep3 0.076, ep5 0.063).

  - lr이 거의 최대(1.8e-3)인 구간이라 base 최종(14.1)보다 나쁜 것은 예상했던 일이다.
    - 예전 5 epoch run도 ep1은 37.0, ep2는 17.2였다. 다만 그때는 5 epoch에 맞춘 감쇠라 lr이 더 빨리 내려갔다.
  - 판단은 lr이 내려가는 ep6~ep10을 보고 한다.
  - 학습 loss: road_transform은 7.0에서 0.54로 내려갔다(상한 수정 효과). plan은 0.70이다(augmentation이 걸린 값).
- 확인할 파일: `logs/mac_base_aug10.log`, `logs/snap_mac_base_aug10.log`, `logs/eval_mac_base_aug10_snaps.txt`(epoch별 평가), `logs/eval_mac_base_aug10.txt`(최종, base 최종과 한 표), `logs/DONE_mac_base_aug10`.

### 00.1 진단: aug10의 plan_y2s가 출시 모델보다 나쁜 이유 (2026-10-29 03:20 ~ 04:00)
사용자 지시: plan 오차가 어디서 나오는지 볼 것.

**결론: 용량 문제가 아니라 일반화 문제이고, 차이는 전부 커브·회전·차선 변경 구간에서 난다.**
직선 구간에서는 이미 small보다 좋다. 처방은 데이터 다양성이다(학습 경로 18개, 차 1대).

스크립트 (모두 CPU, 평가 경로 7청크에 모델당 약 5분):
- `scripts/diag_y2s_dump.py`: 프레임별 plan을 `outputs/diag_y2s/dump/<이름>/`에 저장한다(806MB). aug10, base5, aug10_nohist(history 0)를 저장해 두었다.
- `scripts/diag_y2s.py`: 그 dump를 구간별로 나눈다. 출력은 `logs/diag_y2s_aug10.txt`.
- `scripts/diag_motion.py`: 프레임 쌍의 간격을 0 / 0.2 / 0.4 s로 바꿔서 움직임을 읽는지 본다.
- `scripts/diag_plan.py`: 성분별 σ 정규화 오차.
- 비교 기준은 small fp32다(9.62, 출시 모델 9.32와 거의 같다). 출시 모델 열은 10-05--00-07 청크의 기록 plan에 NaN이 있어서 전체 평균이 nan으로 나온다.

1. **학습 경로에서는 aug10이 가장 좋다.** plan_y2s는 aug10 7.17, base5 8.29, 출시 모델 9.24, small 9.63이다.
   평가 경로에서는 aug10 13.65, base5 14.12, small 9.62이다. 학습 경로와 평가 경로의 차이는 augmentation으로 조금만 줄었다(8.3/14.1 → 7.2/13.7).
2. **차이(13.65 - 9.62 = 4.0cm)는 전부 teacher |y2s| > 0.3 m인 프레임에서 난다.** 평가 프레임의 33%다.

   | teacher \|y2s\| | 비율 | small | aug10 | 전체 차이에 기여 |
   |---|---|---|---|---|
   | < 0.1 m (직선) | 48% | 5.21 | **4.65** | -0.27 |
   | 0.1~0.3 m | 19% | 8.97 | 8.75 | -0.04 |
   | 0.3~1 m | 21% | 12.64 | 18.91 | +1.31 |
   | 1~3 m | 10% | 18.08 | 35.09 | +1.70 |
   | > 3 m | 2% | 50.42 | 121.3 | +1.35 |

   - desire가 있는 프레임(차선 변경 등, 6%)은 49.5 대 25.6이다.
   - 5~10 m/s 구간은 25.0 대 16.6이다.
   - 평가 청크 7개 중 6개에서 small보다 나쁘다. 09-27--02-44__000만 같다.
3. **오차는 느린 성분이다.** 2초 이동평균으로 나눈 오차가 slow 12.0 대 7.5, fast 4.9 대 5.1이다.
   출력 떨림은 오히려 small보다 적다. 커브에서 덜 꺾는다: teacher에 대한 기울기가 aug10 0.885, base5 0.849, small 1.02다.
4. **현재 움직임 추정이 새 경로에서 무너진다.** 단위는 vx0 cm/s, r0 mrad/s다.

   | | small 학습 | small 평가 | aug10 학습 | aug10 평가 |
   |---|---|---|---|---|
   | vx0 | 62 | 65 | 38 | 159 |
   | r0 | 1.65 | 1.60 | 3.20 | 4.16 |

   평가 경로에서 y2s 오차와 (r0 오차 × v × 2)의 상관은 0.47이다. 학습 경로에서는 0.03이다.
5. **diag_motion 결과 (aug10):**
   - 두 프레임을 같게 주면(움직임 0) |r0|는 14~19에서 6~7 mrad/s로 줄고 plan_y2s는 29~32cm가 된다. yaw rate는 움직임에서 읽는다.
   - 속도는 움직임 0이어도 15.3 대 17.7 m/s로 거의 그대로다. 간격을 두 배로 해도 17.7에서 18.0으로 그대로다.
     **속도는 흐름(flow)이 아니라 장면이나 history에서 추정한다.** 학습 경로에서는 맞고, 새 경로에서는 틀린다.
   - 평가 경로 10-05--16-28에서 teacher vx0 16.3에 대해 student는 17.7이다(+9%).
6. history를 끄면(features_buffer 0) plan_y2s는 24.5cm다. history 의존이 크다.
7. 처방 후보:
   - (a) **데이터 다양성을 늘린다.** 효과가 가장 크다.
     - 남은 21청크 라벨링은 +10%라 효과가 작다.
     - comma2k19(33h, EON 카메라)를 기기 시점으로 warp해서 teacher 라벨을 만들면 데이터가 약 3배가 된다. 파이프라인 작업이 필요하다.
   - (b) 커브와 desire 프레임 oversampling. 플래그 하나로 되고 Mac에서 바로 돌릴 수 있다.
     덜 꺾는 문제(기울기 0.885)에는 도움이 될 수 있다. 하지만 학습 경로에서는 커브도 이미 small보다 좋아서, 일반화 효과는 작을 것으로 본다.
   - (c) 좌우 반전 augmentation. 라벨의 y, yaw, 차선, desire 좌우를 바꾸고 반전 프레임용 bank가 따로 필요하다. 복잡하다.

### 00.2 차선 overlay (2026-10-29 04:20, `scripts/viz_lanes.py`, `outputs/lane_overlay/grid_*.png`)
- 평가 경로 4개에서 aug10을 closed loop로 돌려 frame에 그렸다. 투영은 medmodel 내부 파라미터(f 910, 256, 47.6)를 쓴다.
- YUV6의 Y 채널 순서는 [Y(짝수 행, 짝수 열), Y(홀수 행, 짝수 열), Y(짝수 행, 홀수 열), Y(홀수 행, 홀수 열)]이다. 이 순서일 때 영상이 매끄럽다.
- 프레임 평균 차선 오차(LD 대비, 6~40 m, student / 출시 모델, cm):
  - 10-05--16-28: 7.5 / 10.1
  - 10-05--00-07: 5.8 / 9.2
  - 09-27--02-44: 7.6 / 14.6
  - 10-25--22-42: 23.6 / 26.5
- 직선과 커브는 페인트에 잘 붙는다. 최악 장면은 모두 차선 변경 중이거나(선을 밟고 있을 때 ego 차선 번호가 바뀐다) 저속 회전이다. 출시 모델도 같은 장면에서 크게 틀린다.

### 0. 끝난 것 (2026-10-28 00:10 ~ 13:20): w48d 6 epoch 학습
결과 요약 (자세한 표는 이 섹션 끝):
- 인식 쪽은 base보다 좋다: lanes 11.3cm, width 12.7cm, lead 6.64m.
- 주행 쪽은 base보다 나쁘다: plan_y2s 16.8cm, lat_acc 0.096.
사용자 판단: "곡선을 보니 한계에 근접한 것 같다. 취소하고 바로 w48d를 학습하자".
- 바로 아래의 base 10 epoch 추가 학습은 bank 계산 도중(00:09)에 취소했다. 결과물이 없어서 `ckpt/mac_base_10ep`는 지웠다.
- 명령 (Mac `~/Documents/supercombo_distill`):
  `nohup caffeinate -i sh -c "(sleep 60; scripts/snap_epochs.sh ckpt/mac_w48d_6ep 6 > logs/snap_mac_w48d_6ep.log 2>&1) & .venv/bin/python scripts/train.py --out ckpt/mac_w48d_6ep --epochs 6 --batch 64 --workers 4 --arch '{\"widths\":[48,96,192,384,768],\"depths\":[1,3,6,9,3]}' && .venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_w48d_6ep/last.pt > logs/eval_mac_w48d_6ep.txt 2>&1; echo DONE > logs/DONE_mac_w48d_6ep" > logs/mac_w48d_6ep.log 2>&1 < /dev/null &`
- 설정:
  - 처음부터 학습한다. 구조가 달라서 base 가중치로 시작할 수 없다. 그래서 첫 epoch의 bank는 0이다.
  - 6 epoch은 base의 누적 epoch 수(1 + 5)에 맞춘 것이다.
  - batch 64와 lr 2e-3은 base와 같다.
  - `--compile-refresh`는 쓰지 않는다. w48d에서는 Metal 컴파일 오류가 난다.
- 실측 (step 200):
  - 63 samples/s. 가짜 데이터 기준은 71이었다.
  - GPU 사용률 99%, GPU 메모리 약 7.2GB, 시스템 여유 메모리 약 20%.
  - 스왑은 3GB 중 2.2GB를 쓰는데, 시작 전부터 쓰던 양이라 추세를 봐야 한다.
  - epoch당 학습 약 98분에 refresh 약 29분이 붙어 약 2시간 7분이다. **13:20쯤 eval까지 끝난다.**
- 중간 점검 (08:05, 4번째 epoch의 64%, 전체 step 22400/34854):
  - epoch당 실제로는 학습 97분에 refresh 34분(2015~2042 s)으로 2시간 10분이다. ep1~ep3 스냅샷은 01:47, 03:57, 06:07에 생겼다.
  - 속도는 64 samples/s로 그대로다. 메모리 여유 22%.
  - 스왑은 4GB 중 2.9GB로 늘었지만 속도에는 영향이 없다.
  - 남은 일정: ep4 08:19, ep5 10:30, ep6 12:41, 마지막 refresh 13:15, 그 뒤에 eval이 돈다.
  - 학습 loss를 누적 epoch 수가 같은 base와 비교했다(base는 1 epoch에서 이어 학습해서 로그의 epoch 번호가 하나 밀린다).
    - plan은 비슷하다. 3번째 epoch에서 base 0.917, w48d 0.895다.
    - road_transform은 base가 7.0에서 막혔지만 w48d는 3.75까지 내려갔다.
    - pose 1.04는 base의 4번째 epoch 1.07과 같은 수준이다.
- ep3 스냅샷 평가 (CPU, 08:07~08:29, 22분; `logs/eval_mac_w48d_6ep_ep3.txt`). 누적 3 epoch이라 base에서는 `mac_base_5ep/ep2`(1 + 2)와 맞춰 본다:

  | model | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld | width_vs_ld | lead_x_m | gas2 |
  |---|---|---|---|---|---|---|---|
  | board NPU (shipped) | 0.069 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |
  | base 누적 3 epoch (5ep/ep2) | 0.101 | 0.298 | 17.22 | 15.58 | 16.48 | 8.04 | 0.127 |
  | base 최종 (누적 6 epoch) | 0.063 | 0.278 | 14.12 | 13.49 | 14.55 | 7.45 | 0.133 |
  | **w48d ep3 (누적 3 epoch)** | 0.124 | 0.342 | 17.79 | **12.85** | 14.65 | **6.70** | 0.135 |

  - 같은 epoch의 base보다 lanes, width, lead는 확실히 좋고, plan, gas2는 비슷하고, lat_acc, long_acc는 나쁘다.
  - lanes와 lead는 벌써 base 최종보다 좋다.
  - lat_acc와 plan은 base에서도 후반 epoch, 즉 lr이 내려가는 구간에서 크게 줄었다(17.2 → 14.1). w48d도 남은 3 epoch을 봐야 한다.
- 완료: 13:20에 끝났다.
  - 모든 epoch이 학습 약 97분에 refresh 약 33분이었다(마지막 refresh는 1961 s).
  - 스냅샷 ep1~ep6이 모두 남았고, `last.pt`는 ep6과 같다.
  - 최종 평가 결과는 `logs/eval_mac_w48d_6ep.txt`에 있다(MPS로 약 9분).

  | model | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld | width_vs_ld | lead_x_m | gas2 |
  |---|---|---|---|---|---|---|---|
  | board NPU (shipped, 14 ms) | 0.069 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |
  | small fp32 | 0.052 | 0.206 | 9.62 | 16.35 | 23.03 | 5.43 | 0.090 |
  | base 최종 (누적 6 epoch, 6.0 ms) | **0.063** | **0.278** | **14.12** | 13.49 | 14.55 | 7.45 | 0.133 |
  | w48d ep3 | 0.124 | 0.342 | 17.79 | 12.85 | 14.65 | 6.70 | 0.135 |
  | **w48d 최종 (6 epoch, 25.6 ms)** | 0.096 | 0.319 | 16.80 | **11.32** | **12.66** | **6.64** | **0.126** |

  - w48d가 base보다 좋은 것: lanes는 16% 줄었고, width는 13% 줄었다. lead는 7.45m에서 6.64m로 줄었다. gas2는 조금 좋다.
  - w48d가 base보다 나쁜 것: plan_y2s(14.1 → 16.8), lat_acc(0.063 → 0.096), long_acc(0.278 → 0.319)다. lat_acc는 출시 모델(0.069)보다도 나쁘다.
  - plan 학습 loss를 누적 epoch별로 비교했다:

    | 누적 epoch | 1 | 2 | 3 | 4 | 5 | 6 |
    |---|---|---|---|---|---|---|
    | base | 3.02 | 1.60 | 0.917 | 0.754 | 0.659 | 0.612 |
    | w48d | 3.03 | 1.34 | 0.895 | 0.804 | 0.723 | 0.689 |

    - w48d는 3 epoch까지 앞서다가 그 뒤로 뒤처졌다.
    - 학습 loss 자체가 높으므로 eval에서만 생기는 문제는 아니다. plan이 덜 맞춰진 것이다.
    - 같은 구간에서 w48d는 lead, lanes, road_edges loss가 더 낮고, road_transform도 base는 7.0, w48d는 3.67이다. 공유 trunk가 인식 쪽에 더 쓰였을 가능성이 있다(추정).
    - base는 1 epoch 학습 뒤 schedule을 새로 시작해서 lr 감쇠를 두 번 거쳤다. w48d는 한 번이다.
- 확인할 파일:
  - `logs/mac_w48d_6ep.log`
  - 스냅샷 `ckpt/mac_w48d_6ep/ep1.pt`~`ep6.pt`, 감시 로그 `logs/snap_mac_w48d_6ep.log`
  - 평가 `logs/eval_mac_w48d_6ep.txt`, 완료 표시 `logs/DONE_mac_w48d_6ep`
  - TensorBoard `ckpt/mac_w48d_6ep/tb`
- 비교 대상: base 최종 `ckpt/mac_base_5ep/last.pt` (plan_y2s 14.1cm, lat_acc 0.063, lead 7.45m). 같은 표로 보려면:
  `.venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_base_5ep/last.pt ckpt/mac_w48d_6ep/last.pt`

### 0.1 취소: base 10 epoch 추가 학습 (2026-10-28 00:00 ~ 00:09)
사용자 지시는 "base를 10에포크 더 돌리자"였다. 1.0의 5 epoch 결과(`ckpt/mac_base_5ep/last.pt`)에서 이어서 학습한다. 끝나면 누적 16 epoch가 된다.
- 명령 (Mac `~/Documents/supercombo_distill`. ep3/ep4 eval 프로세스가 끝나기를 기다렸다가 시작하게 했다):
  `nohup caffeinate -i sh -c "…; (sleep 60; scripts/snap_epochs.sh ckpt/mac_base_10ep 10 > logs/snap_mac_base_10ep.log 2>&1) & .venv/bin/python scripts/train.py --out ckpt/mac_base_10ep --epochs 10 --batch 64 --workers 4 --compile-refresh --init ckpt/mac_base_5ep/last.pt && .venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_base_10ep/last.pt > logs/eval_mac_base_10ep.txt 2>&1; echo DONE > logs/DONE_mac_base_10ep" > logs/mac_base_10ep.log 2>&1 < /dev/null &`
- 새 OneCycle 스케줄이다 (lr 2e-3, warm-up 5%). 그래서 처음 1~2 epoch은 지표가 일시적으로 나빠질 수 있다 (1.0의 ep1 참고).
- 예상: epoch당 약 50분 (학습 39분 + refresh 11분). **08:40쯤 eval까지 끝난다.** 충전기를 연결한 상태에서 시작했다 (배터리 34%, 충전 중).
- 확인할 파일:
  - `logs/mac_base_10ep.log`
  - 스냅샷 `ckpt/mac_base_10ep/ep1.pt`~`ep10.pt`, 감시 로그 `logs/snap_mac_base_10ep.log`
  - 평가 `logs/eval_mac_base_10ep.txt`, 완료 표시 `logs/DONE_mac_base_10ep`
  - TensorBoard `ckpt/mac_base_10ep/tb` (서버는 6006 포트에서 돌고 있다)
- 판단 기준: 아래 5 epoch 곡선은 끝으로 갈수록 평평해졌다. 이번에도 plan_y2s가 12cm 근처에서 멈추면 모델 용량 문제로 보고 w48d를 검토한다.

### 1.0 끝난 것: base 5 epoch (2026-10-27 19:43 ~ 23:47)
**Mac에서 base 5 epoch 학습.** 1 epoch 체크포인트에서 이어서 학습한다. 사용자 지시: "PC쪽은 무시하고 1 epoch에 이어서 학습".
- 명령 (Mac `~/Documents/supercombo_distill`, nohup과 caffeinate로 떼어서 실행):
  `nohup caffeinate -i sh -c ".venv/bin/python scripts/train.py --out ckpt/mac_base_5ep --epochs 5 --batch 64 --workers 4 --compile-refresh --init ckpt/scale/s05_base/last.pt && .venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_base_5ep/last.pt > logs/eval_mac_base_5ep.txt 2>&1; echo DONE > logs/DONE_mac_base_5ep" > logs/mac_base_5ep.log 2>&1 < /dev/null &`
- 새 OneCycle 스케줄(lr 2e-3)을 쓴다. 학습 전에 1 epoch 가중치로 bank를 먼저 계산하고, 마지막에 eval이 자동으로 돈다.
- 실측: 처음 bank 계산 659초, 학습 198 samples/s.
  - 가짜 데이터 벤치마크(261 samples/s)보다 느리다. 실제 데이터는 디스크 읽기(약 205MB/s)가 병목이다. bank refresh도 디스크가 병목이라 compile 효과가 거의 없다.
  - epoch당 학습 약 31분에 refresh 약 11분이 붙는다. **23:35쯤 eval까지 끝날 것**으로 본다.
  - step 200의 loss는 plan 2.66, lanes 0.80, lead 0.91이다. 1 epoch 끝 무렵과 비슷하고 튀지 않았다. warm-up(1,452 step)으로 lr이 2e-3까지 오르는 동안은 잠깐 올라갈 수 있다.
- 확인할 파일: `logs/mac_base_5ep.log`, `ckpt/mac_base_5ep/train.log`, 평가 결과 `logs/eval_mac_base_5ep.txt`, 완료 표시 `logs/DONE_mac_base_5ep`. TensorBoard는 `ckpt/mac_base_5ep/tb`.
- `last.pt`는 매 epoch 끝에 덮어쓴다. 그래서 `scripts/snap_epochs.sh`가 감시하다가 `ckpt/mac_base_5ep/ep1.pt` … `ep5.pt`로 복사해 둔다.
  - 감시 스크립트는 20:06에 시작했고, 로그는 `logs/snap_mac_base_5ep.log`다.
  - N은 이번 run 안에서 센 번호다. ep1이 누적 2 epoch(s05_base 1 epoch + 1)이다.
  - 끝나면 epoch별 곡선을 본다: `.venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_base_5ep/ep{1,2,3,4}.pt`
  - OneCycle이라 중간 epoch은 lr이 높은 상태다. 그래서 그 시점에서 멈춘 모델보다 조금 나쁘게 나온다.
    eval의 TensorBoard step은 체크포인트의 epoch+1이다. 그래서 ep1~ep4는 step 1~4, last.pt(=ep5)는 step 5로 들어가 한 곡선이 된다.
- 학습 도중 스냅샷 평가 (21:38~21:51, CPU, `OMP_NUM_THREADS=4 … --device cpu`. 학습 속도 영향 거의 없음. 결과 `logs/eval_mac_base_5ep_ep12.txt`):

  | model | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
  |---|---|---|---|---|---|---|---|
  | board NPU (출시) | 0.069 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |
  | small fp32 | 0.052 | 0.206 | 9.62 | 16.35 | 23.03 | 5.43 | 0.090 |
  | s05_base (누적 1 epoch) | 0.109 | 0.363 | 26.95 | 24.87 | 21.21 | 9.36 | 0.142 |
  | ep1 (누적 2 epoch, lr 최고점) | 0.254 | 0.282 | 36.96 | 18.43 | 19.81 | 8.63 | 0.163 |
  | ep2 (누적 3 epoch) | 0.101 | 0.298 | 17.22 | 15.58 | 16.48 | 8.04 | 0.127 |
  | ep3 (누적 4 epoch) | 0.071 | 0.319 | 15.35 | 14.61 | 15.59 | 8.04 | 0.130 |
  | ep4 (누적 5 epoch) | 0.064 | 0.279 | 14.26 | 13.63 | 14.77 | 7.43 | 0.134 |
  | **ep5 = last.pt (누적 6 epoch, 최종)** | **0.063** | 0.278 | 14.12 | **13.49** | **14.55** | 7.45 | **0.133** |

  - **최종 (2026-10-27 23:47 완료, `logs/eval_mac_base_5ep.txt`):**
    - 출시 모델보다 나은 것: 횡가속(0.063 vs 0.069), 차선 위치, 차선 폭, gas2.
    - 아직 뒤지는 것: plan 2초 횡위치 1.52배(14.1 vs 9.3cm), lead 거리 1.25배(7.45 vs 5.94m).
    - 마지막 step의 학습 loss(epoch 평균)는 plan 0.61, lead 0.54, lanes 0.49로 아직 내려가는 중이었다. 과적합은 보이지 않는다.
  - 23:01~23:31 Mac이 배터리 1%로 절전과 hibernate에 들어갔다. 학습은 그 직전(23:01:11)에 끝났고, 마지막 refresh만 늘어났다(2444초).
    `caffeinate -i`는 배터리 부족으로 잠드는 것을 막지 못한다. **긴 학습은 충전기를 연결하고 돌릴 것.**

  - ep2에서 출시 모델보다 나은 것: 차선 위치, 차선 폭, gas2. 차선 위치는 teacher(17.4)보다도 낫다. 차선은 LD를 직접 보고 학습하기 때문이다.
  - 아직 뒤지는 것: plan 횡위치 1.85배, 횡가속 1.47배, lead 거리 1.35배.
  - ep1은 lr이 최고점(1.9e-3) 근처라 1 epoch 결과보다 오히려 나빴다.
  - `eval_student.py`를 고쳤다 (2026-10-27 20:10). 한 폴더의 체크포인트를 여러 개 넣으면 표 이름이 `student <run>`으로 겹쳐서 하나만 남았다.
    이제 last.pt가 아닌 것은 `student <run>/ep1`처럼 파일 이름을 붙인다.
- 멈출 때는 바깥 `sh -c`를 먼저 죽인다. 그래야 eval이 이어서 돌지 않는다 (10절).

### 1.1 끝난 것: base 1 epoch
**Mac에서 base 학습 1 epoch (짧은 비교 실행).** 18:17에 시작했다.
- 구성: 학습 경로 58개(Mac에 있는 65개 청크 중 평가 경로 7개 청크를 뺀 것), 샘플 371,837개.
- 학습 step은 18:52에 끝났다 (5,809 step, 185 samples/s, 오류 없음).
- 이어서 bank refresh(약 10분)와 `eval_student.py`(약 10분)가 돈다. **19:10쯤 평가 결과**가 나온다.
- 1 epoch loss (처음 200 step → 마지막 400 step 평균):

  | head | 처음 → 마지막 |
  |---|---|
  | lanes | 5.68 → 0.80 |
  | road_edges | 5.47 → 0.92 |
  | lead | 5.86 → 0.93 |
  | lane_prob | 0.59 → 0.23 |
  | meta | 0.48 → 0.24 |
  | **plan** | **4.12 → 2.68 (1.5배로 가장 더딤)** |
  | pose | 6.67 → 3.57 |
  | road_transform | 13.4 → 7.10 |

  - plan과 ego-motion head는 시간 정보가 필요하다. 1 epoch째는 bank(과거 hidden)가 0이라, 2 epoch부터 제대로 배운다.
  - 판단은 eval 지표로 한다 (출시 모델 board NPU, small fp32와 비교).
- **1 epoch eval 결과** (평가 경로 4개, `logs/scale_eval_s05_base.txt`, TensorBoard `eval/*`):

  | model | lat_acc | long_acc | plan_y2s_cm | lanes_vs_ld_cm | width_vs_ld_cm | lead_x_m | gas2 |
  |---|---|---|---|---|---|---|---|
  | board NPU (출시) | 0.069 | – | 9.32 | 18.49 | 26.91 | 5.94 | 0.246 |
  | small fp32 | 0.052 | 0.206 | 9.62 | 16.35 | 23.03 | 5.43 | 0.090 |
  | big teacher | 0 | 0 | 0 | 17.36 | 23.93 | 0 | 0 |
  | **base 1 epoch** | 0.109 | 0.363 | 26.95 | 24.87 | **21.21** | 9.36 | **0.142** |

  - 출시 모델보다 이미 나은 것: 차선 폭(width_vs_ld), gas2.
  - 아직 뒤지는 것: plan(횡위치 2.9배, 횡가속 1.6배), lead, 차선 위치.
  - 이유 추정: epoch 0은 features_buffer가 전부 0인 상태로 학습했다. 그런데 eval은 자기 hidden을 넣는 closed loop라 입력 분포가 다르다.
    → 여러 epoch 학습(bank 사용)으로 확인해야 한다.
- Mac 속도 비교 sweep은 19:17에 끝났다. 결과와 결론은 12절, 원본은 Mac `logs/bench_sweep.txt`에 있다.
- 19:40까지 Mac에서는 아무 작업도 돌지 않았다. 지금은 1.0의 5 epoch 학습이 돈다. TensorBoard 서버는 꺼져 있다. 지시 없이 학습이나 서버를 새로 시작하지 말 것.
- 실행 명령 (Mac, 당시 `~/supercombo_distill`, 지금은 삭제된 py3.9 `.venv`):
  `caffeinate -i sh -c ".venv/bin/python scripts/train.py --out ckpt/scale/s05_base --epochs 1 --batch 64 --workers 4 --arch {} ; .venv/bin/python scripts/eval_student.py --ckpt ckpt/scale/s05_base/last.pt > logs/scale_eval_s05_base.txt; echo MACDONE > logs/MACDONE"`
- 확인할 파일:
  - `ckpt/scale/s05_base/train.log`
  - `logs/scale_s05_base.log`
  - `logs/scale_eval_s05_base.txt`
  - `logs/MACDONE` (끝나면 생김)
- **TensorBoard**: http://192.168.219.111:6006. 켤 때는 Mac `~/Documents/supercombo_distill`에서 `nohup .venv/bin/tensorboard --logdir ckpt --bind_all --port 6006 > logs/tensorboard.log 2>&1 &`를 실행한다. 지금은 꺼져 있다.
  - 이 run은 TensorBoard 기능을 넣기 전에 시작했다. 그래서 `scripts/log2tb.py ckpt/scale/s05_base/train.log --follow`가
    train.log를 따라가며 `ckpt/scale/s05_base/tb`에 기록한다.
  - 끝나고 도는 eval도 같은 tb에 `eval/*`를 남긴다.
- 학습에 쓴 청크 목록:
  - Mac `ckpt/scale/s05_base/units.txt`
  - PC `configs/units_mac_base.txt` (65개)

### 1.2 PC
- **PC에서는 아무 작업도 돌지 않는다.** 라벨링은 보류 상태다 (1.3의 1번).
- 사용자가 배터리 때문에 라벨링을 멈췄다. 충전기보다 라벨링 소비 전력이 커서 배터리가 줄어든다.
- 다시 시작할 때는 사용자 지시를 받을 것.

### 1.3 남은 작업 순서
1. **라벨링은 사용자 결정으로 보류 (2026-10-27 18:30).** "나중에 데이터가 더 필요하면 진행". 지금 데이터(65개 청크, 학습 샘플의 93%, 평가 경로 4/5)로
   가능성을 본다. 다시 할 때: 21개 청크(79 세그먼트, 약 9.6만 프레임)가 남았고 worker 2개로 약 30분 걸린다.
   - 남은 경로: 10-06--01-05, 10-15--03-14, **10-22--23-22 (평가 경로)**, 10-03--00-11, 10-16--23-00, 10-02--00-21, 10-16--23-16, 10-16--23-07, 10-22--22-52, 10-02--23-59, 10-03--00-05, 09-25--22-23, 10-16--23-13, 09-25--23-05, 10-04--08-59
   - 실행 명령:
     ```
     cd /home/chan/pulsar2/work/supercombo_distill
     setsid nohup scripts/thermal_log.sh >> logs/thermal.log 2>&1 < /dev/null &
     setsid nohup sh -c 'cat configs/ep_routes.txt | xargs -P 2 -I{} sh -c "scripts/py scripts/ep_label.py data/edgepilot_drives/experiments/{} data/ep_label > logs/label/{}.log 2>&1"; echo ALLDONE > logs/label/ALLDONE' > /dev/null 2>&1 < /dev/null &
     ```
   - 이미 라벨링된 세그먼트와 중복으로 표시된 세그먼트는 ep_label.py가 알아서 건너뛴다.
   - 끝나면 `grep -l Traceback logs/label/*.log`로 실패한 경로가 없는지 꼭 확인할 것. 실패한 경로가 있어도 xargs는 계속 진행해서 놓치기 쉽다.
2. **Mac으로 나머지 전송:** `SSHPASS=<Mac 비밀번호> scripts/py scripts/sync_mac.py chan@192.168.219.111`
   - 이미 보낸 청크는 건너뛴다. 약 20GB이고 5분 정도 걸린다.
   - 전송 후 Mac 여유 디스크는 약 30GB가 된다.
3. **w48d 1 epoch을 PC에서 학습.** base와 같은 65개 청크를 써야 공정하게 비교된다.
   ```
   mkdir -p data/ep_label_65 && while read u; do ln -sfn $PWD/data/ep_label/$u data/ep_label_65/$u; done < configs/units_mac_base.txt
   scripts/py scripts/train.py --data data/ep_label_65 --out ckpt/scale/s26_w48d --epochs 1 --batch 32 --workers 14 \
     --arch '{"widths":[48,96,192,384,768],"depths":[1,3,6,9,3]}' > logs/scale_s26_w48d.log 2>&1
   ```
   - `scripts/scaling_study.sh`는 base와 w48d를 둘 다 학습한다. base는 Mac에서 이미 했으므로 그대로 실행하지 말 것.
   - PC 속도 추정: w48d는 epoch당 30~45분이다. 라벨링과 GPU를 같이 쓸 때 110샘플/s를 측정했다.
4. **두 모델을 같은 평가 세트로 비교한다.**
   - Mac 체크포인트를 PC로 가져와서 PC에서 둘 다 평가한다:
     `scripts/py scripts/eval_student.py --ckpt ckpt/scale/s05_base/last.pt ckpt/scale/s26_w48d/last.pt`
   - 이렇게 하면 두 student와 board NPU, small, teacher 기준값이 한 표에 나온다.
   - 평가 경로는 라벨링된 4개(10-22--23-22 제외)다. 라벨링을 재개하면 5개가 된다.
5. 사용자 지시는 "짧게 돌려서 어느 모델이 더 우수한지 판단하고, 이후 그 모델로 쭉 학습"이다.
   - 이긴 모델을 PC에서 이어서 학습한다: `--init ckpt/scale/<winner>/last.pt --epochs 20`. 전체 데이터(`data/ep_label`)를 쓴다.
   - 그다음 할 일:
     - QAT 검토
     - export: `student.py`의 `export_onnx`, fuse 후 CPU heads 분리
     - Pulsar2 AX620E 빌드
     - MaixCAM2 실측 및 평가
6. 학습 장소:
   - PyTorch 기준으로 Mac은 PC보다 2.2배 이상 느리다. 그래서 긴 학습은 PC에서 하는 게 기본이다.
   - Mac에서 학습할 때는 **`.venv`(py3.14, torch 2.14.1)와 `--compile-refresh`**를 쓴다 (12절). MLX는 이득이 작아 이식하지 않았다.
   - base 여러 epoch 학습은 19:43에 시작했다 (1.0). plan이 따라잡는지 확인하는 것이 목적이다. 당시 제안한 명령 (경로와 venv는 지금 기준):
     `cd ~/Documents/supercombo_distill && caffeinate -i .venv/bin/python scripts/train.py --out ckpt/mac_base_5ep --epochs 5 --batch 64 --workers 4 --compile-refresh --init ckpt/scale/s05_base/last.pt`
     - epoch당 약 28~30분이다.
     - `--init`을 주면 학습 전에 그 가중치로 bank를 먼저 계산한다.

---

## 2. 사용자 결정 이력 (중요)

1. Phase 1 (완료): edgepilot supercombo를 Axera **M57 (QNX)**로 포팅했다.
   - 위치: `/home/chan/pulsar2/work/edgepilot_m57` (README.md, release/)
   - 결과: CPU heads 분리(최종 Gemm은 fp32 CPU)를 하지 않으면 횡방향 오차가 11배가 되고 plan yaw가 뭉개진다. 이것이 edgepilot 핸들 흔들림의 원인이다.
2. Phase 2 (완료): 담당자 공유용 보고서 (Claude Doc): https://claude.ai/artifact/G55oNcUXzbSbvgLaYTdc5V
3. Phase 3 (현재): big 모델로 증류해서 AX630 최적화 student를 만든다.
   - 차선 teacher는 **LD 모델**(`/home/chan/Downloads/Models/FVRFBGRLEDT0001819`)로 한다. 사용자 판단: LKA14는 구형 Mobileye 출력이라 품질이 낮다.
   - 차선은 **선의 안쪽 edge** 기준으로 학습한다.
   - latency 예산은 **최대 30~35ms**이고, 그 안에서 효율이 가장 좋은 지점을 고른다. → 후보는 **base와 w48d 두 개만** 쓴다.
   - 라벨링이 끝나면 두 모델을 짧게 학습해서 비교하고, 이긴 모델로 길게 학습한다.
   - **중복 구간 제거**: 같은 장소를 다시 주행한 구간은 조도나 시간대가 다를 때만 남긴다.
   - 학습·평가 데이터를 Mac으로 옮겨서 Mac에서 base를 학습한다 (현재 진행 중).

---

## 3. 머신과 접속 (비밀번호는 사용자에게 받을 것, 파일에 적지 않음)

| 머신 | 접속 | 용도 / 메모 |
|---|---|---|
| PC (Linux, RTX 4070 Laptop) | 로컬 | 라벨링, 학습, Pulsar2. 작업 폴더 `/home/chan/pulsar2/work/supercombo_distill`, venv `.venv` (torch cu128, onnxruntime-gpu) |
| MacBook Pro M5 (기본형, GPU 10코어, 16GB, Wi-Fi) | `chan@192.168.219.111`, 비밀번호 인증 | `~/Documents/supercombo_distill`이 PC 폴더의 미러다. `.venv`는 Homebrew python 3.14 + torch 2.14.1 (MPS). zstd는 없다. brew는 `/opt/homebrew/bin` |
| MaixCAM2 (AX630C) | `root@192.168.219.107`, 비밀번호 인증 | latency 실측용. `/opt/bin/ax_run_model`. **측정 전에 `edgepilot.service`를 멈추고 끝나면 다시 켤 것.** modeld가 NPU를 잡고 있다 |
| M57 보드 (QNX) | `root@10.20.20.170` | Phase 1에서 썼다. QNX SDP는 `/home/chan` 아래 |

- 비밀번호 SSH는 `sshpass`로 한다. 비밀번호는 `SSHPASS` 환경변수와 `sshpass -e`로 넘긴다.
  - Mac에는 sshpass가 없다. `SSH_ASKPASS`에 `echo "$SSHPASS"`만 하는 스크립트를 주고 `SSH_ASKPASS_REQUIRE=force`로 쓴다(`board_eval.py`가 이렇게 한다).
- PC와 Mac은 둘 다 Wi-Fi(192.168.219.x)에 붙어 있다. PC의 유선 랜은 다른 네트워크(192.168.0.x)다.

---

## 4. 디렉터리와 데이터 (PC 기준, Mac도 같은 상대경로)

```
models/   big_driving_supercombo.onnx (teacher, sha256 6fee5937…, commit 64f9b47), driving_supercombo.onnx (edgepilot small),
          master_vision/policy.onnx, arch_*.onnx (구조 탐색용 export)
configs/  ep_routes.txt (라벨링 순서, 37 경로), arch_*.json (Pulsar2 빌드 설정), units_mac_base.txt
scripts/  아래 5절
data/edgepilot_drives/experiments/  원본 주행 기록 (USB에서 rsync, `._*` AppleDouble 제외), 38 경로 (영상 있는 것 37), 14.5h
data/ep_label/<route>__<첫 세그먼트>/  라벨링 결과 (예전 방식으로 경로 전체를 한 번에 한 것은 <route>/)
data/dedup/   emb_*.npz, thumb_*.npy, matches.npy, verify_scores.npy, dedup.json
data/comma2k19/  내려받기만 했고 아직 안 씀 (카메라가 달라서)
ckpt/ logs/ outputs/ (Pulsar2 빌드 결과) calib/dummy
```

### 4.1 라벨 형식 (청크마다)
- `frames.npy` uint8 [M,2,6,128,256]
  - 짝수 프레임만 들어 있다. 기기와 똑같이 warp한 road/wide YUV6 영상이다.
  - `frames_idx.npy`에 각 영상의 프레임 번호가 있다.
- `labels.npz`: 프레임 단위 배열이다. 압축하지 않은 npz라 `train.npz_mmap`으로 mmap할 수 있다.
  - `t_ns`, `seg`
  - `big` [N,2066]: teacher 출력. 16384 hidden은 뺐다.
  - `small` [N,2576]: edgepilot small, fp32
  - `ld_raw` [N,2071]: Mac에는 보내지 않았다
  - `ld_lanes` [N,4,33]: 안쪽 edge의 y. LD가 못 본 곳은 NaN이다.
  - `ld_lanes_centre`, `ld_lane_prob` [N,4], `ld_edges` [N,2,33]
  - `desire`, `pulse`, `v_ego`, `yaw_rate` (Localization이 없는 경로는 NaN), `engaged`, `rpy`, `mount`
  - `rec_*`: 보드에서 실제로 돈 U16 NPU 모델의 ModelState 기록
- 최종 규모: 968k 프레임 (13.4h)
  - 학습용: 중복 마스킹 후 799k 프레임 (11.1h), epoch당 샘플 약 40만 개
  - 평가용: 96.5k 프레임
  - 디스크: PC 약 220GB, Mac 약 200GB

### 4.2 평가 경로 (학습에서 제외)
`EVAL_ROUTES` (train.py):
- 2026-10-25--22-42-26-175
- 2026-10-05--00-07-29-810
- 2026-10-22--23-22-40-918
- 2026-10-05--16-28-44-410
- 2026-09-27--02-44-40-679

---

## 5. 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/py` | PC 전용 래퍼. ORT CUDA 13용 `LD_LIBRARY_PATH`(nvidia cu13, cudnn)를 잡고 `.venv/bin/python`을 실행한다. **PC에서 python은 항상 이걸로 실행할 것** (안 그러면 ORT가 조용히 CPU로 돈다). Mac에서는 `.venv/bin/python`을 쓴다 |
| `ep_eval.py` | `Stateful` (원본 ONNX를 상태 queue와 함께 closed loop로 실행, CUDA 필수), `recorded()` (기록된 ControlState, ModelState, Localization, LKA14) |
| `ld_teacher.py` | LD 모델 추론, 디코딩 (중심 x, half width), 도로 평면 투영 |
| `ep_label.py` | 라벨링. 청크 15 세그먼트, 직전 세그먼트 200프레임으로 warm-up. 끝난 청크와 중복 세그먼트는 건너뛴다 |
| `dedup.py` | 중복 구간 탐지 (6절) |
| `student.py` | student 모델 (7절), `export_onnx`, `count_macs`, `load_student`, S8 양자화 시뮬레이션 `fake_quant_s8`/`quant_sim` (QAT 겸용) |
| `train.py` | 증류 학습 (8절) |
| `eval_student.py` | 평가 (9절) |
| `bench_arch.py`, `p2build.sh` | 구조 후보를 ONNX로 export하고 Pulsar2 AX620E로 빌드한 뒤 max_cycle을 본다 |
| `bench_train.py` | 가짜 데이터로 학습 처리량을 잰다 (장비 비교용) |
| `scaling_study.sh` | base와 w48d를 1 epoch씩 학습하고 평가한다 (지금은 base를 Mac에서 따로 돌림) |
| `sync_mac.py` | Mac으로 청크 단위 전송 (tar, pigz, ssh). 이어받기 가능 |
| `thermal_log.sh` | 1분마다 온도 기록 → `logs/thermal.log` |
| `log2tb.py` | TensorBoard 기능이 없던 예전 run의 train.log를 TensorBoard 기록으로 바꾼다 (`--follow`로 진행 중인 run도 가능) |
| `snap_epochs.sh` | 학습 중 epoch마다 last.pt를 epN.pt로 복사하고, 세 번째 인자 `eval`을 주면 CPU로 평가한다 (`eval-s8`은 S8 평가) |
| `diag_plan.py`, `diag_y2s_dump.py`, `diag_y2s.py`, `diag_motion.py` | plan 오차 진단 (00.1절). `diag_y2s_dump.py --s8`는 lead도 저장하고, `diag_y2s.py <dump 이름들>`로 모델을 고른다 |
| `diag_lead.py`, `diag_gap.py` | lead_x 오차와 헛 lead 분석, 프레임 간격을 바꿔 pose가 움직임을 읽는지 시험 (000000절) |
| `quant_build.py` | 체크포인트를 ONNX로 export하고, 실제 프레임 보정 데이터를 만들고, Pulsar2 6.0 AX620E 빌드까지 한다. Mac에서는 docker(colima와 Rosetta)로 돌린다 (0000절) |
| `quant_eval_p2.py` | Pulsar2 QuantAxModel(Reference)을 평가 unit에 closed loop로 돌리고, fp32와 S8 시뮬레이션과 함께 지표를 비교한다 |
| `board_eval.py` | MaixCAM2에서 axmodel의 latency(ax_run_model)와 평가 unit closed loop 출력(`board_run.py`)을 재고, fp32, S8 시뮬레이션, Pulsar2와 비교한다. 비밀번호는 `SSHPASS` 환경변수로 받는다. edgepilot.service를 멈췄다가 다시 켠다 |
| `board_run.py` | 보드에서 도는 쪽(python3.13, pyaxengine, numpy). 한 번의 pass로 closed loop를 돌린다 |
| `viz_lanes.py` | 평가 경로 프레임에 student, 출시 모델, teacher, LD 차선을 그린다 (직선, 커브, 차선 변경, 최악 장면). 결과는 `outputs/lane_overlay/` |

---

## 6. 중복 구간 제거 (dedup.py, GPS 없음)
1. embed: 1초마다 키프레임 하나(GOP 1s)를 torchvision ResNet-50(IMAGENET1K_V2)에 넣고 layer3+layer4를 GeM pooling해서 descriptor를 만든다. 밝기(luma)도 같이 저장한다.
2. match: cosine > 0.86이고 순서가 맞는 구간이 8초 이상 이어지면 후보로 잡는다. 같은 경로 안에서는 600초 이상 떨어진 구간끼리만 비교한다. → 2346개 run.
3. verify: SIFT, ratio 0.75, RANSAC fundamental matrix로 확인한다.
   - 영상 위쪽 200행만 쓰고, 3px 미만으로 움직인 매칭(유리, 본넷)은 버린다.
   - `accept()` 기준: 3쌍 중 2쌍이 inlier 15 이상이거나, 한 쌍이 30 이상이면 같은 장소로 본다.
   - 기준은 사람이 눈으로 확인한 표본으로 잡았다. 다른 장소는 0~11 (비슷한 터널이나 밤거리는 ~14까지), 같은 장소는 27~80이다.
   - 결과: 1040 run이 통과했다.
4. decide: 시간대(낮 07–17, 박명 05–07/17–19, 밤)나 밝기(dark<45, dim<90, bright)가 다르면 두 번 지나간 것을 모두 남긴다.
   - 우선순위는 평가 경로, 이미 라벨링된 것, 이름 순이다. 평가 경로는 절대 버리지 않는다.
   - 결과: **52,155초 중 6,861초 제거 (13.2%)**, 67개 세그먼트는 라벨링하지 않음.
   - 버린 구간은 `dedup.json`의 `routes[r].dropped_ns`([start,end) capture time)에 있고, train.py가 마스킹한다.
- 밤 영상은 AI-ISP 때문에 대부분 "bright"로 분류된다. 그래서 조건이 달라서 남은 중복은 728초뿐이다.
- 임계값만 바꿀 때는 `accept()`를 고치고 `decide`만 다시 돌리면 된다. SIFT를 다시 돌릴 필요가 없다.

---

## 7. Student 구조와 latency

- RepVGG 블록을 쓴다 (dense 3x3 + 1x1 + identity BN, ReLU). export 때 `fuse()`로 conv 하나로 합친다.
- 다음 연산은 쓰지 않는다: depthwise, GELU, LayerNorm, layer scale, attention. 기존 모델의 AX620E 프로파일에서 depthwise가 26%, Slice/assign tiling이 26%, GELU가 4.5%를 차지했다.
- 입력 스케일(1/64)은 첫 conv에 접어 넣는다. Sub 연산이 없다.
- hidden 정규화(LpNorm)는 AX620E가 3D/4D만 tiling하므로 3D view로 한다.
- 각 head는 NPU에서 ReLU feature까지 계산하고, 마지막 Linear는 **CPU에서 fp32**로 돈다 (`cpu_heads`). Phase 1의 교훈이다.
- policy는 최근 8개 hidden과 현재 hidden(총 9 tokens)을 FC로 처리한다. desire는 25x8, traffic은 2다.
- 기본값(base): widths (32,48,96,192,384), depths (1,2,3,4,2), neck 64, feat 1024, hidden 512. **w48d**: widths [48,96,192,384,768], depths [1,3,6,9,3].
- MaixCAM2 실측 latency (ax_run_model, 대략 4ms + GMAC당 3.8ms):

  | 모델 | 연산량 | latency |
  |---|---|---|
  | 현재 출시 모델 | - | 14.0 ms (실제 운행 중 15.5) |
  | slim | - | 4.25 ms |
  | base | 0.85 GMAC | 6.0 ms |
  | deep | - | 7.24 ms |
  | wide | - | 8.22 ms |
  | wide2 | 2.03 G | 9.6 ms |
  | wide_deep | - | 10.6 ms |
  | hires32 | - | 12.5 ms |
  | w48 | 4.51 G | 19.0 ms |
  | **w48d** | **6.55 G** | **25.6 ms** |
  | hires48 | 7.06 G | 26.3 ms |
  | w64 | 7.98 G | 31.3 ms |
  | w64d | 11.6 G | 42.9 ms |

- Pulsar2의 max_cycle은 큰 conv 모델의 latency를 과소평가한다 (1M cycle당 1.4~1.9ms). **반드시 보드에서 잰다.**
- Pulsar2 6.0과 8.0은 docker로 돈다: `/home/chan/pulsar2/scripts/pulsar2-6.0.sh`. `/home/chan/pulsar2/work`가 `/data`로 마운트된다.
  - 설정: AX620E, NPU1, U16 activation, S8 per-channel weight, MinMax 보정.

---

## 8. 학습 (train.py)

- 샘플: 짝수 프레임 t. 입력은 road/wide (t-4, t), 최근 100 desire pulse를 4:1로 max-pool한 것, features_buffer [24,512].
- features_buffer는 **student 자신의 hidden 기록**(bank)에서 t-96 … t-4를 4프레임 간격으로 가져온다.
  - bank는 매 epoch 끝에 현재 weight로 다시 계산한다. 첫 epoch에는 0이다.
  - 기기에서 closed loop로 도는 것을 흉내 낸 것이다. 버린 구간의 프레임도 bank는 계산하므로, 그 뒤 프레임의 기록은 끊기지 않는다.
- loss:
  - plan, lead, pose, road_transform, wide_from_device: teacher 평균에 대해 L1을 teacher σ로 나눈다 (상한 20). 여기에 0.1 × logstd L1을 더한다.
  - lanes, road_edges: y는 LD가 본 곳은 LD(σ = 0.05 + 0.004x), 나머지는 teacher를 따른다.
  - lane_prob: 홀수 인덱스(존재 확률)는 LD에서, 나머지는 teacher에서 가져온다.
  - meta, lead_prob: BCE
  - desire_state, desire_pred: KL
  - 가중치는 `WEIGHTS`에 있다 (plan 1, lanes 1, road_edges 0.5, lead 0.5, …).
- 최적화: AdamW (lr 2e-3, wd 1e-4), OneCycle (pct_start 0.05), grad clip 5, bf16 autocast.
- 배치: base 64, w48d 32. PC workers 14, Mac 4.
- **TensorBoard**: `<out>/tb`에 기록한다.
  - `--tb-every` step(기본 20)마다 head별 loss 구간 평균, grad norm, lr, samples/s, epoch 진행률을 남긴다.
  - epoch이 끝나면 걸린 시간과 bank refresh 시간을 남긴다.
  - `eval_student.py`는 기본으로 `<ckpt dir>/tb`에 `eval/<지표>`를 쓴다. student, board_npu, small_fp32가 한 차트에 나오고 step은 epoch이다.
  - 보기: `tensorboard --logdir ckpt --bind_all`. PC에서는 `.venv/bin/tensorboard`, Mac에서는 `~/Documents/supercombo_distill/.venv/bin/tensorboard`.
- 주요 옵션:
  - `--device` (기본 cuda > mps > cpu)
  - `--dedup` (기본 data/dedup/dedup.json)
  - `--init`: 이전 체크포인트 weight에서 새 OneCycle 스케줄로 시작한다. **학습 전에 그 weight로 bank를 먼저 계산한다.**
    0으로 시작하면 1 epoch 결과처럼 plan이 망가진다.
  - `--compile-refresh`: bank refresh의 backbone+neck을 torch.compile한다. 마지막 배치는 패딩한다.
    Mac base에서 refresh가 2.5배 빨라진다. w48d는 Metal 컴파일 오류가 나고, 학습 step에는 쓰지 않는다 (느려짐).
  - `--tb-every` (TensorBoard 간격, 기본 20)
  - `--max-steps` (smoke test)
  - `--flip p`, `--curve-sampling` (000절), `--qat` (S8 가중치 QAT, 0000절), `--gap-aug p` (프레임 간격 augmentation, 0000000절)
- 라벨과 영상은 **프로세스마다 처음 쓸 때 mmap**한다. 라벨은 `npz_mmap`, 영상은 `np.load(mmap_mode)`.
  - macOS DataLoader는 spawn이라 Route를 pickle할 때 데이터를 빼고 보낸다.
  - 청크 하나가 fd 7개를 쓰므로 `raise_open_files()`로 open file 한도를 올린다 (macOS 기본값 256).

---

## 9. 평가 (eval_student.py)

평가 경로의 짝수 프레임 중 v > 5 m/s인 프레임을 쓴다. student는 closed loop(자기 hidden)로 돈다.

| 지표 | 정의 | 기준 |
|---|---|---|
| lat_acc | plan으로 계산한 횡가속 (openpilot 방식, action_t 0.3) | teacher와의 차이 |
| long_acc | 종가속 | teacher와의 차이 |
| plan_y2s_cm | 2초 시점 plan의 횡위치 | teacher와의 차이 |
| lanes_vs_ld_cm | 6~40m 구간 ego 차선 | LD와의 차이 |
| width_vs_ld_cm | 차선 폭 | LD와의 차이 |
| lead_x_m | 앞차 거리 (teacher 확률 > 0.5) | teacher와의 차이 |
| gas2 | 2초 시점 gas press 확률 | teacher와의 차이 |

비교 대상: 보드 NPU (출시 모델 기록), small fp32, big teacher (lanes/width는 LD와의 차이 자체), student들.

---

## 10. 함정과 주의사항

- **PC 발열과 정지**: 01:39에 하드 프리즈가 있었다. 로그에는 원인이 없었고 CPU가 88~93°C였다.
  - 그 뒤로 라벨링 worker는 **2개**, 청크 단위로 저장, 온도 기록을 상시로 한다.
  - 2 worker 기준 지금 온도는 CPU 67~73°C, GPU 60°C다.
- **PC 전원**: 라벨링 중에는 충전보다 방전이 빠르다. 사용자가 중간에 멈추라고 하면 **진행 중인 청크만 마치고**(labels.npz가 생긴 뒤) 멈춘다. 막 시작한 청크는 사용자 사정(배터리)에 따라 바로 끊어도 된다. 끊긴 청크는 다음 실행 때 `.tmp`를 지우고 다시 만든다.
- **프로세스 죽이기**: `pkill -f`는 자기 셸 명령줄과도 매칭돼서 자기 자신을 죽인다 (실제로 겪음). PID로 죽이거나 `ps | grep "[x]yz"` 패턴을 쓸 것.
  - **디스크 함정(2026-10-30 20:05)**: 학습 중인 프로세스가 매핑해 둔 파일은 지우거나 바꿔도 프로세스가 끝날 때까지 공간이 풀리지 않는다.
    - 학습 중에 `scripts/slim_labels.py`로 labels.npz를 줄였다. ld_raw와 NaN뿐인 small을 외장 HDD `labels_extra/`로 옮기고 검증한 뒤 다시 썼다(14.9 GB 절약).
    - 그런데 새 파일만큼 오히려 디스크가 줄어 147 MB까지 떨어졌다.
    - 다시 만들 수 있는 CoreML 캐시(4.8 GB), 임시 시험 폴더, 예전 ckpt를 외장으로 옮겨 넘겼다.
    - 큰 파일은 학습이 돌지 않을 때 정리할 것.
  - `pgrep -f "train.py --out X"`로 끝나기를 기다리는 루프도 같다. 그 문자열이 든 다른 대기 셸(그리고 `sh -c "... train.py --out X ..."` 체인 셸 자체)과 매칭돼 영원히 기다린다(2026-10-30 Run A가 30분 늦게 시작했다). PID를 기다릴 것.
  - xargs 체인을 멈출 때는 **바깥 sh를 먼저** 죽인 다음 xargs를 죽인다. 순서가 반대면 ALLDONE과 이후 명령이 실행된다.
- **xargs**: 경로 하나가 실패해도 계속 진행한다. 끝나면 꼭 Traceback을 grep할 것. 09-27--21-39는 Localization이 없어서 처음에 실패했고 지금은 고쳤다.
- **ORT**: 반드시 `scripts/py`로 실행한다. teacher는 13ms/프레임, small은 2.6ms/프레임이다 (CUDA 기준).
- **torch.hub (DINOv2 등)**: auto-mode 분류기가 막았다. torchvision 가중치만 쓴다.
- **teacher 출처**: GitHub LFS에 없다. HF LFS batch API(`huggingface.co/commaai/openpilot-lfs.git/info/lfs`)에서 받았다.
- **onnx2torch**: opset 19/20을 변환하지 못한다. teacher는 ORT로 실행한다.
- **USB 원본**: macOS `._*` 파일이 reader를 깨뜨린다. rsync할 때 제외할 것.
- **M57 (Phase 1)**: S16 weight는 깨진다. AdaRound도 모델을 망가뜨린다. QNX `/tmp`에는 디렉터리를 만들 수 없다. `AX_SYS_CmmMem_Dev_Init`가 필요하다.
- **벤치마크는 GPU가 비어 있을 때만 잴 것**: 학습과 GPU를 같이 쓰면서 잰 MLX 151 vs PyTorch 46 샘플/초는 틀린 신호였다. 단독으로 재면 비슷하다.
- **Mac PyTorch**: channels_last는 3배 넘게 느려진다. torch.compile은 학습 step을 느리게 만들고 w48d에서는 컴파일이 실패한다.
  bf16과 fp16, 배치 64와 128은 차이가 거의 없다.
- **Mac 환경**: `.venv` 하나다 (py3.14, torch 2.14.1). 예전 `.venv`(py3.9, torch 2.8)와 `.venv314`는 지웠다.
  venv는 만든 경로가 실행 파일에 박히므로, 폴더를 옮기면 다시 만들어야 한다 (12.2).
- **Mac 전송**: Wi-Fi로 원본 기준 약 75MB/s 나온다 (pigz -1, 압축률 1.65). 디스크 여유가 적으니 큰 파일을 추가하기 전에 `df`로 확인할 것.

---

## 11. 메모리 (Claude Code auto-memory)
`/home/chan/.claude/projects/-home-chan-pulsar2/memory/sc-distill-project.md`에 요약이 있다. teacher, 환경, latency, dedup, Mac 정보.

---

## 12. Mac 학습 속도와 MLX (2026-10-27)

### 12.1 측정한 것

| 장비 / 프레임워크 | base | w48d | 비고 |
|---|---|---|---|
| Mac M5, PyTorch 2.8 MPS (py3.9 venv, 지금은 삭제) bf16 | 175~185 samples/s (epoch 약 50분) | 52 samples/s (epoch 약 2.7시간) | |
| PC RTX 4070 Laptop, CUDA bf16 | 383 samples/s | 110 samples/s | 라벨링과 GPU 공유 중에 잼 → 실제는 더 빠름 |

- 숫자는 `bench_train.py`(가짜 데이터, 학습 step + bank refresh)로 쟀다. Mac 실제 학습도 185 samples/s로 같았다.
- **Mac 실제 학습 중 자원 사용**:
  - GPU (`ioreg` "Device Utilization %"): 94~96%로 이것이 한계다.
  - CPU: 메인 44%, DataLoader 워커 4개 각 2%로 여유가 많다.
  - 디스크 읽기 160~175MB/s, 메모리 여유 29%, 스왑 0, 발열 경고 없음.
  - 즉 Mac은 GPU-bound다. 다만 base는 채널이 좁아서(32~48) GPU 연산 효율이 약 10%로 낮다. GPU가 바쁘지만 일을 효율적으로 하지는 못한다.
- Neural Engine(ANE)은 PyTorch나 MLX 학습에 쓸 수 없다.

### 12.2 환경
- `~/Documents/supercombo_distill/.venv` (2026-10-27 19:40 새로 만듦): Homebrew Python 3.14.6 (`/opt/homebrew/opt/python@3.14`),
  **torch 2.14.1 (그날 기준 최신)**, numpy 2.5.4, tensorboard 2.21.0, MLX 0.32.4.
  - 다시 만드는 법: `/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv && .venv/bin/pip install -U pip torch numpy tensorboard mlx`
  - 확인한 것: train, student, eval_student, log2tb import, MPS 사용 가능, `bench_train.py --batch 8 --steps 3` 통과, `.venv/bin/tensorboard` 실행.
  - 실제 데이터 학습은 아직 이 환경에서 돌려보지 않았다.
- 1 epoch base 학습(`ckpt/scale/s05_base`)은 지금은 지운 py3.9 venv(CommandLineTools Python 3.9, torch 2.8.0)로 돌렸다.
  12.4의 벤치마크는 지운 `.venv314`로 쟀고, 패키지 버전은 지금 `.venv`와 같다.
- Homebrew `brew`는 `/opt/homebrew/bin`에 있다. ssh 비대화 셸의 PATH에는 없으니 전체 경로를 쓴다.
  brew에는 python@3.14 3.14.10이 나와 있지만, 설치된 것은 3.14.6이고 업그레이드하지 않았다.

### 12.3 벤치마크 스크립트
- `scripts/bench_train.py`: PyTorch. `--device`, `--arch`, `--batch`, `--dtype bf16|fp16|fp32`, `--channels-last`, `--compile` (backbone과 neck에 torch.compile).
- `scripts/bench_mlx.py`: **student를 MLX로 그대로 옮긴 벤치마크**다.
  - 학습 형태 RepVGG (3x3+BN, 1x1+BN, identity BN), neck, vis_fc, summarizer, trunk, 모든 head와 CPU linear, temporal MLP를 포함한다.
  - 레이아웃은 NHWC다. loss는 같은 대용 loss(출력 제곱 평균), optimizer는 AdamW, `mx.compile`을 쓴다.
  - MLX에는 autocast가 없다. `--dtype bf16|fp16`은 모델 전체를 그 dtype으로 바꾼다 (순수 저정밀 학습).
  - 옵션: `--no-compile`, `--batch`, `--arch`.
- `scripts/bench_mac_sweep.sh`: base와 w48d 각각에 대해 다음을 한 줄씩 `logs/bench_sweep.txt`에 기록한다.
  - torch2.8 vs 최신 torch (같은 배치)
  - 최신 torch: 배치 2배, fp16, channels_last, compile
  - MLX: 배치 두 가지, bf16, fp16, fp32, no-compile
- 사전 신호(학습과 GPU를 같이 쓰면서 배치 8로 잼): MLX 151 vs PyTorch 46. **이 값은 틀린 신호였다.** 단독으로 재면 비슷하다 (아래).

### 12.4 sweep 결과 (2026-10-27 19:10~19:17, 가짜 데이터, epoch = 학습 40만 샘플 + bank refresh 48.4만 프레임)

| 모델 | 설정 | 학습 samples/s | refresh frames/s | epoch |
|---|---|---|---|---|
| base | torch 2.8 bf16 b64 | 197 | 845 | 43분 |
| base | **torch 2.14 bf16 b64** | **261** | 856 | **35분** |
| base | torch 2.14 b128 / fp16 b128 | 263 / 264 | 853 / 862 | 35분 |
| base | torch 2.14 channels_last | 72 | 805 | 103분 (훨씬 느림) |
| base | torch 2.14 compile | 157 | **2119** | 46분 (학습은 느려지고 refresh는 빨라짐) |
| base | MLX bf16 b64 / b128 / fp16 | 246 / 244 / 247 | 1508 / 1482 / 1434 | 32~33분 |
| base | MLX fp32 | 144 | 1131 | 53분 |
| base | MLX bf16 no-compile | 194 | 867 | 44분 |
| w48d | torch 2.8 bf16 b32 | 56 | 276 | 148분 |
| w48d | **torch 2.14 bf16 b64** | **71** | 276 | **122분** |
| w48d | torch 2.14 channels_last | 27 | 269 | 281분 |
| w48d | torch 2.14 compile | – | – | Metal 셰이더 컴파일 오류 ("no 'buffer' resource location available") |
| w48d | MLX bf16 b32 / b64 / fp16 | 61 / 62 / 63 | 355~357 | 129~132분 |
| w48d | MLX fp32 | 37 | 258 | 211분 |

**결론**
- PyTorch를 2.8에서 2.14로 올린 것이 가장 큰 개선이다. base는 학습이 +32%, w48d는 +27%(배치 64 포함) 빨라졌다.
- 배치 크기와 fp16은 효과가 거의 없다. channels_last는 크게 느려지니 쓰지 말 것.
- MLX는 학습 속도가 PyTorch 2.14와 비슷하거나 약간 느리다. refresh만 1.8배 빠르다. 이식 비용에 비해 이득이 작아서 **MLX 이식은 하지 않기로 했다.**
- torch.compile은 학습에는 손해지만 forward만 하는 bank refresh는 2.5배 빨라진다.
  - 그래서 train.py에 `--compile-refresh`를 추가했다. backbone+neck만 compile하고, 마지막 배치는 패딩해서 shape를 고정한다.
  - base는 Mac 실데이터 smoke에서 refresh가 22s → 10s(컴파일 포함)로 줄었다. w48d는 Metal 컴파일 오류가 나므로 쓰지 말 것.
- Mac 권장 설정:
  - base: `.venv` + `--batch 64 --compile-refresh`로 epoch 약 28~30분
  - w48d: `.venv` + `--batch 64`로 epoch 약 2시간. w48d는 PC가 낫다.

### 12.5 MLX로 옮기게 될 경우 (지금은 보류)
1. 모델: `bench_mlx.py`의 `Student`를 바탕으로 쓴다. PyTorch `student.py`와 **가중치 변환기**가 필요하다 (export와 Pulsar2는 PyTorch/ONNX 경로를 유지).
   - conv weight: PyTorch [O,I,kh,kw] ↔ MLX [O,kh,kw,I]
   - vis_fc 입력 flatten 순서: NCHW (C,H,W) ↔ NHWC (H,W,C). 첫 Linear의 열 순서를 재배열해야 한다.
   - BN running stats와 head 이름을 맞춘다.
2. 데이터: train.py의 `Route`/`Samples`(mmap)는 그대로 쓸 수 있다.
   - DataLoader 대신 numpy 배치를 만들고 `mx.array`로 넘긴다. 입력은 NCHW → NHWC transpose가 필요하다.
   - CPU는 여유가 있으니 스레드나 프로세스로 prefetch한다.
3. loss: `train.losses`(MDN L1/σ, LD 차선 마스킹, BCE, KL)를 mlx.core로 옮긴다.
4. bank refresh: forward만 쓰므로 그대로 옮긴다.
5. 정밀도: autocast가 없다. bf16 순수 학습이 PyTorch AMP(bf16 autocast, fp32 master weight)만큼 수렴하는지 짧게 확인해야 한다.
   - 대안: fp32 가중치를 두고 forward에서 bf16로 캐스팅하는 수동 혼합 정밀도.
6. 같은 데이터, 같은 epoch에서 PyTorch base 1 epoch(`ckpt/scale/s05_base`)과 eval 지표를 비교해서 동등한지 확인한다.

