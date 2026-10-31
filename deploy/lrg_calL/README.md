# 실차 시험 패키지: lrg_calL (2026-11-01)

출시 모델(supercombo, 14.0 ms) 대신 학생 모델 lrg_calL을 edgepilot에서 돌리는 패키지다.
보드 rehearsal 검증(`check`)은 아직 하지 않았다. 아래 "검증한 것"을 보고 판단한다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `lrg_calL.axmodel` | NPU 모델(Pulsar2, NPU1, U16). `outputs/quant_lrg_calL_ext/m.axmodel`과 같다 |
| `lrg_calL.heads` | 헤드별 마지막 선형층(fp32, CPU). `ckpt/mac_lrg/ema_calL.pt`에서 `scripts/export_cpu_heads.py`로 만들었다 |
| `modeld` | 학생 모델을 읽는 edgepilot modeld. 버전은 `v0.0.9-1-gbeccd3d+student-cpu-heads`다. 출시 모델도 그대로 돌린다 |
| `SHA256SUMS`, `source.json` | 체크섬, 출처 |

edgepilot 코드는 `~/Documents/ax630/edgepilot`의 브랜치 `student-cpu-heads`에 있다. 아직 커밋하지 않았다.
- `src/model/cpu_heads.*`: 새 파일
- `supercombo_model.*`, `model_output_assembly.h`: CPU 헤드 코어 계약 추가
- `tests/gtest_cpu_heads.cc`: 새 테스트
- 문서 3개

보드에 설치된 버전은 v0.0.9다. HEAD와 다른 점은 HUD 커밋 하나뿐이라 modeld만 바꿔도 다른 프로세스와 맞는다.

## 검증한 것

- **모델 성능**: NPU(p2sim)와 보드에서 5경로 합산의 모든 지표가 출시 모델보다 낫다. 보드 latency는 11.21 ms다(`reports/experiments.html`).
- **CPU 헤드**: 보드 NPU 특징 1,497프레임에 C++ 구현(`CpuHeads` + `kCpuHeadParts`)을 돌린 결과가 PyTorch와 최대 6e-5 차이다.
- **edgepilot 호스트 테스트**: 새 테스트 5개가 통과했다. 기존 테스트 중 녹화 관련 몇 개는 Mac 디스크 여유가 10% 미만이라 실패한다. 변경과는 관계없다.
- **보드용 빌드**: 성공했다(docker, `tools/docker_ax630/build.sh`).
- **plan·차선·앞차 외 출력**(8 unit, teacher 기준, `logs/diag_aux_heads_lrg_calL.txt`):

  | 출력 | 학생 | 출시(fp32) | 비고 |
  | --- | --- | --- | --- |
  | pose 회전 오차(mrad/s, 롤/피치/요, 1 unit) | 7.1 / 10.4 / 3.4 | 2.6 / 3.8 / 1.5 | 학생이 약 2.5배 잡음이 크다. 보고하는 표준편차는 teacher와 비슷해 잡음을 덜 보고한다 |
  | 보정 각 평균 치우침(teacher 대비) | 피치 -0.12, 요 +0.25 mrad | – | 보정기가 평균을 내므로 영향이 작다 |
  | 도로 높이 오차 | 2.1 cm | 4.3 cm | 학생이 낫다 |
  | 도로 경계 y 오차 | 75 cm | 80 cm | 비슷하다 |
  | 차선 변경 desire 일치 | 99.2% | 98.1% | |

  조향 제어는 차량 CAN 요레이트(ESP12)를 쓴다. pose는 locationd(학습기 lagd 등)와 온라인 보정에 들어간다.
  그래서 pose 잡음은 조향에 바로 영향을 주지 않는다. 다만 시험하는 동안 학습기와 보정이 `params/`에 저장한다.
  `select`가 처음에 `params/`를 `params.before_student/`로 복사하고, `rollback --params`가 이것을 되돌린다.

## 아직 안 한 것

- **보드 설치와 rehearsal**: 2026-11-01 03시에 보드가 응답하지 않아 하지 못했다. 시험 전에 `install`과 `check`를 먼저 돌린다.
- **실제 카메라 프레임**: 학습과 평가는 녹화(H.264)를 디코딩한 프레임으로 했다. 차에서는 압축하지 않은 ISP 프레임이 들어간다. 이 차이는 실차에서 처음 확인한다.

## 명령 (Mac에서, 비밀번호는 SSHPASS로만)

```sh
cd ~/Documents/supercombo_distill
SSHPASS=<보드 비밀번호> .venv/bin/python scripts/deploy_student.py install lrg_calL   # modeld 교체(기존은 modeld.orig), 모델 복사
SSHPASS=<보드 비밀번호> .venv/bin/python scripts/deploy_student.py check lrg_calL     # 서비스 멈춤 → 두 모델 로드 확인 → 40초 리허설(clip34) → 서비스 다시 켬
SSHPASS=<보드 비밀번호> .venv/bin/python scripts/deploy_student.py select lrg_calL --shadow   # 학생 모델, CAN 송신 없음
SSHPASS=<보드 비밀번호> .venv/bin/python scripts/deploy_student.py select lrg_calL            # 학생 모델로 조향
SSHPASS=<보드 비밀번호> .venv/bin/python scripts/deploy_student.py rollback [--params] [--binary]   # 출시 모델로
SSHPASS=<보드 비밀번호> .venv/bin/python scripts/deploy_student.py status
```

- `check` 결과의 `== load models/lrg_calL.axmodel`에 `Supercombo CPU-heads axmodel`이 나와야 한다.
  출시 모델 쪽에는 `openpilot-master axmodel`이 나와야 한다.
- 리허설에서 `profile ms`의 npu+outputs는 15 ms 근처여야 한다(출시 모델은 약 15.5 ms + 출력 처리).
- `select`는 systemd drop-in(`/etc/systemd/system/edgepilot.service.d/driving-model.conf`)과 `params/driving_model.json`을 쓴다.
  녹화된 주행마다 `params/driving_model.json`이 복사되므로 어떤 모델로 달렸는지 남는다.

## 실차 시험 권장 순서

1. 정차 상태에서 `select --shadow`로 켠다. HUD에서 경로, 차선, 앞차가 출시 모델과 비슷하게 나오는지 본다.
2. 주행하며 섀도 모드로 비교한다. CAN 송신이 없으므로 조향하지 않는다.
   야간 시내, 주간 도로, 오르막과 내리막을 지나며 차선이 흔들리지 않는지 본다.
3. 문제가 없으면 `select`(송신)로 바꿔 짧은 구간부터 조향한다. 운전자는 언제든 개입할 수 있어야 한다.
4. 끝나면 `rollback --params`로 출시 모델과 시험 전 학습 상태로 돌린다. 녹화는 오프라인으로 비교한다.

## 알려진 약점

- **주간 경로의 먼 앞차**: 주간 경로 10-05--16-28(터널 포함)에서 헛 lead가 출시 모델보다 많다(13.6 vs 6.0%).
  100 m 넘는 차를 짧게 보고 앞차로 잡는다. 크루즈가 앞차 거리를 보는 비전 크루즈 동작에 영향을 줄 수 있다.
- **앞차 판별력**: AUC가 0.9767로 출시 모델(0.9772)보다 조금 낮다.
- **pose 잡음**: 위 표대로 회전 잡음이 출시 모델보다 크다.
