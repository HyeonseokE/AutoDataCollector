# Phase2 데이터 수집 셋업 가이드 — fresh clone 부터

새 머신에 리포를 clone 한 상태에서 Method3 **Phase2 (MI-based selection)** 데이터
수집까지 가는 전 과정. 이 문서는 *환경 준비 + 실행 순서* 를 다룬다.

- 이론/설계: `ours_method/final_method3_spec.md` (§6 re-embedding, §7-14 MI selection)
- 모듈 구조: `method3/README.md`
- **기존(옛 paradigm) 세션을 phase2 로 전환**하는 경우: 이 문서 대신
  `docs/METHOD3_PARADIGM_MIGRATION.md` 의 Step 0~4 를 따를 것.

## 0. 요약 체크리스트

| # | 항목 | 출처 |
|---|---|---|
| 1 | `robot_configs/` — git 에 포함됨. 단 **머신별 값 재확인 필요** (§2) | tracked |
| 2 | VLA 체크포인트 — phase1 데이터로 학습해 생성 (§5) | 재생성 |
| 3 | Phase1 데이터셋 — phase1 수집으로 생성 (§4) | 재생성 |
| 4 | API config — 템플릿 복사 후 수정 (§1) | tracked (template) |
| 5 | API 키 — `google_aistudio_key.json` 수동 배치 (§1) | **수동** |
| 6 | Phase1 세션 산출물 — phase1 실행이 만듦, `RESUME_SESSION` 으로 지정 (§6) | 재생성 |

git 에 없는 것은 전부 "비밀(5)" 아니면 "그 환경에서 생성(2·3·6)" 이다.
추가로 올려야 할 config/yaml 은 없다 (2026-08 기준 전수 확인).

## 1. 환경 + API 설정

```bash
conda activate lerobot_cap        # pyrealsense2 포함. lerobot 은 PYTHONPATH 로 잡힘
                                  # (run_*.sh 가 자동 export — PARALLEL_TASKS.md §3(c))
```

**API config** — 실제 파일은 gitignore. 템플릿을 복사해 채운다:

```bash
cp pipeline_config/paid_api_config.yaml.template pipeline_config/paid_api_config.yaml
cp pipeline_config/free_api_config.yaml.template pipeline_config/free_api_config.yaml   # vLLM 서버 쓸 때만
```

**API 키** — codegen/judge 가 gemini 모델이므로 필수. 둘 중 하나:

```bash
# (a) 리포 루트에 JSON 파일
echo '{"api_key": "<YOUR_GOOGLE_AISTUDIO_KEY>"}' > google_aistudio_key.json
# (b) 환경변수 (파일보다 우선)
export GOOGLE_API_KEY=<YOUR_KEY>
```

OpenAI 모델을 쓰면 같은 요령으로 `openai_api_key.json` 또는 `OPENAI_API_KEY`.

## 2. 로봇/카메라 — 머신별 값 확인

`robot_configs/` 는 git 에 있지만 **커밋된 값은 원본 머신 기준**이다. 새 머신에서:

1. **시리얼 포트**: `robot_configs/robot/so101_robot<N>.yaml` 의 `port:` 를
   `/dev/serial/by-id/...` 로 (ACM 번호는 재부팅에 흔들림). 현재 연결 확인:
   ```bash
   ls -l /dev/serial/by-id/
   ```
2. **모터 캘리브레이션**: arm 이 다르면 `run_calibration.sh` 로 재캘리브레이션
   (`robot_configs/motor_calibration/so101/robot<N>_calibration.json` 갱신).
3. **RealSense 시리얼**: `pipeline_config/recording_config_ws<N>.yaml` 의
   `cameras.shared[].serial_number`. 연결된 카메라 조회:
   ```bash
   python -c "
   import pyrealsense2 as rs
   for d in rs.context().query_devices():
       print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name))"
   ```
   config 에 넣는 값은 이 명령이 출력하는 librealsense 시리얼이다
   (`lsusb`/sysfs 의 USB 디스크립터 시리얼과 다름 — 혼동 주의).
4. **pix2robot / charuco 행렬**: 카메라·로봇 배치가 바뀌었으면
   `pix2robot_charuco_calibrator/` 로 재보정.

## 3. Phase1 수집 (phase2 의 입력을 만드는 단계)

phase2 는 빈 상태에서 시작할 수 없다 — phase1 세션의 buffer 를 이어받는다.

`run_forward_and_reset_ws<N>.sh` 에서:

```bash
PHASE="phase1"
RECORD_DATASET=true      # ★ 필수 — 여기서 만든 dataset 이 §5 학습 + VDB re-embedding 입력
RESUME_SESSION=""        # 새 세션
```

`RECORD_DATASET=true` 가 핵심이다. recording config 의 `dataset_repo_id` 가
phase2 config 의 `phase1_dataset_path` 와 **일치해야** 나중에 vector DB 재구축이
그 dataset 을 찾는다 (예: `CoRL2026/table1/ours_open_drawer`).

> 참고: 이 repo_id 형식은 슬래시 2개라 HF Hub 업로드가 불가능한 이름이다.
> 로컬 캐시(`~/.cache/huggingface/lerobot/`) 운용 전제. HF 배포로 전환하려면 개명 필요.

산출물:
- LeRobot dataset → `~/.cache/huggingface/lerobot/<dataset_repo_id>/`
- 세션 디렉토리 → `results/session_<timestamp>/` (subgoal_buffer.npz 포함)

phase1 종료 판정(B₁_min/max, readiness)은 `pipeline_config/phase1_config.yaml` 참조.

## 4. VLA 체크포인트 학습

phase1 dataset 으로 DCT-tuned SmolVLA 를 학습한다:

```bash
bash lerobot/scripts/train_DCT_smolvla.sh   # DATASET_REPO_ID 등은 스크립트 상단에서 설정
```

학습 결과 경로를 phase2 config 의 SoT 키에 기록:

```yaml
# pipeline_config/phase2_config_<task>.yaml
phase1_trained_vla_path: "lerobot/outputs/train/<job>/checkpoints/<step>/pretrained_model"
```

이 키가 preselective filter / re-embedding / DCT denoise 가 공유하는 **유일한
체크포인트 출처**다 (server 쪽에 따로 적지 않는다 — mismatch 방지).

## 5. phase2 config 심링크 전환

`pipeline_config/phase2_config.yaml` 은 task 별 config 를 가리키는 심링크다:

```bash
ln -sfn phase2_config_open_drawer.yaml pipeline_config/phase2_config.yaml
ls -l pipeline_config/phase2_config.yaml   # → phase2_config_open_drawer.yaml 확인
```

## 6. (gRPC 모드) 서버 준비

`skill_planner_transport.mode: grpc` 인 경우. 연결 정보는 phase2 config 의
`remote:` 섹션이 SoT 이고, launcher 가 알아서 읽는다:

```bash
bash grpc_server/launch_remote_server.sh        # 서버 tmux + SSH tunnel
bash grpc_server/launch_remote_server.sh stop   # 정리
```

서버 머신에 필요한 것: **동일 리포 + §4 체크포인트** (config `remote.project_path`
경로에). P_phase1 vector DB 는 client 가 로컬 빌드 후 서버 cache 로 **자동 scp**
하므로 (옵션 C, 2026-05-21) 서버에서 따로 만들 필요 없다.
`mode: local` 이면 이 절은 통째로 건너뛴다.

## 7. Phase2 launch

`run_forward_and_reset_ws<N>.sh`:

```bash
PHASE="phase2"
RESUME_SESSION="./results/session_<phase1-세션>"   # §3 에서 만든 세션
```

실행하면 `_setup_phase2_session` 이:

1. `<session>/skill_wise_vector_db.npz` 캐시 hit → 그대로 로드
2. miss → `phase1_trained_vla_path` + `phase1_dataset_path` 로 §6 re-embedding
   자동 수행 → VDB 생성·저장 (→ grpc 모드면 서버로 scp)
3. `<session>/subgoal_buffer.npz` 를 MI selection reference 로 사용

phase2 에피소드는 `<session>/phase2/episode_*` 아래에 쌓인다.

## 트러블슈팅

| 증상 | 원인/해법 |
|---|---|
| `ModuleNotFoundError: lerobot.utils` | PYTHONPATH 누락 — `run_*.sh` 경유로 실행할 것 |
| codegen 첫 호출에서 인증 에러 | §1 API 키 미배치 (`google_aistudio_key.json` / `GOOGLE_API_KEY`) |
| VDB 재구축이 dataset 못 찾음 | `phase1_dataset_path` ≠ 수집 때 `dataset_repo_id`, 또는 phase1 을 `RECORD_DATASET=false` 로 돌렸음 |
| phase2 가 subgoal buffer 없다고 함 | `RESUME_SESSION` 미지정/오타. forward 로그만 있으면 `scripts/rebuild_subgoal_buffer_from_forward_log.py` 로 복원 |
| 로봇이 엉뚱한 자세로 감 | 다른 arm 에 옛 캘리브레이션 사용 중 — §2-2 재캘리브레이션 |
