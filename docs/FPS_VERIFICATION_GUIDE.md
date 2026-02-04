# LeRobot Dataset FPS 검증 가이드

데이터 취득이 정말로 30fps로 진행되었는지 확인하는 모든 방법을 설명합니다.

## 현재 상태 요약

**최근 실행 결과 (session_20260112_120104):**
- 목표 FPS: 30 fps
- 실제 effective FPS: **11.65 ± 3.03 fps**
- 편차: **61.2%** (POOR)

**원인:**
- Control Hz (50Hz)와 Target FPS (30fps)의 불일치로 인한 프레임 스키핑
- 카메라 캡처 지연
- 시스템 리소스 병목 현상

---

## 검증 방법 1: 자동 검증 스크립트 (추천)

가장 빠르고 포괄적인 방법입니다.

### 사용법

```bash
# 기본 실행 (자동으로 최근 데이터셋과 세션 찾기)
python scripts/verify_dataset_fps.py

# 특정 데이터셋 지정
python scripts/verify_dataset_fps.py \
  --dataset-path ~/.cache/huggingface/lerobot/local/pnp_15epis_original

# 특정 실행 세션 지정
python scripts/verify_dataset_fps.py \
  --session-dir results/session_20260112_120104

# 둘 다 지정
python scripts/verify_dataset_fps.py \
  --dataset-path ~/.cache/huggingface/lerobot/local/pnp_15epis_original \
  --session-dir results/session_20260112_120104

# 결과를 JSON으로 저장
python scripts/verify_dataset_fps.py \
  --session-dir results/session_20260112_120104 \
  --output fps_verification_results.json
```

### 검증 항목

1. **Metadata (info.json)**: 설정된 FPS 값 확인
2. **Parquet timestamps**: 실제 데이터의 타임스탬프 분석으로 FPS 계산
3. **Video files**: ffprobe로 저장된 비디오 파일의 FPS 확인
4. **Execution logs**: 실행 중 기록된 effective FPS 추출

---

## 검증 방법 2: 실행 로그 수동 확인

가장 직접적이고 간단한 방법입니다.

### 단계

```bash
# 1. 최근 세션 찾기
ls -lt results/ | head -5

# 2. 특정 에피소드의 로그 확인
grep -i "effective FPS\|recorded frames" results/session_20260112_120104/episode_*/forward/forward_log.txt

# 3. 모든 에피소드의 FPS 요약
grep "effective FPS:" results/session_20260112_120104/episode_*/forward/forward_log.txt
```

### 예시 출력

```
results/session_20260112_120104/episode_01/forward/forward_log.txt:[Recording] Recorded 572 frames, effective FPS: 13.0
results/session_20260112_120104/episode_02/forward/forward_log.txt:[Recording] Recorded 575 frames, effective FPS: 13.0
results/session_20260112_120104/episode_03/forward/forward_log.txt:[Recording] Recorded 587 frames, effective FPS: 13.0
```

### 해석

- **effective FPS**: 실제 레코딩된 FPS (타겟 30fps 대비 실제 달성한 값)
- **Recorded frames**: 저장된 프레임 수
- **Skipped frames**: FPS 동기화를 위해 건너뛴 프레임 수

---

## 검증 방법 3: 메타데이터 파일 확인

데이터셋 설정값을 확인합니다.

```bash
# info.json 확인
cat ~/.cache/huggingface/lerobot/local/pnp_15epis_original/meta/info.json | jq '.fps, .total_episodes, .total_frames'
```

### 예시 출력

```json
30
15
8580
```

이 값은 **설정값**이지 **실제 달성한 FPS**가 아닙니다.

---

## 검증 방법 4: Parquet 파일 타임스탬프 분석 (가장 정확)

실제 저장된 데이터의 타임스탬프를 분석하여 FPS를 계산합니다.

### Python 스크립트

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Parquet 파일 읽기
dataset_path = Path.home() / ".cache/huggingface/lerobot/local/pnp_15epis_original"
parquet_files = list(dataset_path.glob("data/**/*.parquet"))

for pf in parquet_files:
    df = pd.read_parquet(pf)

    # 에피소드별 FPS 계산
    for episode_idx in df['episode_index'].unique():
        ep_df = df[df['episode_index'] == episode_idx]
        timestamps = ep_df['timestamp'].values

        # 연속 프레임 간 시간 간격
        time_diffs = np.diff(timestamps)
        avg_interval = np.median(time_diffs[time_diffs > 0])

        # FPS = 1 / interval
        fps = 1.0 / avg_interval

        print(f"Episode {episode_idx}: {fps:.2f} fps ({len(ep_df)} frames)")
```

---

## 검증 방법 5: 비디오 파일 FPS 확인

저장된 비디오 파일의 FPS를 ffprobe로 확인합니다.

### 명령어

```bash
# ffprobe 설치 (필요시)
sudo apt install ffmpeg

# 비디오 FPS 확인
find ~/.cache/huggingface/lerobot/local/pnp_15epis_original/videos -name "*.mp4" | head -5 | while read video; do
  echo "File: $(basename $video)"
  ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video"
done
```

### 예시 출력

```
File: chunk-000-file-000.mp4
30/1
```

`30/1` = 30 fps

---

## 검증 방법 6: RecordingContext 통계 확인

실행 중 출력되는 RecordingContext 통계를 확인합니다.

### 실행 중 출력 예시

```
[RecordingContext] Setup complete
  Target FPS: 30, Control Hz: 50
  Frame skip ratio: 1.67
  Cameras: ['realsense', 'innomaker']

[Recording] Recorded 572 frames, effective FPS: 13.0
  Recorded frames: 572
  Skipped frames: 381
```

### 해석

- **Target FPS**: 목표 레코딩 FPS (설정값)
- **Control Hz**: 제어 루프 주파수
- **Frame skip ratio**: Control Hz / Target FPS = 50/30 = 1.67
  - 약 1.67번의 제어 루프마다 1프레임을 레코딩
- **Recorded frames**: 실제 저장된 프레임 수
- **Skipped frames**: FPS 동기화를 위해 건너뛴 제어 스텝 수
- **Effective FPS**: 실제 달성한 FPS

---

## 검증 방법 7: 세션 요약 JSON 확인

전체 세션의 요약 정보를 확인합니다.

```bash
# 세션 요약 파일 확인
cat results/session_20260112_120104/session_summary.json | jq '.summary, .episodes[] | select(.episode <= 3)'
```

---

## FPS가 낮은 원인과 해결 방법

### 원인 분석

1. **Control Hz와 Target FPS 불일치**
   - Control Hz: 50 Hz (20ms마다 제어)
   - Target FPS: 30 fps (33.3ms마다 프레임)
   - Frame skip ratio: 1.67 (일부 제어 스텝은 스킵됨)

2. **카메라 캡처 지연**
   - 멀티 카메라 동기화 지연
   - OpenCV `async_read()` timeout

3. **시스템 리소스 부족**
   - CPU 병목
   - 디스크 I/O 병목
   - CUDA OOM (GPU 메모리 부족)

### 해결 방법

#### 방법 1: Control Hz 낮추기 (추천)

`record_dataset/context.py` 수정:

```python
# execution_forward_and_reset.py
RecordingContext.setup(
    recorder=self.dataset_recorder,
    camera_manager=recording_camera,
    target_fps=30,
    control_hz=30,  # 50 → 30으로 변경
)
```

또는 `skills/skills_lerobot.py`의 `time.sleep(0.02)` 조정:

```python
# 0.02초 (50Hz) → 0.033초 (30Hz)
time.sleep(0.033)
```

#### 방법 2: Target FPS 낮추기

실제 달성 가능한 FPS로 설정:

```yaml
# pipeline_config/recording_config.yaml
recording_fps: 15  # 30 → 15로 낮춤
```

#### 방법 3: 카메라 최적화

```yaml
# pipeline_config/recording_config.yaml
cameras:
  - name: "realsense"
    width: 320      # 640 → 320 (해상도 낮춤)
    height: 240     # 480 → 240
    fps: 30
```

#### 방법 4: 시스템 최적화

```bash
# GPU 메모리 단편화 방지
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU governor 설정
sudo cpupower frequency-set -g performance

# SSD 사용 (HDD 대신)
# 데이터셋을 빠른 SSD에 저장
```

---

## 빠른 체크리스트

실행 후 이 순서로 확인하세요:

1. ✅ **실행 로그 확인**
   ```bash
   grep "effective FPS" results/session_*/episode_*/forward/forward_log.txt | tail -15
   ```

2. ✅ **자동 검증 스크립트**
   ```bash
   python scripts/verify_dataset_fps.py
   ```

3. ✅ **메타데이터 확인**
   ```bash
   cat ~/.cache/huggingface/lerobot/local/*/meta/info.json | jq '.fps, .total_episodes, .total_frames'
   ```

4. ✅ **Parquet 분석** (데이터가 있는 경우)
   ```bash
   python scripts/verify_dataset_fps.py --dataset-path <path>
   ```

---

## 기대 결과

### 이상적인 경우 (30fps 목표)

```
Target FPS: 30
Actual FPS: 28-30 fps
Deviation: < 10%
Status: ✓ EXCELLENT
```

### 허용 가능한 경우

```
Target FPS: 30
Actual FPS: 25-28 fps
Deviation: 10-15%
Status: ⚠ ACCEPTABLE
```

### 개선 필요한 경우

```
Target FPS: 30
Actual FPS: < 25 fps
Deviation: > 15%
Status: ✗ POOR
```

---

## 추가 도구

### 실시간 FPS 모니터링

실행 중 FPS를 실시간으로 모니터링하려면:

```bash
# 터미널에서 실시간 로그 확인
tail -f results/session_*/episode_*/forward/forward_log.txt | grep --line-buffered "effective FPS"
```

### CSV 내보내기

모든 에피소드의 FPS를 CSV로 저장:

```bash
echo "episode,effective_fps,recorded_frames,skipped_frames" > fps_summary.csv
grep -h "effective FPS:" results/session_20260112_120104/episode_*/forward/forward_log.txt | \
  sed -E 's/.*episode_([0-9]+).*Recorded ([0-9]+) frames, effective FPS: ([0-9.]+).*/\1,\3,\2,0/' >> fps_summary.csv
```

---

## 참고 자료

- LeRobot 공식 문서: https://github.com/huggingface/lerobot
- RecordingContext 구현: `record_dataset/context.py`
- DatasetRecorder 구현: `record_dataset/recorder.py`
- FPS 검증 스크립트: `scripts/verify_dataset_fps.py`
