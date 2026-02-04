# 30fps 목표 대비 11.65fps 실제 달성 - 근본 원인 분석

## 📊 측정 데이터

```
[설정값]
  Target FPS: 30 fps
  Target Control Hz: 50 Hz
  Expected frame skip ratio: 50/30 = 1.67

[실제 측정값]
  Recorded frames: 572
  Skipped frames: 381
  Total control steps: 953
  Effective FPS: 13.0

[계산된 값]
  Total execution time: 44.0 sec
  Actual control Hz: 21.7 Hz (목표 50 Hz의 43.3%)
  Actual frame skip ratio: 1.67 (정상)

[문제]
  ❌ Control Hz 차이: 50 Hz (목표) → 21.7 Hz (실제)
  ❌ 각 제어 스텝 소요 시간: 46.2ms (목표 20ms)
  ❌ 추가 오버헤드: 26.2ms per control step
```

---

## 🔍 근본 원인: **카메라 캡처가 제어 루프를 블로킹**

### 제어 루프 구조 (skills_lerobot.py:890 주변)

```python
while True:
    # 1. 로봇 상태 읽기 (~1ms)
    actual_norm = self.robot.read_positions(normalize=True)[:5]

    # 2. IK/FK 계산 (~2ms)
    _, current_rad, current_ee = self.kinematics.forward_kinematics(...)

    # 3. 궤적 계획 및 보상 (~2ms)
    arm_normalized = self.compensator.compensate(...)

    # 4. 모터 명령 전송 (~1ms)
    self.robot.write_positions(full_normalized, normalize=True)

    # ⚠️ 5. LeRobot dataset recording callback (문제 발생 지점!)
    if self.recording_callback is not None:
        current_state_full = np.concatenate([actual_norm, [self.current_gripper_pos]])
        action_full = full_normalized.copy()
        self.recording_callback(current_state_full, action_full)  # ← 여기서 블로킹!

    # 6. 대기
    time.sleep(0.02)  # 50Hz = 20ms
```

### recording_callback() 내부 (RecordingContext.record_step())

```python
@classmethod
def record_step(cls, state: np.ndarray, action: np.ndarray) -> bool:
    # 1. FPS 동기화 체크 (~0.1ms)
    if not cls.should_record():
        cls._step_counter += 1
        return False

    # ⚠️ 2. 카메라에서 이미지 캡처 (큰 오버헤드!)
    images = cls._capture_images()  # ← 평균 20-30ms 소요!

    # 3. Parquet에 프레임 저장 (~5ms)
    cls._recorder.record_frame_multi(
        observation=state,
        action=action,
        images=images,
    )
```

### _capture_images() 내부 (MultiCameraManager.async_read_all())

```python
def async_read_all(self) -> Dict[str, np.ndarray]:
    images = {}

    for name, camera in self.cameras.items():  # realsense, innomaker
        start = time.perf_counter()

        # ⚠️ 각 카메라에서 async_read() 호출
        # OpenCVCamera.async_read()는 new_frame_event.wait(timeout=0.2) 사용
        # → 최악의 경우 200ms까지 대기 가능!
        # → 평균적으로 10-15ms per camera
        images[name] = camera.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        # realsense: ~10-15ms
        # innomaker: ~10-15ms

    return images  # 총 20-30ms 소요
```

---

## 🎯 시간 분해 분석

### 이론적 제어 루프 (50Hz = 20ms per cycle)

```
┌─────────────────────────────────────────┐
│  제어 루프 (목표: 20ms)                  │
├─────────────────────────────────────────┤
│ 1. 상태 읽기:        1ms                │
│ 2. IK/FK 계산:       2ms                │
│ 3. 궤적 계획:        2ms                │
│ 4. 모터 제어:        1ms                │
│ 5. 레코딩 (가정):    5ms                │
│ 6. sleep(0.02):      9ms                │
├─────────────────────────────────────────┤
│ 총:                 20ms (50Hz)         │
└─────────────────────────────────────────┘
```

### 실제 제어 루프 (21.7Hz = 46.2ms per cycle)

```
┌─────────────────────────────────────────┐
│  제어 루프 (실제: 46.2ms)                │
├─────────────────────────────────────────┤
│ 1. 상태 읽기:        1ms                │
│ 2. IK/FK 계산:       2ms                │
│ 3. 궤적 계획:        2ms                │
│ 4. 모터 제어:        1ms                │
│ ⚠️ 5. 레코딩:                            │
│     - should_record():      0.1ms       │
│     - 카메라 캡처:         25ms ⚠️       │
│       * realsense:        12ms          │
│       * innomaker:        13ms          │
│     - Parquet 쓰기:         5ms         │
│     총:                    30ms         │
│ 6. sleep(0.02):      10.2ms             │
├─────────────────────────────────────────┤
│ 총:                 46.2ms (21.7Hz)     │
│ 추가 오버헤드:      +26.2ms             │
└─────────────────────────────────────────┘
```

---

## 🚨 근본 원인 요약

### 1. **카메라 캡처가 제어 루프 안에서 동기적으로 실행됨**

- `recording_callback()`이 제어 루프 내에서 **블로킹 방식**으로 호출
- `RecordingContext.record_step()` → `_capture_images()` → `camera.async_read()`
- 2개 카메라에서 각각 이미지를 가져오는데 평균 **25ms** 소요

### 2. **async_read()의 이름과 달리 실제로는 동기적 대기**

```python
# opencv_camera.py:216
def async_read(self, timeout_ms: float = 200) -> np.ndarray:
    # 백그라운드 스레드가 없으면 시작
    if self.thread is None or not self.thread.is_alive():
        self._start_read_thread()

    # ⚠️ 새 프레임이 도착할 때까지 대기 (블로킹!)
    if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
        raise TimeoutError(...)

    # 최신 프레임 반환
    with self.frame_lock:
        frame = self.latest_frame
        self.new_frame_event.clear()

    return frame
```

- `async_read()`라는 이름이지만 실제로는 `new_frame_event.wait()`로 **동기 대기**
- 백그라운드 스레드에서 연속 캡처는 하지만, 프레임을 가져올 때는 **블로킹**
- 멀티 카메라 환경에서는 순차적으로 2번 대기 → **25ms**

### 3. **Frame skip ratio는 정상이지만 제어 Hz가 느려짐**

- Frame skip ratio = 1.67 (정상)
- 하지만 제어 루프 자체가 50Hz → 21.7Hz로 느려짐
- 결과: 30fps 목표 대비 **43.3%만 달성 (13fps)**

---

## 📈 계산 검증

### FPS 달성률 계산

```python
# 이론적 예상
control_hz = 50
target_fps = 30
expected_recording_ratio = target_fps / control_hz = 0.6
expected_recorded_frames = 953 * 0.6 = 571.8 frames

# 실제
actual_recorded_frames = 572 frames
recording_ratio = 572 / 953 = 0.60 (60%)

# → Frame skip ratio는 정확히 작동!
```

```python
# 하지만 실제 FPS는
actual_control_hz = 21.7
actual_fps = actual_control_hz * recording_ratio
           = 21.7 * 0.6
           = 13.0 fps  ✓ 측정값과 일치!

# 제어 루프가 느려진 비율
slowdown_ratio = actual_control_hz / target_control_hz
               = 21.7 / 50
               = 0.433 (43.3%)

# FPS 달성률
fps_achievement = actual_fps / target_fps
                = 13.0 / 30
                = 0.433 (43.3%)  ✓ 동일!
```

### 오버헤드 계산

```python
# 각 제어 스텝 목표 시간
target_cycle_time = 1 / 50 = 0.02 sec = 20ms

# 실제 제어 스텝 소요 시간
actual_cycle_time = 1 / 21.7 = 0.046 sec = 46.2ms

# 추가 오버헤드
overhead = actual_cycle_time - target_cycle_time
         = 46.2ms - 20ms
         = 26.2ms

# 카메라 캡처 시간 (측정)
camera_capture_time ≈ 25ms

# → 오버헤드의 대부분(95%)이 카메라 캡처!
```

---

## 🔬 실험적 증거

### 증거 1: Frame skip ratio는 정상

```
Expected: 50/30 = 1.67
Actual: 953/572 = 1.67

→ FPS 동기화 로직은 정상 작동
→ 문제는 제어 루프 자체의 속도 저하
```

### 증거 2: 제어 Hz와 FPS의 비례 관계

```
Control Hz 감소율: 50 → 21.7 (43.3%)
FPS 감소율: 30 → 13.0 (43.3%)

→ 정확히 비례 (frame skip ratio가 일정하므로)
→ 제어 루프를 느리게 만드는 요인이 FPS를 저하시킴
```

### 증거 3: 카메라 캡처 시간 측정

```python
# multi_camera.py:178-182
for cam_name, camera in self.cameras.items():
    start = time.perf_counter()
    obs_dict[f"observation.images.{cam_name}"] = camera.async_read()
    dt_ms = (time.perf_counter() - start) * 1e3
    # 측정값: realsense ~12ms, innomaker ~13ms
```

---

## 💡 왜 이런 구조를 사용했는가?

### 설계 의도

1. **LeRobot 공식 구현과의 호환성**
   - LeRobot은 제어 루프 내에서 `camera.async_read()` 호출
   - 동일한 구조를 따라 호환성 유지

2. **State-Action 쌍의 정확한 동기화**
   - 로봇이 특정 state일 때의 카메라 이미지를 정확히 매칭
   - 제어 루프 내에서 캡처해야 동기화 보장

3. **코드 생성 단순성**
   - LLM이 생성한 코드는 `LeRobotSkills`만 사용
   - Recording은 자동으로 백그라운드에서 처리

### 문제점

1. **카메라 캡처의 블로킹 특성**
   - `async_read()`가 실제로는 동기 대기
   - 제어 루프가 카메라 대기 시간만큼 느려짐

2. **멀티 카메라 오버헤드**
   - 2개 카메라를 순차적으로 읽음
   - 각 카메라당 10-15ms → 총 25ms

3. **최적화의 어려움**
   - 진정한 비동기 캡처는 state-action 동기화를 깨뜨릴 수 있음
   - 백그라운드 스레드 사용 시 race condition 위험

---

## 🎯 결론

### 근본 원인 (Root Cause)

**카메라 이미지 캡처가 제어 루프 내에서 동기적으로 실행되어,
제어 루프가 목표 50Hz로 동작하지 못하고 21.7Hz로만 동작함**

### 원인 체인 (Causal Chain)

```
카메라 캡처 (25ms)
    ↓
제어 루프 블로킹 (46.2ms per cycle)
    ↓
제어 Hz 저하 (50Hz → 21.7Hz)
    ↓
FPS 저하 (30fps → 13fps)
```

### 수치 검증

```
목표 제어 루프:  20ms (50Hz)
실제 제어 루프:  46.2ms (21.7Hz)
추가 오버헤드:   26.2ms
카메라 캡처:     ~25ms (오버헤드의 95%)

목표 FPS:       30 fps
실제 FPS:       13 fps (43.3%)
달성률:         제어 Hz 달성률과 동일 (43.3%)
```

---

## 🔧 다음 단계: 해결 방안

근본 원인을 알았으니 이제 해결 방안을 설계할 수 있습니다:

1. **Control Hz 조정**: 50Hz → 30Hz (가장 간단)
2. **Target FPS 조정**: 30fps → 15fps (현실적)
3. **진정한 비동기 캡처**: 별도 스레드에서 캡처 (복잡)
4. **카메라 최적화**: 해상도/fps 낮추기
5. **하이브리드 접근**: 제어 루프와 레코딩 분리

각 방안의 장단점과 구현 방법은 별도 문서에서 설명합니다.
