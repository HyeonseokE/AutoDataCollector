# Joint Recording: 엔코더 Raw → Normalized → Radian 변환 과정

## 1. Raw → Normalized (observation.state, action)

```
엔코더 raw 값 (예: 2067)
      │
      ▼
 clamp [range_min, range_max]     ← 예: [1128, 3006] 밖이면 잘라냄
      │
      ▼
 bounded_val (예: 2067)
      │
      ▼
 ((bounded_val - range_min) / (range_max - range_min)) * 200 - 100
 ((2067 - 1128) / (3006 - 1128)) * 200 - 100
 = (939 / 1878) * 200 - 100
 = 0.0 (범위 정중앙)
      │
      ▼
 drive_mode == 1 이면 부호 반전
      │
      ▼
 normalized 값: 0.0  ← observation.state / action에 저장
```

## 2. Normalized → 캘리브레이션 기준 Radian (observation.radian.state, observation.radian.action)

```
normalized 값 (예: 50.0)
      │
      ▼
 (normalized / 100.0) * half_range_radians
 (50.0 / 100.0) * 1.625
 = 0.8125 rad
      │
      ▼
 radian 값: 0.8125  ← observation.radian.state / observation.radian.action에 저장
```

- 기준점: 캘리브레이션 중앙(normalized=0.0) = 0.0 rad
- homing offset 보정 없음 (URDF 0°와는 다름)
- half_range_radians = 관절별 가동 범위의 절반 (캘리브레이션에서 산출)
- gripper: `(normalized / 100.0) * π`

## 3. Normalized → URDF 기준 Radian (observation.radian.state_urdf0, observation.radian.action_urdf0)

```
normalized 값 (예: 50.0)
      │
      ▼
 corrected = normalized - offset_normalized
           = 50.0 - (-2.0) = 52.0
      │
      ▼
 (corrected / 100.0) * half_range_radians
 = (52.0 / 100.0) * 1.625
 = 0.845 rad
      │
      ▼
 radian 값: 0.845  ← observation.radian.state_urdf0 / observation.radian.action_urdf0에 저장
```

- 기준점: URDF 0° (엔코더 2048) = 0.0 rad
- offset_normalized = ((2048 - range_min) / (range_max - range_min)) * 200 - 100
- offset_normalized = URDF 0° (엔코더 2048)가 calibration기준 normalization값으로 얼마인지.
- 관절마다 offset이 다름 (예: shoulder_pan=-2.0, elbow_flex=11.3)
- IK/FK 계산에 사용되는 것과 동일한 라디안 값

## 정리

| 단계 | 값 | 기준점 (0의 의미) | 저장 feature |
|------|-----|-------------------|-------------|
| raw | 0 ~ 4095 | - | (저장 안 됨) |
| normalized | -100 ~ +100 | 캘리브레이션 range 중앙 | observation.state, action |
| radian (calib) | -half_range ~ +half_range | 캘리브레이션 중앙 = 0 rad | observation.radian.state, .action |
| radian (urdf) | 가변 | URDF 0° (엔코더 2048) = 0 rad | observation.radian.state_urdf0, .action_urdf0 |

## 설정 (recording_config.yaml)

```yaml
observation_features:
  radian:
    state: true          # observation.radian.state (캘리브레이션 중앙 기준)
    action: true         # observation.radian.action (캘리브레이션 중앙 기준)
    state_urdf0: true    # observation.radian.state_urdf0 (URDF 0° 기준)
    action_urdf0: true   # observation.radian.action_urdf0 (URDF 0° 기준)
```
