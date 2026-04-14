## 1. 로깅 키   
type: state(urdf0)
Key: observation.state.radian_urdf0
Shape: (6,) float32
Dim 의미: [shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper]
단위: URDF 0° radian
────────────────────────────────────────
type: action(urdf0)
Key: action.radian_urdf0
Shape: (6,) float32
Dim 의미: 동일 (target)
단위: URDF 0° radian

공식 (6 dim 공통): rad = ((norm - offset_normalized)
/ 100) * half_range_radians

## 배포시 고려사항
배포 시 반드시 "타겟 로봇 자신의" calibration 으로 변환 
— 소스 로봇 calibration 을 쓰면 homing offset 차이만큼 모든 모션이 어긋남. 
Policy 는 URDF radian 만 알고, "어느 로봇인지"는 변환 단계 부재.

● 배포 과정 핵심                                       
                                               
  필요한 2가지 파일                                    
   
  ┌────────────────┬───────────────────────────────┐   
  │      파일      │             역할                │ 
  ├────────────────┼───────────────────────────────┤   
  │                │ 링크 길이·조인트 축·FK 체인        │ 
  │ URDF (타겟     │ 정의. 학습 로봇과 동일 URDF        │
  │ 로봇)          │ 여야 함 (기하학적 호환성            │   
  │                │ 보장)                          │
  ├────────────────┼───────────────────────────────┤   
  │ Calibration    │ range_min/max, homing_offset, │   
  │ JSON (타겟     │  drive_mode — 각 개체 고유의  │
  │ 로봇)          │ encoder 편차 흡수             │   
  └────────────────┴───────────────────────────────┘ 

  Calibration이 제공하는 2개의 변환 파라미터          
   
  load_calibration_limits() +                          
  load_gripper_radian_params() 가 JSON 을 읽어 계산: 
                                                       
  - half_range_radians[i] = (range_max - range_min) /  
  4096 × 360° / 2 (radian) — 각 joint 물리 가동 범위의
  절반                                                 
  - offset_normalized[i] = URDF 0° 가 normalized 공간
  어디에 위치하는지 (drive_mode 반전 포함)             
   
  핵심 변환 공식 (양방향)                              
                                                     
  radian     = ((normalized - offset_normalized) / 100)
   × half_range_radians                                
  normalized = (radian / half_range_radians) × 100 +
  offset_normalized                                    
                                                     
  이 공식이 하드웨어 차이를 흡수하는 지점. 같은 URDF   
  radian 값이 로봇 A·B 에서 서로 다른 normalized 값으로
   매핑됨 — 바로 이 매핑 차이가 각 로봇의 개체 편차.   
                                                     
  제어 tick 당 수행 단계                               
   
  [서보 encoder]                                       
       │  Feetech state read (normalized)                    
       ↓
  [state_norm, 6D]                                     
       │  ← 타겟 calibration 정변환                    
       │    arm[:5]:                                   
  target_calib.normalized_to_radians()                 
       │    gripper:  ((norm - g_offset)/100) * g_half 
       ↓                                               
  [state_rad, 6D]  ─── URDF 0° 기준 radian (학습 분포와
   일치)                                               
       │                                               
       ↓  policy.predict() -> action(radian)                             
       │                                               
  [action_rad, 6D] ─── URDF 0° 기준 radian
       │  ← 타겟 calibration 역변환                    
       │    arm[:5]:                                 
  target_calib.radians_to_normalized()                 
       │    gripper:  (rad / g_half) * 100 + g_offset
       ↓                                               
  [action_norm, 6D] → clip(-100, +100) → Feetech write