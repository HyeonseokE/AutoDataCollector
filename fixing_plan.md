# Fixing Plan

## 1. Gemini 3 Flash Thinking Budget 제한

### 문제
- `gemini-3-flash-preview`가 thinking에 62K+ 토큰을 소비하고 실제 응답 0 토큰 (빈 응답)
- 294초 소요 후 에러 발생 (`content has no parts`)
- `GenerationConfig`에 `thinking_config` 미설정 → thinking 토큰 무제한

### 해결
- `gemini_chat_start()`에서 `GenerationConfig`에 thinking budget 설정
- 또는 모델을 `gemini-2.0-flash`로 변경 (thinking 없음)

### 파일
- `code_gen_lerobot/llm_utils/gemini.py` — `GenerationConfig`에 `thinking_config` 추가
- `pipeline_config/paid_api_config.yaml` — 모델명 변경 (선택)

---

## 2. 프롬프트 패턴 구조 통일 (START/PREPARE/EXECUTE/END)

### 문제
- 현재 패턴이 스킬별 독립 블록으로 나열되어 있어 구조가 불명확
- LLM이 일관된 코드 구조를 생성하지 못함

### 목표 구조
```
# START — move_to_initial_state
# PREPARE <interaction> — gripper open/close + move_to_position (approach)
# EXECUTE <interaction> — execute_pick/place/push/press (core skill)
# END — move_to_free_state
```

For multi-step tasks, repeat PREPARE → EXECUTE for each interaction.

### 변경 방향
- 디테일한 코드 예시 대신 **구조 규칙**만 간결하게 명시
- 기존 스킬별 패턴 블록 → 통합된 4단계 구조 설명으로 교체

### 파일
- `code_gen_lerobot/forward_execution/user_prompt.py`
  - Turn 1: `lerobot_code_gen_prompt()` — "6. Skill Composition Patterns" 섹션 + "7. Executable Code Skeleton" 섹션
  - Turn 3: `turn3_code_gen_prompt()` — "Skill Composition Patterns" 섹션
