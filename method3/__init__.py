"""Method3 — Pre-selective Real-world Data Acquisition (문서 final_method3_spec).

현재 buffer 가 충분히 커버하지 못한 state/action region 을 확장하면서도, 유사
state 에서 action ambiguity 를 과도하게 키우지 않는 trajectory 를 선별 수집하는
two-phase online acquisition 파이프라인.

하위 폴더:
    phase1_state_seeding/  — Phase1: subgoal diversity 로 state coverage H(S)↑
                              (§4-5). skill-wise subgoal buffer B_{g,t}^{(m)}.
    phase2_mi_selection/   — Phase2: full vector DB B_t^{(m)} 기준 MI-style score
                              Q2 = β·ΔH_A − λ·ΔH_A|S 로 후보 선별 (§7-14).
    storage/               — §3: raw trajectory dataset (source of truth).
    reembedding/           — §6: Phase1 raw 를 re-embedding → P_phase1 seed DB.
    phase_control/         — §15: phase saturation & adaptive transition.
    acquisition/           — §2: 위 모듈을 묶는 two-phase acquisition 오케스트레이터.

자세한 명세는 ours_method/final_method3_spec.md 및 ours_method/details/ 참고.
"""
