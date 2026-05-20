"""method3.phase2_server_inference — H100 gRPC transport client + demo ingest.

Phase2 의 candidate generation + selection 을 원격 H100 server 에서 돌릴 때 쓰는
client-side adapter 들. server 본체는 ``preselective_rpc.server`` 에서 booting,
이 패키지는 그것을 client 측에서 *호출* 하는 thin 어댑터만 담는다.

  - ``grpc_planner_adapter.GrpcPlannerClient``
       skills_lerobot 의 plan_batch contract 를 만족하도록 server.PlanAndSelect
       를 1-element list 로 wrapping 하는 어댑터.
  - ``demo_ingest`` (helper 모듈)
       client 가 server.IngestEpisode 스트림을 만들 때 LeRobot dataset 을 frame
       단위로 읽기 위한 helper. (``all_episode_indices``, ``open_chunked_dataset``)

History: ``preselective_filter.integration.{grpc_planner_adapter, demo_ingest}``
에서 이동 (2026-05-20). 옛 IG·AC selector / score_metric / FAISS buffer 는
method3 의 phase2 useful-OOD selector 로 대체됐다.
"""
from method3.phase2_server_inference.grpc_planner_adapter import GrpcPlannerClient

__all__ = ["GrpcPlannerClient"]
