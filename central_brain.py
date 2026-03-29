from __future__ import annotations

import uuid

from app.ann.predictor import AnnPredictor
from app.commands.planner import build_command_package
from app.core.schemas import OptimizeRequest, OptimizeResponse


class CentralBrain:
    """Coordinates intent -> ANN inference -> command package."""

    def __init__(self) -> None:
        self.ann_predictor = AnnPredictor()

    def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        session_id = request.session_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        ann = self.ann_predictor.predict(request.target_spec)
        command_package = build_command_package(request, ann, session_id=session_id, trace_id=trace_id)

        return OptimizeResponse(
            status="accepted",
            session_id=session_id,
            trace_id=trace_id,
            current_stage="planning_commands",
            ann_prediction=ann,
            command_package=command_package,
        )
