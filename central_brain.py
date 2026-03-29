from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.ann.predictor import AnnPredictor
from app.commands.planner import build_command_package
from app.core.refinement import evaluate_acceptance, refine_prediction
from app.core.schemas import AnnPrediction, OptimizeRequest, OptimizeResponse
from app.core.session_store import SessionStore


class CentralBrain:
    """Coordinates intent -> ANN inference -> command package and iterative refinement."""

    def __init__(self) -> None:
        self.ann_predictor = AnnPredictor()
        self.session_store = SessionStore()

    def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        session_id = request.session_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        ann = self.ann_predictor.predict(request.target_spec)
        command_package = build_command_package(request, ann, session_id=session_id, trace_id=trace_id, iteration_index=0)

        self.session_store.create(
            session_id=session_id,
            trace_id=trace_id,
            request_payload=request.model_dump(mode="json"),
            ann_payload=ann.model_dump(mode="json"),
            command_package=command_package,
            max_iterations=request.optimization_policy.max_iterations,
        )

        return OptimizeResponse(
            status="accepted",
            session_id=session_id,
            trace_id=trace_id,
            current_stage="planning_commands",
            ann_prediction=ann,
            command_package=command_package,
        )

    def process_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        session = self.session_store.load(session_id)

        request = OptimizeRequest.model_validate(session["request"])
        current_iteration = int(session["current_iteration"])
        reported_iteration = int(payload["iteration_index"])
        if reported_iteration != current_iteration:
            raise ValueError(
                f"Feedback iteration mismatch: expected {current_iteration}, got {reported_iteration}"
            )

        current_ann = AnnPrediction.model_validate(session["current_ann_prediction"])
        evaluation = evaluate_acceptance(request, payload)

        history_item: dict[str, Any] = {
            "type": "feedback_evaluation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration_index": current_iteration,
            "feedback": payload,
            "evaluation": evaluation,
        }
        session["history"].append(history_item)

        if bool(evaluation["accepted"]):
            session["status"] = "completed"
            session["current_iteration"] = current_iteration
            self.session_store.save(session_id, session)
            return {
                "status": "completed",
                "session_id": session_id,
                "trace_id": session["trace_id"],
                "accepted": True,
                "iteration_index": current_iteration,
                "evaluation": evaluation,
                "message": "Acceptance criteria met. No further refinement needed.",
            }

        if current_iteration + 1 >= int(session["max_iterations"]):
            session["status"] = "max_iterations_reached"
            self.session_store.save(session_id, session)
            return {
                "status": "stopped",
                "session_id": session_id,
                "trace_id": session["trace_id"],
                "accepted": False,
                "iteration_index": current_iteration,
                "evaluation": evaluation,
                "message": "Max iterations reached before acceptance.",
            }

        next_iteration = current_iteration + 1
        refined_ann = refine_prediction(request, current_ann, evaluation, next_iteration_index=next_iteration)
        next_command_package = build_command_package(
            request,
            refined_ann,
            session_id=session_id,
            trace_id=str(session["trace_id"]),
            iteration_index=next_iteration,
        )

        session["current_iteration"] = next_iteration
        session["status"] = "refining_design"
        session["current_ann_prediction"] = refined_ann.model_dump(mode="json")
        session["current_command_package"] = next_command_package
        session["history"].append(
            {
                "type": "refinement_plan",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration_index": next_iteration,
                "ann_prediction": refined_ann.model_dump(mode="json"),
                "command_package": next_command_package,
            }
        )
        self.session_store.save(session_id, session)

        return {
            "status": "refining",
            "session_id": session_id,
            "trace_id": session["trace_id"],
            "accepted": False,
            "iteration_index": next_iteration,
            "evaluation": evaluation,
            "next_command_package": next_command_package,
            "message": "Generated refined command package for next iteration.",
        }
