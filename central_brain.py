from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, cast

from app.ann.live_retraining import OnlineRetrainingManager
from app.ann.predictor import AnnPredictor
from app.commands.planner import build_command_package
from app.core.feedback_features import derive_feedback_features
from app.core.family_registry import apply_family_profile
from app.core.objectives import build_initial_objective_state, evaluate_objective_state
from app.core.refinement import evaluate_acceptance, refine_prediction_with_strategy
from app.core.schemas import AnnPrediction, OptimizeRequest, OptimizeResponse
from app.core.session_store import SessionStore
from app.core.surrogate_validator import validate_with_surrogate
from app.llm.intent_parser import summarize_user_intent
from app.planning.dynamic_planner import plan_refinement_strategy


class CentralBrain:
    """Coordinates intent -> ANN inference -> command package and iterative refinement."""

    def __init__(self) -> None:
        self.ann_predictor = AnnPredictor()
        self.live_retraining = OnlineRetrainingManager(predictor=self.ann_predictor)
        self.session_store = SessionStore()

    @staticmethod
    def _surrogate_warnings(surrogate: dict[str, Any]) -> list[str]:
        confidence = float(surrogate.get("confidence", 0.0))
        threshold = float(surrogate.get("threshold", 0.0))
        residual = surrogate.get("residual", {})
        freq_error = float(residual.get("center_frequency_abs_error_ghz", 0.0))
        bw_error = float(residual.get("bandwidth_abs_error_mhz", 0.0))
        decision_reason = str(surrogate.get("decision_reason", "unknown"))

        return [
            f"surrogate_confidence={confidence:.3f} (threshold={threshold:.3f})",
            (
                "surrogate_residuals: "
                f"center_frequency_abs_error_ghz={freq_error:.4f}, "
                f"bandwidth_abs_error_mhz={bw_error:.2f}"
            ),
            f"surrogate_decision={decision_reason}",
        ]

    def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        session_id = request.session_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        normalized_request = apply_family_profile(request)
        intent_summary = summarize_user_intent(normalized_request.user_request)
        initial_objective_state = build_initial_objective_state(normalized_request)

        ann = self.ann_predictor.predict(normalized_request)
        surrogate = validate_with_surrogate(normalized_request, ann)

        if not bool(surrogate["accepted"]):
            fallback_behavior = normalized_request.optimization_policy.fallback_behavior

            if fallback_behavior == "require_user_confirmation":
                decision_reason = "surrogate_confidence_below_threshold"
                stop_reason = "requires_user_confirmation"
                self.session_store.create(
                    session_id=session_id,
                    trace_id=trace_id,
                    request_payload=normalized_request.model_dump(mode="json"),
                    ann_payload=ann.model_dump(mode="json"),
                    surrogate_validation=surrogate,
                    command_package=None,
                    max_iterations=normalized_request.optimization_policy.max_iterations,
                    initial_status="clarification_required",
                    stop_reason=stop_reason,
                    decision_reason="surrogate_confidence_below_threshold_requires_confirmation",
                    objective_state=initial_objective_state,
                )
                return OptimizeResponse(
                    status="clarification_required",
                    session_id=session_id,
                    trace_id=trace_id,
                    current_stage="clarification_required",
                    ann_prediction=ann,
                    objective_state=initial_objective_state,
                    warnings=self._surrogate_warnings(surrogate),
                    clarification={
                        "reason": "Surrogate confidence is below safety threshold for automatic execution.",
                        "missing_fields": [],
                        "suggested_questions": [
                            "Do you want to continue with best-effort execution despite low surrogate confidence?",
                            "Can you relax target bandwidth or center-frequency tolerance?",
                        ],
                        "safe_next_step": "Submit updated request constraints or switch fallback_behavior to best_effort.",
                    },
                )

            if fallback_behavior == "return_error":
                decision_reason = "surrogate_confidence_below_threshold"
                stop_reason = "surrogate_rejected_by_policy"
                self.session_store.create(
                    session_id=session_id,
                    trace_id=trace_id,
                    request_payload=normalized_request.model_dump(mode="json"),
                    ann_payload=ann.model_dump(mode="json"),
                    surrogate_validation=surrogate,
                    command_package=None,
                    max_iterations=normalized_request.optimization_policy.max_iterations,
                    initial_status="error",
                    stop_reason=stop_reason,
                    decision_reason="surrogate_confidence_below_threshold_rejected_by_policy",
                    objective_state=initial_objective_state,
                )
                return OptimizeResponse(
                    status="error",
                    session_id=session_id,
                    trace_id=trace_id,
                    current_stage="failed",
                    ann_prediction=ann,
                    objective_state=initial_objective_state,
                    warnings=self._surrogate_warnings(surrogate),
                    error={
                        "code": "LOW_SURROGATE_CONFIDENCE",
                        "message": "Request rejected by policy because surrogate confidence is below threshold.",
                        "retryable": True,
                        "details": surrogate,
                    },
                )

        command_package = build_command_package(normalized_request, ann, session_id=session_id, trace_id=trace_id, iteration_index=0)

        self.session_store.create(
            session_id=session_id,
            trace_id=trace_id,
            request_payload=normalized_request.model_dump(mode="json"),
            ann_payload=ann.model_dump(mode="json"),
            surrogate_validation=surrogate,
            command_package=command_package,
            max_iterations=normalized_request.optimization_policy.max_iterations,
            initial_status="accepted",
            stop_reason=None,
            decision_reason="family_profile_applied_and_surrogate_confidence_sufficient",
            objective_state=initial_objective_state,
        )
        session = self.session_store.load(session_id)
        session["intent_summary"] = intent_summary
        self.session_store.save(session_id, session)

        return OptimizeResponse(
            status="accepted",
            session_id=session_id,
            trace_id=trace_id,
            current_stage="planning_commands",
            ann_prediction=ann,
            command_package=command_package,
            objective_state=initial_objective_state,
            warnings=self._surrogate_warnings(surrogate),
        )

    def _append_manifest_history(
        self,
        *,
        session: dict[str, Any],
        iteration_index: int,
        decision_reason: str,
        stop_reason: str | None,
        command_package: dict[str, Any] | None,
        planning_provenance: dict[str, Any] | None = None,
    ) -> None:
        manifest = session.get("artifact_manifest")
        if not isinstance(manifest, dict):
            return
        manifest_typed = cast(dict[str, Any], manifest)

        now = datetime.now(timezone.utc).isoformat()
        checksum: str | None = None
        if command_package is not None:
            checksum = self.session_store.payload_checksum(command_package)
            manifest_typed["latest_command_package_checksum_sha256"] = checksum

        manifest_typed["latest_iteration_index"] = int(iteration_index)
        manifest_typed["updated_at"] = now
        history = manifest_typed.get("history")
        if not isinstance(history, list):
            history = []
            manifest_typed["history"] = history
        manifest_history = cast(list[dict[str, Any]], history)
        manifest_history.append(
            {
                "timestamp": now,
                "iteration_index": int(iteration_index),
                "decision_reason": decision_reason,
                "stop_reason": stop_reason,
                "command_package_checksum_sha256": checksum,
                "planning_provenance": planning_provenance,
            }
        )
        if planning_provenance is not None:
            manifest_typed["latest_planning_decision"] = planning_provenance

    def process_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        session = self.session_store.load(session_id)

        request = OptimizeRequest.model_validate(session["request"])
        current_iteration = int(session["current_iteration"])
        notes_text = str(payload.get("notes") or "").strip().lower()
        completion_requested = bool(payload.get("completion_requested", False)) or any(
            marker in notes_text
            for marker in (
                "marked this design done",
                "session marked complete",
                "session completed",
                "finish the session",
                "user completed",
            )
        )

        reported_iteration = int(payload["iteration_index"])
        if reported_iteration != current_iteration:
            # The QML client can restore a locally saved session after CST execution,
            # which leaves the local iteration one step ahead of the server until
            # feedback is posted. Be tolerant for explicit Done/completion requests.
            if completion_requested and reported_iteration == current_iteration + 1:
                reported_iteration = current_iteration
            else:
                raise ValueError(
                    f"Feedback iteration mismatch: expected {current_iteration}, got {reported_iteration}"
                )

        current_ann = AnnPrediction.model_validate(session["current_ann_prediction"])
        evaluation = evaluate_acceptance(request, payload)
        objective_state = evaluate_objective_state(request, payload, evaluation)
        session["objective_state"] = objective_state
        decision_reason = "acceptance_criteria_not_met"
        stop_reason: str | None = None
        if bool(evaluation["accepted"]):
            decision_reason = "acceptance_criteria_met"
            stop_reason = "acceptance_criteria_met"

        history_item: dict[str, Any] = {
            "type": "feedback_evaluation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration_index": current_iteration,
            "decision_reason": decision_reason,
            "stop_reason": stop_reason,
            "feedback": payload,
            "evaluation": evaluation,
            "objective_state": objective_state,
        }
        session["history"].append(history_item)
        dataset_feedback = self.live_retraining.ingest_result(
            request=request,
            ann_prediction=current_ann,
            payload=payload,
            evaluation=evaluation,
        )
        history_item["dataset_feedback"] = dataset_feedback

        if bool(evaluation["accepted"]):
            session["status"] = "completed"
            session["stop_reason"] = "acceptance_criteria_met"
            session["current_iteration"] = current_iteration
            self._append_manifest_history(
                session=session,
                iteration_index=current_iteration,
                decision_reason="acceptance_criteria_met",
                stop_reason="acceptance_criteria_met",
                command_package=session.get("current_command_package"),
                planning_provenance=None,
            )
            self.session_store.save(session_id, session)
            return {
                "status": "completed",
                "session_id": session_id,
                "trace_id": session["trace_id"],
                "accepted": True,
                "iteration_index": current_iteration,
                "decision_reason": "acceptance_criteria_met",
                "stop_reason": "acceptance_criteria_met",
                "evaluation": evaluation,
                "objective_state": objective_state,
                "dataset_feedback": dataset_feedback,
                "message": "Acceptance criteria met. No further refinement needed.",
            }

        if completion_requested:
            session["status"] = "completed"
            session["stop_reason"] = "user_marked_done"
            session["current_iteration"] = current_iteration
            session["history"][-1]["decision_reason"] = "user_marked_done"
            session["history"][-1]["stop_reason"] = "user_marked_done"
            self._append_manifest_history(
                session=session,
                iteration_index=current_iteration,
                decision_reason="user_marked_done",
                stop_reason="user_marked_done",
                command_package=session.get("current_command_package"),
                planning_provenance=None,
            )
            self.session_store.save(session_id, session)
            return {
                "status": "completed",
                "session_id": session_id,
                "trace_id": session["trace_id"],
                "accepted": bool(evaluation["accepted"]),
                "iteration_index": current_iteration,
                "decision_reason": "user_marked_done",
                "stop_reason": "user_marked_done",
                "evaluation": evaluation,
                "objective_state": objective_state,
                "dataset_feedback": dataset_feedback,
                "message": "Session completed by explicit client request.",
            }

        if current_iteration + 1 >= int(session["max_iterations"]):
            session["status"] = "max_iterations_reached"
            session["stop_reason"] = "max_iterations_reached"
            session["history"][-1]["decision_reason"] = "max_iterations_reached_without_acceptance"
            session["history"][-1]["stop_reason"] = "max_iterations_reached"
            self._append_manifest_history(
                session=session,
                iteration_index=current_iteration,
                decision_reason="max_iterations_reached_without_acceptance",
                stop_reason="max_iterations_reached",
                command_package=session.get("current_command_package"),
                planning_provenance=None,
            )
            self.session_store.save(session_id, session)
            return {
                "status": "stopped",
                "session_id": session_id,
                "trace_id": session["trace_id"],
                "accepted": False,
                "iteration_index": current_iteration,
                "decision_reason": "max_iterations_reached_without_acceptance",
                "stop_reason": "max_iterations_reached",
                "evaluation": evaluation,
                "objective_state": objective_state,
                "dataset_feedback": dataset_feedback,
                "message": "Max iterations reached before acceptance.",
            }

        next_iteration = current_iteration + 1
        features = derive_feedback_features(request, payload, evaluation)
        refinement_plan = plan_refinement_strategy(
            session=session,
            features=features,
            iteration_index=next_iteration,
        )
        strategy = refinement_plan.get("strategy")
        strategy_typed = strategy if isinstance(strategy, dict) else None
        refined_ann = refine_prediction_with_strategy(
            request,
            current_ann,
            evaluation,
            next_iteration_index=next_iteration,
            strategy=strategy_typed,
            action_name=str(refinement_plan.get("selected_action", "generic_refinement")),
        )
        next_command_package = build_command_package(
            request,
            refined_ann,
            session_id=session_id,
            trace_id=str(session["trace_id"]),
            iteration_index=next_iteration,
            previous_ann=current_ann,
        )

        session["current_iteration"] = next_iteration
        session["status"] = "refining_design"
        session["stop_reason"] = None
        session["current_ann_prediction"] = refined_ann.model_dump(mode="json")
        session["current_command_package"] = next_command_package
        session["history"].append(
            {
                "type": "refinement_plan",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration_index": next_iteration,
                "decision_reason": "apply_refinement_strategy_due_to_unmet_acceptance",
                "stop_reason": None,
                "feedback_features": features,
                "objective_state": objective_state,
                "planning_provenance": refinement_plan,
                "ann_prediction": refined_ann.model_dump(mode="json"),
                "command_package": next_command_package,
            }
        )
        self._append_manifest_history(
            session=session,
            iteration_index=next_iteration,
            decision_reason="apply_refinement_strategy_due_to_unmet_acceptance",
            stop_reason=None,
            command_package=next_command_package,
            planning_provenance=refinement_plan,
        )
        self.session_store.save(session_id, session)

        return {
            "status": "refining",
            "session_id": session_id,
            "trace_id": session["trace_id"],
            "accepted": False,
            "iteration_index": next_iteration,
            "decision_reason": "apply_refinement_strategy_due_to_unmet_acceptance",
            "stop_reason": None,
            "evaluation": evaluation,
            "planning_summary": {
                "selected_action": refinement_plan.get("selected_action"),
                "decision_source": refinement_plan.get("decision_source"),
                "confidence": refinement_plan.get("confidence"),
                "rule_id": refinement_plan.get("rule_id"),
                "focus_area": objective_state.get("focus_area"),
            },
            "objective_state": objective_state,
            "dataset_feedback": dataset_feedback,
            "next_command_package": next_command_package,
            "message": "Generated refined command package for next iteration.",
        }
