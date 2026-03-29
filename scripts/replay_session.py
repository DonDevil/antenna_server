#!/usr/bin/env python3
"""
Session Replay and Audit Utility

Reconstructs and displays the complete decision path from a saved session JSON file.
Shows all iterations, feedback submissions, evaluations, and refinement decisions.

Usage:
    python3 scripts/replay_session.py <session_id>
"""

import sys
import json
from pathlib import Path
from datetime import datetime


def inject_repository_root():
    """Add repository root to sys.path for imports."""
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))


inject_repository_root()

from config import SESSIONS_DIR


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(ts_str)


def print_header(text: str, width: int = 80) -> None:
    """Print formatted header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_separator(width: int = 80) -> None:
    """Print separator line."""
    print("-" * width)


def print_dimensions(dims: dict, indent: str = "    ") -> None:
    """Pretty-print dimensions."""
    if not dims:
        print(f"{indent}(no dimensions)")
        return
    for key, value in dims.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            for subkey, subval in value.items():
                print(f"{indent}  {subkey}: {subval}")
        else:
            print(f"{indent}{key}: {value}")


def print_metrics(metrics: dict, indent: str = "    ") -> None:
    """Pretty-print metrics."""
    if not metrics:
        print(f"{indent}(no metrics)")
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{indent}{key}: {value:.2f}" if isinstance(value, float) else f"{indent}{key}: {value}")
        else:
            print(f"{indent}{key}: {value}")


def replay_session(session_id: str) -> None:
    """Load and replay a session from disk."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    
    if not session_file.exists():
        print(f"Error: Session file not found: {session_file}")
        sys.exit(1)
    
    with open(session_file, 'r') as f:
        session = json.load(f)
    
    print_header(f"SESSION REPLAY: {session_id}")
    
    print(f"\nStatus: {session.get('status', 'unknown')}")
    print(f"Max Iterations: {session.get('max_iterations', 'N/A')}")
    print(f"Current Iteration: {session.get('current_iteration', 0)}")
    print(f"Total History Entries: {len(session.get('history', []))}")
    
    history = session.get("history", [])
    if not history:
        print("\nNo history entries found.")
        return
    
    for idx, entry in enumerate(history, 1):
        entry_type = entry.get("type", "unknown")
        timestamp = entry.get("timestamp", "unknown")
        
        print_separator()
        print(f"\nEntry {idx}: {entry_type} ({format_timestamp(timestamp)})")
        print_separator()
        
        if entry_type == "initial_plan":
            print("\nInitial ANN Prediction:")
            print_dimensions(entry.get("ann_prediction", {}), "  ")
            print("\nInitial Command Package:")
            cmd_pkg = entry.get("command_package", {})
            print(f"  Iteration Index: {cmd_pkg.get('iteration_index', 'N/A')}")
            print(f"  Commands Count: {len(cmd_pkg.get('cst_commands', []))}")
        
        elif entry_type == "feedback_received":
            print("\nClient Feedback:")
            feedback = entry.get("feedback", {})
            print(f"  Session ID: {feedback.get('session_id', 'N/A')}")
            print(f"  Reported Iteration: {feedback.get('reported_iteration', 'N/A')}")
            print("\n  Metrics Received:")
            print_metrics(feedback.get("metrics", {}), "    ")
        
        elif entry_type == "feedback_evaluation":
            print("\nFeedback Evaluation:")
            evaluation = entry.get("evaluation", {})
            print(f"  Accepted: {evaluation.get('accepted', False)}")
            if not evaluation.get("accepted"):
                print("  Error Metrics:")
                print_metrics(evaluation.get("metrics", {}), "    ")
        
        elif entry_type == "refinement_plan":
            print("\nRefinement Decision:")
            decision = entry.get("decision", {})
            print(f"  Next Iteration Index: {decision.get('next_iteration_index', 'N/A')}")
            print(f"  Reason: {decision.get('reason', 'N/A')}")
            print("\nRefined ANN Prediction:")
            print_dimensions(entry.get("refined_ann", {}), "  ")
            print("\nNext Command Package:")
            cmd_pkg = entry.get("next_command_package", {})
            print(f"  Iteration Index: {cmd_pkg.get('iteration_index', 'N/A')}")
            print(f"  Commands Count: {len(cmd_pkg.get('cst_commands', []))}")
        
        elif entry_type == "final_feedback":
            print("\nFinal Feedback:")
            feedback = entry.get("feedback", {})
            print(f"  Session ID: {feedback.get('session_id', 'N/A')}")
            print(f"  Reported Iteration: {feedback.get('reported_iteration', 'N/A')}")
            print("\n  Metrics Received:")
            print_metrics(feedback.get("metrics", {}), "    ")
            print("\nFinal Evaluation:")
            evaluation = entry.get("evaluation", {})
            print(f"  Accepted: {evaluation.get('accepted', False)}")
            print(f"  Final Status: {entry.get('final_status', 'N/A')}")
        
        else:
            print(f"\nEntry Data:")
            for key, value in entry.items():
                if key != "timestamp":
                    if isinstance(value, (dict, list)):
                        print(f"  {key}: {json.dumps(value, indent=4)}")
                    else:
                        print(f"  {key}: {value}")
    
    print_header("END OF REPLAY")
    print(f"\nFinal Status: {session.get('status', 'unknown')}")
    print(f"Total Iterations: {session.get('current_iteration', 0)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/replay_session.py <session_id>")
        print("\nExample:")
        print("  python3 scripts/replay_session.py abc12345def")
        sys.exit(1)
    
    session_id = sys.argv[1]
    replay_session(session_id)
