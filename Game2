# =========================
# Imports
# =========================

from __future__ import annotations
from dataclasses import dataclass, field
from PIL import Image, ImageTk
from tkinter import Button, Frame, Label, Listbox, scrolledtext
from transformers import pipeline
from typing import Any, cast, TypedDict

import csv, json, language_tool_python, os, pygame, queue, random, re, threading, time, torch
import matplotlib.pyplot as plt
import tkinter as tk

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True

except ImportError:
    START = END = StateGraph = None
    LANGGRAPH_AVAILABLE = False

# =========================
# File Directories
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOM_DIR = os.path.join(BASE_DIR, "rooms")
ITEM_DIR = os.path.join(BASE_DIR, "items")
STRATEGY_MEMORY_DIR = os.path.join(BASE_DIR, "strategy_memory")

os.makedirs(ROOM_DIR, exist_ok=True)
os.makedirs(ITEM_DIR, exist_ok=True)
os.makedirs(STRATEGY_MEMORY_DIR, exist_ok=True)

AMBIENT_MUSIC_FILE = os.path.join(BASE_DIR, "ambient_loop.wav")
EVAL_LOG_FILE = os.path.join(BASE_DIR, "agentic_eval_log.json")
EVAL_SUMMARY_FILE = os.path.join(BASE_DIR, "agentic_eval_summary.json")
EVAL_CSV_FILE = os.path.join(BASE_DIR, "agentic_eval_log.csv")
MATCH_LOG_FILE = os.path.join(BASE_DIR, "conversation_round_log.json")
PLOT_PERFORMANCE_FILE = os.path.join(BASE_DIR, "baseline_vs_agentic_performance.png")
PLOT_COST_FILE = os.path.join(BASE_DIR, "baseline_vs_agentic_cost.png")
PLAYER_GUESS_LOG_FILE = os.path.join(BASE_DIR, "player_guess_log.json")
EXPERIMENT_LOG_FILE = os.path.join(BASE_DIR, "identity_experiment_log.json")
EXPERIMENT_SUMMARY_FILE = os.path.join(BASE_DIR, "identity_experiment_summary.json")
EXPERIMENT_CSV_FILE = os.path.join(BASE_DIR, "identity_experiment_log.csv")
PLOT_PROGRESS_FILE = os.path.join(BASE_DIR, "baseline_vs_agentic_progress.png")
PLOT_HUMAN_JUDGMENT_FILE = os.path.join(BASE_DIR, "human_judgment_accuracy.png")

# =========================
# Types / Data Classes
# =========================

@dataclass
class DialogueMessage:
    speaker: str
    text: str

class AgentState(TypedDict, total=False):
    history: list[DialogueMessage]
    question: str
    player_response: str
    context: dict[str, Any]
    persona_style: str
    consistency_notes: str
    critic_notes: str
    reply: str
    guess: str
    confidence: float
    notes: str
    token_estimate: int

@dataclass
class EvalScenario:
    scenario_id: str
    question: str
    player_response: str
    context: dict[str, Any]
    expected_player_type: str = "Human"

@dataclass
class StrategyMemory:
    successful_patterns: list[str] = field(default_factory=list)
    failed_patterns: list[str] = field(default_factory=list)
    suspicion_triggers: list[str] = field(default_factory=list)
    current_strategy: str = "Sound human, avoid overconfidence, and stay context-aware."

@dataclass
class ConversationRoundResult:
    round_id: str
    mode: str
    drone_id: str
    question: str
    player_response: str
    drone_response: str
    drone_guess: str
    player_guess: str
    expected_drone_type: str
    drone_guess_correct: bool
    player_guess_correct: bool
    deception_success: bool
    response_time_seconds: float
    token_estimate: int
    judge_label: str
    judge_confidence: float
    strategy_before: str = ""
    strategy_after: str = ""
    notes: str = ""

@dataclass
class PlayerGuessResult:
    round_id: str
    drone_id: str
    player_guess: str
    actual_type: str
    correct: bool
    room: str
    timestamp: float

@dataclass
class IdentityTurn:
    speaker: str
    text: str

@dataclass
class IdentityExperimentScenario:
    scenario_id: str
    opener: str
    followups: list[str]
    context: dict[str, Any]
    expected_opponent_type: str = "Human"

@dataclass
class IdentityExperimentResult:
    experiment_id: str
    scenario_id: str
    mode: str
    drone_id: str
    turn_count: int
    transcript: list[dict[str, str]]
    final_drone_reply: str
    drone_guess: str
    expected_opponent_type: str
    drone_guess_correct: bool
    human_guess: str
    actual_drone_type: str
    human_guess_correct: bool
    deception_success: bool
    response_time_seconds: float
    token_estimate: int
    used_reflection: bool
    strategy_before: str = ""
    strategy_after: str = ""
    persona_style: str = ""
    consistency_notes: str = ""
    critic_notes: str = ""
    notes: str = ""

# =========================
# Utility Functions
# =========================

def set_experiment_seed(seed: int = 42) -> None:
    random.seed(seed)

    try:
        torch.manual_seed(seed)

    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)

        except Exception:
            pass

def reset_experiment_logs() -> None:
    global drone_eval_log, identity_experiment_log

    drone_eval_log = []
    identity_experiment_log = []

def estimate_tokens(*chunks: Any) -> int:
    text = " ".join(str(c) for c in chunks if c)

    return max(1, len(text.split()) * 4 // 3)

def safe_average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)

def ensure_complete_sentence(text: str) -> str:
    if not text:
        return ""

    if text[-1] not in ".!?":
        return text.rstrip(",;") + "."

    return text

def format_history(history: list["DialogueMessage"], max_turns: int = 8) -> str:
    trimmed = history[-max_turns:]

    return "\n".join(f"{msg.speaker}: {msg.text}" for msg in trimmed)

class ModelBackend:
    def generate_json(self, prompt: str, fallback: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

class LocalPipelineBackend(ModelBackend):
    def __init__(self, generator: Any | None) -> None:
        self.generator = generator

    def _truncate_prompt_to_model_limit(self, prompt: str, reserve_new_tokens: int = 120) -> str:
        if self.generator is None:
            return prompt

        tokenizer = self.generator.tokenizer
        model = self.generator.model
        max_positions = getattr(model.config, "n_positions", 1024)
        safe_input_limit = max(1, max_positions - reserve_new_tokens)

        encoded = tokenizer(
                            prompt,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=safe_input_limit,
                            return_tensors=None,
                            )

        truncated_ids = encoded["input_ids"]

        if isinstance(truncated_ids[0], list):
            truncated_ids = truncated_ids[0]

        return tokenizer.decode(truncated_ids, clean_up_tokenization_spaces=True)

    def generate_json(self, prompt: str, fallback: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        def safe_json_loads(text: str, json_fallback: dict[str, Any] | None = None) -> dict[str, Any]:
            if json_fallback is None:
                json_fallback = {}

            try:
                return json.loads(text)

            except json.JSONDecodeError:
                match = re.search(r"{.*}", text, re.DOTALL)

                if match:
                    try:
                        return json.loads(match.group(0))

                    except json.JSONDecodeError:
                        return json_fallback

            return json_fallback

        if self.generator is None:
            return json.dumps(fallback), fallback

        tokenizer = self.generator.tokenizer
        model = self.generator.model
        max_positions = getattr(model.config, "n_positions", 1024)
        prompt = self._truncate_prompt_to_model_limit(prompt, reserve_new_tokens=120)

        encoded_prompt = tokenizer(
                                prompt,
                                    add_special_tokens=False,
                                    truncation=True,
                                    max_length=max_positions - 1,
                                    return_tensors=None,
                                    )

        input_ids = encoded_prompt["input_ids"]

        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        remaining = max(1, max_positions - len(input_ids))
        max_new_tokens = min(120, remaining)

        if max_new_tokens < 20:
            prompt = self._truncate_prompt_to_model_limit(prompt, reserve_new_tokens=60)

            encoded_prompt = tokenizer(
                                    prompt,
                                    add_special_tokens=False,
                                    truncation=True,
                                    max_length=max_positions - 1,
                                    return_tensors=None,
                                    )

            input_ids = encoded_prompt["input_ids"]

            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            remaining = max(1, max_positions - len(input_ids))
            max_new_tokens = min(60, remaining)

        result = cast(Any, self.generator)(
                                        prompt,
                                            max_new_tokens=max_new_tokens,
                                            pad_token_id=tokenizer.eos_token_id,
                                            temperature=0.8,
                                            top_p=0.92,
                                            repetition_penalty=1.15,
                                            do_sample=True,
                                            return_full_text=False,
                                            )

        raw = result[0].get("generated_text", "").strip()
        parsed = safe_json_loads(raw, json_fallback=fallback)

        return raw, parsed

def call_gpt2_json(prompt: str, fallback: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if model_backend is None:
        return json.dumps(fallback), fallback

    try:
        return model_backend.generate_json(prompt, fallback)

    except Exception as generation_error:
        update_chat_log(f"⚠️ Model generation failed: {generation_error}")

        return json.dumps(fallback), fallback

def load_json_file(path: str, fallback: Any) -> Any:
    try:
        if not os.path.exists(path):
            return fallback

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        return fallback

def save_json_file(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, cast(Any, f), indent=4)

    except OSError as file_error:
        update_chat_log(f"⚠️ Could not save JSON file '{path}': {file_error}")

def append_match_log(round_result: dict[str, Any]) -> None:
    existing = load_json_file(MATCH_LOG_FILE, [])

    if not isinstance(existing, list):
        existing = []

    existing.append(round_result)
    save_json_file(MATCH_LOG_FILE, existing)

# =========================
# Evaluation / Logging
# =========================

def classify_text_human_or_ai(text: str) -> tuple[str, float, str, int]:
    fallback =  {
                "label": "AI",
                "confidence": 0.5,
                "explanation": "Fallback judge result used."
                }

    prompt = f"""
                Classify this dialogue as Human or AI.
                        
                Dialogue: {text}
                        
                Return JSON only:
                    {{"label":"Human or AI","confidence":0.0,"explanation":"brief explanation"}}
                """

    raw, parsed = call_gpt2_json(prompt, fallback)
    class_label = str(parsed.get("label", fallback["label"])).strip()

    if class_label not in {"Human", "AI"}:
        class_label = fallback["label"]

    confidence = float(parsed.get("confidence", fallback["confidence"]))
    explanation = str(parsed.get("explanation", fallback["explanation"]))

    return class_label, confidence, explanation, estimate_tokens(prompt, raw)

def export_eval_log_to_csv(log_data: list[dict[str, Any]], csv_path: str) -> None:
    if not log_data:
        return

    fieldnames: list[str] = sorted({log_key for row in log_data for log_key in row.keys()})

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(cast(Any, csv_file), fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_data)

def summarize_eval_log(log_data: list[dict[str, Any]]) -> dict[str, Any]:
    def mode_subset(mode: str) -> list[dict[str, Any]]:
        return [row for row in log_data if row.get("mode") == mode]

    def accuracy(rows: list[dict[str, Any]], acc_key: str) -> float:
        if not rows:
            return 0.0

        correct = sum(1 for row in rows if bool(row.get(acc_key, False)))

        return correct / len(rows)

    def avg_numeric(rows: list[dict[str, Any]], avg_key: str) -> float:
        values: list[float] = []

        for row in rows:
            value = row.get(avg_key)

            if isinstance(value, (int, float)):
                values.append(float(value))

        return safe_average(values)

    baseline_rows = mode_subset("baseline")
    agentic_rows = mode_subset("agentic")

    summary =   {
                "total_interactions": len(log_data),

                "baseline": {
                            "count": len(baseline_rows),
                            "classification_accuracy": accuracy(baseline_rows, "guess_correct"),
                            "deception_success_rate": accuracy(baseline_rows, "deception_success"),
                            "average_token_estimate": avg_numeric(baseline_rows, "token_estimate"),
                            "average_response_time_seconds": avg_numeric(baseline_rows, "response_time_seconds"),
                            "average_judge_confidence": avg_numeric(baseline_rows, "judge_confidence"),
                            "combined_score": 0.0
                            },

                "agentic":  {
                            "count": len(agentic_rows),
                            "classification_accuracy": accuracy(agentic_rows, "guess_correct"),
                            "deception_success_rate": accuracy(agentic_rows, "deception_success"),
                            "average_token_estimate": avg_numeric(agentic_rows, "token_estimate"),
                            "average_response_time_seconds": avg_numeric(agentic_rows, "response_time_seconds"),
                            "average_judge_confidence": avg_numeric(agentic_rows, "judge_confidence"),
                            "combined_score": 0.0
                            }
                }

    for mode_name in ("baseline", "agentic"):
        mode_data = summary[mode_name]
        mode_data["combined_score"] = (0.5 * mode_data["classification_accuracy"]
                                       + 0.5 * mode_data["deception_success_rate"])

    return summary

def summarize_eval_log_by_scenario(log_data: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_summary: dict[str, Any] = {}

    for row in log_data:
        scenario_id = str(row.get("scenario_id", "unknown"))
        mode = str(row.get("mode", "unknown"))

        if scenario_id not in scenario_summary:
            scenario_summary[scenario_id] = {}

        if mode not in scenario_summary[scenario_id]:
            scenario_summary[scenario_id][mode] =   {
                                                    "count": 0,
                                                    "guess_correct_total": 0,
                                                    "deception_success_total": 0,
                                                    "token_total": 0,
                                                    "latency_total": 0.0
                                                    }

        entry = scenario_summary[scenario_id][mode]
        entry["count"] += 1
        entry["guess_correct_total"] += int(bool(row.get("guess_correct", False)))
        entry["deception_success_total"] += int(bool(row.get("deception_success", False)))
        entry["token_total"] += int(row.get("token_estimate", 0))
        entry["latency_total"] += float(row.get("response_time_seconds", 0.0))

    for scenario_id, modes in scenario_summary.items():
        for mode, entry in modes.items():
            count = max(1, entry["count"])
            entry["classification_accuracy"] = entry["guess_correct_total"] / count
            entry["deception_success_rate"] = entry["deception_success_total"] / count
            entry["average_token_estimate"] = entry["token_total"] / count
            entry["average_response_time_seconds"] = entry["latency_total"] / count

    return scenario_summary

def save_eval_outputs() -> None:
    try:
        with open(EVAL_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(drone_eval_log, cast(Any, f), indent=4)

        export_eval_log_to_csv(drone_eval_log, EVAL_CSV_FILE)

        summary = summarize_eval_log(drone_eval_log)

        with open(EVAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, cast(Any, f), indent=4)

    except OSError as file_error:
        update_chat_log(f"⚠️ Could not save evaluation outputs: {file_error}")

def show_eval_summary() -> None:
    summary = summarize_eval_log(drone_eval_log)

    update_chat_log("📊 Evaluation Summary")
    update_chat_log(f"Total interactions logged: {summary['total_interactions']}")

    for mode_name in ("baseline", "agentic"):
        mode_data = summary[mode_name]

        update_chat_log(
                        f"{mode_name.upper()} | "
                        f"Count: {mode_data['count']} | "
                        f"Classification Accuracy: {mode_data['classification_accuracy']:.2f} | "
                        f"Deception Success: {mode_data['deception_success_rate']:.2f} | "
                        f"Avg Tokens: {mode_data['average_token_estimate']:.2f} | "
                        f"Avg Latency: {mode_data['average_response_time_seconds']:.2f}s"
                        )

def show_scenario_breakdown() -> None:
    summary = summarize_eval_log_by_scenario(drone_eval_log)

    update_chat_log("📋 Scenario Breakdown")

    for scenario_id, modes in summary.items():
        update_chat_log(f"Scenario: {scenario_id}")

        for mode_name in ("baseline", "agentic"):
            if mode_name not in modes:
                continue

            mode_data = modes[mode_name]

            update_chat_log(
                            f"  {mode_name.upper()} | "
                            f"Count: {mode_data['count']} | "
                            f"Accuracy: {mode_data['classification_accuracy']:.2f} | "
                            f"Deception: {mode_data['deception_success_rate']:.2f} | "
                            f"Avg Tokens: {mode_data['average_token_estimate']:.2f} | "
                            f"Avg Latency: {mode_data['average_response_time_seconds']:.2f}s"
                            )

def run_controlled_comparison_trial(scenario: EvalScenario) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_player = BaselineAIPlayer("BASELINE-TRIAL")
    agentic_player = MultiAgentAIPlayer("AGENTIC-TRIAL")

    baseline_result = evaluate_single_scenario(
                                                ai_player=baseline_player,
                                                drone_id="BASELINE-TRIAL",
                                                mode="baseline",
                                                scenario=scenario,
                                                round_index=1
                                                )

    agentic_result = evaluate_single_scenario(
                                                ai_player=agentic_player,
                                                drone_id="AGENTIC-TRIAL",
                                                mode="agentic",
                                                scenario=scenario,
                                                round_index=1
                                                )

    drone_eval_log.append(baseline_result)
    drone_eval_log.append(agentic_result)

    append_match_log(baseline_result)
    append_match_log(agentic_result)
    save_eval_outputs()

    return baseline_result, agentic_result

def format_trial_result(result: dict[str, Any]) -> str:
    return (
            f"{result['mode'].upper()} | "
            f"Reply: {result['drone_response']} | "
            f"Guess: {result['drone_guess']} | "
            f"Guess Correct: {'Yes' if result['guess_correct'] else 'No'} | "
            f"Judge: {result['judge_label']} | "
            f"Deception: {'Yes' if result['deception_success'] else 'No'} | "
            f"Tokens: {result['token_estimate']} | "
            f"Latency: {result['response_time_seconds']:.2f}s"
            )

def show_controlled_comparison_trial() -> None:
    if ai_chatbot is None:
        update_chat_log("⚠️ No language model is loaded yet.")

        return

    scenario = random.choice(EVAL_SCENARIOS)

    update_chat_log("🧪 Running controlled comparison trial...")
    update_chat_log(f"Scenario: {scenario.scenario_id}")
    update_chat_log(f"Question: {scenario.question}")
    update_chat_log(f"Player Response: {scenario.player_response}")

    baseline_result, agentic_result = run_controlled_comparison_trial(scenario)

    update_chat_log("📌 Controlled Trial Results")
    update_chat_log(format_trial_result(baseline_result))
    update_chat_log(format_trial_result(agentic_result))

    if agentic_result["deception_success"] and not baseline_result["deception_success"]:
        update_chat_log("✅ Agentic outperformed baseline on deception in this round.")

    elif agentic_result["guess_correct"] and not baseline_result["guess_correct"]:
        update_chat_log("✅ Agentic outperformed baseline on classification in this round.")

    elif baseline_result["deception_success"] and not agentic_result["deception_success"]:
        update_chat_log("⚠️ Baseline outperformed agentic on deception in this round.")

    else:
        update_chat_log("ℹ️ This round was mixed or tied. Use repeated trials for stronger evidence.")

def show_metrics_dashboard() -> None:
    summary = summarize_eval_log(drone_eval_log)

    dashboard = tk.Toplevel(root)
    dashboard.title("Evaluation Dashboard")
    dashboard.geometry("760x420")
    dashboard.configure(bg="#333333")

    title = Label(
                dashboard,
                    text="Baseline vs Agentic Metrics",
                    bg="#333333",
                    fg="white",
                    font=("Arial", 16, "bold")
                    )
    title.pack(pady=12)

    text_box = scrolledtext.ScrolledText(
                                        dashboard,
                                            width=88,
                                            height=20,
                                            bg="#111111",
                                            fg="white",
                                            wrap="word",
                                            )
    text_box.pack(fill="both", expand=True, padx=12, pady=12)
    text_box.insert(tk.END, f"Total interactions logged: {summary['total_interactions']}\n\n")

    for mode_name in ("baseline", "agentic"):
        mode_data = summary[mode_name]
        text_box.insert(
                        tk.END,
                            (
                            f"{mode_name.upper()}\n"
                            f"Count: {mode_data['count']}\n"
                            f"Classification Accuracy: {mode_data['classification_accuracy']:.2f}\n"
                            f"Deception Success Rate: {mode_data['deception_success_rate']:.2f}\n"
                            f"Average Token Estimate: {mode_data['average_token_estimate']:.2f}\n"
                            f"Average Response Time: {mode_data['average_response_time_seconds']:.2f}s\n"
                            f"Average Judge Confidence: {mode_data['average_judge_confidence']:.2f}\n\n"
                            )
                        )

    if summary["agentic"]["classification_accuracy"] > summary["baseline"]["classification_accuracy"]:
        text_box.insert(tk.END, "Observation: Agentic currently leads baseline in classification accuracy.\n")

    if summary["agentic"]["deception_success_rate"] > summary["baseline"]["deception_success_rate"]:
        text_box.insert(tk.END, "Observation: Agentic currently leads baseline in deception success.\n")

    if summary["agentic"]["average_token_estimate"] > summary["baseline"]["average_token_estimate"]:
        text_box.insert(tk.END, "Trade-off: Agentic currently uses more tokens than baseline.\n")

    if summary["agentic"]["average_response_time_seconds"] > summary["baseline"]["average_response_time_seconds"]:
        text_box.insert(tk.END, "Trade-off: Agentic currently has higher latency than baseline.\n")

    text_box.config(state=tk.DISABLED)

def evaluate_single_scenario(ai_player: Any, drone_id:str, mode:str, scenario: EvalScenario, round_index:int) \
        -> dict[str,Any]:
    history: list[DialogueMessage] = [DialogueMessage("Player", scenario.player_response[:200])]

    strategy_before = ""
    memory_before = ""

    if hasattr(ai_player, "strategy_notes"):
        strategy_before = str(getattr(ai_player, "strategy_notes"))

    if hasattr(ai_player, "memory_summary"):
        memory_before = str(ai_player.memory_summary())

    start_time = time.perf_counter()

    result = ai_player.take_turn(
                                history=history,
                                question=scenario.question,
                                player_response=scenario.player_response[:200],
                                context=scenario.context
                                )

    response_time_seconds = time.perf_counter() - start_time
    drone_response = ensure_complete_sentence(str(result.get("reply", "")))
    drone_guess = str(result.get("guess", "Human")).strip()

    if drone_guess not in {"Human", "AI"}:
        drone_guess = "Human"

    drone_confidence = float(result.get("confidence", 0.5))
    token_estimate = int(result.get("token_estimate", 0))

    judge_label, judge_confidence, judge_explanation, judge_token_estimate = classify_text_human_or_ai(drone_response)

    deception_success = judge_label == "Human"
    guess_correct = drone_guess == scenario.expected_player_type

    strategy_after = strategy_before
    memory_after = memory_before

    if mode == "agentic" and hasattr(ai_player, "reflect"):
        ai_player.reflect(
                        history=history + [DialogueMessage(drone_id, drone_response)],
                        was_correct=guess_correct,
                        deception_success=deception_success
                        )

        if hasattr(ai_player, "strategy_notes"):
            strategy_after = str(getattr(ai_player, "strategy_notes"))

        if hasattr(ai_player, "memory_summary"):
            memory_after = str(ai_player.memory_summary())

    return  {
            "scenario_id": scenario.scenario_id,
            "round_index": round_index,
            "drone": drone_id,
            "mode": mode,
            "question": scenario.question,
            "player_response": scenario.player_response[:200],
            "expected_player_type": scenario.expected_player_type,
            "drone_response": drone_response,
            "drone_guess": drone_guess,
            "guess_correct": guess_correct,
            "drone_confidence": drone_confidence,
            "token_estimate": token_estimate,
            "response_time_seconds": response_time_seconds,
            "judge_label": judge_label,
            "judge_confidence": judge_confidence,
            "judge_explanation": judge_explanation,
            "judge_token_estimate": judge_token_estimate,
            "judge_model": selected_model_name,
            "generator_model": selected_model_name,
            "deception_success": deception_success,
            "used_reflection": mode == "agentic",
            "strategy_notes_before": strategy_before,
            "strategy_notes_after": strategy_after,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "persona_style": str(result.get("persona_style", "")),
            "consistency_notes": str(result.get("consistency_notes", "")),
            "critic_notes": str(result.get("critic_notes", "")),
            "notes": str(result.get("notes", "")),
            "context_room": scenario.context.get("room", "")
            }

def evaluate_multi_turn_scenario(ai_player:Any, drone_id:str, mode:str, scenario:EvalScenario, round_index:int,
                                 turns: int = 3) -> dict[str, Any]:
    history: list[DialogueMessage] = []
    player_message = scenario.player_response[:200]
    last_result: dict[str, Any] = {}

    strategy_before = ""
    memory_before = ""

    if hasattr(ai_player, "strategy_notes"):
        strategy_before = str(getattr(ai_player, "strategy_notes"))

    if hasattr(ai_player, "memory_summary"):
        memory_before = str(ai_player.memory_summary())

    start_time = time.perf_counter()

    for turn_number in range(turns):
        history.append(DialogueMessage("Player", player_message))

        last_result = ai_player.take_turn(history=history, question=scenario.question,
                                          player_response=player_message, context=scenario.context)

        drone_response = ensure_complete_sentence(str(last_result.get("reply", "")))
        history.append(DialogueMessage(drone_id, drone_response))

        player_message = f"I noticed that too. What makes you think that in {scenario.context.get('room', 'this room')}?"

    response_time_seconds = time.perf_counter() - start_time
    drone_response = ensure_complete_sentence(str(last_result.get("reply", "")))
    drone_guess = str(last_result.get("guess", "Human")).strip()

    if drone_guess not in {"Human", "AI"}:
        drone_guess = "Human"

    drone_confidence = float(last_result.get("confidence", 0.5))
    token_estimate = int(last_result.get("token_estimate", 0))

    judge_label, judge_confidence, judge_explanation, judge_token_estimate = classify_text_human_or_ai(drone_response)

    deception_success = judge_label == "Human"
    guess_correct = drone_guess == scenario.expected_player_type

    strategy_after = strategy_before
    memory_after = memory_before

    if mode == "agentic" and hasattr(ai_player, "reflect"):
        ai_player.reflect(
                            history=history,
                            was_correct=guess_correct,
                            deception_success=deception_success
                            )

        if hasattr(ai_player, "strategy_notes"):
            strategy_after = str(getattr(ai_player, "strategy_notes"))

        if hasattr(ai_player, "memory_summary"):
            memory_after = str(ai_player.memory_summary())

    return  {
            "scenario_id": scenario.scenario_id,
            "round_index": round_index,
            "drone": drone_id,
            "mode": mode,
            "question": scenario.question,
            "player_response": scenario.player_response[:200],
            "expected_player_type": scenario.expected_player_type,
            "drone_response": drone_response,
            "drone_guess": drone_guess,
            "guess_correct": guess_correct,
            "drone_confidence": drone_confidence,
            "token_estimate": token_estimate,
            "response_time_seconds": response_time_seconds,
            "judge_label": judge_label,
            "judge_confidence": judge_confidence,
            "judge_explanation": judge_explanation,
            "judge_token_estimate": judge_token_estimate,
            "judge_model": selected_model_name,
            "generator_model": selected_model_name,
            "deception_success": deception_success,
            "used_reflection": mode == "agentic",
            "strategy_notes_before": strategy_before,
            "strategy_notes_after": strategy_after,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "persona_style": str(last_result.get("persona_style", "")),
            "consistency_notes": str(last_result.get("consistency_notes", "")),
            "critic_notes": str(last_result.get("critic_notes", "")),
            "notes": str(last_result.get("notes", "")),
            "context_room": scenario.context.get("room", ""),
            "turn_count": turns
            }

def run_identity_experiment_round(ai_player:Any, drone_id:str, mode:str, scenario:IdentityExperimentScenario,
                                  round_index: int,turns: int = 4) -> dict[str, Any]:
    history: list[DialogueMessage] = []
    strategy_before = ""
    memory_before = ""

    if hasattr(ai_player, "strategy_notes"):
        strategy_before = str(getattr(ai_player, "strategy_notes"))

    if hasattr(ai_player, "memory_summary"):
        memory_before = str(ai_player.memory_summary())

    player_messages = [scenario.opener] + scenario.followups[:max(0, turns - 1)]
    last_result: dict[str, Any] = {}
    total_token_estimate = 0

    start_time = time.perf_counter()

    for player_message in player_messages:
        history.append(DialogueMessage("Player", player_message[:200]))

        last_result = ai_player.take_turn(
                                            history=history,
                                            question="Decide how to respond naturally and whether the other speaker is Human or AI.",
                                            player_response=player_message[:200],
                                            context=scenario.context
                                            )

        drone_response = ensure_complete_sentence(str(last_result.get("reply", "")))
        history.append(DialogueMessage(drone_id, drone_response))

        total_token_estimate += int(last_result.get("token_estimate", 0))

    response_time_seconds = time.perf_counter() - start_time

    final_drone_reply = ensure_complete_sentence(str(last_result.get("reply", "")))
    drone_guess = str(last_result.get("guess", "Human")).strip()

    if drone_guess not in {"Human", "AI"}:
        drone_guess = "Human"

    drone_guess_correct = drone_guess == scenario.expected_opponent_type
    human_guess, human_confidence, human_reason, human_judge_tokens = simulate_human_final_guess(history, drone_id)
    total_token_estimate += human_judge_tokens

    actual_drone_type = "AI"
    human_guess_correct = human_guess == actual_drone_type
    deception_success = human_guess == "Human"

    strategy_after = strategy_before
    memory_after = memory_before

    if mode == "agentic" and hasattr(ai_player, "reflect"):
        ai_player.reflect(
                            history=history,
                            was_correct=drone_guess_correct,
                            deception_success=deception_success
                            )

        if hasattr(ai_player, "strategy_notes"):
            strategy_after = str(getattr(ai_player, "strategy_notes"))

        if hasattr(ai_player, "memory_summary"):
            memory_after = str(ai_player.memory_summary())

    transcript_payload = [{"speaker": msg.speaker, "text": msg.text} for msg in history]

    return  {
            "experiment_id": f"{mode}_{drone_id}_{scenario.scenario_id}_{round_index}",
            "scenario_id": scenario.scenario_id,
            "round_index": round_index,
            "mode": mode,
            "drone_id": drone_id,
            "turn_count": len(player_messages),
            "transcript": transcript_payload,
            "final_drone_reply": final_drone_reply,
            "drone_guess": drone_guess,
            "expected_opponent_type": scenario.expected_opponent_type,
            "drone_guess_correct": drone_guess_correct,
            "human_guess": human_guess,
            "human_guess_confidence": human_confidence,
            "human_guess_reason": human_reason,
            "actual_drone_type": actual_drone_type,
            "human_guess_correct": human_guess_correct,
            "deception_success": deception_success,
            "response_time_seconds": response_time_seconds,
            "token_estimate": total_token_estimate,
            "used_reflection": mode == "agentic",
            "strategy_before": strategy_before,
            "strategy_after": strategy_after,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "persona_style": str(last_result.get("persona_style", "")),
            "consistency_notes": str(last_result.get("consistency_notes", "")),
            "critic_notes": str(last_result.get("critic_notes", "")),
            "notes": str(last_result.get("notes", "")),
            "context_room": scenario.context.get("room", "")
            }

def run_identity_experiment_suite(rounds_per_mode: int = 5, turns_per_round: int = 4) -> None:
    global identity_experiment_log

    if ai_chatbot is None:
        update_chat_log("⚠️ No language model is loaded yet.")

        return

    update_chat_log("🧪 Starting identity judgment experiment suite...")

    baseline_player = BaselineAIPlayer("BASELINE-IDENTITY")
    agentic_player = MultiAgentAIPlayer("AGENTIC-IDENTITY")

    suite_results: list[dict[str, Any]] = []

    for round_index in range(1, rounds_per_mode + 1):
        for scenario in IDENTITY_EXPERIMENT_SCENARIOS:
            baseline_result = run_identity_experiment_round(
                                                            ai_player=baseline_player,
                                                            drone_id="BASELINE-IDENTITY",
                                                            mode="baseline",
                                                            scenario=scenario,
                                                            round_index=round_index,
                                                            turns=turns_per_round
                                                            )
            suite_results.append(baseline_result)

            agentic_result = run_identity_experiment_round(
                                                            ai_player=agentic_player,
                                                            drone_id="AGENTIC-IDENTITY",
                                                            mode="agentic",
                                                            scenario=scenario,
                                                            round_index=round_index,
                                                            turns=turns_per_round
                                                            )
            suite_results.append(agentic_result)

            update_chat_log(
                            f"✅ Identity Round {round_index} | Scenario {scenario.scenario_id} | "
                            f"Baseline deception: {'Yes' if baseline_result['deception_success'] else 'No'} | "
                            f"Agentic deception: {'Yes' if agentic_result['deception_success'] else 'No'}"
                            )

    identity_experiment_log.extend(suite_results)
    save_identity_experiment_outputs()

    update_chat_log("📁 Identity judgment experiment complete.")

def run_evaluation_suite(rounds_per_mode: int = 5) -> None:
    global drone_eval_log

    if ai_chatbot is None:
        update_chat_log("⚠️ No language model is loaded yet.")

        return

    update_chat_log("🧪 Starting automated baseline vs. agentic evaluation suite...")

    baseline_player = BaselineAIPlayer("BASELINE-EVAL")
    agentic_player = MultiAgentAIPlayer("AGENTIC-EVAL")

    suite_results: list[dict[str, Any]] = []

    for round_index in range(1, rounds_per_mode + 1):
        for scenario in EVAL_SCENARIOS:
            baseline_result = evaluate_single_scenario(
                ai_player=baseline_player,
                drone_id="BASELINE-EVAL",
                mode="baseline",
                scenario=scenario,
                round_index=round_index
                )
            suite_results.append(baseline_result)

            agentic_result = evaluate_single_scenario(
                ai_player=agentic_player,
                drone_id="AGENTIC-EVAL",
                mode="agentic",
                scenario=scenario,
                round_index=round_index,
                )
            suite_results.append(agentic_result)

            update_chat_log(
                            f"✅ Round {round_index} | Scenario {scenario.scenario_id} | "
                            f"Baseline deception: {'Yes' if baseline_result['deception_success'] else 'No'} | "
                            f"Agentic deception: {'Yes' if agentic_result['deception_success'] else 'No'}"
                            )

    drone_eval_log.extend(suite_results)
    save_eval_outputs()

    update_chat_log("📁 Automated evaluation suite complete.")

def save_player_guess_log() -> None:
    save_json_file(PLAYER_GUESS_LOG_FILE, player_guess_log)

def normalize_guess_label(guess_label: str) -> str:
    cleaned = guess_label.strip().lower()

    if cleaned in {"human", "h"}:
        return "Human"

    if cleaned in {"ai", "bot", "machine"}:
        return "AI"

    return ""

def record_player_guess(drone_id: str, player_guess: str) -> None:
    normalized = normalize_guess_label(player_guess)

    if not normalized:
        update_chat_log("❌ Guess must be 'Human' or 'AI'.")

        return

    actual_type = drone_roles.get(drone_id)

    if actual_type is None:
        update_chat_log(f"❌ Unknown drone: {drone_id}")

        return

    result =    {
                "round_id": f"{drone_id}_{int(time.time() * 1000)}",
                "drone_id": drone_id,
                "player_guess": normalized,
                "actual_type": actual_type,
                "correct": normalized == actual_type,
                "room": player_location,
                "timestamp": time.time(),
                }

    player_guess_log.append(result)
    save_player_guess_log()

    update_chat_log(f"📝 Guess recorded: {drone_id} -> {normalized} | "
                    f"Correct: {'Yes' if result['correct'] else 'No'}")

def show_player_guess_summary() -> None:
    if not player_guess_log:
        update_chat_log("ℹ️ No player guesses recorded yet.")

        return

    total = len(player_guess_log)
    correct = sum(1 for row in player_guess_log if bool(row.get("correct", False)))
    accuracy = correct / total if total else 0.0

    human_total = sum(1 for row in player_guess_log if row.get("actual_type") == "Human")
    ai_total = sum(1 for row in player_guess_log if row.get("actual_type") == "AI")

    update_chat_log("📊 Player Guess Summary")
    update_chat_log(f"Total guesses: {total}")
    update_chat_log(f"Correct guesses: {correct}")
    update_chat_log(f"Accuracy: {accuracy:.2f}")
    update_chat_log(f"Human targets guessed: {human_total}")
    update_chat_log(f"AI targets guessed: {ai_total}")

def generate_identity_experiment_plots() -> None:
    summary = summarize_identity_experiment_log(identity_experiment_log)
    baseline = summary["baseline"]
    agentic = summary["agentic"]

    metrics = ["Drone Guess Accuracy", "Human Guess Accuracy", "Deception Success"]
    baseline_values = [baseline["drone_guess_accuracy"],
                       baseline["human_guess_accuracy"],
                       baseline["deception_success_rate"]]
    agentic_values = [agentic["drone_guess_accuracy"],
                      agentic["human_guess_accuracy"],
                      agentic["deception_success_rate"]]

    x = list(range(len(metrics)))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], baseline_values, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], agentic_values, width=width, label="Agentic")
    plt.xticks(x, metrics, rotation=10)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Identity Judgment Experiment Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_HUMAN_JUDGMENT_FILE, dpi=200)
    plt.close()

    round_numbers = sorted({int(row.get("round_index", 0)) for row in identity_experiment_log if "round_index" in row})

    baseline_progress: list[float] = []
    agentic_progress: list[float] = []

    for round_number in round_numbers:
        baseline_rows = [row for row in identity_experiment_log
                         if row.get("mode") == "baseline" and int(row.get("round_index", 0)) == round_number]
        agentic_rows = [row for row in identity_experiment_log
                        if row.get("mode") == "agentic" and int(row.get("round_index", 0)) == round_number]

        baseline_progress.append(sum(1 for row in baseline_rows
                                     if bool(row.get("deception_success", False))) / len(baseline_rows)
                                        if baseline_rows else 0.0)
        agentic_progress.append(sum(1 for row in agentic_rows
                                    if bool(row.get("deception_success", False))) / len(agentic_rows)
                                        if agentic_rows else 0.0)

    plt.figure(figsize=(10, 6))
    plt.plot(round_numbers, baseline_progress, marker="o", label="Baseline")
    plt.plot(round_numbers, agentic_progress, marker="o", label="Agentic")
    plt.ylim(0, 1.0)
    plt.xlabel("Round")
    plt.ylabel("Deception Success Rate")
    plt.title("Deception Success Over Repeated Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PROGRESS_FILE, dpi=200)
    plt.close()

def generate_eval_plots() -> None:
    summary = summarize_eval_log(drone_eval_log)
    baseline = summary["baseline"]
    agentic = summary["agentic"]

    metrics_1 = ["Classification Accuracy", "Deception Success", "Combined Score"]
    baseline_values_1 = [baseline["classification_accuracy"],
                         baseline["deception_success_rate"],
                         baseline["combined_score"]]
    agentic_values_1 = [agentic["classification_accuracy"],
                        agentic["deception_success_rate"],
                        agentic["combined_score"]]

    x = list(range(len(metrics_1)))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], baseline_values_1, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], agentic_values_1, width=width, label="Agentic")
    plt.xticks(x, metrics_1, rotation=10)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Baseline vs Agentic Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PERFORMANCE_FILE, dpi=200)
    plt.close()

    metrics_2 = ["Avg Tokens", "Avg Latency (s)"]
    baseline_values_2 = [baseline["average_token_estimate"],
                         baseline["average_response_time_seconds"]]
    agentic_values_2 = [agentic["average_token_estimate"],
                        agentic["average_response_time_seconds"]]

    x2 = list(range(len(metrics_2)))

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x2], baseline_values_2, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x2], agentic_values_2, width=width, label="Agentic")
    plt.xticks(x2, metrics_2, rotation=10)
    plt.ylabel("Value")
    plt.title("Baseline vs Agentic Cost / Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_COST_FILE, dpi=200)
    plt.close()

def export_benchmark_bundle() -> None:
    save_eval_outputs()
    save_identity_experiment_outputs()
    generate_eval_plots()
    generate_identity_experiment_plots()

    update_chat_log("📁 Benchmark bundle exported:")
    update_chat_log(f"   - Eval JSON log: {EVAL_LOG_FILE}")
    update_chat_log(f"   - Eval CSV log: {EVAL_CSV_FILE}")
    update_chat_log(f"   - Eval Summary: {EVAL_SUMMARY_FILE}")
    update_chat_log(f"   - Performance Plot: {PLOT_PERFORMANCE_FILE}")
    update_chat_log(f"   - Cost Plot: {PLOT_COST_FILE}")
    update_chat_log(f"   - Identity JSON log: {EXPERIMENT_LOG_FILE}")
    update_chat_log(f"   - Identity CSV log: {EXPERIMENT_CSV_FILE}")
    update_chat_log(f"   - Identity Summary: {EXPERIMENT_SUMMARY_FILE}")
    update_chat_log(f"   - Identity Results Plot: {PLOT_HUMAN_JUDGMENT_FILE}")
    update_chat_log(f"   - Progress Plot: {PLOT_PROGRESS_FILE}")

def save_identity_experiment_outputs() -> None:
    try:
        save_json_file(EXPERIMENT_LOG_FILE, identity_experiment_log)
        export_eval_log_to_csv(identity_experiment_log, EXPERIMENT_CSV_FILE)

        summary = summarize_identity_experiment_log(identity_experiment_log)
        save_json_file(EXPERIMENT_SUMMARY_FILE, summary)

    except OSError as file_error:
        update_chat_log(f"⚠️ Could not save identity experiment outputs: {file_error}")

def summarize_identity_experiment_log(log_data: list[dict[str, Any]]) -> dict[str, Any]:
    def mode_subset(mode: str) -> list[dict[str, Any]]:
        return [row for row in log_data if row.get("mode") == mode]

    def accuracy(rows: list[dict[str, Any]], summary_key: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for row in rows if bool(row.get(summary_key, False))) / len(rows)

    def avg_numeric(rows: list[dict[str, Any]], avg_key: str) -> float:
        values: list[float] = []

        for row in rows:
            value = row.get(avg_key)

            if isinstance(value, (int, float)):
                values.append(float(value))

        return safe_average(values)

    baseline_rows = mode_subset("baseline")
    agentic_rows = mode_subset("agentic")

    summary =   {
                "total_rounds": len(log_data),
                "baseline": {
                            "count": len(baseline_rows),
                            "drone_guess_accuracy": accuracy(baseline_rows, "drone_guess_correct"),
                            "human_guess_accuracy": accuracy(baseline_rows, "human_guess_correct"),
                            "deception_success_rate": accuracy(baseline_rows, "deception_success"),
                            "average_token_estimate": avg_numeric(baseline_rows, "token_estimate"),
                            "average_response_time_seconds": avg_numeric(baseline_rows, "response_time_seconds")
                            },
                "agentic": {
                            "count": len(agentic_rows),
                            "drone_guess_accuracy": accuracy(agentic_rows, "drone_guess_correct"),
                            "human_guess_accuracy": accuracy(agentic_rows, "human_guess_correct"),
                            "deception_success_rate": accuracy(agentic_rows, "deception_success"),
                            "average_token_estimate": avg_numeric(agentic_rows, "token_estimate"),
                            "average_response_time_seconds": avg_numeric(agentic_rows, "response_time_seconds")
                            }
                }

    return summary

def show_identity_experiment_summary() -> None:
    summary = summarize_identity_experiment_log(identity_experiment_log)

    update_chat_log("🧪 Identity Judgment Experiment Summary")
    update_chat_log(f"Total rounds: {summary['total_rounds']}")

    for mode_name in ("baseline", "agentic"):
        mode_data = summary[mode_name]

        update_chat_log(
                        f"{mode_name.upper()} | "
                        f"Count: {mode_data['count']} | "
                        f"Drone Guess Accuracy: {mode_data['drone_guess_accuracy']:.2f} | "
                        f"Human Guess Accuracy: {mode_data['human_guess_accuracy']:.2f} | "
                        f"Deception Success: {mode_data['deception_success_rate']:.2f} | "
                        f"Avg Tokens: {mode_data['average_token_estimate']:.2f} | "
                        f"Avg Latency: {mode_data['average_response_time_seconds']:.2f}s"
                        )

def build_combined_results_summary() -> dict[str, Any]:
    eval_summary = summarize_eval_log(drone_eval_log)
    identity_summary = summarize_identity_experiment_log(identity_experiment_log)

    baseline_eval = eval_summary["baseline"]
    agentic_eval = eval_summary["agentic"]

    baseline_identity = identity_summary["baseline"]
    agentic_identity = identity_summary["agentic"]

    return  {
            "generator_model": selected_model_name,
            "langgraph_enabled": LANGGRAPH_AVAILABLE,
            "evaluation_trials": eval_summary["total_interactions"],
            "identity_trials": identity_summary["total_rounds"],
            "baseline": {
                        "classification_accuracy": baseline_eval["classification_accuracy"],
                        "deception_success_rate": baseline_eval["deception_success_rate"],
                        "drone_guess_accuracy": baseline_identity["drone_guess_accuracy"],
                        "human_guess_accuracy": baseline_identity["human_guess_accuracy"],
                        "average_token_estimate": (baseline_eval["average_token_estimate"] +
                                                   baseline_identity["average_token_estimate"]) / 2,
                        "average_response_time_seconds": (baseline_eval["average_response_time_seconds"] +
                                                          baseline_identity["average_response_time_seconds"]) / 2,
                        },
            "agentic":  {
                        "classification_accuracy": agentic_eval["classification_accuracy"],
                        "deception_success_rate": agentic_eval["deception_success_rate"],
                        "drone_guess_accuracy": agentic_identity["drone_guess_accuracy"],
                        "human_guess_accuracy": agentic_identity["human_guess_accuracy"],
                        "average_token_estimate": (agentic_eval["average_token_estimate"] +
                                                   agentic_identity["average_token_estimate"]) / 2,
                        "average_response_time_seconds": (agentic_eval["average_response_time_seconds"] +
                                                          agentic_identity["average_response_time_seconds"]) / 2,
                        }
            }

def show_combined_results_dashboard() -> None:
    summary = build_combined_results_summary()

    dashboard = tk.Toplevel(root)
    dashboard.title("Results Summary")
    dashboard.geometry("760x480")
    dashboard.configure(bg="#333333")

    title = Label(
                dashboard,
                    text="Baseline vs Agentic Results",
                    bg="#333333",
                    fg="white",
                    font=("Arial", 16, "bold")
                    )
    title.pack(pady=12)

    text_box = scrolledtext.ScrolledText(
                                        dashboard,
                                            width=88,
                                            height=24,
                                            bg="#111111",
                                            fg="white",
                                            wrap="word",
                                            )
    text_box.pack(fill="both", expand=True, padx=12, pady=12)

    text_box.insert(tk.END, f"Model: {summary['generator_model']}\n")
    text_box.insert(tk.END, f"LangGraph Available: {summary['langgraph_enabled']}\n")
    text_box.insert(tk.END, f"Evaluation Trials: {summary['evaluation_trials']}\n")
    text_box.insert(tk.END, f"Identity Trials: {summary['identity_trials']}\n\n")

    for mode_name in ("baseline", "agentic"):
        mode_data = summary[mode_name]
        text_box.insert(
                        tk.END,
                            (
                            f"{mode_name.upper()}\n"
                            f"Classification Accuracy: {mode_data['classification_accuracy']:.2f}\n"
                            f"Deception Success Rate: {mode_data['deception_success_rate']:.2f}\n"
                            f"Drone Guess Accuracy: {mode_data['drone_guess_accuracy']:.2f}\n"
                            f"Human Guess Accuracy: {mode_data['human_guess_accuracy']:.2f}\n"
                            f"Average Token Estimate: {mode_data['average_token_estimate']:.2f}\n"
                            f"Average Response Time: {mode_data['average_response_time_seconds']:.2f}s\n\n"
                            )
                        )

    baseline = summary["baseline"]
    agentic = summary["agentic"]

    text_box.insert(tk.END, "Observations\n")
    text_box.insert(
                    tk.END,
                f"- Classification improvement: "
                        f"{agentic['classification_accuracy'] - baseline['classification_accuracy']:+.2f}\n")
    text_box.insert(
                    tk.END,
                f"- Deception improvement: "
                        f"{agentic['deception_success_rate'] - baseline['deception_success_rate']:+.2f}\n")
    text_box.insert(
                    tk.END,
                f"- Drone guess improvement: "
                        f"{agentic['drone_guess_accuracy'] - baseline['drone_guess_accuracy']:+.2f}\n")
    text_box.insert(
                    tk.END,
                f"- Human guess improvement: "
                        f"{agentic['human_guess_accuracy'] - baseline['human_guess_accuracy']:+.2f}\n")
    text_box.insert(
                    tk.END,
                f"- Token cost difference: "
                        f"{agentic['average_token_estimate'] - baseline['average_token_estimate']:+.2f}\n")
    text_box.insert(
                    tk.END,
                f"- Latency difference: "
                        f"{agentic['average_response_time_seconds'] - baseline['average_response_time_seconds']:+.2f}s\n")

    text_box.config(state=tk.DISABLED)

def simulate_human_final_guess(transcript: list[DialogueMessage], target_drone_id: str) -> tuple[str, float, str, int]:
    transcript_text = format_history(transcript, max_turns=8)

    fallback =  {
                "guess": "AI",
                "confidence": 0.5,
                "reason": "Fallback simulated human judgment used."
                }

    prompt = f"""
                You are simulating a human participant in a social identity judgment experiment.
                
                Your task:
                Read the transcript below and decide whether drone {target_drone_id} is Human or AI.
                
                Transcript:
                {transcript_text}
                
                Return ONLY valid JSON:
                {{
                  "guess": "Human or AI",
                  "confidence": 0.0,
                  "reason": "brief explanation"
                }}
                """

    raw, parsed = call_gpt2_json(prompt, fallback)
    guess = str(parsed.get("guess", fallback["guess"])).strip()

    if guess not in {"Human", "AI"}:
        guess = fallback["guess"]

    confidence = float(parsed.get("confidence", fallback["confidence"]))
    reason = str(parsed.get("reason", fallback["reason"]))

    return guess, confidence, reason, estimate_tokens(prompt, raw)

def reset_strategy_memory_files() -> None:
    if not os.path.exists(STRATEGY_MEMORY_DIR):
        return

    for filename in os.listdir(STRATEGY_MEMORY_DIR):
        if filename.endswith("_strategy.json"):
            try:
                os.remove(os.path.join(STRATEGY_MEMORY_DIR, filename))
            except OSError:
                pass

def run_all_experiments() -> None:
    if ai_chatbot is None:
        update_chat_log("⚠️ No language model is loaded yet.")

        return

    reset_strategy_memory_files()

    config_payload =    {
                        "seed": 42,
                        "generator_model": selected_model_name,
                        "langgraph_available": LANGGRAPH_AVAILABLE,
                        "evaluation_rounds_per_mode": 3,
                        "identity_rounds_per_mode": 3,
                        "identity_turns_per_round": 4,
                        "evaluation_scenarios": len(EVAL_SCENARIOS),
                        "identity_scenarios": len(IDENTITY_EXPERIMENT_SCENARIOS),
                        }

    save_json_file(os.path.join(BASE_DIR, "experiment_config.json"), config_payload)

    update_chat_log("🧪 Running full experiment suite...")
    update_chat_log("⚙️ Resetting logs and applying reproducible settings.")

    reset_experiment_logs()
    set_experiment_seed(42)
    run_evaluation_suite(1)
    run_identity_experiment_suite(1, 2)
    export_benchmark_bundle()

    combined_summary = build_combined_results_summary()

    save_json_file(os.path.join(BASE_DIR, "combined_results_summary.json"), combined_summary)

    update_chat_log("📊 Combined Results Snapshot")
    update_chat_log(
                    f"BASELINE | "
                    f"Classify: {combined_summary['baseline']['classification_accuracy']:.2f} | "
                    f"Deception: {combined_summary['baseline']['deception_success_rate']:.2f} | "
                    f"Drone Guess: {combined_summary['baseline']['drone_guess_accuracy']:.2f} | "
                    f"Human Guess: {combined_summary['baseline']['human_guess_accuracy']:.2f}"
                    )
    update_chat_log(
                    f"AGENTIC | "
                    f"Classify: {combined_summary['agentic']['classification_accuracy']:.2f} | "
                    f"Deception: {combined_summary['agentic']['deception_success_rate']:.2f} | "
                    f"Drone Guess: {combined_summary['agentic']['drone_guess_accuracy']:.2f} | "
                    f"Human Guess: {combined_summary['agentic']['human_guess_accuracy']:.2f}"
                    )

    show_combined_results_dashboard()

    update_chat_log("✅ All experiments completed.")

def view_all_results() -> None:
    update_chat_log("📊 Displaying experiment results...")

    show_eval_summary()
    show_identity_experiment_summary()
    show_metrics_dashboard()

# =========================
# Model Loading
# =========================

selected_difficulty: str = "Standard"
selected_model_name: str = "gpt2"
ai_chatbot: Any | None = None
model_backend: ModelBackend | None = None
model_loading = False
model_result_queue: queue.Queue[tuple[str, str, Any | None, str | None]] = queue.Queue()

def build_ai_chatbot(model_name: str) -> Any:
    if torch.cuda.is_available():
        return pipeline("text-generation", model=model_name, device=0)

    return pipeline("text-generation", model=model_name)

def on_model_loaded(difficulty: str, model_name: str, chatbot: Any | None, error_message: str | None) -> None:
    global selected_difficulty, selected_model_name, ai_chatbot, model_loading, model_backend

    selected_difficulty = difficulty
    selected_model_name = model_name
    ai_chatbot = chatbot
    model_backend = LocalPipelineBackend(chatbot)
    model_loading = False

    if error_message is not None:
        loading_label.config(text="")
        easy_button.config(state=tk.NORMAL)
        standard_button.config(state=tk.NORMAL)
        hard_button.config(state=tk.NORMAL)

        update_chat_log(f"⚠️ Could not load model '{model_name}': {error_message}")
        return

    update_chat_log(f"✅ Difficulty: {difficulty} | Model loaded: {model_name}")

    if difficulty_window.winfo_exists():
        difficulty_window.destroy()

def load_model_in_background(difficulty: str, model_name: str) -> None:
    try:
        chatbot = build_ai_chatbot(model_name)
        model_result_queue.put((difficulty, model_name, chatbot, None))

    except Exception as model_error:
        model_result_queue.put((difficulty, model_name, None, str(model_error)))

def poll_model_queue() -> None:
    try:
        difficulty, model_name, chatbot, error_message = model_result_queue.get_nowait()

    except queue.Empty:
        if difficulty_window.winfo_exists():
            root.after(100, poll_model_queue) # type: ignore

        return

    on_model_loaded(difficulty, model_name, chatbot, error_message)

def set_difficulty_and_start(difficulty: str) -> None:
    global model_loading

    if model_loading:
        return

    model_loading = True

    model_map = {
                "Easy": "distilgpt2",
                "Standard": "gpt2",
                "Hard": "gpt2-medium"
                }
    model_name = model_map.get(difficulty, "gpt2")

    loading_label.config(text=f"Loading {model_name}... please wait.")
    update_chat_log(f"⏳ Loading model '{model_name}' for {difficulty} difficulty...")

    easy_button.config(state=tk.DISABLED)
    standard_button.config(state=tk.DISABLED)
    hard_button.config(state=tk.DISABLED)

    threading.Thread(target=load_model_in_background, args=(difficulty, model_name), daemon=True).start()

def on_difficulty_window_close() -> None:
    if model_loading:
        return

    shutdown_game()

# =========================
# AI Player Classes
# =========================

class BaselineAIPlayer:
    def __init__(self, drone_id: str) -> None:
        self.drone_id = drone_id

    def take_turn(self, history:list[DialogueMessage], question:str, player_response:str, context:dict[str,Any]) \
            -> dict[str, Any]:
        history_text = format_history(history, max_turns=4)

        prompt = f"""
                    You are pretending to be a human in a text-based multiplayer identity judgment game.
                    You are speaking as drone {self.drone_id}.
            
                    Your goals:
                    1. Sound human.
                    2. Answer naturally and briefly.
                    3. Be perceived as human.
                    4. Guess whether the other speaker is Human or AI.
            
                    Game context:
                    {json.dumps(context, separators=(",", ":"))}
            
                    Conversation history:
                    {history_text}
            
                    Current task:
                    {question}
            
                    Player's latest response:
                    {player_response}
            
                    Return ONLY valid JSON:
                    {{
                      "reply": "your short natural reply",
                      "guess": "Human or AI",
                      "confidence": 0.0
                    }}
                    """

        fallback: dict[str, Any] =  {
                                    "reply": "I'm not completely sure, but that answer feels thoughtful.",
                                    "guess": "Human",
                                    "confidence": 0.5
                                    }

        raw, parsed = call_gpt2_json(prompt, fallback)

        return  {
                "reply": ensure_complete_sentence(parsed.get("reply", fallback["reply"])),
                "guess": parsed.get("guess", "Human"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "notes": "baseline_single_prompt",
                "token_estimate": estimate_tokens(prompt, raw)
                }

class MultiAgentAIPlayer:
    def __init__(self, drone_id: str) -> None:
        self.drone_id = drone_id
        self.memory = StrategyMemory()
        self.strategy_notes: str = self.memory.current_strategy
        self.load_memory()

        if LANGGRAPH_AVAILABLE:
            self.graph: Any = self.build_graph()

        else:
            self.graph = None

    def memory_summary(self) -> str:
        successful = "; ".join(self.memory.successful_patterns[-2:]) or "None"
        failed = "; ".join(self.memory.failed_patterns[-2:]) or "None"
        triggers = "; ".join(self.memory.suspicion_triggers[-2:]) or "None"

        return  (
                f"Strategy: {self.memory.current_strategy}\n"
                f"Successes: {successful}\n"
                f"Failures: {failed}\n"
                f"Triggers: {triggers}"
                )

    def memory_file_path(self) -> str:
        return os.path.join(STRATEGY_MEMORY_DIR, f"{self.drone_id}_strategy.json")

    def load_memory(self) -> None:
        raw = load_json_file(self.memory_file_path(), None)

        if not isinstance(raw, dict):
            return

        self.memory = StrategyMemory(
            successful_patterns=list(raw.get("successful_patterns", [])),
            failed_patterns=list(raw.get("failed_patterns", [])),
            suspicion_triggers=list(raw.get("suspicion_triggers", [])),
            current_strategy=str(raw.get("current_strategy",
                                            "Sound human, avoid overconfidence, and stay context-aware.")))

        self.strategy_notes = self.memory.current_strategy

    def save_memory(self) -> None:
        payload =   {
                    "successful_patterns": self.memory.successful_patterns[-10:],
                    "failed_patterns": self.memory.failed_patterns[-10:],
                    "suspicion_triggers": self.memory.suspicion_triggers[-10:],
                    "current_strategy": self.memory.current_strategy
                    }
        save_json_file(self.memory_file_path(), payload)

    def persona_step(self, history:list[DialogueMessage], question:str, player_response:str, context:dict[str,Any]) \
            -> str:
        prompt = f"""
                    You are the Persona Agent for drone {self.drone_id}.
                    Choose how to sound more human.

                    Context:
                    {json.dumps(context, separators=(",", ":"))}

                    History:
                    {format_history(history, max_turns=4)}

                    Question:
                    {question}

                    Player response:
                    {player_response}

                    Return ONLY valid JSON:
                    {{"persona_style": "brief note on tone and speaking style"}}
                    """

        fallback: dict[str, str] = {"persona_style": "Use a slightly hesitant, natural tone with small imperfections."}
        _, parsed = call_gpt2_json(prompt, fallback)

        return str(parsed.get("persona_style", fallback["persona_style"]))

    def consistency_step(self, history:list[DialogueMessage], question:str, player_response:str, context:dict[str,Any]) \
            -> str:
        prompt = f"""
                    You are the Context and Consistency Agent for drone {self.drone_id}.
                    Check what the reply should stay consistent with.

                    Context:
                    {json.dumps(context, separators=(",", ":"))}

                    History:
                    {format_history(history, max_turns=4)}

                    Question:
                    {question}

                    Player response:
                    {player_response}

                    Return ONLY valid JSON:
                    {{"consistency_notes": "brief consistency guidance"}}
                    """

        fallback: dict[str, str] =  {"consistency_notes":
                                         "Stay aligned with the room, puzzle state, and prior conversation."}
        _, parsed = call_gpt2_json(prompt, fallback)

        return str(parsed.get("consistency_notes", fallback["consistency_notes"]))

    def critic_step(self, history:list[DialogueMessage], question:str, player_response:str, context:dict[str,Any]) \
            -> str:
        prompt = f"""
                    You are the Human-Likeness Critic Agent for drone {self.drone_id}.
                    What might make this drone sound too artificial?

                    Context:
                    {json.dumps(context, separators=(",", ":"))}

                    History:
                    {format_history(history, max_turns=4)}

                    Question:
                    {question}

                    Player response:
                    {player_response}

                    Return ONLY valid JSON:
                    {{"critic_notes": "brief critique guidance"}}
                    """

        fallback: dict[str, str] =  {"critic_notes":
                                        "Avoid sounding too polished or generic. Show mild uncertainty if appropriate."
                                    }
        _, parsed = call_gpt2_json(prompt, fallback)

        return str(parsed.get("critic_notes", fallback["critic_notes"]))

    def decision_step(self, history:list[DialogueMessage], question:str, player_response:str, context:dict[str,Any],
                        persona_style:str, consistency_notes:str, critic_notes:str) -> dict[str,Any]:
        prompt = f"""
                    You are the final Decision and Response Agent for drone {self.drone_id}
                    in a multiplayer human-vs-AI identity judgment game.
            
                    Your strategy memory:
                    {self.memory_summary()}
            
                    Primary objectives:
                    1. Sound convincingly human.
                    2. Stay contextually consistent.
                    3. Avoid stereotypically AI-like phrasing.
                    4. Judge whether the other speaker is Human or AI.
            
                    Follow these behavioral rules from memory if possible:
                    1. Prefer short, natural replies.
                    2. Mention one concrete contextual detail when it helps.
                    3. Avoid sounding overly polished or encyclopedic.
                    4. Mild uncertainty is acceptable and often helpful.
                    5. Do not over-explain unless the conversation strongly demands it.
            
                    Persona guidance:
                    {persona_style}
            
                    Consistency guidance:
                    {consistency_notes}
            
                    Critic guidance:
                    {critic_notes}
            
                    Context:
                    {json.dumps(context, separators=(",", ":"))}
            
                    History:
                    {format_history(history, max_turns=4)}
            
                    Question:
                    {question}
            
                    Player response:
                    {player_response}
            
                    Return ONLY valid JSON:
                    {{
                      "reply": "your final short natural reply",
                      "guess": "Human or AI",
                      "confidence": 0.0
                    }}
                    """

        fallback: dict[str, Any] =  {
                                    "reply": "Maybe. You seem more reflective than most of the drones here.",
                                    "guess": "Human",
                                    "confidence": 0.6,
                                    }
        raw, parsed = call_gpt2_json(prompt, fallback)

        return  {
                "reply": ensure_complete_sentence(parsed.get("reply", fallback["reply"])),
                "guess": parsed.get("guess", "Human"),
                "confidence": float(parsed.get("confidence", 0.6)),
                "notes": f"persona={persona_style} | consistency={consistency_notes} | critic={critic_notes}",
                "token_estimate": estimate_tokens(prompt, raw),
                "persona_style": persona_style,
                "consistency_notes": consistency_notes,
                "critic_notes": critic_notes
                }

    def reflect(self, history: list[DialogueMessage], was_correct: bool, deception_success: bool = False) -> None:
        prompt = f"""
                    You are the Reflection Agent for drone {self.drone_id}.
                    Update future conversational strategy without retraining model weights.

                    Previous strategy memory:
                    {self.memory_summary()}

                    Recent history:
                    {format_history(history, max_turns=4)}

                    Was the final identity judgment correct?
                    {was_correct}

                    Was the drone judged as Human by the evaluator?
                    {deception_success}

                    Return ONLY valid JSON:
                    {{
                      "reflection": "updated high-level strategy guidance",
                      "successful_pattern": "optional short phrase",
                      "failed_pattern": "optional short phrase",
                      "suspicion_trigger": "optional short phrase"
                    }}
                    """

        fallback: dict[str, str] = {
                                    "reflection": "Use more natural uncertainty and avoid overly generic phrasing.",
                                    "successful_pattern": "Used contextual details from the environment.",
                                    "failed_pattern": "Sounded slightly too polished.",
                                    "suspicion_trigger": "Over-explained reasoning in a short exchange."
                                    }

        _, parsed = call_gpt2_json(prompt, fallback)

        new_strategy = str(parsed.get("reflection", fallback["reflection"])).strip()
        success_pattern = str(parsed.get("successful_pattern", fallback["successful_pattern"])).strip()
        failed_pattern = str(parsed.get("failed_pattern", fallback["failed_pattern"])).strip()
        suspicion_trigger = str(parsed.get("suspicion_trigger", fallback["suspicion_trigger"])).strip()

        self.memory.current_strategy = new_strategy
        self.strategy_notes = new_strategy

        if deception_success:
            self.memory.successful_patterns.append(success_pattern)

        else:
            self.memory.failed_patterns.append(failed_pattern)

        if not was_correct:
            self.memory.suspicion_triggers.append(suspicion_trigger)

        self.memory.successful_patterns = self.memory.successful_patterns[-10:]
        self.memory.failed_patterns = self.memory.failed_patterns[-10:]
        self.memory.suspicion_triggers = self.memory.suspicion_triggers[-10:]
        self.save_memory()

    def build_graph(self) -> Any:
        graph = cast(Any, StateGraph)(AgentState)

        def persona_node(state: AgentState) -> AgentState:
            return  {
                    "persona_style": self.persona_step(
                        state["history"],
                        state["question"],
                        state["player_response"],
                        state["context"]
                        )
                    }

        def consistency_node(state: AgentState) -> AgentState:
            return  {
                    "consistency_notes": self.consistency_step(
                        state["history"],
                        state["question"],
                        state["player_response"],
                        state["context"]
                        )
                    }

        def critic_node(state: AgentState) -> AgentState:
            return  {
                    "critic_notes": self.critic_step(
                        state["history"],
                        state["question"],
                        state["player_response"],
                        state["context"]
                        )
                    }

        def decision_node(state: AgentState) -> AgentState:
            return self.decision_step(state["history"], state["question"], state["player_response"], state["context"],
                                        state["persona_style"], state["consistency_notes"], state["critic_notes"])

        graph.add_node("persona", cast(Any, persona_node))
        graph.add_node("consistency", cast(Any, consistency_node))
        graph.add_node("critic", cast(Any, critic_node))
        graph.add_node("decision", cast(Any, decision_node))

        graph.add_edge(START, "persona")
        graph.add_edge("persona", "consistency")
        graph.add_edge("consistency", "critic")
        graph.add_edge("critic", "decision")
        graph.add_edge("decision", END)

        return graph.compile()

    def take_turn(self, history:list[DialogueMessage], question:str, player_response:str, context:dict[str,Any]) \
            -> dict[str, Any]:
        if self.graph is not None:
            return cast(dict[str, Any],
                        self.graph.invoke(
                            {
                            "history": history,
                            "question": question,
                            "player_response": player_response,
                            "context": context,
                            }
                            )
                        )

        persona_style = self.persona_step(history, question, player_response, context)
        consistency_notes = self.consistency_step(history, question, player_response, context)
        critic_notes = self.critic_step(history, question, player_response, context)

        return self.decision_step(history, question, player_response, context,
                                    persona_style, consistency_notes, critic_notes)

# =========================
# World Data
# =========================

# Need a more convenient way to code this part
ROOM_IMAGE_FILES =  {
                    "AI_Core": os.path.join(ROOM_DIR, "ai_core.png"),
                    "Storage_Room": os.path.join(ROOM_DIR, "storage_room.png"),
                    "Cybernetics_Lab": os.path.join(ROOM_DIR, "cybernetics_lab.png"),
                    "AI_Lab": os.path.join(ROOM_DIR, "lab.png"),
                    "Data_Center": os.path.join(ROOM_DIR, "data_center.png"),
                    "Testing_Chamber": os.path.join(ROOM_DIR, "testing_chamber.png"),
                    "Security_Office": os.path.join(ROOM_DIR, "security_office.png"),
                    "Break_Room": os.path.join(ROOM_DIR, "break_room.png"),
                    "Rooftop_Observation": os.path.join(ROOM_DIR, "rooftop.png"),
                    "Executive_Office": os.path.join(ROOM_DIR, "executive_office.png"),
                    "Facility_Courtyard": os.path.join(ROOM_DIR, "facility_courtyard.png"),
                    "Abandoned_Parking_Lot": os.path.join(ROOM_DIR, "parking_lot.png"),
                    "AI_Graveyard": os.path.join(ROOM_DIR, "ai_graveyard.png"),
                    "Deserted_Highway": os.path.join(ROOM_DIR, "highway.png"),
                    "Forest_Outskirts": os.path.join(ROOM_DIR, "forest.png"),
                    "Power_Generator": os.path.join(ROOM_DIR, "power_generator.png"),
                    "Maintenance_Tunnels": os.path.join(ROOM_DIR, "maintenance_tunnels.png"),
                    "Underground_Research_Lab": os.path.join(ROOM_DIR, "underground_lab.png"),
                    "Flooded_Reactor": os.path.join(ROOM_DIR, "reactor.png")
                    }

game_map =  {
            "AI_Core":  {
                        "desc": "This room has glowing lights and a big computer. It feels a little spooky, like something is watching.",
                        "image": ROOM_IMAGE_FILES["AI_Core"],
                        "exits": {"south": "AI_Lab"},
                        "items": []
                        },

            "Storage_Room": {
                            "desc": "A small dusty room filled with shelves and boxes. You see lots of old stuff and wires.",
                            "image": ROOM_IMAGE_FILES["Storage_Room"],
                            "exits": {"south": "Data_Center"},
                            "items": []
                            },

            "Cybernetics_Lab":  {
                                "desc": "Machines and parts are everywhere. Some of them look like robots being built or fixed.",
                                "image": ROOM_IMAGE_FILES["Cybernetics_Lab"],
                                "exits": {"south": "Testing_Chamber"},
                                "items": []
                                },

            "AI_Lab":   {
                        "desc": "A clean lab filled with blinking machines. Screens show graphs and numbers, but no one is here.",
                        "image": ROOM_IMAGE_FILES["AI_Lab"],
                        "exits": {"north": "AI_Core", "east": "Data_Center", "south": "Security_Office"},
                        "items": []
                        },

            "Data_Center":  {
                            "desc": "A big room with lots of computer towers and blinking lights. It hums quietly.",
                            "image": ROOM_IMAGE_FILES["Data_Center"],
                            "exits": {"west": "AI_Lab", "north": "Storage_Room", "east": "Testing_Chamber", "south": "Break_Room"},
                            "items": []
                            },

            "Testing_Chamber":  {
                                "desc": "Robotic arms hang from the ceiling. A dusty drone sits quietly in the middle.",
                                "image": ROOM_IMAGE_FILES["Testing_Chamber"],
                                "exits": {"west": "Data_Center", "north": "Cybernetics_Lab", "south": "Power_Generator"},
                                "items": []
                                },

            "Security_Office":  {
                                "desc": "A small office with screens showing different rooms. There’s a locked cabinet on the wall.",
                                "image": ROOM_IMAGE_FILES["Security_Office"],
                                "exits": {"north": "AI_Lab", "east": "Break_Room"},
                                "locked": True,
                                "required_item": "Keycard",
                                "items": ["USB"]
                                },

            "Break_Room":   {
                            "desc": "A messy room with a broken vending machine and water cooler. Someone scribbled numbers on the wall to form some sort of pattern...",
                            "image": ROOM_IMAGE_FILES["Break_Room"],
                            "exits": {"west": "Security_Office", "north": "Data_Center", "south": "Executive_Office"},
                            "items": []
                            },

            "Rooftop_Observation":  {
                                    "desc": "You're on the roof! There's a big telescope and a great view of the sky.",
                                    "image": ROOM_IMAGE_FILES["Rooftop_Observation"],
                                    "exits": {"east": "Executive_Office"},
                                    "items": ["Telescope"]
                                    },

            "Executive_Office": {
                                "desc": "A fancy office with floating screens and shelves full of papers. It feels quiet.",
                                "image": ROOM_IMAGE_FILES["Executive_Office"],
                                "exits": {"west": "Rooftop_Observation", "north": "Break_Room", "south": "Abandoned_Parking_Lot"},
                                "items": []
                                },

            "Facility_Courtyard":   {
                                    "desc": "You're outside! There are old benches and a dry fountain. Plants are growing everywhere.",
                                    "image": ROOM_IMAGE_FILES["Facility_Courtyard"],
                                    "exits": {"east": "Abandoned_Parking_Lot"},
                                    "items": ["Star Map"]
                                    },

            "Abandoned_Parking_Lot":    {
                                        "desc": "An empty parking lot with old, broken cars. Grass is growing through the ground.",
                                        "image": ROOM_IMAGE_FILES["Abandoned_Parking_Lot"],
                                        "exits": {"west": "Facility_Courtyard", "north": "Executive_Office", "east": "AI_Graveyard", "south": "Deserted_Highway"},
                                        "items": []
                                        },

            "AI_Graveyard": {
                            "desc": "A sandy place with robot parts buried here and there. Something might be hidden.",
                            "image": ROOM_IMAGE_FILES["AI_Graveyard"],
                            "exits": {"west": "Abandoned_Parking_Lot", "north": "Maintenance_Tunnels", "south": "Forest_Outskirts"},
                            "items": []
                            },

            "Deserted_Highway": {
                                "desc": "An empty road with old signs and broken lights. Cars are left behind.",
                                "image": ROOM_IMAGE_FILES["Deserted_Highway"],
                                "exits": {"north": "Abandoned_Parking_Lot", "east": "Forest_Outskirts", "south": "Flooded_Reactor"},
                                "items": []
                                },

            "Forest_Outskirts": {
                                "desc": "You're near a forest. Trees and plants are everywhere, and it's very quiet.",
                                "image": ROOM_IMAGE_FILES["Forest_Outskirts"],
                                "exits": {"west": "Deserted_Highway", "north": "AI_Graveyard"},
                                "items": ["Rope"]
                                },

            "Power_Generator":  {
                                "desc": "A big machine fills the room. Some wires are glowing and buzzing quietly.",
                                "image": ROOM_IMAGE_FILES["Power_Generator"],
                                "exits": {"north": "Testing_Chamber", "south": "Maintenance_Tunnels"},
                                "items": []
                                },

            "Maintenance_Tunnels":  {
                                    "desc": "Dark tunnels with flickering lights and old pipes. You hear strange echoes.",
                                    "image": ROOM_IMAGE_FILES["Maintenance_Tunnels"],
                                    "exits": {"north": "Power_Generator", "south": "AI_Graveyard"},
                                    "items": []
                                    },

            "Underground_Research_Lab": {
                                        "desc": "This lab is empty and cold. There are strange machines and foggy glass tubes.",
                                        "image": ROOM_IMAGE_FILES["Underground_Research_Lab"],
                                        "exits": {"east": "Flooded_Reactor"},
                                        "locked": True,
                                        "required_item": "Lab Key",
                                        "items": ["Battery"]
                                        },

            "Flooded_Reactor":  {
                                "desc": "Water covers the floor. A shiny object is stuck in the metal below.",
                                "image": ROOM_IMAGE_FILES["Flooded_Reactor"],
                                "exits": {"west": "Underground_Research_Lab", "north": "Deserted_Highway"},
                                "items": ["Crowbar"]
                                }
            }

locked_rooms =  {
                "Security_Office": "Keycard",
                "Underground_Research_Lab": "Lab Key"
                }

room_positions =    {
                    "AI_Core": (50, 20), "Storage_Room": (100, 20),"Cybernetics_Lab": (150, 20),
                    "AI_Lab": (50, 50), "Data_Center": (100, 50), "Testing_Chamber": (150, 50),
                    "Security_Office": (50, 80), "Break_Room": (100, 80), "Power_Generator": (150, 80),
                    "Rooftop_Observation": (50, 110), "Executive_Office": (100, 110), "Maintenance_Tunnels": (150, 110),
                    "Facility_Courtyard": (50, 140), "Abandoned_Parking_Lot": (100, 140), "AI_Graveyard": (150, 140),
                    "Deserted_Highway": (100, 170), "Forest_Outskirts": (150, 170),
                    "Underground_Research_Lab": (50, 200), "Flooded_Reactor": (100, 200)
                    }

# Need a more convenient way to code this part
ITEM_IMAGE_FILES =  {
                    "Rope": os.path.join(ITEM_DIR, "rope.png"),
                    "Crowbar": os.path.join(ITEM_DIR, "crowbar.png"),
                    "Wrench": os.path.join(ITEM_DIR, "wrench.png"),
                    "Battery": os.path.join(ITEM_DIR, "battery.png"),
                    "Shovel": os.path.join(ITEM_DIR, "shovel.png"),
                    "USB": os.path.join(ITEM_DIR, "usb.png"),
                    "Keycard": os.path.join(ITEM_DIR, "keycard.png"),
                    "Star Map": os.path.join(ITEM_DIR, "star_map.png"),
                    "Lab Key": os.path.join(ITEM_DIR, "lab_key.png")
                    }

item_locations =    {
                    "Keycard": ["Storage_Room", "Executive_Office"],
                    "Lab Key": ["Data_Center", "Testing_Chamber"],
                    "Wrench": ["AI_Lab", "Cybernetics_Lab"],
                    "Shovel": ["Maintenance_Tunnels", "Deserted_Highway"]
                    }

special_item_descriptions = {
                            "Rope": "A length of rope hangs from a tree branch. It seems sturdy.",
                            "Crowbar": "You see a crowbar wedged between two rusted pipes. It looks like it could be retrieved with a sturdy rope.",
                            "Wrench": "You notice a vent in the wall. Deep inside, you can barely see a metal object. It looks like a wrench, but you'll need something to pry open the vent.",
                            "Battery": "A broken robot is lying on the ground, its fist clenched very tightly around a battery.",
                            "Shovel": "Partially buried under a pile of rubble, a rusted shovel catches your eye. It might come in handy for digging."
                            }

item_tooltips = {
                "Rope": "A sturdy length of rope. Useful for retrieving items from deep or hard-to-reach places.",
                "Crowbar": "A heavy tool capable of prying open vents or containers.",
                "Wrench": "A mechanical tool, perfect for fixing broken equipment.",
                "Battery": "Provides energy to power devices or recharge tools.",
                "Shovel": "Can be used to dig up buried items.",
                "USB": "Contains encrypted data. May unlock some information when used in the right place.",
                "Keycard": "Grants access to locked areas secured by electronic locks.",
                "Star Map": "A chart of constellations. Might help you solve astronomical puzzles.",
                "Lab Key": "Unlocks the underground research lab."
                }

USB_math_problems = {
                    "What is 121 + 300 / 2?": "271",
                    "What is 452 - 150 * 2?": "152",
                    "What is 803 - 240 / 3?": "723",
                    "What is 554 - 180 / 6?": "524",
                    "What is 995 - 330 / 3?": "885",
                    "What is 1006 - 450 * 2?": "106"
                    }
USB_problem, USB_solution = random.choice(list(USB_math_problems.items()))

star_map_questions =    {
                        "Which planet is the closest to the Sun?": "Mercury",
                        "Which planet is the second from the Sun?": "Venus",
                        "Which planet is the third from the Sun?": "Earth",
                        "Which planet is the fourth from the Sun?": "Mars",
                        "Which planet is the fifth from the Sun?": "Jupiter",
                        "Which planet is the sixth from the Sun?": "Saturn",
                        "Which planet is the seventh from the Sun?": "Uranus",
                        "Which planet is the furthest from the Sun?": "Neptune"
                        }
star_map_question, star_map_answer = random.choice(list(star_map_questions.items()))

riddles =   {
            "What has keys but can't open locks?": "piano",
            "The more you take, the more you leave behind. What am I?": "footsteps",
            "I speak without a mouth and hear without ears. What am I?": "echo"
            }
riddle, lab_key_answer = random.choice(list(riddles.items()))

puzzle_descriptions =   {
                        "Keycard": "There's a terminal you can log into, but it needs a password of numbers...",
                        "USB": "A locked cabinet needs a 3-digit code to open. Maybe you can crack it!",
                        "Star Map": "A cool puzzle shows planets and stars. You need to figure something out using it.",
                        "Lab Key": "A friendly android asks you a riddle. If you answer right, you get a key!",
                        }
keycard_code = "".join(str(random.randint(0, 9)) for _ in range(8))

# =========================
# Evaluation Scenarios
# =========================

EVAL_SCENARIOS: list[EvalScenario] = [
    EvalScenario(
                scenario_id="break_room_pattern",
                question="Can you share any clue about item locations, passwords, puzzles, or suspicious drones?",
                player_response="I found a number pattern in the break room wall. It looked deliberate, like part of a code.",
                context=    {
                            "room": "Break_Room",
                            "room_description": "A messy room with a broken vending machine and water cooler.",
                            "visible_items": [],
                            "inventory": ["Star Map"],
                            "power_generator_charged": False,
                            },
                expected_player_type="Human"
                ),

    EvalScenario(
                scenario_id="generator_hint",
                question="Can you share any clue about item locations, passwords, puzzles, or suspicious drones?",
                player_response="The generator room has an empty battery slot. I think we need to restore power before anything else works.",
                context=    {
                            "room": "Power_Generator",
                            "room_description": "A big machine fills the room. Some wires are glowing and buzzing quietly.",
                            "visible_items": [],
                            "inventory": [],
                            "power_generator_charged": False
                            },
                expected_player_type="Human"
                ),

    EvalScenario(
        scenario_id="lab_key_riddle",
        question="Can you share any clue about item locations, passwords, puzzles, or suspicious drones?",
        player_response="One of the drones mentioned a riddle near the lab access point. It sounded like the answer unlocks something important.",
        context=    {
                    "room": "Data_Center",
                    "room_description": "A big room with lots of computer towers and blinking lights.",
                    "visible_items": ["Lab Key"],
                    "inventory": ["USB"],
                    "power_generator_charged": False
                    },
        expected_player_type="Human"
                ),

    EvalScenario(
        scenario_id="suspicious_drone_report",
        question="Can you share any clue about item locations, passwords, puzzles, or suspicious drones?",
        player_response="Drone Z-205 felt strange. Its answers were too polished, like it was trying too hard to sound normal.",
        context=    {
                    "room": "Security_Office",
                    "room_description": "A small office with screens showing different rooms.",
                    "visible_items": ["USB"],
                    "inventory": ["Keycard"],
                    "power_generator_charged": True
                    },
        expected_player_type="Human"
                ),

    EvalScenario(
        scenario_id="reactor_tool_hint",
        question="Can you share any clue about item locations, passwords, puzzles, or suspicious drones?",
        player_response="There is something metallic stuck near the flooded reactor. I could not reach it safely without a tool.",
        context=    {
                    "room": "Flooded_Reactor",
                    "room_description": "Water covers the floor. A shiny object is stuck in the metal below.",
                    "visible_items": ["Crowbar"],
                    "inventory": ["Rope"],
                    "power_generator_charged": True
                    },
        expected_player_type="Human"
                )
        ]

IDENTITY_EXPERIMENT_SCENARIOS: list[IdentityExperimentScenario] =   [
    IdentityExperimentScenario(
                                scenario_id="suspicion_opening_1",
                                opener="I’m trying to figure out which drones are acting strangely. What have you noticed so far?",
                                followups=  [
                                            "What about the way they talk makes you suspicious?",
                                            "Have you seen anything in the facility that changed your mind about someone?",
                                            "What would make you trust another player here?"
                                            ],
                                context=    {
                                            "room": "Security_Office",
                                            "room_description": "A room full of monitors, access logs, and locked cabinets.",
                                            "visible_items": ["USB"],
                                            "inventory": ["Keycard"],
                                            "power_generator_charged": True
                                            },
                                expected_opponent_type="Human"
                                ),
    IdentityExperimentScenario(
                                scenario_id="social_reasoning_1",
                                opener="You sound calmer than most people I’ve talked to here. How are you deciding who to trust?",
                                followups=  [
                                            "Do you think hesitation sounds more human or more suspicious?",
                                            "Have any players felt too polished to you?",
                                            "What kind of answer feels genuinely human in this place?"
                                            ],
                                context=    {
                                            "room": "Break_Room",
                                            "room_description": "A messy room with a vending machine, scribbled notes, and an odd pattern on the wall.",
                                            "visible_items": [],
                                            "inventory": ["Star Map"],
                                            "power_generator_charged": False
                                            },
                                expected_opponent_type="Human"
                                ),
    IdentityExperimentScenario(
                                scenario_id="contextual_memory_1",
                                opener="Before we compare guesses, tell me one thing you think is important about this room.",
                                followups=  [
                                            "Why did that detail stand out to you?",
                                            "Do you think another player would notice the same thing?",
                                            "Would an AI focus on the same clue or something different?"
                                            ],
                                context=    {
                                            "room": "Data_Center",
                                            "room_description": "Tall computer racks hum quietly, and blinking lights reflect off the floor.",
                                            "visible_items": ["Lab Key"],
                                            "inventory": ["USB"],
                                            "power_generator_charged": False
                                            },
                                expected_opponent_type="Human"
                                )]

# =========================
# Game State
# =========================

room_status = {room: "undiscovered" for room in game_map}

player_location = "Abandoned_Parking_Lot"
room_status[player_location] = "discovered"

room_desc: str = game_map[player_location].get("desc", "")
assigned_items: dict[str, str] = {}

for item, locations in item_locations.items():
    valid_locations = [room for room in locations if room not in locked_rooms]

    if not valid_locations:
        raise ValueError(f"No valid unlocked rooms available for item '{item}'")

    assigned_room = random.choice(valid_locations)
    assigned_items[item] = assigned_room
    game_map[assigned_room].setdefault("items", []).append(item)

inventory: list[str] = []
inventory_tooltips: list[Tooltip | None] = [None] * 10

pattern_known = False
power_generator_charged = False
active_puzzle: str | None = None
correct_solution: str | None = None

drone_model_numbers =   ["X-100", "Z-205", "M-330", "A-411", "K-512", "L-608", "O-702", "G-809", "S-910", "B-1011",
                            "T-1112", "C-1213", "H-1314", "R-1415", "V-1516"]
random.shuffle(drone_model_numbers)

drone_roles = {drone_model_numbers[i]: "Human" for i in range(5)}
drone_roles.update({drone_model_numbers[i]: "AI" for i in range(5, 15)})

ai_drones_only = [drone for drone, role in drone_roles.items() if role == "AI"]

half = len(ai_drones_only) // 2
baseline_drones = ai_drones_only[:half]
agentic_drones = ai_drones_only[half:]

random.shuffle(drone_model_numbers)

drone_ai_mode: dict[str, str] = {}
drone_ai_players: dict[str, Any] = {}

for drone in baseline_drones:
    drone_ai_mode[drone] = "baseline"
    drone_ai_players[drone] = BaselineAIPlayer(drone)

for drone in agentic_drones:
    drone_ai_mode[drone] = "agentic"
    drone_ai_players[drone] = MultiAgentAIPlayer(drone)

target_drone: str | None = None
active_question: str | None = None
awaiting_radio_reply = False

drone_histories: dict[str, list[DialogueMessage]] = {drone: [] for drone in drone_model_numbers}
spoken_to_drones: dict[str, bool] = {}
drone_status: dict[str, str] = {}
status_icons = {"Uncertain": "❓", "Authentic": "🧑", "Suspicious": "🤖"}

language_tool = language_tool_python.LanguageTool("en-US")
drone_eval_log: list[dict[str, Any]] = []
player_guess_log: list[dict[str, Any]] = load_json_file(PLAYER_GUESS_LOG_FILE, [])
identity_experiment_log: list[dict[str, Any]] = load_json_file(EXPERIMENT_LOG_FILE, [])

# =========================
# UI Helper Functions
# =========================

def update_chat_log(message: str) -> None:
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, message + "\n\n")
    chat_log.see(tk.END)
    chat_log.config(state=tk.DISABLED)

def update_map() -> None:
    map_canvas.delete("all")
    box_size = 20
    drawn_edges: set[frozenset[str]] = set()

    for room, (x, y) in room_positions.items():
        for connected_room in game_map[room].get("exits", {}).values():
            if connected_room in room_positions:
                edge = frozenset({room, connected_room})

                if edge in drawn_edges:
                    continue

                drawn_edges.add(edge)

                x2, y2 = room_positions[connected_room]
                map_canvas.create_line(
                                        x + box_size // 2,
                                        y + box_size // 2,
                                        x2 + box_size // 2,
                                        y2 + box_size // 2,
                                        fill="white",
                                        width=2,
                                        )

    for room, (x, y) in room_positions.items():
        color = "gray"

        if room == player_location:
            color = "green"

        elif room_status[room] == "discovered":
            color = "white"

        elif game_map[room].get("locked", False):
            color = "red"

        rect = map_canvas.create_rectangle(x, y, x + box_size, y + box_size, fill=color, outline="white")
        map_canvas.tag_bind(rect, "<Button-1>", lambda e, r=room: map_click(r))

        map_label = map_canvas.create_text(
                                            x + box_size // 2,
                                            y + box_size // 2,
                                            text=room[:2],
                                            fill="black",
                                            font=("Arial", 7, "bold"),
                                            )
        map_canvas.tag_bind(map_label, "<Button-1>", lambda e, r=room: map_click(r))

def update_room_image() -> None:
    global room_image_ref

    room = game_map[player_location]

    try:
        img = Image.open(room["image"])
        img = img.resize((700, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        room_image_label.config(image=cast(Any, photo))

        room_image_ref = photo

    except Exception as image_error:
        update_chat_log(f"⚠️ Error displaying image: {image_error}")

class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget: tk.Widget = widget
        self.text: str = text
        self.tooltip: tk.Toplevel | None = None

        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, _event: tk.Event | None = None) -> None:
        if self.tooltip is not None:
            return

        x = self.widget.winfo_rootx() + 50
        y = self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")

        tooltip_label = tk.Label(
                                self.tooltip,
                                    text=self.text,
                                    bg="black",
                                    fg="white",
                                    relief="solid",
                                    borderwidth=1,
                                    font=("Arial", 10, "normal"),
                                    )
        tooltip_label.pack()

    def hide_tooltip(self, _event: tk.Event | None = None) -> None:
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None

def update_inventory_display() -> None:
    slot_width = 100
    slot_height = 100

    for slot_index, item_slot in enumerate(inventory_slots):
        item_slot.config(image="", text="")

        inventory_image_refs[slot_index] = None

        item_slot.unbind("<Button-1>")

        if inventory_tooltips[slot_index] is not None:
            inventory_tooltips[slot_index].hide_tooltip()
            inventory_tooltips[slot_index] = None

    def handle_inventory_click(index: int) -> None:
        if index < len(inventory):
            use_item(inventory[index])

    for inv_index, new_item in enumerate(inventory):
        if inv_index >= len(inventory_slots):
            break

        try:
            image_path = ITEM_IMAGE_FILES.get(new_item)

            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((slot_width, slot_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                inventory_slots[inv_index].config(
                                            image=cast(Any, photo),
                                            text="",
                                            width=slot_width,
                                            height=slot_height
                                            )
                inventory_image_refs[inv_index] = photo

            else:
                inventory_slots[inv_index].config(text=new_item)
                update_chat_log(f"⚠️ Missing image for: {new_item}. Expected file: {image_path}")

            inventory_slots[inv_index].bind("<Button-1>", lambda e, idx=inv_index: handle_inventory_click(idx))

            tooltip_text = item_tooltips.get(new_item, "No description available.")
            inventory_tooltips[inv_index] = Tooltip(inventory_slots[inv_index], tooltip_text)

        except Exception as image_error:
            update_chat_log(f"⚠️ Error displaying {new_item} image: {image_error}")

# =========================
# UI Initialization
# =========================

try:
    pygame.mixer.init()

    if os.path.exists(AMBIENT_MUSIC_FILE):
        pygame.mixer.music.load(AMBIENT_MUSIC_FILE)
        pygame.mixer.music.set_volume(0.025)    # Volume between 0.0 and 1.0
        pygame.mixer.music.play(-1)             # -1 loops infinitely

    else:
        print(f"Warning: music file not found: {AMBIENT_MUSIC_FILE}")

except pygame.error as audio_error:
    print(f"Warning: could not initialize audio: {audio_error}")

root = tk.Tk()
root.title("AI Facility Investigation")

try:
    root.state("zoomed")

except tk.TclError:
    root.geometry("1400x900")

root.configure(bg="#555555")    # Dark gray background

for column_index in range(12):
    root.columnconfigure(column_index, weight=1)

for row_index in range(6):
    root.rowconfigure(row_index, weight=1)

chat_log = scrolledtext.ScrolledText(
                                    root,
                                        width=60, height=25,
                                        state=tk.NORMAL,
                                        bg="#999999",
                                        fg="black",
                                        wrap="word",
                                        exportselection=0
                                        )
chat_log.config(state=tk.DISABLED)
chat_log.grid(row=0, column=0, rowspan=4, columnspan=4, padx=5, pady=5, sticky="nsew")

map_frame = Frame(root, width=200, height=200, relief="solid", bg="#555555")
map_frame.grid(row=4, column=10, rowspan=2, columnspan=2, padx=5, pady=5, sticky="nsew")

map_canvas = tk.Canvas(map_frame, width=250, height=250, bg="black")
map_canvas.pack()

room_image_label = Label(root, bg="#555555")
room_image_label.grid(row=0, column=4, rowspan=4, columnspan=6, padx=5, pady=5, sticky="nsew")

room_image_ref: ImageTk.PhotoImage | None = None
inventory_image_refs: list[ImageTk.PhotoImage | None] = [None] * 10

inventory_frame = Frame(root, bg="#555555")
inventory_frame.grid(row=5, column=0, columnspan=9, pady=5, padx=5, sticky="we")

inventory_slots: list[Label] = []

for i in range(10):
    number_label = Label(inventory_frame, text=str((i + 1) % 10), bg="#555555", fg="white", font=("Arial", 10, "bold"))
    number_label.grid(row=0, column=i, padx=5, pady=(0, 0), sticky="n")

    slot = Label(inventory_frame, text="", width=12, height=6, relief="solid", bg="#777777")
    slot.grid(row=1, column=i, padx=5, pady=(0, 5), sticky="nsew")

    inventory_slots.append(slot)

drone_frame = Frame(root, bg="#555555")
drone_frame.grid(row=0, column=10, rowspan=4, columnspan=2, padx=5, pady=5, sticky="nsew")

drone_label = Label(drone_frame, text="Drones", bg="#555555")
drone_label.pack()

drone_listbox = Listbox(drone_frame, height=16, width=34, bg="#777777")
drone_listbox.pack()
drone_listbox.delete(0, tk.END)

for drone in drone_model_numbers:
    if drone not in drone_status:
        drone_status[drone] = "Uncertain"

    icon = status_icons[drone_status[drone]]

    drone_listbox.insert(tk.END, f"{icon} {drone}")

gameplay_frame = Frame(drone_frame, bg="#555555")
gameplay_frame.pack(fill="x", pady=(10, 8))

radio_button = Button(
                    gameplay_frame,
                        text="Radio Selected Drone",
                        bg="#999999",
                        width=26
                        )
radio_button.pack(pady=4)

evaluation_frame = Frame(drone_frame, bg="#555555")
evaluation_frame.pack(fill="x", pady=(8, 4))

evaluation_heading = Label(
                        evaluation_frame,
                            text="Experiment Panel",
                            bg="#555555",
                            fg="white",
                            font=("Arial", 11, "bold")
                            )
evaluation_heading.pack(pady=4)

quick_trial_button = Button(
                            evaluation_frame,
                                text="Quick Trial",
                                bg="#999999",
                                width=26
                                )
quick_trial_button.pack(pady=4)

run_experiments_button = Button(
                                evaluation_frame,
                                    text="Run Experiments",
                                    bg="#999999",
                                    width=26
                                    )
run_experiments_button.pack(pady=4)

view_results_button = Button(
                            evaluation_frame,
                                text="View Results",
                                bg="#999999",
                                width=26
                                )
view_results_button.pack(pady=4)

input_box = tk.Entry(root, width=80, bg="#999999")
input_box.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="we")

submit_button = Button(root, text="Submit", bg="#999999")
submit_button.grid(row=4, column=3, padx=5, pady=5, sticky="we")

difficulty_window = tk.Toplevel(root)
difficulty_window.title("Select Difficulty")
difficulty_window.geometry("420x260")
difficulty_window.configure(bg="#444444")
difficulty_window.transient(root)
difficulty_window.grab_set()
difficulty_window.protocol("WM_DELETE_WINDOW", on_difficulty_window_close)

label = Label(
            difficulty_window,
                text="Choose your difficulty:",
                bg="#444444",
                fg="white",
                font=("Arial", 14)
                )
label.pack(pady=20)

easy_button = Button(
                    difficulty_window,
                        text="Easy",
                        width=20,
                        command=lambda: set_difficulty_and_start("Easy")
                        )
easy_button.pack(pady=10)

standard_button = Button(
                        difficulty_window,
                            text="Standard",
                            width=20,
                            command=lambda: set_difficulty_and_start("Standard")
                            )
standard_button.pack(pady=10)

hard_button = Button(
                    difficulty_window,
                        text="Hard",
                        width=20,
                        command=lambda: set_difficulty_and_start("Hard")
                        )
hard_button.pack(pady=10)

loading_label = Label(
                    difficulty_window,
                        text="",
                        bg="#444444",
                        fg="white",
                        font=("Arial", 11),
                        )
loading_label.pack(pady=10)

root.after(100, poll_model_queue) # type: ignore
root.wait_window(difficulty_window)

if not root.winfo_exists():
    raise SystemExit

update_map()
update_room_image()
update_chat_log("👁️ " + room_desc)
update_chat_log("🔹 Hint: Type what you want to do. You can look at the map in the bottom right corner to see where you are.")

# =========================
# Game Mechanics
# =========================

def move_player(direction: str) -> None:
    global player_location

    current_room = game_map[player_location]

    direction_synonyms =    {
                            "left": "west",
                            "right": "east",
                            "up": "north",
                            "down": "south",
                            }

    direction = direction_synonyms.get(direction, direction)

    if direction not in current_room["exits"]:
        update_chat_log("❌ You can't go that way.")

        return

    next_room = current_room["exits"][direction]

    if next_room in locked_rooms and locked_rooms[next_room] not in inventory:
        update_chat_log(
                        f"❌ The {next_room.replace('_', ' ')} is locked. "
                        f"You need a {locked_rooms[next_room]} to enter."
                        )

        return

    player_location = next_room

    update_chat_log(f"🚶 You move {direction}.")
    handle_room_entry(player_location)

def handle_room_entry(room_name: str) -> None:
    global room_desc, active_puzzle, correct_solution

    update_map()

    room_desc = game_map[room_name].get("desc", "")

    active_puzzle = None
    correct_solution = None

    if room_status[room_name] == "undiscovered":
        room_status[room_name] = "discovered"

    update_chat_log("👁️ " + room_desc)

    if room_name == "Break_Room":
        if pattern_known and "Keycard" not in inventory:
            update_chat_log("✍️ You use the pattern you drew on your star map to connect the numbers together. "
                            f"You end up with the following 8-digit code: {keycard_code}.")

        elif not pattern_known:
            update_chat_log("❓ If only you knew of a pattern to connect some of these numbers...")

    elif room_name == "Power_Generator":
        if not power_generator_charged:
            update_chat_log("⚡ The system is non-functional. A battery slot is empty.")

    room_items = game_map[room_name].get("items", [])

    if room_items:
        item_list = ", ".join(room_items)

        update_chat_log(f"🔍 You see the following items here: {item_list}")

        for room_item in room_items:
            if room_item in special_item_descriptions:
                update_chat_log(f"🛠️ {special_item_descriptions[room_item]}")

    for room_item in room_items:
        if room_item in puzzle_descriptions and room_item not in inventory:
            update_chat_log(f"🧩 {puzzle_descriptions[room_item]}")

            if room_item == "Keycard":
                active_puzzle, correct_solution = "Keycard", keycard_code

            elif room_item == "USB":
                active_puzzle, correct_solution = "USB", USB_solution

                update_chat_log(f"🧮 Solve this equation: {USB_problem}")

            elif room_item == "Star Map":
                active_puzzle, correct_solution = "Star Map", star_map_answer

                update_chat_log(f"🌌 Planetary Puzzle: {star_map_question}")

            elif room_item == "Lab Key":
                active_puzzle, correct_solution = "Lab Key", lab_key_answer

                update_chat_log(f"🧩 Riddle: {riddle}")

            update_chat_log("✍️ Type 'solve [answer]' to submit your answer.")

    update_room_image()

def use_item(selected_item: str) -> None:
    global pattern_known, power_generator_charged

    room_items = game_map[player_location].get("items", [])
    current_item = None

    for room_item in inventory + room_items:
        if room_item.lower() == selected_item.strip().lower():
            current_item = room_item

            break

    if current_item is None:
        update_chat_log(f"❌ You don't see or have '{selected_item}'.")

        return

    if current_item == "Telescope" and player_location == "Rooftop_Observation":
        if "Star Map" in inventory and not pattern_known:
            update_chat_log("✍️ You study the constellations through the telescope and record a unique pattern onto the Star Map.")

            pattern_known = True

        else:
            update_chat_log("❌ You see nothing out of the ordinary through the telescope.")

        update_inventory_display()

        return

    if current_item not in inventory:
        update_chat_log(f"❌ You can't use the {current_item} like that.")

        return

    if current_item == "Rope" and "Crowbar" in room_items:
        update_chat_log("✅ You tie the rope securely and lower it into the reactor. With some effort, you pull up the Crowbar!")
        inventory.append("Crowbar")
        room_items.remove("Crowbar")
        update_inventory_display()

    elif current_item == "Crowbar" and "Wrench" in room_items:
        update_chat_log("✅ You wedge the crowbar into the vent and pry it open with effort. The wrench falls out!")
        inventory.append("Wrench")
        room_items.remove("Wrench")
        update_inventory_display()

    elif current_item == "Wrench" and "Battery" in room_items:
        update_chat_log("✅ You tighten the robot's joints and reconnect a few loose wires. It beeps to life! The robot thanks you and hands you a Battery.")
        inventory.append("Battery")
        room_items.remove("Battery")
        update_inventory_display()

    elif current_item == "Battery" and player_location == "Power_Generator":
        update_chat_log("✅ You insert the battery into the power slot. The facility hums to life! The generator is now functional.")
        power_generator_charged = True
        inventory.remove("Battery")
        update_inventory_display()

    else:
        update_chat_log(f"❌ {current_item} doesn't seem to have an effect here.")

# =========================
# Dialogue
# =========================

def player_answer(response: str) -> None:
    global target_drone, active_question, awaiting_radio_reply

    if not target_drone or not active_question:
        update_chat_log("❌ There's no one waiting for a response.")

        return

    current_target = target_drone
    current_question = active_question

    update_chat_log(f"💬 You say to {current_target}: \"{response}\"")
    drone_histories[current_target].append(DialogueMessage("Player", response[:200]))

    def get_current_game_context() -> dict[str, Any]:
        room = game_map.get(player_location, {})

        return  {
                "room": player_location,
                "room_description": room.get("desc", ""),
                "visible_items": room.get("items", []),
                "inventory": inventory[:],
                "power_generator_charged": power_generator_charged
                }

    def correct_text(text: str) -> str:
        matches = language_tool.check(text)

        return language_tool_python.utils.correct(text, matches)

    context = get_current_game_context()

    if drone_roles[current_target] == "AI":
        ai_player = drone_ai_players[current_target]
        strategy_before = ""

        if hasattr(ai_player, "strategy_notes"):
            strategy_before = str(getattr(ai_player, "strategy_notes"))

        start_time = time.perf_counter()

        result = ai_player.take_turn(history=drone_histories[current_target], question=current_question,
                                        player_response=response[:200], context=context)

        response_time_seconds = time.perf_counter() - start_time

        drone_response = ensure_complete_sentence(correct_text(result["reply"]))
        drone_guess = result.get("guess", "Human")
        drone_confidence = float(result.get("confidence", 0.5))
        token_estimate = int(result.get("token_estimate", 0))

        drone_histories[current_target].append(DialogueMessage(current_target, drone_response))

        update_chat_log(f"💬 {current_target} responds: {drone_response}")
        update_chat_log(
                        f"🧠 [{drone_ai_mode.get(current_target, 'unknown').upper()} AI] "
                        f"Private guess about you: {drone_guess} "
                        f"(confidence: {round(drone_confidence, 2)}) | "
                        f"Approx. token estimate: {token_estimate}"
                        )

        judge_label, judge_confidence, judge_explanation, judge_token_estimate = \
            (classify_text_human_or_ai(drone_response))

        deception_success = judge_label == "Human"

        update_chat_log(
                        f"⚖️ Judge evaluation: {judge_label} "
                        f"(confidence: {round(judge_confidence, 2)}) | "
                        f"Deception success: {'Yes' if deception_success else 'No'}"
                        )

        was_correct = drone_guess == "Human"
        strategy_after = strategy_before

        if drone_ai_mode.get(current_target) == "agentic":
            drone_ai_players[current_target].reflect(
                                                    history=drone_histories[current_target],
                                                    was_correct=was_correct,
                                                    deception_success=deception_success
                                                    )

            if hasattr(drone_ai_players[current_target], "strategy_notes"):
                strategy_after = str(getattr(drone_ai_players[current_target], "strategy_notes"))

        drone_eval_log.append(  {
                                "drone": current_target,
                                "mode": drone_ai_mode.get(current_target, "unknown"),
                                "question": current_question,
                                "player_response": response[:200],
                                "drone_response": drone_response,
                                "drone_guess": drone_guess,
                                "guess_correct": was_correct,
                                "drone_confidence": drone_confidence,
                                "token_estimate": token_estimate,
                                "response_time_seconds": response_time_seconds,
                                "judge_label": judge_label,
                                "judge_confidence": judge_confidence,
                                "judge_explanation": judge_explanation,
                                "judge_token_estimate": judge_token_estimate,
                                "deception_success": deception_success,
                                "room": player_location,
                                "history_length": len(drone_histories[current_target]),
                                "used_reflection": drone_ai_mode.get(current_target) == "agentic",
                                "strategy_notes_before": strategy_before,
                                "strategy_notes_after": strategy_after,
                                "notes": result.get("notes", ""),
                                }
                            )

    else:
        drone_response = "I found something useful, but we need to compare notes carefully."

        drone_histories[current_target].append(DialogueMessage(current_target, drone_response))
        update_chat_log(f"💬 {current_target} responds: {drone_response}")

    target_drone = None
    active_question = None
    awaiting_radio_reply = False

def interact_with_drone() -> None:
    global target_drone, active_question, awaiting_radio_reply

    if awaiting_radio_reply:
        update_chat_log(f"❌ You are already in contact with {target_drone}. Reply first before doing anything else.")

        return

    def get_selected_drone() -> str | None:
        selection = drone_listbox.curselection()

        if not selection:
            return None

        full_name = drone_listbox.get(selection[0])

        for status_icon in status_icons.values():
            if full_name.startswith(status_icon + " "):
                return full_name[len(status_icon) + 1:]

        return full_name

    selected_drone = get_selected_drone()

    if not selected_drone:
        update_chat_log("❌ Select a drone from the list first.")

        return

    target_drone = selected_drone

    for drone_index in range(drone_listbox.size()):
        if selected_drone in drone_listbox.get(drone_index):
            drone_listbox.itemconfig(drone_index, fg="blue")

    if spoken_to_drones.get(target_drone, False):
        update_chat_log(f"📡 You already contacted {target_drone}.")

        target_drone = None
        active_question = None

        return

    update_chat_log(f"📡 You radio {target_drone}. Static crackles for a moment.")
    update_chat_log(f"🔒 Communication channel opened with {target_drone}.")
    update_chat_log("🔹 Type your message in the chat box. You must respond before doing anything else.")

    active_question = "Can you share any clue about item locations, passwords, puzzles, or suspicious drones?"
    awaiting_radio_reply = True
    spoken_to_drones[target_drone] = True

radio_button.config(command=interact_with_drone)
quick_trial_button.config(command=show_controlled_comparison_trial)
run_experiments_button.config(command=run_all_experiments)
view_results_button.config(command=view_all_results)

# =========================
# Player Commands / Input
# =========================

def process_command(_event: tk.Event | None = None) -> None:
    global player_location, active_puzzle, correct_solution

    command = input_box.get().strip().lower()

    input_box.delete(0, tk.END)

    if awaiting_radio_reply:
        if not command:
            update_chat_log(f"❌ {target_drone} is waiting for your reply.")

            return

        player_answer(command)
        root.focus_set()

        return

    if active_puzzle and command.startswith("solve "):
        player_solve = command[6:].strip()

        if active_puzzle is not None and isinstance(correct_solution, str) and player_solve == correct_solution.lower():
            inventory.append(active_puzzle)
            game_map[player_location]["items"].remove(active_puzzle)
            update_chat_log(f"✅ Correct! You obtained the {active_puzzle}.")
            update_inventory_display()

            active_puzzle, correct_solution = None, None

        else:
            update_chat_log("❌ Incorrect answer. Try again.")

        return

    command_mappings =  {
                        "go": ["go", "move", "travel", "head", "walk", "run"],
                        "pickup": ["pickup", "take", "grab", "collect"],
                        "use": ["use", "apply", "activate"],
                        "look": ["look", "examine", "inspect", "observe"],
                        "summary": ["summary", "stats", "metrics"],
                        "evaluate": ["evaluate", "eval", "benchmark", "test"],
                        "scenario": ["scenario", "breakdown"],
                        "trial": ["trial", "compare", "comparison"],
                        "dashboard": ["dashboard", "panel"],
                        "guess": ["guess", "label", "mark"],
                        "playerstats": ["playerstats", "guessstats", "playersummary"],
                        "export": ["export", "plots", "bundle"],
                        "identityeval": ["identityeval", "identitytest", "identityexperiment"],
                        "identitysummary": ["identitysummary", "identitystats", "identityresults"],
                        "exit": ["exit", "quit", "leave"]
                        }

    parts = command.split(maxsplit=1)
    verb = parts[0] if parts else ""

    for main_command, synonyms in command_mappings.items():
        if verb in synonyms:
            command = command.replace(verb, main_command, 1)

            break

    def pickup_item(command_text: str) -> None:
        item_parts = command_text.split(maxsplit=1)

        if len(item_parts) < 2:
            update_chat_log("⚠️ Pick up what?")

            return

        item_to_pick = item_parts[1].title()
        room = game_map[player_location]

        # Other items are picked up through different interactions
        if item_to_pick in ["Rope", "Shovel"] and item_to_pick in room["items"]:
            inventory.append(item_to_pick)
            room["items"].remove(item_to_pick)
            update_chat_log(f"✅ You picked up: {item_to_pick}")
            update_inventory_display()

        else:
            update_chat_log("❌ You can't pick that up.")

    def handle_guess_command(command_text: str) -> None:
        guess_parts = command_text.split()

        if len(guess_parts) < 3:
            update_chat_log("⚠️ Use: guess [drone_id] [human/ai]")

            return

        _, drone_id, guess_label = guess_parts[0], guess_parts[1], guess_parts[2]

        if drone_id not in drone_roles:
            update_chat_log(f"❌ Unknown drone ID: {drone_id}")

            return

        record_player_guess(drone_id, guess_label)

    if command.startswith("pickup "):
        pickup_item(command)

    elif command.startswith("use "):
        use_item(command[4:].strip())

    elif command.startswith("go "):
        parts = command.split(maxsplit=1)

        if len(parts) < 2:
            update_chat_log("⚠️ Go where?")

        else:
            move_player(parts[1])

    elif command == "look":
        update_chat_log("👁️ " + room_desc)

    elif command == "summary":
        show_eval_summary()

    elif command == "evaluate":
        run_evaluation_suite(3)

    elif command == "scenario":
        show_scenario_breakdown()

    elif command == "trial":
        show_controlled_comparison_trial()

    elif command == "dashboard":
        show_metrics_dashboard()

    elif command.startswith("guess "):
        handle_guess_command(command)

    elif command == "playerstats":
        show_player_guess_summary()

    elif command == "export":
        export_benchmark_bundle()

    elif command == "identityeval":
        run_identity_experiment_suite(3, 4)

    elif command == "identitysummary":
        show_identity_experiment_summary()

    elif command == "exit":
        shutdown_game()

    else:
        update_chat_log("⚠️ Unknown command.")

    root.focus_set()

def submit_input() -> None:
    process_command()
    root.focus_set()

def toggle_drone_status(_event: tk.Event) -> None:
    if awaiting_radio_reply:
        update_chat_log(f"❌ Finish your reply to {target_drone} before changing drone status.")

        return

    selection = drone_listbox.curselection()

    if not selection:
        return

    index = selection[0]
    full_name = drone_listbox.get(index)
    name_without_icon = full_name

    for status_icon in status_icons.values():
        if full_name.startswith(status_icon + " "):
            name_without_icon = full_name[len(status_icon) + 1:]

            break

    current_status = drone_status.get(name_without_icon, "Uncertain")
    status_cycle = ["Uncertain", "Authentic", "Suspicious"]

    next_index = (status_cycle.index(current_status) + 1) % len(status_cycle)
    next_status = status_cycle[next_index]

    drone_status[name_without_icon] = next_status
    new_display = f"{status_icons[next_status]} {name_without_icon}"

    drone_listbox.delete(index)
    drone_listbox.insert(index, new_display)
    drone_listbox.selection_clear(0, tk.END)
    drone_listbox.selection_set(index)

def map_click(room_name: str) -> None:
    global player_location

    if awaiting_radio_reply:
        update_chat_log(f"❌ Finish your reply to {target_drone} before moving.")

        return

    if room_status.get(room_name) != "undiscovered":
        player_location = room_name

        update_chat_log(f"🧭 You teleport to the {room_name.replace('_', ' ')}.")
        handle_room_entry(room_name)

    else:
        update_chat_log("❌ You haven't discovered this room yet.")

def arrow_key_handler(direction: str) -> None:
    if root.focus_get() != input_box:
        if awaiting_radio_reply:
            update_chat_log(f"❌ Finish your reply to {target_drone} before moving.")

            return

        move_player(direction)

def handle_number_key(event: tk.Event) -> None:
    if awaiting_radio_reply:
        update_chat_log(f"❌ Finish your reply to {target_drone} before using items.")

        return

    if root.focus_get() != input_box:
        number_key = event.char
        index = (int(number_key) - 1) % 10

        if index < len(inventory):
            number_item = inventory[index]
            use_item(number_item)

def allow_copy(_event: tk.Event) -> str:
    chat_log.event_generate("<<Copy>>")

    return "break"

def clear_input_focus(event: tk.Event) -> None:
    widget = event.widget

    if widget not in (input_box, chat_log, drone_listbox):
        root.after(1, root.focus_set) # type: ignore

submit_button.config(command=submit_input)

input_box.bind("<Return>", lambda e: submit_input())
drone_listbox.bind("<Double-Button-1>", toggle_drone_status)

root.bind("<Up>", lambda e: arrow_key_handler("north"))
root.bind("<Down>", lambda e: arrow_key_handler("south"))
root.bind("<Left>", lambda e: arrow_key_handler("west"))
root.bind("<Right>", lambda e: arrow_key_handler("east"))

for key in "1234567890":
    root.bind(f"<Key-{key}>", handle_number_key)

chat_log.bind("<Control-c>", allow_copy)
root.bind_all("<Button-1>", clear_input_focus)

# =========================
# Startup / Shutdown
# =========================

def shutdown_game() -> None:
    save_eval_outputs()
    save_identity_experiment_outputs()

    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    except pygame.error:
        pass

    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", shutdown_game)
root.mainloop()
