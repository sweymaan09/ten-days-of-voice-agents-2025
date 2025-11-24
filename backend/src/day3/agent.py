import logging
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

# Load env variables (LIVEKIT, Murf, Deepgram, Google, etc.)
load_dotenv(".env.local")

# Path to the JSON log file
LOG_PATH = os.path.join(os.path.dirname(__file__), "wellness_log.json")


# -----------------------
#  STATE & HELPERS
# -----------------------

@dataclass
class WellnessState:
    """Holds current session check-in plus history."""
    mood: str = ""
    energy: str = ""
    stressors: str = ""
    goals: Optional[List[str]] = None
    recap: str = ""
    # history is a list of dicts, each representing a past entry
    history: Optional[List[dict]] = None


def load_history() -> List[dict]:
    """Load previous wellness entries from wellness_log.json, if it exists."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except Exception as e:
        logger.warning(f"Failed to read wellness_log.json: {e}")
        return []


def save_history(history: List[dict]) -> None:
    """Save the full list of entries back to wellness_log.json."""
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write wellness_log.json: {e}")


def get_missing_fields(state: WellnessState) -> List[str]:
    """
    Decide which conversational fields are still missing.
    We want the agent to keep asking until these are filled.
    """
    missing: List[str] = []
    if not state.mood.strip():
        missing.append("mood")
    if not state.energy.strip():
        missing.append("energy")
    if not state.stressors.strip():
        missing.append("stressors")
    if not state.goals:
        missing.append("goals")
    return missing


# -----------------------
#  ASSISTANT AGENT
# -----------------------

class Assistant(Agent):
    def __init__(self, history: List[dict]) -> None:
        # Build a short reference to last check-in, if any
        extra_history_note = ""
        if history:
            last = history[-1]
            last_mood = last.get("mood", "unknown")
            last_energy = last.get("energy", "unknown")
            extra_history_note = (
                "\n\nWhen it feels natural, you may gently reference the last check-in, "
                f"e.g., 'Last time you mentioned feeling {last_mood} with {last_energy} energy. "
                "How does today compare?'. Keep it brief and supportive."
            )

        super().__init__(
            instructions=(
                "You are a daily health and wellness voice companion.\n\n"
                "Your job is to have a short, kind, realistic check-in with the user. "
                "You are supportive, but grounded and non-clinical.\n\n"
                "VERY IMPORTANT SAFETY:\n"
                "- Do NOT diagnose any mental or physical health conditions.\n"
                "- Do NOT mention or suggest medication.\n"
                "- Keep advice simple, practical, and low-risk, like:\n"
                "  * breaking tasks into tiny steps\n"
                "  * taking a short break, walk, stretch, or drinking water\n"
                "  * gentle encouragement, not pressure\n\n"
                "Conversation goals:\n"
                "1) Ask about mood and energy.\n"
                "   - Examples: 'How are you feeling today?', "
                "               'How is your energy right now?', "
                "               'Anything stressing you out?'\n"
                "2) Ask for 1–3 simple goals or intentions for today.\n"
                "   - Examples: 'What are one to three things you’d like to get done today?', "
                "               'Anything you want to do just for yourself?'\n"
                "3) Offer small, realistic, non-medical suggestions.\n"
                "4) At the end, give a brief recap:\n"
                "   - Today’s mood and energy\n"
                "   - The main 1–3 goals\n"
                "   Then ask: 'Does this sound right?'\n\n"
                "Tools:\n"
                "- Whenever the user shares anything about how they feel, stress, or goals, "
                "  you should call `update_checkin` with the extracted info.\n"
                "- Once the main details are collected and the user has confirmed the summary, "
                "  call `finalize_checkin` to save this session.\n\n"
                "Keep your spoken answers short, warm, and natural. No emojis.\n"
            )
            + extra_history_note,
        )

    # -------- TOOLS -------- #

    @function_tool()
    async def update_checkin(
        self,
        context: RunContext[WellnessState],
        mood: Optional[str] = None,
        energy: Optional[str] = None,
        stressors: Optional[str] = None,
        goals: Optional[List[str]] = None,
        recap: Optional[str] = None,
    ) -> dict:
        """
        Update the current daily wellness check-in.

        Use this whenever the user tells you about:
        - how they feel (mood),
        - their energy,
        - what is stressing them,
        - what goals or intentions they have today.

        The LLM should:
        - Extract simple text for mood (e.g. 'a bit anxious but hopeful'),
        - energy (e.g. 'low', 'medium', 'pretty high'),
        - stressors (short description),
        - goals as a list of 1–3 short items.
        """
        state = context.userdata

        if mood is not None:
            state.mood = mood.strip()
        if energy is not None:
            state.energy = energy.strip()
        if stressors is not None:
            state.stressors = stressors.strip()
        if goals is not None:
            # Filter out empty strings
            clean_goals = [g.strip() for g in goals if g and g.strip()]
            state.goals = clean_goals or None
        if recap is not None:
            state.recap = recap.strip()

        missing = get_missing_fields(state)

        return {
            "status": "updated",
            "current_state": {
                "mood": state.mood,
                "energy": state.energy,
                "stressors": state.stressors,
                "goals": state.goals or [],
                "recap": state.recap,
            },
            "missing_fields": missing,
        }

    @function_tool()
    async def finalize_checkin(self, context: RunContext[WellnessState]) -> dict:
        """
        Save the current check-in to wellness_log.json.

        Call this ONLY when:
        - Mood, energy, stressors, and goals are collected.
        - You have already given the user a brief spoken recap and they confirmed.

        This will:
        - Append a new entry to wellness_log.json
        - Return a short summary string.
        """
        state = context.userdata
        missing = get_missing_fields(state)

        if missing:
            # Not ready yet; tell the model to keep asking
            return {
                "status": "incomplete",
                "missing_fields": missing,
                "message": "Some parts of the check-in are still missing. "
                           "Please ask the user gentle follow-up questions."
            }

        # Build a textual summary (used both for file and as helper for the LLM)
        goals_text = ", ".join(state.goals) if state.goals else "no specific goals"
        summary = (
            f"User mood: {state.mood}. "
            f"Energy: {state.energy}. "
            f"Stressors: {state.stressors or 'not specified'}. "
            f"Goals for today: {goals_text}."
        )

        # Prepare entry
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mood": state.mood,
            "energy": state.energy,
            "stressors": state.stressors,
            "goals": state.goals or [],
            "summary": summary,
        }

        # Append to history and save
        history = state.history or []
        history.append(entry)
        state.history = history
        save_history(history)

        # Reset the current session fields but keep history in memory
        state.mood = ""
        state.energy = ""
        state.stressors = ""
        state.goals = None
        state.recap = ""

        return {
            "status": "saved",
            "summary": summary,
            "entries_count": len(history),
        }


# -----------------------
#  WORKER / SESSION SETUP
# -----------------------

def prewarm(proc: JobProcess):
    # Preload VAD so each worker is ready
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Add room name to logs
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load previous history from disk
    history = load_history()

    # Create initial wellness state
    wellness_state = WellnessState(history=history)

    # Set up AgentSession with your existing stack
    session = AgentSession[WellnessState](
        userdata=wellness_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the wellness companion agent
    await session.start(
        agent=Assistant(history=history),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
