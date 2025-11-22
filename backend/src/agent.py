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

load_dotenv(".env.local")

# -----------------------
# ORDER STATE
# -----------------------

@dataclass
class OrderState:
    drinkType: str = ""
    size: str = ""
    milk: str = ""
    extras: Optional[List[str]] = None   # None until user responds
    name: str = ""

    def is_complete(self) -> bool:
        return (
            self.drinkType.strip() != ""
            and self.size.strip() != ""
            and self.milk.strip() != ""
            and self.name.strip() != ""
        )


def get_missing_fields(state: OrderState) -> List[str]:
    """Return only required missing fields. Extras are optional."""
    missing = []
    if not state.drinkType:
        missing.append("drinkType")
    if not state.size:
        missing.append("size")
    if not state.milk:
        missing.append("milk")
    if not state.name:
        missing.append("name")
    return missing


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly coffee shop barista for the brand 'Sweymaan Coffee Roasters'. "
                "You take a voice order and update the JSON order object:\n"
                '{ "drinkType": "string", "size": "string", "milk": "string", '
                '"extras": ["string"], "name": "string" }.\n\n'
                "Rules:\n"
                "- Always ask ONE question at a time.\n"
                "- ALWAYS call `update_order` when the user gives information.\n"
                "- Use missing_fields from tool result to decide the next question.\n"
                "- Extras are optional; user may say 'no extras'.\n"
                "- When everything is filled and the user confirms, call `finalize_order`.\n"
                "- Keep responses short and natural like a real barista.\n"
            ),
        )

    # -------- TOOLS -------- #

    @function_tool()
    async def update_order(
        self,
        context: RunContext[OrderState],
        drinkType: Optional[str] = None,
        size: Optional[str] = None,
        milk: Optional[str] = None,
        extras: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> dict:
        """Update coffee order whenever user provides details."""
        state = context.userdata

        if drinkType:
            state.drinkType = drinkType.strip()
        if size:
            state.size = size.strip()
        if milk:
            state.milk = milk.strip()

        # Handle extras
        if extras is not None:
            lowered = [e.lower().strip() for e in extras]
            if any(w in ["no extras", "none", "no extra", "nothing"] for w in lowered):
                state.extras = []
            else:
                state.extras = extras

        if name:
            state.name = name.strip()

        missing = get_missing_fields(state)

        return {
            "order": asdict(state),
            "missing_fields": missing,
        }

    @function_tool()
    async def finalize_order(self, context: RunContext[OrderState]) -> dict:
        """Finalize order once user confirms and all fields are filled."""
        state = context.userdata
        missing = get_missing_fields(state)

        if missing:
            return {
                "status": "incomplete",
                "missing_fields": missing,
                "order": asdict(state),
            }

        # Ensure orders dir
        orders_dir = os.path.join(os.path.dirname(__file__), "orders")
        os.makedirs(orders_dir, exist_ok=True)

        filename = f"order_{int(time.time())}.json"
        filepath = os.path.join(orders_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=4)

        # Prepare summary
        extras_text = "no extras"
        if state.extras and len(state.extras) > 0:
            extras_text = ", ".join(state.extras)

        summary = (
            f"Order summary: a {state.size} {state.drinkType} with {state.milk} milk, "
            f"{extras_text}, for {state.name}."
        )

        # SAFE RESET (do NOT replace userdata object)
        state.drinkType = ""
        state.size = ""
        state.milk = ""
        state.extras = None
        state.name = ""

        return {
            "status": "saved",
            "filename": filename,
            "order_summary": summary,
        }


# -----------------------
# SESSION SETUP
# -----------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    order_state = OrderState()

    session = AgentSession[OrderState](
        userdata=order_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
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
    def _on(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
