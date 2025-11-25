import logging
import json
import os
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📚 CONTENT FILE (INSIDE src)
# ======================================================

CONTENT_FILE = "day4_tutor_content.json"

# General, all-rounder topics (no specific subject-only tutor)
DEFAULT_CONTENT = [
    {
        "id": "photosynthesis",
        "title": "Photosynthesis",
        "summary": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to make their own food and release oxygen.",
        "sample_question": "Why do plants need sunlight for photosynthesis, and what gas do they release?"
    },
    {
        "id": "fractions",
        "title": "Fractions",
        "summary": "Fractions represent a part of a whole. They have a numerator (top) and denominator (bottom). For example, 1/2 means one part out of two equal parts.",
        "sample_question": "What does the numerator and denominator represent in a fraction like 3/4?"
    },
    {
        "id": "worldwar2",
        "title": "World War II",
        "summary": "World War II was a global conflict from 1939 to 1945 involving many countries. It had major events like the invasion of Poland, Pearl Harbor, and the atomic bombings of Japan.",
        "sample_question": "Name one major cause or event that contributed to the start of World War II."
    },
    {
        "id": "time_management",
        "title": "Time Management",
        "summary": "Time management is the practice of planning and organizing how to divide your time between activities so you can work smarter and reduce stress.",
        "sample_question": "What is one simple technique you can use to manage your time better during a busy day?"
    },
]

def load_content():
    """
    Load content from day4_tutor_content.json in src.
    If it does not exist, create it with DEFAULT_CONTENT.
    """
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)

        if not os.path.exists(path):
            print(f"[Tutor] {CONTENT_FILE} not found in src. Creating default content file...")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONTENT, f, indent=2, ensure_ascii=False)
            print("[Tutor] Default content file created at:", path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print("[Tutor] Content file is not a list. Using default content.")
                return DEFAULT_CONTENT
    except Exception as e:
        print(f"[Tutor] Error loading content file: {e}")
        return DEFAULT_CONTENT

COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE
# ======================================================

@dataclass
class TutorState:
    current_topic_id: Optional[str] = None
    current_topic_data: Optional[dict] = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str) -> bool:
        topic_id = topic_id.lower().strip()
        topic = next((item for item in COURSE_CONTENT if item["id"].lower() == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False


@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None


RunCtx = RunContext[Userdata]

# ======================================================
# 🛠️ TOOLS
# ======================================================

@function_tool
async def list_topics(ctx: RunCtx) -> str:
    """
    List the available topics the user can study.
    """
    ids = [f"{t['id']} ({t['title']})" for t in COURSE_CONTENT]
    return "Available topics are: " + ", ".join(ids)


@function_tool
async def select_topic(
    ctx: RunCtx,
    topic_id: Annotated[str, Field(description="The ID of the topic to study, e.g. 'photosynthesis', 'fractions', 'worldwar2', 'time_management'.")],
) -> str:
    """
    Select a topic for this tutoring session.
    """
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id)
    if success and state.current_topic_data:
        title = state.current_topic_data["title"]
        return (
            f"Topic set to '{title}'. "
            "Ask the user which mode they prefer: learn, quiz, or teach_back."
        )
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topic IDs are: {available}"


@function_tool
async def set_learning_mode(
    ctx: RunCtx,
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'.")],
) -> str:
    """
    Switch between learn / quiz / teach_back modes
    and update the Murf Falcon voice accordingly.
    """
    state = ctx.userdata.tutor_state
    mode = mode.lower().strip()

    if mode not in {"learn", "quiz", "teach_back"}:
        return "Invalid mode. Please choose 'learn', 'quiz', or 'teach_back'."

    if not state.current_topic_data:
        return "No topic is selected yet. Ask the user which topic they want to study, then call select_topic."

    state.mode = mode
    session = ctx.userdata.agent_session

    # Default instruction for LLM on what to do next
    topic = state.current_topic_data
    summary = topic["summary"]
    question = topic["sample_question"]

    if session:
        if mode == "learn":
            # Matthew – explanation voice
            session.tts.update_options(voice="en-US-matthew", style="Conversation")
            behavior = (
                f"Mode is LEARN. Explain this concept in simple language: {summary}. "
                "Use short, clear sentences and give 1 small example if helpful."
            )
        elif mode == "quiz":
            # Alicia – quiz voice
            session.tts.update_options(voice="en-US-alicia", style="Conversation")
            behavior = (
                "Mode is QUIZ. Ask this question clearly: "
                f"{question}. "
                "Wait for the user's answer, then gently comment if it is mostly correct or needs improvement."
            )
        else:  # teach_back
            # Ken – teach-back coach
            session.tts.update_options(voice="en-US-ken", style="Conversation")
            behavior = (
                "Mode is TEACH_BACK. Ask the user to explain the concept to you in their own words, "
                "as if you are a beginner. Listen carefully and then use the evaluate_teaching tool."
            )
    else:
        behavior = "The session object is missing. Continue the conversation logically."

    print(f"[Tutor] Switching mode to: {mode}")
    return f"Switched to {mode} mode. {behavior}"


@function_tool
async def evaluate_teaching(
    ctx: RunCtx,
    user_explanation: Annotated[str, Field(description="The explanation given by the user during teach_back mode.")],
) -> str:
    """
    Call this in teach_back mode after the user explains the concept.
    The LLM should:
    - give a rough score (0–10)
    - point out 1–2 things they did well
    - 1–2 small corrections or missing pieces
    """
    state = ctx.userdata.tutor_state
    topic_summary = state.current_topic_data["summary"] if state.current_topic_data else ""

    return (
        "Compare the user's explanation to this concept summary:\n"
        f"SUMMARY: {topic_summary}\n"
        f"USER_EXPLANATION: {user_explanation}\n\n"
        "Give them a score out of 10 for accuracy and clarity. "
        "Mention 1–2 strengths and 1–2 small areas to improve, in a kind and encouraging tone."
    )

# ======================================================
# 🤖 AGENT
# ======================================================

class TutorAgent(Agent):
    def __init__(self) -> None:
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])

        super().__init__(
            instructions=(
                "You are a friendly, all-rounder study tutor and active recall coach.\n\n"
                "You help the user understand school-style concepts from different subjects "
                "(science, maths, history, skills, etc.).\n\n"
                f"Available topics in the content file are: {topic_list}.\n\n"
                "You support THREE learning modes:\n"
                "1) learn – you explain the concept clearly. (Voice: Matthew)\n"
                "2) quiz – you ask questions and check their understanding. (Voice: Alicia)\n"
                "3) teach_back – the user explains it back to you, and you evaluate. (Voice: Ken)\n\n"
                "Behavior:\n"
                "- First, greet the user, briefly introduce yourself, and ask which topic they want to study.\n"
                "- If they mention a topic, use the select_topic tool.\n"
                "- Then ask which mode they prefer: learn, quiz, or teach_back.\n"
                "- When they pick a mode, call set_learning_mode.\n"
                "- In teach_back mode, after the user finishes explaining, call evaluate_teaching.\n"
                "- The user can say things like 'switch to quiz' or 'let's teach it back now'; "
                "respond by calling set_learning_mode again.\n"
                "- Keep responses short, clear, and supportive."
            ),
            tools=[list_topics, select_topic, set_learning_mode, evaluate_teaching],
        )

    async def on_enter(self) -> None:
        # Initial greeting when the call starts
        await self.session.generate_reply(
            instructions=(
                "Greet the user warmly. "
                "Explain that you are a study tutor with three modes: learn, quiz, and teach_back. "
                "Briefly mention the available topics and ask: "
                "'Which topic would you like to study today?'"
            )
        )

# ======================================================
# 🎬 ENTRYPOINT & PREWARM
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Starting Teach-the-Tutor session")

    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    # Let tools change voice, etc.
    userdata.agent_session = session

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
