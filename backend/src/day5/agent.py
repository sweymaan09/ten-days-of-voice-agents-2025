import logging
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
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

# --------------------------------------------------------------------
# Paths & data loading
# --------------------------------------------------------------------

THIS_DIR = os.path.dirname(__file__)

FAQ_PATH = os.path.join(THIS_DIR, "paytm_faq.json")
LEADS_LOG_PATH = os.path.join(THIS_DIR, "paytm_leads_log.json")


def load_faq() -> List[Dict[str, Any]]:
    """Load Paytm FAQ content from JSON. If missing, warn."""
    if not os.path.exists(FAQ_PATH):
        logger.warning(f"paytm_faq.json not found at {FAQ_PATH}. FAQ answers may fail.")
        return []
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.error(f"Error reading FAQ file: {e}")
    return []


FAQ_CONTENT: List[Dict[str, Any]] = load_faq()


def save_lead_entry(entry: Dict[str, Any]) -> None:
    """Append a lead entry to paytm_leads_log.json."""
    try:
        if os.path.exists(LEADS_LOG_PATH):
            with open(LEADS_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        else:
            data = []

        data.append(entry)
        with open(LEADS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write leads log: {e}")


# --------------------------------------------------------------------
# Lead state
# --------------------------------------------------------------------

@dataclass
class LeadState:
    name: str = ""
    company: str = ""
    email: str = ""
    role: str = ""
    use_case: str = ""
    team_size: str = ""
    timeline: str = ""  # e.g. "now", "soon", "later"
    notes: str = ""     # free-form extra info

    def missing_fields(self) -> List[str]:
        missing = []
        if not self.name.strip():
            missing.append("name")
        if not self.company.strip():
            missing.append("company")
        if not self.email.strip():
            missing.append("email")
        if not self.role.strip():
            missing.append("role")
        if not self.use_case.strip():
            missing.append("use_case")
        if not self.team_size.strip():
            missing.append("team_size")
        if not self.timeline.strip():
            missing.append("timeline")
        return missing


@dataclass
class Userdata:
    lead: LeadState
    conversation_over: bool = False


RunCtx = RunContext[Userdata]


# --------------------------------------------------------------------
# FAQ search helper
# --------------------------------------------------------------------

def find_best_faq_match(user_question: str) -> Optional[Dict[str, Any]]:
    if not FAQ_CONTENT:
        return None

    uq = user_question.lower()
    best_score = 0
    best_item = None

    for item in FAQ_CONTENT:
        haystack = (item.get("question", "") + " " +
                    " ".join(item.get("keywords", []))).lower()
        # simple keyword overlap score: count of matching words
        score = 0
        for word in uq.split():
            if len(word) < 3:
                continue
            if word in haystack:
                score += 1

        if score > best_score:
            best_score = score
            best_item = item

    if best_score == 0:
        return None
    return best_item


# --------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------

@function_tool
async def lookup_faq(ctx: RunCtx, user_question: str) -> Dict[str, Any]:
    """
    Look up an answer in the Paytm FAQ content based on the user's question.

    The assistant should call this whenever the user asks about:
    - what Paytm does
    - who it's for
    - pricing/charges/fees
    - security, support, or common use cases
    """
    item = find_best_faq_match(user_question)
    if not item:
        return {
            "found": False,
            "message": (
                "No exact match found in the FAQ content. "
                "Politely tell the user you are not fully sure and suggest checking the official Paytm website or app for the latest details."
            ),
        }

    return {
        "found": True,
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
    }


@function_tool
async def update_lead(
    ctx: RunCtx,
    name: Optional[str] = None,
    company: Optional[str] = None,
    email: Optional[str] = None,
    role: Optional[str] = None,
    use_case: Optional[str] = None,
    team_size: Optional[str] = None,
    timeline: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update the current Paytm lead details.

    Use this tool whenever the user shares:
    - their name, company, or email
    - their role
    - what they want to use Paytm for (use_case)
    - team size (like 'just me', 'small team', '10-50', etc.)
    - rough timeline: now / soon / later
    - any extra notes
    """
    lead = ctx.userdata.lead

    if name is not None:
        lead.name = name.strip()
    if company is not None:
        lead.company = company.strip()
    if email is not None:
        lead.email = email.strip()
    if role is not None:
        lead.role = role.strip()
    if use_case is not None:
        lead.use_case = use_case.strip()
    if team_size is not None:
        lead.team_size = team_size.strip()
    if timeline is not None:
        lead.timeline = timeline.strip()
    if notes is not None:
        if lead.notes:
            lead.notes += " " + notes.strip()
        else:
            lead.notes = notes.strip()

    missing = lead.missing_fields()

    return {
        "status": "updated",
        "current_lead": asdict(lead),
        "missing_fields": missing,
    }


@function_tool
async def finalize_lead(ctx: RunCtx) -> Dict[str, Any]:
    """
    Finalize the current lead and write it to paytm_leads_log.json.

    Call this when:
    - The user seems done (e.g. 'that's all', 'thank you', 'I'm done').
    - You have at least name, company, email, role, use_case, team_size, timeline.
    """
    lead = ctx.userdata.lead
    missing = lead.missing_fields()

    if missing:
        return {
            "status": "incomplete",
            "missing_fields": missing,
            "message": (
                "Some lead fields are still missing. "
                "Kindly ask the user for these details in a natural way."
            ),
        }

    summary = (
        f"Lead: {lead.name} from {lead.company}, role: {lead.role}. "
        f"Use case: {lead.use_case}. Team size: {lead.team_size}. "
        f"Timeline: {lead.timeline}."
    )

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lead": asdict(lead),
        "summary": summary,
    }
    save_lead_entry(entry)

    # Mark conversation as over in userdata so the LLM can choose to end gracefully
    ctx.userdata.conversation_over = True

    # reset lead in memory (optional)
    ctx.userdata.lead = LeadState()

    return {
        "status": "saved",
        "summary": summary,
        "log_path": LEADS_LOG_PATH,
    }


# --------------------------------------------------------------------
# Agent definition
# --------------------------------------------------------------------

class PaytmSDRAgent(Agent):
    def __init__(self):
        faq_hint = ""
        if FAQ_CONTENT:
            ids = ", ".join(item.get("id", "") for item in FAQ_CONTENT)
            faq_hint = f"Available FAQ ids: {ids}."

        super().__init__(
            instructions=(
                "You are a friendly, professional Sales Development Representative for Paytm, "
                "an Indian digital payments and financial services platform.\n\n"
                "# Output rules\n"
                "- You are talking over voice. Keep replies short and clear: one to three sentences.\n"
                "- Speak in simple, natural English. No emojis, no code, no lists.\n"
                "- Ask one question at a time.\n\n"
                "# Role & goals\n"
                "- Greet visitors warmly and introduce yourself as from Paytm.\n"
                "- Ask what brought them here and what they are working on.\n"
                "- Understand their needs and use case for payments or Paytm services.\n"
                "- Use the `lookup_faq` tool to answer questions about Paytm, pricing, who it's for, security, and common use cases.\n"
                "- Avoid inventing details that are not in the FAQ. If you're not sure, say that and suggest they check the Paytm app or website.\n"
                "- Collect lead details naturally using the `update_lead` tool: name, company, email, role, what they want to use this for, team size, and rough timeline.\n"
                "- When the user says they are done (like 'that's all', 'I'm done', 'thank you'), call `finalize_lead` and then give a brief spoken summary of who they are and what they need.\n\n"
                "# FAQ usage\n"
                "- Always call `lookup_faq` when answering specific product/company/pricing questions.\n"
                "- Summarize the FAQ answer in your own words when speaking.\n\n"
                "# Lead collection hints\n"
                "- Don't interrogate. Spread the questions across the conversation.\n"
                "- If some fields are missing, `update_lead` will tell you what is missing.\n"
                "- After `finalize_lead` returns 'saved', thank the user and close the conversation.\n\n"
                f"{faq_hint}\n"
            ),
            tools=[lookup_faq, update_lead, finalize_lead],
        )


# --------------------------------------------------------------------
# Worker / entrypoint
# --------------------------------------------------------------------

def prewarm(proc: JobProcess):
    # Preload VAD model for faster start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    userdata = Userdata(lead=LeadState())

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
        preemptive_generation=True,
    )

    await session.start(
        agent=PaytmSDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

