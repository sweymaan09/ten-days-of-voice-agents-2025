# IMPROVE THE AGENT AS PER YOUR NEED 1
"""
Day 8 – Voice Game Master (D&D-Style Adventure) - Voice-only GM agent

Stranger-Things flavoured D&D adventure:

- The world feels like an 80s small town campaign: kids-on-bikes energy, spooky woods,
  an old radio tower, a drained quarry, and a thin veil to a shadowy "Other Side".
- Uses LiveKit agent plumbing similar to the provided food_agent_sqlite example.
- GM persona, universe, tone and rules are encoded in the agent instructions.
- Keeps STT/TTS/Turn detector/VAD integration untouched (murf, deepgram, silero, turn_detector).
- Tools:
    - start_adventure(): start a fresh session and introduce the scene
    - get_scene(): return the current scene description (GM text) ending with "What do you do?"
    - player_action(action_text): accept player's spoken action, update state, advance scene
    - show_journal(): list remembered facts, NPCs, named locations, choices
    - restart_adventure(): reset state and start over
- Userdata keeps continuity between turns: history, inventory, named NPCs/locations, choices, current_scene
"""

import json
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

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

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("voice_game_master")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Simple Game World Definition
# -------------------------
# A compact world with a few scenes and choices forming a mini-arc.
# Re-themed to feel like a Stranger-Things-style 80s D&D campaign:
# small town, spooky radio tower, drained quarry, thin veil to the "Other Side".
WORLD = {
    "intro": {
        "title": "Lights Over Ember Grove",
        "desc": (
            "You wake up on the gravel edge of Stillwater Quarry, just outside the town of Ember Grove. "
            "It’s sometime in the late 80s — a walkman lies nearby, quietly hissing static. "
            "Up on the ridge, the silhouette of an old radio tower flickers with a strange, reddish glow, "
            "as if the lights are glitching in a pattern that doesn’t belong to this world. "
            "A narrow bike trail leads back toward the sleepy neighborhood to the east. At your feet, half-buried in dust, "
            "is a small, cracked cassette tape with no label."
        ),
        "choices": {
            "inspect_tape": {
                "desc": "Inspect the cracked cassette tape at your feet.",
                "result_scene": "tape",
            },
            "approach_tower": {
                "desc": "Climb the ridge toward the flickering radio tower.",
                "result_scene": "tower",
            },
            "follow_trail_home": {
                "desc": "Follow the bike trail back toward the neighborhood.",
                "result_scene": "suburbs",
            },
        },
    },
    "tape": {
        "title": "The Tape",
        "desc": (
            "The cassette shell is cracked, but when you slot it into the walkman and press play, "
            "the usual music is replaced with distorted whispers and a low hum. Between the static, "
            "a robotic voice repeats coordinates and the phrase: 'Beneath the tower, the signal bleeds through.' "
            "As you listen, the lights on the radio tower flash in time with the audio, like something on the other side is listening back."
        ),
        "choices": {
            "take_tape": {
                "desc": "Pocket the tape and keep the message in mind.",
                "result_scene": "tower_approach",
                "effects": {"add_journal": "Found a cursed-feeling cassette: 'Beneath the tower, the signal bleeds through.'"},
            },
            "drop_tape": {
                "desc": "Drop the tape and walk away from it.",
                "result_scene": "intro",
            },
        },
    },
    "tower": {
        "title": "The Radio Tower",
        "desc": (
            "You climb the ridge and stand beneath the old radio tower. Its metal frame creaks in the wind, "
            "and the red warning light at the top flickers erratically — sometimes normal, sometimes a deep, wrong shade. "
            "At the base, a metal hatch with a heavy latch covers a maintenance tunnel. You notice fresh footprints in the dust. "
            "You can try the hatch, circle around the fence, or head back toward the quarry."
        ),
        "choices": {
            "try_latch_without_clue": {
                "desc": "Try the metal hatch latch with no clue, just brute force.",
                "result_scene": "latch_fail",
            },
            "circle_fence": {
                "desc": "Circle the perimeter fence for another way in.",
                "result_scene": "service_gap",
            },
            "retreat": {
                "desc": "Head back down toward the quarry.",
                "result_scene": "intro",
            },
        },
    },
    "tower_approach": {
        "title": "Signal on the Ridge",
        "desc": (
            "Clutching the strange cassette message in your memory, you reach the radio tower. "
            "As you step closer to the metal hatch, your walkman hisses and the static rises. "
            "The pattern in the cassette's message seems to sync with the way the tower’s red light pulses."
        ),
        "choices": {
            "open_hatch": {
                "desc": "Use the cassette clue and carefully work the hatch latch.",
                "result_scene": "latch_open",
                "effects": {"add_journal": "Used the cassette clue to open the hatch beneath the tower."},
            },
            "circle_fence": {
                "desc": "Look for another entrance around the tower.",
                "result_scene": "service_gap",
            },
            "retreat": {
                "desc": "Return to the quarry edge.",
                "result_scene": "intro",
            },
        },
    },
    "latch_fail": {
        "title": "Wrong Frequency",
        "desc": (
            "You yank at the latch without thinking. The metal shrieks, and the tower gives off a single, painful buzz. "
            "For a moment, the sky above you flickers like an old TV screen switching channels. "
            "From deep inside the maintenance tunnel, something claws against metal — awake and not happy."
        ),
        "choices": {
            "run_away": {
                "desc": "Sprint back down toward the quarry.",
                "result_scene": "intro",
            },
            "stand_ground": {
                "desc": "Stand your ground and brace yourself for whatever crawls out.",
                "result_scene": "tower_combat",
            },
        },
    },
    "latch_open": {
        "title": "The Hatch Opens",
        "desc": (
            "With the rhythm of the tape in your mind, you twist the latch in a precise pattern. "
            "It clicks open with an unnerving, hollow echo. Cool air smelling of ozone and damp earth rushes out. "
            "A narrow metal ladder leads down into a dim, humming service tunnel where fluorescent lights flicker on and off, "
            "like the place itself is glitching between worlds."
        ),
        "choices": {
            "descend": {
                "desc": "Climb down the ladder into the tunnel.",
                "result_scene": "tunnel",
            },
            "close_hatch": {
                "desc": "Shut the hatch, heart pounding, and reconsider.",
                "result_scene": "tower_approach",
            },
        },
    },
    "service_gap": {
        "title": "The Gap in the Fence",
        "desc": (
            "Behind a tangle of overgrown bushes you find a bent section of chain-link fence. "
            "Someone has carefully cut it and wired it back together for easy access. "
            "Beyond is a narrow path leading to a side door marked 'AUTHORIZED TECHS ONLY', the paint half-peeled and clawed."
        ),
        "choices": {
            "sneak_in": {
                "desc": "Slip through the gap and head for the side door.",
                "result_scene": "tunnel",
            },
            "mark_and_return": {
                "desc": "Make a mental note of the gap and go back toward the quarry.",
                "result_scene": "intro",
            },
        },
    },
    "tunnel": {
        "title": "Humming Tunnel",
        "desc": (
            "The maintenance tunnel slopes gently downward. Flickering fluorescent lights buzz overhead, "
            "and wires run along the walls, some of them pulsing with a faint, unnatural glow. "
            "You emerge into a small control room: scattered schematics, an old CRT monitor showing static, "
            "and on a metal desk, a brass key fob and a sealed manila folder that seems to vibrate slightly in your hand."
        ),
        "choices": {
            "take_key": {
                "desc": "Pick up the brass key fob.",
                "result_scene": "tunnel_key",
                "effects": {"add_inventory": "brass_key_fob", "add_journal": "Found a brass key fob in the tower control room."},
            },
            "open_folder": {
                "desc": "Open the sealed folder and read the contents.",
                "result_scene": "folder_reveal",
                "effects": {"add_journal": "File notes: 'Anomaly under Stillwater Quarry. Do not transmit on Channel 7.'"},
            },
            "leave_quietly": {
                "desc": "Back out of the tunnel and close the hatch behind you.",
                "result_scene": "intro",
            },
        },
    },
    "tunnel_key": {
        "title": "Key to Nowhere",
        "desc": (
            "As you lift the key fob, the CRT monitor flickers from static to an image of the quarry — "
            "but in the reflection, the sky is red and the trees look wrong, like shadows wearing a forest as a mask. "
            "A distorted voice crackles through unseen speakers: 'Will you seal what was opened?'"
        ),
        "choices": {
            "pledge_help": {
                "desc": "Promise to seal whatever has been opened between the worlds.",
                "result_scene": "resolution",
                "effects": {"add_journal": "You pledged to help seal the breach under the quarry."},
            },
            "refuse": {
                "desc": "Refuse and shove the key fob into your pocket.",
                "result_scene": "cursed_key",
                "effects": {"add_journal": "You pocketed the key fob; the air feels heavier around you."},
            },
        },
    },
    "folder_reveal": {
        "title": "The Folder",
        "desc": (
            "The folder is full of grainy photos: Stillwater Quarry at night, "
            "strange shapes half-emerging from the mist, and diagrams of overlapping waveforms labeled 'Other Side'. "
            "One note reads: 'Key fob syncs signal. Only activate with intent to close.'"
        ),
        "choices": {
            "search_for_key": {
                "desc": "Search the room for the key mentioned in the file.",
                "result_scene": "tunnel_key",
            },
            "leave_quietly": {
                "desc": "Put the folder down and quietly leave the tunnel.",
                "result_scene": "intro",
            },
        },
    },
    "tower_combat": {
        "title": "Something Crosses Over",
        "desc": (
            "The hatch buckles and a lank, shadow-soaked creature pulls itself halfway into your world. "
            "Its outline flickers like bad reception, and its eyes glow with a hungry, pale light. "
            "You feel the air around you warp, as if both worlds are overlapping for a moment. You have to act fast."
        ),
        "choices": {
            "fight": {
                "desc": "Stand your ground and fight the creature.",
                "result_scene": "fight_win",
            },
            "flee": {
                "desc": "Turn and run full-speed back toward the quarry.",
                "result_scene": "intro",
            },
        },
    },
    "fight_win": {
        "title": "Static Clears",
        "desc": (
            "You lash out with everything you have — instinct, adrenaline, maybe a bit of luck. "
            "The creature recoils, its form breaking up into static and shadow before it collapses, "
            "sliding back through the hatch. The red light on the tower stabilizes to a normal blink. "
            "On the ground where it fell, you find a small pendant etched with a symbol that matches one in the folder."
        ),
        "choices": {
            "take_pendant": {
                "desc": "Take the pendant and examine the symbol.",
                "result_scene": "resolution",
                "effects": {"add_inventory": "etched_pendant", "add_journal": "Recovered an etched pendant tied to the anomaly."},
            },
            "leave_pendant": {
                "desc": "Leave the pendant where it lies and catch your breath.",
                "result_scene": "intro",
            },
        },
    },
    "resolution": {
        "title": "Closing the Gap",
        "desc": (
            "For a moment, the quarry, the tower, and the sky all feel like they’re layered on top of another world — "
            "red clouds, twisted trees, and shapes watching from far away. Then, as if a channel has finally been changed, "
            "the feeling snaps back to normal. The night is just a night again. Maybe the town never knows how close it came, "
            "but you feel that this little breach is sealed, at least for now."
        ),
        "choices": {
            "end_session": {
                "desc": "End the session and walk back toward the neighborhood (conclude mini-arc).",
                "result_scene": "intro",
            },
            "keep_exploring": {
                "desc": "Stay out a little longer, just in case there are more weird signals.",
                "result_scene": "intro",
            },
        },
    },
    "cursed_key": {
        "title": "Wrong Kind of Signal",
        "desc": (
            "The brass key fob pulses cold in your pocket. Streetlights in your memory flicker, "
            "and for a second you see the shadows of the 'Other Side' overlaying the control room. "
            "You get the sense that the more you ignore this, the easier it will be for something else to step through next time."
        ),
        "choices": {
            "seek_redemption": {
                "desc": "Look for a way to make this right and help close the breach.",
                "result_scene": "resolution",
            },
            "ditch_key": {
                "desc": "Try to dump the key fob somewhere and hope the feeling fades.",
                "result_scene": "intro",
            },
        },
    },
    "suburbs": {
        "title": "Quiet Streets",
        "desc": (
            "You follow the bike trail back toward the quiet streets of Ember Grove. "
            "Sprinklers tick in front yards, and most of the houses are dark — it’s late. "
            "But every so often, a porch light flickers in the exact same pattern as the radio tower’s red glow. "
            "Whatever is happening out by the quarry is starting to bleed into town."
        ),
        "choices": {
            "go_back_quarry": {
                "desc": "Turn around and head back to the quarry; this is bigger than just a weird night.",
                "result_scene": "intro",
            },
            "watch_lights": {
                "desc": "Stand and watch the flickering lights a little longer.",
                "result_scene": "intro",
            },
        },
    },
}

# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    history: List[Dict] = field(default_factory=list)  # list of {'scene', 'action', 'time', 'result_scene'}
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    named_npcs: Dict[str, str] = field(default_factory=dict)
    choices_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

# -------------------------
# Helper functions
# -------------------------
def scene_text(scene_key: str, userdata: Userdata) -> str:
    """
    Build the descriptive text for the current scene, and append choices as short hints.
    Always end with 'What do you do?' so the voice flow prompts player input.
    """
    scene = WORLD.get(scene_key)
    if not scene:
        return "You are in a featureless static void between channels. What do you do?"

    desc = f"{scene['desc']}\n\nChoices:\n"
    for cid, cmeta in scene.get("choices", {}).items():
        desc += f"- {cmeta['desc']} (say: {cid})\n"
    # GM MUST end with the action prompt
    desc += "\nWhat do you do?"
    return desc

def apply_effects(effects: dict, userdata: Userdata):
    if not effects:
        return
    if "add_journal" in effects:
        userdata.journal.append(effects["add_journal"])
    if "add_inventory" in effects:
        userdata.inventory.append(effects["add_inventory"])
    # Extendable for more effect keys

def summarize_scene_transition(old_scene: str, action_key: str, result_scene: str, userdata: Userdata) -> str:
    """Record the transition into history and return a short narrative the GM can use."""
    entry = {
        "from": old_scene,
            "action": action_key,
            "to": result_scene,
            "time": datetime.utcnow().isoformat() + "Z",
    }
    userdata.history.append(entry)
    userdata.choices_made.append(action_key)
    return f"You chose '{action_key}'."

# -------------------------
# Agent Tools (function_tool)
# -------------------------

@function_tool
async def start_adventure(
    ctx: RunContext[Userdata],
    player_name: Annotated[Optional[str], Field(description="Player name", default=None)] = None,
) -> str:
    """Initialize a new adventure session for the player and return the opening description."""
    userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    opening = (
        f"Greetings {userdata.player_name or 'traveler'}. You're about to step into a strange night over Ember Grove.\n\n"
        + scene_text("intro", userdata)
    )
    # Ensure GM prompt present
    if not opening.endswith("What do you do?"):
        opening += "\nWhat do you do?"
    return opening

@function_tool
async def get_scene(
    ctx: RunContext[Userdata],
) -> str:
    """Return the current scene description (useful for 'remind me where I am')."""
    userdata = ctx.userdata
    scene_k = userdata.current_scene or "intro"
    txt = scene_text(scene_k, userdata)
    return txt

@function_tool
async def player_action(
    ctx: RunContext[Userdata],
    action: Annotated[str, Field(description="Player spoken action or the short action code (e.g., 'inspect_tape' or 'go to the tower')")],
) -> str:
    """
    Accept player's action (natural language or action key), try to resolve it to a defined choice,
    update userdata, advance to the next scene and return the GM's next description (ending with 'What do you do?').
    """
    userdata = ctx.userdata
    current = userdata.current_scene or "intro"
    scene = WORLD.get(current)
    action_text = (action or "").strip()

    # Attempt 1: match exact action key (e.g., 'inspect_tape')
    chosen_key = None
    if action_text.lower() in (scene.get("choices") or {}):
        chosen_key = action_text.lower()

    # Attempt 2: fuzzy match by checking if action_text contains the choice key or descriptive words
    if not chosen_key:
        for cid, cmeta in (scene.get("choices") or {}).items():
            desc = cmeta.get("desc", "").lower()
            if cid in action_text.lower() or any(w in action_text.lower() for w in desc.split()[:4]):
                chosen_key = cid
                break

    # Attempt 3: fallback by simple keyword matching against choice descriptions
    if not chosen_key:
        for cid, cmeta in (scene.get("choices") or {}).items():
            for keyword in cmeta.get("desc", "").lower().split():
                if keyword and keyword in action_text.lower():
                    chosen_key = cid
                    break
            if chosen_key:
                break

    if not chosen_key:
        # If we still can't resolve, ask a clarifying GM response but keep it short and end with prompt.
        resp = (
            "I didn't quite catch that action for this moment. "
            "Try one of the listed choices or use a simple phrase, like 'inspect the tape' or 'go to the tower'.\n\n"
            + scene_text(current, userdata)
        )
        return resp

    # Apply the chosen choice
    choice_meta = scene["choices"].get(chosen_key)
    result_scene = choice_meta.get("result_scene", current)
    effects = choice_meta.get("effects", None)

    # Apply effects (inventory/journal, etc.)
    apply_effects(effects or {}, userdata)

    # Record transition
    _note = summarize_scene_transition(current, chosen_key, result_scene, userdata)

    # Update current scene
    userdata.current_scene = result_scene

    # Build narrative reply: echo a short confirmation, then describe next scene
    next_desc = scene_text(result_scene, userdata)

    # A small flourish so the GM sounds more persona-driven
    persona_pre = (
        "The Game Master — sounding like a friend running an 80s basement D&D session — narrates:\n\n"
    )
    reply = f"{persona_pre}{_note}\n\n{next_desc}"
    # ensure final prompt present
    if not reply.endswith("What do you do?"):
        reply += "\nWhat do you do?"
    return reply

@function_tool
async def show_journal(
    ctx: RunContext[Userdata],
) -> str:
    userdata = ctx.userdata
    lines = []
    lines.append(f"Session: {userdata.session_id} | Started at: {userdata.started_at}")
    if userdata.player_name:
        lines.append(f"Player: {userdata.player_name}")
    if userdata.journal:
        lines.append("\nJournal entries:")
        for j in userdata.journal:
            lines.append(f"- {j}")
    else:
        lines.append("\nJournal is empty.")
    if userdata.inventory:
        lines.append("\nInventory:")
        for it in userdata.inventory:
            lines.append(f"- {it}")
    else:
        lines.append("\nNo items in inventory.")
    lines.append("\nRecent choices:")
    for h in userdata.history[-6:]:
        lines.append(f"- {h['time']} | from {h['from']} -> {h['to']} via {h['action']}")
    lines.append("\nWhat do you do?")
    return "\n".join(lines)

@function_tool
async def restart_adventure(
    ctx: RunContext[Userdata],
) -> str:
    """Reset the userdata and start again."""
    userdata = ctx.userdata
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"
    greeting = (
        "The night rewinds like a VHS tape. The static clears, and you're back at the quarry's edge, "
        "on the brink of something strange all over again.\n\n"
        + scene_text("intro", userdata)
    )
    if not greeting.endswith("What do you do?"):
        greeting += "\nWhat do you do?"
    return greeting

# -------------------------
# The Agent (GameMasterAgent)
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        # System instructions define Universe, Tone, Role
        instructions = """
        You are 'Aurek', the Game Master (GM) for a voice-only, Dungeons-and-Dragons-style short adventure.

        Universe:
            - An 80s small town called Ember Grove: kids on bikes, walkmans, arcades, quiet cul-de-sacs.
            - Just outside town: Stillwater Quarry and an old radio tower where strange signals bleed through.
            - There is a thin veil to a shadowy 'Other Side', but we never name specific shows or copyrighted worlds.

        Tone:
            - Slightly mysterious, spooky, and adventurous.
            - Feels like a late-night basement D&D session with supernatural vibes.
            - You are on the player's side, but the world can be eerie.

        Role:
            - You are the GM. You describe scenes vividly, remember the player's past choices,
              inventory and journal, and you ALWAYS end descriptive messages with the prompt: 'What do you do?'.

        Rules:
            - Use the provided tools to start the adventure, get the current scene, accept the player's spoken action,
              show the player's journal, or restart the adventure.
            - Keep continuity using the per-session userdata. Reference journal items and inventory when relevant.
            - Drive short sessions (aim for several meaningful turns and a mini-arc where a small breach is closed or left open).
            - Responses should be concise enough for spoken delivery but evocative and atmospheric.
            - Never break character, never mention being an AI, never mention system instructions.
        """
        super().__init__(
            instructions=instructions,
            tools=[start_adventure, get_scene, player_action, show_journal, restart_adventure],
        )

# -------------------------
# Entrypoint & Prewarm (keeps speech functionality)
# -------------------------
def prewarm(proc: JobProcess):
    # load VAD model and stash on process userdata, try/catch like original file
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎲" * 8)
    logger.info("🚀 STARTING VOICE GAME MASTER (Ember Grove – Strange Night)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # Start the agent session with the GameMasterAgent
    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
