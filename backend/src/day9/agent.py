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

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logger = logging.getLogger("ecommerce_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# ------------------------------------------------------------
# PRODUCT CATALOG — UPDATED SIZES
# ------------------------------------------------------------

CLOTHING_SIZES = ["small", "medium", "large", "xl", "double xl"]

CATALOG = [
    {
        "id": "hoodie-001",
        "name": "Black Oversized Hoodie",
        "price": 1199,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": CLOTHING_SIZES
    },
    {
        "id": "hoodie-002",
        "name": "Blue Classic Hoodie",
        "price": 999,
        "currency": "INR",
        "category": "hoodie",
        "color": "blue",
        "sizes": CLOTHING_SIZES
    },
    {
        "id": "tee-001",
        "name": "Dragon Print T-Shirt",
        "price": 699,
        "currency": "INR",
        "category": "tshirt",
        "color": "black",
        "sizes": CLOTHING_SIZES
    },
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "price": 499,
        "currency": "INR",
        "category": "mug",
        "color": "white",
        "sizes": ["free size"]
    },
    {
        "id": "mug-002",
        "name": "Dark Matte Coffee Mug",
        "price": 599,
        "currency": "INR",
        "category": "mug",
        "color": "black",
        "sizes": ["free size"]
    }
]

# ------------------------------------------------------------
# ORDER FILE — GUARANTEED CREATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORDERS_FILE = os.path.join(BASE_DIR, "orders.json")


logger.info(f"📁 Orders will be saved to: {ORDERS_FILE}")

def load_orders():
    if not os.path.exists(ORDERS_FILE):
        logger.info("orders.json not found, creating new file...")
        save_orders([])
        return []
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_orders(order_list):
    with open(ORDERS_FILE, "w") as f:
        json.dump(order_list, f, indent=2)
    logger.info(f"💾 Order saved successfully → {ORDERS_FILE}")

# ------------------------------------------------------------
# FILTERING
# ------------------------------------------------------------

def filter_products(text: str):
    text = text.lower()
    results = CATALOG

    if "mug" in text:
        results = [p for p in results if p["category"] == "mug"]
    if "hoodie" in text:
        results = [p for p in results if p["category"] == "hoodie"]
    if "tshirt" in text or "t-shirt" in text:
        results = [p for p in results if p["category"] == "tshirt"]

    if "black" in text:
        results = [p for p in results if p.get("color") == "black"]
    if "white" in text:
        results = [p for p in results if p.get("color") == "white"]
    if "blue" in text:
        results = [p for p in results if p.get("color") == "blue"]

    if "under" in text:
        try:
            limit = int(text.split("under")[1].strip().split()[0])
            results = [p for p in results if p["price"] <= limit]
        except:
            pass

    return results[:5]

# ------------------------------------------------------------
# SESSION DATA
# ------------------------------------------------------------

@dataclass
class SessionData:
    last_shown_products: List[dict] = field(default_factory=list)
    last_order: Optional[dict] = None

# ------------------------------------------------------------
# TOOLS
# ------------------------------------------------------------

@function_tool
async def browse_products(
    ctx: RunContext[SessionData],
    query: Annotated[str, Field(description="User shopping query")]
) -> str:
    products = filter_products(query)
    ctx.userdata.last_shown_products = products

    if not products:
        return "I couldn't find anything matching your request."

    summary = "Here are some items I found:\n"
    for idx, p in enumerate(products, start=1):
        size_info = f" | Sizes: {', '.join(p['sizes'])}"
        summary += f"{idx}. {p['name']} — {p['price']} {p['currency']}{size_info}\n"

    summary += "\nSay: 'Buy item 1 in size medium'."
    return summary


@function_tool
async def create_order(
    ctx: RunContext[SessionData],
    product_index: Annotated[int, Field(description="Index of item from result list")],
    quantity: Annotated[int, Field(description="Qty")] = 1,
    size: Annotated[Optional[str], Field(description="small, medium, large, xl, double xl or free size", default=None)] = None
) -> str:
    products = ctx.userdata.last_shown_products
    if not products or product_index < 1 or product_index > len(products):
        return "Invalid item number."

    product = products[product_index - 1]

    # Validate size
    if size:
        size = size.lower()
        if size not in product["sizes"]:
            return f"{product['name']} does not come in '{size}'. Available sizes: {', '.join(product['sizes'])}"

    order = {
        "id": str(uuid.uuid4())[:8],
        "items": [{
            "product_id": product["id"],
            "quantity": quantity,
            "size": size
        }],
        "total": product["price"] * quantity,
        "currency": product["currency"],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "product_name": product["name"]
    }

    orders = load_orders()
    orders.append(order)
    save_orders(orders)

    ctx.userdata.last_order = order

    size_msg = f" in size {size}" if size else ""
    return f"Your order for {product['name']}{size_msg} has been placed successfully!"


@function_tool
async def last_order(
    ctx: RunContext[SessionData],
):
    if not ctx.userdata.last_order:
        return "You haven't placed any orders yet."
    o = ctx.userdata.last_order
    size = o["items"][0].get("size")
    size_msg = f" (size {size})" if size else ""
    return f"You ordered {o['product_name']}{size_msg} for {o['total']} {o['currency']}."

# ------------------------------------------------------------
# AGENT
# ------------------------------------------------------------

class EcommerceAgent(Agent):
    def __init__(self):
        instructions = """
        Your name is STEVE.
        You must always greet users with:
        "Hi, I'm Steve, your shopping assistant."

        Help users browse items, buy items, and check last orders.
        Use the tools when needed.
        """
        super().__init__(
            instructions=instructions,
            tools=[browse_products, create_order, last_order]
        )

# ------------------------------------------------------------
# SPEECH PIPELINE
# ------------------------------------------------------------

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except:
        logger.warning("VAD preload failed.")

async def entrypoint(ctx: JobContext):
    userdata = SessionData()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-marcus", style="Conversational", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata
    )

    await session.start(
        agent=EcommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
