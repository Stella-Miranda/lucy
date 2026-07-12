import os
import sys
import time
import math
import json
import hashlib
import asyncio
import threading
from flask import Flask
import discord
from discord.ext import commands
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models

# ==============================================================================
# 1. WEB SERVER FOR RENDER
# ==============================================================================
app = Flask(__name__)
@app.route('/')
def home(): return "Bot is alive!"
def run_flask(): threading.Thread(target=lambda: app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000))), daemon=True).start()
run_flask()

# ==============================================================================
# 2. CLIENT CONFIGURATIONS
# ==============================================================================
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HF_SPACE_URL = os.getenv('HF_SPACE_URL')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

ai_client = AsyncOpenAI(base_url=f"{HF_SPACE_URL}/v1", api_key="not-needed")

memory_db = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
MEMORY_COLLECTION = "lucy_memories"
PROFILE_COLLECTION = "lucy_profiles"
VECTOR_SIZE = 384

# --- Tunable memory parameters ---
IMPORTANCE_THRESHOLD = 4          # exchanges scoring below this are never saved
MEMORY_CAP_PER_USER = 1000        # hard cap before LILRU cleanup kicks in
DECAY_HALF_LIFE_DAYS = 14         # a memory's "weight" halves every N days
DECAY_LAMBDA = math.log(2) / (DECAY_HALF_LIFE_DAYS * 86400)
PROFILE_UPDATE_INTERVAL = 10      # summarize the rolling profile every N messages

for collection_name, size in [(MEMORY_COLLECTION, VECTOR_SIZE), (PROFILE_COLLECTION, VECTOR_SIZE)]:
    if not memory_db.collection_exists(collection_name):
        memory_db.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True       # Enforces receiving direct messages cleanly
intents.dm_reactions = True
bot = commands.Bot(command_prefix="!", intents=intents)
ai_lock = asyncio.Lock()


def stable_id(*parts: str) -> int:
    """Deterministic point ID (unlike Python's hash(), which is randomized
    per-process and would silently break profile lookups after a restart)."""
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return int(digest, 16) % (10 ** 12)


# ==============================================================================
# 3. MEMORY HELPERS (retrieval + gated saving)
# ==============================================================================

async def get_memories(user_id: str, prompt: str) -> str:
    try:
        results = memory_db.query_points(
            collection_name=MEMORY_COLLECTION,
            query=prompt,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
            ),
            limit=3
        ).points

        memories = [r.payload["text"] for r in results]
        return "\n".join(memories) if memories else "No relevant past memories found."
    except Exception as e:
        print(f"Memory retrieval error: {e}")
        return "No relevant past memories found."


async def save_memory(user_id: str, user_prompt: str, ai_response: str, importance: int):
    """Only ever called for exchanges that already cleared IMPORTANCE_THRESHOLD."""
    try:
        memory_text = f"User said: {user_prompt} | Lucy responded: {ai_response}"
        point_id = stable_id(str(user_id), memory_text, str(time.time()))

        memory_db.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=memory_db.embed(memory_text)[0].tolist(),
                    payload={
                        "user_id": str(user_id),
                        "text": memory_text,
                        "importance": importance,
                        "created_at": time.time()
                    }
                )
            ]
        )
        print(f"💡 Memory archived (importance={importance}).")

        # Make sure this user hasn't drifted over the per-user cap
        asyncio.create_task(cleanup_old_memories(user_id))
    except Exception as e:
        print(f"Failed to save memory: {e}")


# ==============================================================================
# 4. LILRU CLEANUP — Importance x Recency Decay
# ==============================================================================

async def cleanup_old_memories(user_id: str):
    """
    Keeps each user's stored memory count under MEMORY_CAP_PER_USER.
    Scores every memory as importance * e^(-lambda * age_seconds) — a fresh,
    important memory scores high; an old, trivial one decays toward zero —
    then deletes the lowest scorers until the user is back under the cap.
    """
    try:
        count_result = memory_db.count(
            collection_name=MEMORY_COLLECTION,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
            ),
            exact=True
        )
        total = count_result.count
        if total <= MEMORY_CAP_PER_USER:
            return

        points, _ = memory_db.scroll(
            collection_name=MEMORY_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
            ),
            limit=total,
            with_payload=True,
            with_vectors=False
        )

        now = time.time()
        scored = []
        for p in points:
            importance = p.payload.get("importance", 1)
            created_at = p.payload.get("created_at", now)
            age_seconds = max(0.0, now - created_at)
            score = importance * math.exp(-DECAY_LAMBDA * age_seconds)
            scored.append((score, p.id))

        scored.sort(key=lambda x: x[0])  # lowest (weakest) memories first
        excess = total - MEMORY_CAP_PER_USER
        ids_to_delete = [pid for _, pid in scored[:excess]]

        if ids_to_delete:
            memory_db.delete(
                collection_name=MEMORY_COLLECTION,
                points_selector=models.PointIdsList(points=ids_to_delete)
            )
            print(f"🧹 Pruned {len(ids_to_delete)} low-value memories for user {user_id}.")
    except Exception as e:
        print(f"Cleanup error: {e}")


# ==============================================================================
# 5. ROLLING USER PROFILE (long-term object permanence)
# ==============================================================================

async def get_user_profile(user_id: str) -> dict:
    """Fetch the compressed long-term profile for a user, or a blank default."""
    point_id = stable_id("profile", str(user_id))
    try:
        result = memory_db.retrieve(collection_name=PROFILE_COLLECTION, ids=[point_id], with_payload=True)
        if result:
            return result[0].payload
    except Exception as e:
        print(f"Profile retrieval error: {e}")

    return {"user_id": str(user_id), "profile_text": "No profile established yet.", "message_count": 0}


async def _write_profile(user_id: str, profile_text: str, message_count: int):
    point_id = stable_id("profile", str(user_id))
    vector = memory_db.embed(profile_text)[0].tolist()
    memory_db.upsert(
        collection_name=PROFILE_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "user_id": str(user_id),
                    "profile_text": profile_text,
                    "message_count": message_count,
                    "updated_at": time.time()
                }
            )
        ]
    )


async def maybe_update_profile(user_id: str, prompt: str, answer: str):
    """
    Bumps the user's interaction counter on every message. Every
    PROFILE_UPDATE_INTERVAL messages, asks the LLM to fold the recent
    exchange into a compressed profile (mood, traits, relationship arc)
    instead of relying purely on scattered vector memories.
    """
    try:
        profile = await get_user_profile(user_id)
        new_count = profile.get("message_count", 0) + 1

        if new_count % PROFILE_UPDATE_INTERVAL != 0:
            await _write_profile(user_id, profile.get("profile_text", ""), new_count)
            return

        summarizer_prompt = (
            "You maintain a compressed long-term profile of a Discord user for an AI companion named Lucy. "
            "Update the profile below using the new exchange. Keep it under 120 words, third person, covering: "
            "Current Mood, Core Personality Traits, and Relationship Arc Status. Respond with plain text only, "
            "no markdown, no preamble.\n\n"
            f"EXISTING PROFILE:\n{profile.get('profile_text', 'None yet.')}\n\n"
            f"NEW EXCHANGE:\nUser: {prompt}\nLucy: {answer}"
        )

        response = await ai_client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": summarizer_prompt}],
            max_tokens=200
        )
        updated_text = response.choices[0].message.content.strip()
        await _write_profile(user_id, updated_text, new_count)
        print(f"🧬 Profile updated for user {user_id}.")
    except Exception as e:
        print(f"Profile update error: {e}")


# ==============================================================================
# 6. RESPONSE PARSING — Lucy self-rates importance in structured JSON
# ==============================================================================

def parse_ai_response(raw_content: str):
    """
    Parses the model's JSON output into (reply_text, importance_int 1-10).
    Falls back gracefully to treating the raw text as the reply (importance=1)
    if the model ever slips and returns non-JSON, so a formatting hiccup
    never crashes the bot or corrupts memory storage.
    """
    try:
        data = json.loads(raw_content)
        reply = str(data.get("reply", "")).strip()
        importance = int(data.get("importance", 1))
        importance = max(1, min(10, importance))
        if not reply:
            raise ValueError("empty reply field")
        return reply, importance
    except Exception as e:
        print(f"JSON parse failed, falling back to raw text: {e}")
        return raw_content.strip(), 1


# ==============================================================================
# 7. DISCORD EVENT HANDLING
# ==============================================================================

# Add this dictionary at the top of your script with your other configurations
# Place these two tracking dictionaries at the top of your script
user_debounce_tasks = {}
user_message_buffers = {}

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.guild is None:
        if not message.content.startswith("!ai"):
            ctx = await bot.get_context(message)
            user_id = message.author.id

            # 1. Append the new message to this user's text bundle
            if user_id not in user_message_buffers:
                user_message_buffers[user_id] = []
            user_message_buffers[user_id].append(message.content)

            # 2. If a timer is already counting down, blow it up and reset it
            if user_id in user_debounce_tasks:
                user_debounce_tasks[user_id].cancel()

            # 3. Create the delayed execution task
            async def delayed_trigger():
                try:
                    await asyncio.sleep(3.0)  # Wait for a pause in typing
                    
                    # Merge all accumulated lines into one clean prompt block
                    # Format the bundled stream clearly so the LLM perceives them as sequential inputs
                    formatted_lines = [f"- {msg}" for msg in user_message_buffers[user_id]]
                    full_bundled_prompt = (
                        "The user sent these messages in rapid succession:\n" + 
                        "\n".join(formatted_lines) + 
                        "\n\nPlease respond naturally to this entire sequence."
                    )
                    
                    recent_messages = []
                    # Pull past history, making sure to skip ALL messages sent during this current bundle flurry
                    async for msg in message.channel.history(limit=12, oldest_first=True):
                        # Skip any message that is part of the current unhandled bundle
                        if msg.author.id == user_id and msg.content in user_message_buffers[user_id]:
                            continue
                        role = "assistant" if msg.author == bot.user else "user"
                        content = msg.content.replace("!ai ", "") if msg.content.startswith("!ai ") else msg.content
                        if content:
                            recent_messages.append({"role": role, "content": content})

                    # Clear the buffer for the next conversation turn before calling the LLM
                    user_message_buffers.pop(user_id, None)

                    # Hand the entire combined prompt block over to Lucy
                    await ask_ai(ctx, prompt=full_bundled_prompt, chat_history=recent_messages)
                    
                except asyncio.CancelledError:
                    pass  # Silently drop out if the user is still actively typing
                except Exception as e:
                    print(f"Error in bundle loop: {e}")
                    user_message_buffers.pop(user_id, None)

            # Assign and launch the timer task
            user_debounce_tasks[user_id] = asyncio.create_task(delayed_trigger())
            return

    await bot.process_commands(message)


@bot.command(name="ai")
async def ask_ai(ctx, *, prompt: str, chat_history: list = None):
    async with ai_lock:
        async with ctx.typing():
            try:
                # 1. Gather historical data elements
                past_memories = await get_memories(ctx.author.id, prompt)
                profile = await get_user_profile(ctx.author.id)

                system_instruction = (
                    "You are Lucy, a funny girl.\n\n"
                    f"LONG-TERM PROFILE OF THIS USER:\n{profile.get('profile_text', 'No profile yet.')}\n\n"
                    f"RELEVANT PAST MEMORIES:\n{past_memories}\n\n"
                    "CRITICAL: The user may send multiple rapid-fire messages bundled together separated by line breaks. "
                    "Acknowledge and respond naturally to the ENTIRE sequence of thoughts or questions they sent, "
                    "incorporating answers to all relevant parts into a cohesive, organic reply.\n\n"
                    "After replying, rate — from your own perspective, as Lucy — how important this exchange "
                    "is to remember long-term, on a 1-10 scale:\n"
                    "  1-3  = trivial small talk ('hey', 'lol', filler)\n"
                    "  4-6  = everyday detail (mood, what someone's wearing, a passing comment)\n"
                    "  7-10 = significant personal disclosure (dreams, life events, relationship shifts, "
                    "deep fears or hopes)\n\n"
                    "Respond with ONLY a JSON object, no markdown fences, no extra text, in exactly this shape:\n"
                    '{"reply": "<your in-character reply to the user>", "importance": <integer 1-10>}'
                )

                # 2. Build standard chat compilation window
                messages_payload = [{"role": "system", "content": system_instruction}]

                if chat_history:
                    messages_payload.extend(chat_history)
                
                # ALWAYS append the fresh incoming prompt at the very end so the LLM knows what to reply to!
                messages_payload.append({"role": "user", "content": prompt})

                # 3. Call execution model
                response = await ai_client.chat.completions.create(
                    model="local-model",
                    messages=messages_payload,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
                raw_content = response.choices[0].message.content
                answer, importance = parse_ai_response(raw_content)

                # 4. Return message down pipeline
                await ctx.send(answer if len(answer) <= 2000 else answer[:1990] + "...")

                # 5. Pipeline Memory Storage Filters
                if importance >= IMPORTANCE_THRESHOLD:
                    asyncio.create_task(save_memory(ctx.author.id, prompt, answer, importance))
                else:
                    print(f"🗑️ Skipped saving low-importance exchange (importance={importance}).")

                # Profile evaluation logic loops automatically
                asyncio.create_task(maybe_update_profile(ctx.author.id, prompt, answer))

            except Exception as e:
                await ctx.send("Lucy is having trouble thinking right now.")
                print(f"Error: {e}")

if not DISCORD_TOKEN or not HF_SPACE_URL:
    print("CRITICAL ERROR: Missing environment variables!")
    sys.exit(1)

bot.run(DISCORD_TOKEN.strip())
