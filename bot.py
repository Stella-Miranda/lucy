import os
import sys
import asyncio
import threading
from flask import Flask
import discord
from discord.ext import commands
from openai import AsyncOpenAI
# IMPORTANT: Import models to use built-in Qdrant filters and structured points
from qdrant_client import QdrantClient, models  

# 1. WEB SERVER FOR RENDER
app = Flask(__name__)
@app.route('/')
def home(): return "Bot is alive!"
def run_flask(): threading.Thread(target=lambda: app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000))), daemon=True).start()
run_flask()

# 2. CLIENT CONFIGURATIONS
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HF_SPACE_URL = os.getenv('HF_SPACE_URL')
QDRANT_URL = os.getenv('QDRANT_URL')        
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')  

ai_client = AsyncOpenAI(base_url=f"{HF_SPACE_URL}/v1", api_key="not-needed")

# Initialize client with local/cloud auto-embedding capabilities
memory_db = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "lucy_memories"

# Safe collection creator
if not memory_db.collection_exists(COLLECTION_NAME):
    memory_db.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
ai_lock = asyncio.Lock()

# 3. HELPER FUNCTIONS FOR MEMORY PROCESSING (Corrected Qdrant Methods)
async def get_memories(user_id: str, prompt: str) -> str:
    try:
        # query_points auto-vectorizes text strings using your chosen embedding model
        results = memory_db.query_points(
            collection_name=COLLECTION_NAME,
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
        return ""

async def save_memory(user_id: str, user_prompt: str, ai_response: str):
    try:
        memory_text = f"User said: {user_prompt} | Lucy responded: {ai_response}"
        
        # upsert method parses a text document directly, embedding it in the cloud automatically
        memory_db.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=hash(memory_text) % 10000000, # Generate a simple numeric ID
                    vector=memory_db.embed(memory_text)[0].tolist(), # Cloud vectorized string representation
                    payload={"user_id": str(user_id), "text": memory_text}
                )
            ]
        )
        print("💡 Memory successfully archived.")
    except Exception as e:
        print(f"Failed to save memory: {e}")

# 4. DISCORD COMMAND OVERHAUL
@bot.command(name="ai")
async def ask_ai(ctx, *, prompt: str):
    if ai_lock.locked():
        await ctx.send("⏳ I'm currently thinking for someone else. You've been placed in the queue!")

    async with ai_lock:
        async with ctx.typing():
            try:
                # STEP A: Retrieve relative memories for this specific user
                past_memories = await get_memories(ctx.author.id, prompt)
                
                system_instruction = (
                    "You are Lucy, a helpful and witty AI assistant. "
                    "You have a continuous memory of past conversations. Rely heavily on the following "
                    f"retrieved past interactions to maintain conversational continuity: \n{past_memories}"
                )

                # STEP B: Ask the Hugging Face Model (LoRA ready)
                response = await ai_client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250
                )
                answer = response.choices[0].message.content
                
                # STEP C: Reply to Discord
                await ctx.send(answer if len(answer) <= 2000 else answer[:1990] + "...")
                
                # STEP D: Commit this exchange to long-term memory asynchronously
                asyncio.create_task(save_memory(ctx.author.id, prompt, answer))
                
            except Exception as e:
                await ctx.send("Lucy is having trouble thinking right now.")
                print(f"Error: {e}")

if not DISCORD_TOKEN or not HF_SPACE_URL:
    print("CRITICAL ERROR: Missing environment variables!")
    sys.exit(1)

bot.run(DISCORD_TOKEN.strip())
