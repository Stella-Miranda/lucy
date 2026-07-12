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
intents.direct_messages = True
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

# ==============================================================================
# 3. HELPER FUNCTIONS FOR MEMORY PROCESSING (Keep your existing save_memory here)
# ==============================================================================
async def save_memory(user_id: str, user_prompt: str, ai_response: str):
    try:
        memory_text = f"User said: {user_prompt} | Lucy responded: {ai_response}"
        
        memory_db.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=hash(memory_text) % 10000000, 
                    vector=memory_db.embed(memory_text)[0].tolist(), 
                    payload={"user_id": str(user_id), "text": memory_text}
                )
            ]
        )
        print("💡 Memory successfully archived.")
    except Exception as e:
        print(f"Failed to save memory: {e}")


# ==============================================================================
# 4. DISCORD COMMAND OVERHAUL
# ==============================================================================

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if the message is in a DM
    if message.guild is None:
        if not message.content.startswith("!ai"):
            ctx = await bot.get_context(message)
            ctx.command = bot.get_command("ai")
            
            # Fetch the last 8 messages to build recent context
            # We use oldest_first=True so they are fed to the AI chronologically
            recent_messages = []
            async for msg in message.channel.history(limit=8, oldest_first=True):
                role = "assistant" if msg.author == bot.user else "user"
                # Strip out the command prefix if they manually typed it earlier
                content = msg.content.replace("!ai ", "") if msg.content.startswith("!ai ") else msg.content
                if content: # Avoid empty embed messages
                    recent_messages.append({"role": role, "content": content})

            # Hand over the context stack and current message to the AI invoker
            await bot.invoke(ctx, prompt=message.content, chat_history=recent_messages)
            return

    await bot.process_commands(message)


@bot.command(name="ai")
async def ask_ai(ctx, *, prompt: str, chat_history: list = None):
    async with ai_lock:
        async with ctx.typing():
            try:
                # Retrieve matching vector memories using the latest prompt
                past_memories = await get_memories(ctx.author.id, prompt)
                
                system_instruction = (
                    "You are Lucy, a funny girl."
                    "You have a continuous memory of past conversations. Rely heavily on the following "
                    f"retrieved past interactions to maintain conversational continuity: \n{past_memories}"
                )

                # Initialize our API payload with the core instruction
                messages_payload = [{"role": "system", "content": system_instruction}]

                if chat_history:
                    # Append the recent rolling context from the DM channel
                    messages_payload.extend(chat_history)
                else:
                    # Fallback for server mode command triggers (!ai)
                    messages_payload.append({"role": "user", "content": prompt})

                # Call the Hugging Face API
                response = await ai_client.chat.completions.create(
                    model="local-model",
                    messages=messages_payload,
                    max_tokens=250
                )
                answer = response.choices[0].message.content
                
                await ctx.send(answer if len(answer) <= 2000 else answer[:1990] + "...")
                
                # Archive the exchange into vector storage
                asyncio.create_task(save_memory(ctx.author.id, prompt, answer))
                
            except Exception as e:
                await ctx.send("Lucy is having trouble thinking right now.")
                print(f"Error: {e}")
if not DISCORD_TOKEN or not HF_SPACE_URL:
    print("CRITICAL ERROR: Missing environment variables!")
    sys.exit(1)

bot.run(DISCORD_TOKEN.strip())
