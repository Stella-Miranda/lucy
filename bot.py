import os
import sys
import asyncio
import threading
from flask import Flask  # Added Flask back to trick Render's port scanner
import discord
from discord.ext import commands
from openai import AsyncOpenAI

# 1. DUMMY WEB SERVER FOR RENDER FREE TIER
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive!"

def run_flask():
    # Render automatically passes a PORT environment variable, defaults to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# Start the web server in a separate background thread
threading.Thread(target=run_flask, daemon=True).start()

# 2. DISCORD BOT LOGIC
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HF_SPACE_URL = os.getenv('HF_SPACE_URL') 

ai_client = AsyncOpenAI(
    base_url=f"{HF_SPACE_URL}/v1", 
    api_key="not-needed"
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

ai_lock = asyncio.Lock()

@bot.event
async def on_ready():
    print(f"Lucy is live on Render Web Service Free Tier!")

@bot.command(name="ai")
async def ask_ai(ctx, *, prompt: str):
    if ai_lock.locked():
        await ctx.send("⏳ I'm currently thinking for someone else. You've been placed in the queue!")

    async with ai_lock:
        async with ctx.typing():
            try:
                response = await ai_client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": "You are Lucy, a helpful and witty AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250
                )
                answer = response.choices[0].message.content
                await ctx.send(answer if len(answer) <= 2000 else answer[:1990] + "...")
            except Exception as e:
                await ctx.send("Lucy is having trouble thinking right now.")
                print(f"Error: {e}")

if not DISCORD_TOKEN or not HF_SPACE_URL:
    print("CRITICAL ERROR: Missing environment variables!")
    sys.exit(1)

bot.run(DISCORD_TOKEN.strip())
