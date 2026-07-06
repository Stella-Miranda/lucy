import os
import sys
import asyncio  # 1. Imported asyncio to use Lock
import discord
from discord.ext import commands
from openai import AsyncOpenAI

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HF_SPACE_URL = os.getenv('HF_SPACE_URL') 

ai_client = AsyncOpenAI(
    base_url=f"{HF_SPACE_URL}/v1", 
    api_key="not-needed"
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 2. CREATE THE LOCK HERE
# This lock ensures only one request hits the model at any given millisecond
ai_lock = asyncio.Lock()

@bot.event
async def on_ready():
    print(f"Lucy is live on Render and queue system is active!")

@bot.command(name="ai")
async def ask_ai(ctx, *, prompt: str):
    # 3. Check if the bot is currently busy to give user feedback
    if ai_lock.locked():
        await ctx.send("⏳ I'm currently thinking for someone else. You've been placed in the queue!")

    # 4. Acquire the lock. If someone else is using it, this waits here.
    async with ai_lock:
        async with ctx.typing():
            try:
                response = await ai_client.chat.completions.create(
                    model="Qwen/Qwen2.5-3B-Instruct-GGUF", 
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
    # 5. The lock is automatically released here when the 'async with' block ends,
    # allowing the next user in line to go.

if not DISCORD_TOKEN or not HF_SPACE_URL:
    print("CRITICAL ERROR: Missing environment variables!")
    sys.exit(1)

bot.run(DISCORD_TOKEN.strip())
