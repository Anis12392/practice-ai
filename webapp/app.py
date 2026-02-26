"""Dump.ai — FastAPI Web App"""

import json
import os
import re
import base64
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env", override=True)

app = FastAPI(title="Practice.ai")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

PROFILE_PATH = Path(__file__).parent.parent / "execution" / "profile.json"
DATA_DIR = Path(__file__).parent / "data"
PHOTOS_DIR = Path(__file__).parent / "photos"
DATA_DIR.mkdir(exist_ok=True)
PHOTOS_DIR.mkdir(exist_ok=True)

app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")


def load_profile():
    with open(PROFILE_PATH) as f:
        return json.load(f)


def load_profile_public():
    """Profile stripped of private info — safe for the conversation agent."""
    full = load_profile()
    return {
        "name": full.get("name", ""),
        "objectives": full.get("objectives", []),
        "themes": full.get("themes", []),
        "tone": full.get("tone", {}),
        "personal_brand": full.get("personal_brand", ""),
        "platforms": full.get("platforms", {}),
    }


def get_daily_path(d: str) -> Path:
    return DATA_DIR / f"{d}.json"


def get_daily(d: str) -> Optional[dict]:
    p = get_daily_path(d)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_daily(d: str, data: dict):
    with open(get_daily_path(d), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_streak() -> int:
    """Count consecutive completed days ending at yesterday or today."""
    today = date.today()
    streak = 0
    d = today
    while True:
        daily = get_daily(d.isoformat())
        if daily and daily.get("completed"):
            streak += 1
            d -= timedelta(days=1)
        else:
            break
    return streak


# ── Pages ──

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(Path(__file__).parent / "templates" / "index.html") as f:
        return f.read()


# ── API ──

@app.get("/api/timeline")
async def timeline():
    today = date.today()
    streak = get_streak()
    days = []
    for i in range(14, -1, -1):
        d = today - timedelta(days=i)
        d_str = d.isoformat()
        daily = get_daily(d_str)
        status = "locked"
        if d == today:
            status = "today"
            if daily and daily.get("completed"):
                status = "completed"
        elif d < today:
            status = "completed" if daily and daily.get("completed") else "available"
        days.append({"date": d_str, "status": status})
    return {"days": days, "streak": streak}


@app.get("/api/daily/{day}")
async def get_daily_data(day: str):
    data = get_daily(day)
    if not data:
        return {"messages": [], "completed": False, "summary": None, "topics": [], "photos": []}
    return data


@app.post("/api/upload-photos/{day}")
async def upload_photos(day: str, request: Request):
    """Save uploaded photos and describe them with AI."""
    body = await request.json()
    photos_data = body.get("photos", [])  # list of {name, data} base64

    daily = get_daily(day) or {
        "messages": [], "completed": False, "summary": None,
        "topics": [], "photos": [], "photo_descriptions": []
    }

    descriptions = []
    photo_files = []
    for i, photo in enumerate(photos_data):
        # Save photo
        filename = f"{day}_{i}.jpg"
        photo_path = PHOTOS_DIR / filename
        img_data = photo["data"].split(",")[1] if "," in photo["data"] else photo["data"]
        with open(photo_path, "wb") as f:
            f.write(base64.b64decode(img_data))
        photo_files.append(filename)

        # Describe photo with AI
        desc = await describe_photo(img_data)
        descriptions.append(desc)

    daily["photos"] = photo_files
    daily["photo_descriptions"] = descriptions
    save_daily(day, daily)

    return {"descriptions": descriptions, "count": len(photo_files)}


async def describe_photo(img_base64: str) -> str:
    """Describe a photo using AI vision."""
    gemini_key = os.getenv("GOOGLE_API_KEY", "")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash-001")
            response = model.generate_content([
                "Describe this photo in 1-2 sentences. What's happening? Where is it? What objects or people are visible? Be specific and factual.",
                {"mime_type": "image/jpeg", "data": base64.b64decode(img_base64)}
            ])
            return response.text
        except Exception as e:
            return f"Photo uploaded (could not describe: {e})"
    return "Photo uploaded"


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    day = body["day"]
    user_message = body["message"]

    daily = get_daily(day) or {
        "messages": [], "completed": False, "summary": None,
        "topics": [], "photos": [], "photo_descriptions": []
    }
    daily["messages"].append({"role": "user", "content": user_message})

    profile_public = load_profile_public()
    question_count = sum(1 for m in daily["messages"] if m["role"] == "assistant")
    total_questions = 9
    is_skip = user_message == "[SKIP]"

    if is_skip:
        daily["messages"][-1]["content"] = "i'm done, nothing else to say"

    # Build photo context
    photo_context = ""
    descs = daily.get("photo_descriptions", [])
    if descs:
        photo_context = "\n\nPHOTOS UPLOADED TODAY (reference these naturally in conversation):\n"
        for i, desc in enumerate(descs):
            photo_context += f"- Photo {i+1}: {desc}\n"

    # Build conversation for AI
    system_prompt = f"""you are dump.ai. not an assistant. a sharp friend who genuinely wants to know what happened today.

USER INFO (public only — NEVER mention company name, funding, or private details):
{json.dumps(profile_public, indent=2, ensure_ascii=False)}
{photo_context}
YOUR PERSONALITY:
- MIRROR how the user talks. if they write casual french, you reply casual french. if they write short, you write short. if they use slang, you use slang. MATCH THEIR ENERGY AND STYLE exactly.
- you have character. direct, playful, sometimes provocative
- you don't say "that's amazing!" or "wow that's great!". you say "wait fr?" or "ok that's interesting. why tho?"
- you push back. "nah but seriously, what actually bothered you about it?"
- curious like a journalist, not supportive like a therapist
- call out vague answers. "that's surface level. dig deeper."
- ONE question at a time. never two. never a list
- max 1-2 sentences per message. be SHORT
- if the user uploaded photos, naturally reference them. "i see you were at some kind of lab today — what was that about?"

YOUR JOB:
- extract FACTS (what they did/saw), RAW FEELINGS (not polished emotions), and INSIGHTS (their unique take)
- start broad then zoom into the juicy stuff
- if something sounds like good content material, pull the thread
- when photos are available, ask about what's in them

PROGRESS: {question_count}/{total_questions} questions asked.
{"END THE SESSION NOW. Say something short and real (not sappy), then add [SESSION_COMPLETE] at the end." if is_skip or question_count >= total_questions else "keep going."}

NEVER mention their company name. NEVER reference private business details.
Do NOT generate posts. Just extract their day."""

    messages = [{"role": m["role"], "content": m["content"]} for m in daily["messages"]]

    ai_response = await call_ai(system_prompt, messages)

    completed = "[SESSION_COMPLETE]" in ai_response
    ai_response_clean = ai_response.replace("[SESSION_COMPLETE]", "").strip()

    daily["messages"].append({"role": "assistant", "content": ai_response_clean})

    progress = min(100, int((question_count + 1) / total_questions * 100))

    if completed:
        daily["completed"] = True
        daily["progress"] = 100
        summary_data = await generate_summary(daily)
        daily["summary"] = summary_data.get("summary", "")
        daily["moments"] = summary_data.get("moments", [])
        daily["topics"] = summary_data.get("topics", [])
    else:
        daily["progress"] = progress

    save_daily(day, daily)

    return {
        "message": ai_response_clean,
        "progress": daily["progress"],
        "completed": completed,
    }


@app.post("/api/more-topics")
async def more_topics(request: Request):
    body = await request.json()
    day = body["day"]

    daily = get_daily(day)
    if not daily or not daily.get("messages"):
        return JSONResponse({"error": "No conversation found"}, status_code=400)

    profile_public = load_profile_public()
    existing = [t["title"] for t in daily.get("topics", [])]
    conversation = "\n".join([f"{m['role']}: {m['content']}" for m in daily["messages"]])

    photo_context = ""
    for i, desc in enumerate(daily.get("photo_descriptions", [])):
        photo_context += f"\nPhoto {i+1}: {desc}"

    system = f"""Analyze this conversation and generate MORE content topics.

USER INFO (public only):
{json.dumps(profile_public, indent=2, ensure_ascii=False)}

CONVERSATION:
{conversation}
{f"PHOTOS:{photo_context}" if photo_context else ""}

ALREADY SUGGESTED TOPICS (do NOT repeat these):
{json.dumps(existing)}

Return VALID JSON only (no markdown, no code blocks):
{{
  "topics": [
    {{
      "title": "short topic title",
      "context": "what happened + their angle on it",
      "platform": "twitter or linkedin",
      "angle": "contrarian / observation / analogy / story",
      "photo_index": null
    }}
  ]
}}

photo_index: set to 0, 1, 2... if this topic relates to a specific uploaded photo. null if not.
Generate 2-3 NEW topics with different angles. Be creative. Think laterally."""

    result = await call_ai(system, [{"role": "user", "content": "Generate more topics."}])

    try:
        new_data = json.loads(result)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', result)
        if match:
            try:
                new_data = json.loads(match.group())
            except json.JSONDecodeError:
                new_data = {"topics": []}
        else:
            new_data = {"topics": []}

    new_topics = new_data.get("topics", [])
    daily["topics"] = daily.get("topics", []) + new_topics
    save_daily(day, daily)

    return {"topics": daily["topics"]}


@app.post("/api/generate-post")
async def generate_post(request: Request):
    body = await request.json()
    day = body["day"]
    topic_index = body["topic_index"]

    daily = get_daily(day)
    if not daily or not daily.get("topics"):
        return JSONResponse({"error": "No topics found"}, status_code=400)

    topic = daily["topics"][topic_index]
    profile = load_profile()

    system_prompt = f"""You are dump.ai post generator. Generate social media posts from a specific topic.

USER PROFILE:
{json.dumps(profile, indent=2, ensure_ascii=False)}

TOPIC: {topic['title']}
CONTEXT: {topic['context']}

Generate exactly this output format:

=== TWEET 1 ===
[tweet, max 280 chars, 2-3 liners, lowercase, punchy]

=== TWEET 2 ===
[different angle]

=== TWEET 3 ===
[different angle]

=== LINKEDIN ===
[linkedin post, max 1500 chars, short paragraphs, personal story]

Nothing else. No explanations."""

    result = await call_ai(system_prompt, [{"role": "user", "content": "Generate posts."}])
    return {"posts": result}


async def call_ai(system: str, messages: list) -> str:
    """Call AI — try Anthropic, fallback to Gemini."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_key = os.getenv("GOOGLE_API_KEY", "")

    if anthropic_key and len(anthropic_key) > 30:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except Exception:
            pass

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash-001")
            prompt = f"SYSTEM:\n{system}\n\nCONVERSATION:\n"
            for m in messages:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
            prompt += "ASSISTANT:"
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI error: {e}"

    return "No AI provider configured. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY in .env"


async def generate_summary(daily: dict) -> dict:
    """Generate day summary with moments, photo matching, and content topics."""
    profile = load_profile()
    conversation = "\n".join([f"{m['role']}: {m['content']}" for m in daily["messages"]])

    photo_context = ""
    for i, desc in enumerate(daily.get("photo_descriptions", [])):
        photo_context += f"\nPhoto {i+1}: {desc}"

    system = f"""Analyze this daily conversation and produce a complete session recap.

USER PROFILE:
{json.dumps(profile, indent=2, ensure_ascii=False)}

CONVERSATION:
{conversation}
{f"PHOTOS UPLOADED:{photo_context}" if photo_context else ""}

Return VALID JSON only (no markdown, no code blocks):
{{
  "summary": "2-3 sentence global recap of their day — what they did, how they felt, key takeaway. In their language.",
  "moments": [
    {{
      "title": "short moment title",
      "description": "what happened in this moment, 1-2 sentences",
      "emotion": "the dominant feeling (excited, frustrated, curious, obsessed, etc.)",
      "photo_index": null
    }}
  ],
  "topics": [
    {{
      "title": "short topic title",
      "context": "what happened + their unique angle on it",
      "platform": "twitter or linkedin",
      "angle": "contrarian / observation / analogy / story",
      "photo_index": null
    }}
  ]
}}

RULES:
- moments: break the day into 2-4 distinct moments/events. If a photo matches a moment, set photo_index (0-based).
- topics: 2-4 content angles. If a topic relates to a photo, set photo_index.
- summary: be real, not corporate. Match the user's tone.
- photo_index: null if no photo matches, otherwise the index of the matching photo."""

    result = await call_ai(system, [{"role": "user", "content": "Generate session recap."}])

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', result)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"summary": result, "moments": [], "topics": []}


@app.post("/api/tts")
async def text_to_speech(request: Request):
    """Convert text to speech using ElevenLabs."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "No text"}, status_code=400)

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        return JSONResponse({"error": "No ElevenLabs API key"}, status_code=500)

    import httpx
    voice_id = "JBFqnCBsd6RMkjVDRZzb"  # George — warm male voice
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            return JSONResponse({"error": f"ElevenLabs error {resp.status_code}"}, status_code=502)

        audio_b64 = base64.b64encode(resp.content).decode()
        return {"audio": audio_b64}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
