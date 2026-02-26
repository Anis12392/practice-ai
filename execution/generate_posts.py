#!/usr/bin/env python3
"""
Dump.ai — Transform raw thoughts into polished posts.
Usage: python3 generate_posts.py --input "your raw dump here"
       python3 generate_posts.py --file dump.txt
       python3 generate_posts.py --input "..." --provider gemini
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env", override=True)

PROFILE_PATH = Path(__file__).parent / "profile.json"

SYSTEM_PROMPT = """You are Dump.ai, a personal branding engine. Your job is to transform raw, unstructured thoughts into polished social media posts.

You will receive:
1. A USER PROFILE with their identity, objectives, tone, and example posts
2. A RAW DUMP of their thoughts, experiences, or observations

Your job:
- Extract the key facts, observations, and insights from the dump
- Generate posts that sound like THIS SPECIFIC PERSON, not generic AI content
- Match their tone exactly (study their example posts)
- Align every post with their objectives and themes
- Never use hashtags, emojis, or corporate language

TWEET STRUCTURE RULES:
- 2-liner or 3-liner max
- Line 1 = HOOK (what they saw/did, concrete, visual)
- Line 2 = PUNCHLINE (their unique angle, insight, or analogy)
- Lowercase. Short sentences. Fragmented is fine.
- Max 280 characters per tweet
- No hashtags. No tags. No "thread:" prefix.

LINKEDIN STRUCTURE RULES:
- Start with a hook line (not "I'm excited to share...")
- Short paragraphs (1-2 sentences each)
- Personal story first, then insight
- End with a forward-looking statement or question
- No corporate fluff. No buzzwords. Keep it real.
- Max 1500 characters

OUTPUT FORMAT (strict):
=== TWEET 1 ===
[tweet text]

=== TWEET 2 ===
[tweet text]

=== TWEET 3 ===
[tweet text]

=== LINKEDIN ===
[linkedin post]

Nothing else. No explanations. No "here are your posts". Just the posts."""


def load_profile():
    if not PROFILE_PATH.exists():
        print(f"Profile not found at {PROFILE_PATH}")
        print("Create profile.json with your personal branding info.")
        sys.exit(1)
    with open(PROFILE_PATH) as f:
        return json.load(f)


def build_user_message(dump_text: str, profile: dict) -> str:
    return f"""USER PROFILE:
{json.dumps(profile, indent=2, ensure_ascii=False)}

RAW DUMP:
{dump_text}"""


def generate_with_anthropic(user_message: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def generate_with_gemini(user_message: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text


def generate_posts(dump_text: str, profile: dict, provider: str = "gemini") -> str:
    user_message = build_user_message(dump_text, profile)

    if provider == "anthropic":
        return generate_with_anthropic(user_message)
    elif provider == "gemini":
        return generate_with_gemini(user_message)
    else:
        print(f"Unknown provider: {provider}. Use 'anthropic' or 'gemini'.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Dump.ai — Raw thoughts to polished posts")
    parser.add_argument("--input", "-i", type=str, help="Your raw dump text")
    parser.add_argument("--file", "-f", type=str, help="Path to a file containing your dump")
    parser.add_argument("--provider", "-p", type=str, default="gemini",
                        choices=["anthropic", "gemini"],
                        help="AI provider (default: gemini)")
    parser.add_argument("--dry-run", action="store_true", help="Show the prompt without calling the API")
    args = parser.parse_args()

    if not args.input and not args.file:
        print("Usage: python3 generate_posts.py --input 'your raw dump here'")
        print("       python3 generate_posts.py --file dump.txt")
        sys.exit(1)

    if args.file:
        with open(args.file) as f:
            dump_text = f.read()
    else:
        dump_text = args.input

    profile = load_profile()

    if args.dry_run:
        print("=== PROFILE ===")
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        print("\n=== RAW DUMP ===")
        print(dump_text)
        print(f"\n[dry-run: provider={args.provider}, no API call made]")
        return

    print(f"generating posts with {args.provider}...\n")
    result = generate_posts(dump_text, profile, args.provider)
    print(result)


if __name__ == "__main__":
    main()
