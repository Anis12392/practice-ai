# Directive: Generate Posts from Raw Dump

## Purpose
Transform raw, unstructured thoughts (text, voice transcripts, photo descriptions) into polished X/LinkedIn posts aligned with the user's personal branding.

## Inputs
- Raw dump text (required) — via `--input` or `--file`
- User profile (`execution/profile.json`) — loaded automatically

## Script
`execution/generate_posts.py`

## Usage
```bash
# Direct text input
python3 execution/generate_posts.py --input "today I visited a robotics lab and their charging system was huge and proprietary"

# From file
python3 execution/generate_posts.py --file .tmp/dump.txt

# Dry run (no API call)
python3 execution/generate_posts.py --input "test" --dry-run
```

## Output
- 3 tweet options (280 chars max, 2-3 liners)
- 1 LinkedIn post (1500 chars max)

## Quality Checklist
- Posts match the user's tone from profile.json
- Posts align with user's objectives and themes
- No hashtags, emojis, or corporate language
- Tweets follow hook/punchline structure
- LinkedIn starts with a hook, not "I'm excited to..."

## Edge Cases
- If dump is too vague, the script will still generate but quality will be lower. Encourage detailed dumps.
- If dump contains multiple topics, focus on the strongest one for tweets, can combine for LinkedIn.

## Learnings
- (to be updated as we iterate)
