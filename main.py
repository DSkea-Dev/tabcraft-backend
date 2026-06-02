from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, httpx

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
# API key read fresh on each request (see add_chords_with_claude)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def transcribe_audio(audio_path: str) -> dict:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(audio_path, beam_size=5, language="en", condition_on_previous_text=True, no_speech_threshold=0.6)
    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text
            })
    return {"segments": segments, "language": info.language}


def detect_key_and_tempo(audio_path: str) -> dict:
    import librosa
    import numpy as np
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Return tempo only — let Claude determine key/capo from musical context
    return {
        "key": "unknown",
        "tempo": f"{int(round(float(tempo)))} BPM"
    }


def group_lyrics_into_sections(segments):
    """Group whisper segments into sections based on gaps."""
    sections = []
    current = []
    prev_end = 0
    for seg in segments:
        if seg["start"] - prev_end > 2.0 and current:
            sections.append(current)
            current = []
        current.append(seg["text"])
        prev_end = seg["end"]
    if current:
        sections.append(current)
    return sections


def add_chords_with_claude(title, key, tempo, sections_lyrics):
    """Use Claude to intelligently assign chords to each lyric line."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"API key present: {bool(api_key)}, length: {len(api_key)}")

    section_names = ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Outro"]

    if not api_key:
        print("No API key — returning lyrics without chords")
        return [
            {
                "name": section_names[min(i, 5)],
                "lines": [{"chords": "", "lyrics": line} for line in lines]
            }
            for i, lines in enumerate(sections_lyrics)
        ]

    # Build lyrics for Claude
    lyrics_text = ""
    for i, lines in enumerate(sections_lyrics):
        name = section_names[i] if i < len(section_names) else f"Section {i+1}"
        lyrics_text += f"[{name}]\n"
        for line in lines:
            lyrics_text += f"{line}\n"
        lyrics_text += "\n"

    prompt = f"""You are an expert guitarist. Add guitar chords to these song lyrics and determine the key and capo.

Song: "{title}"
Tempo: {tempo}

YOUR TASKS:
1. Determine the best guitar key and capo position for this song
2. Add chords above each lyric line — typically 2 chords per line
3. Use common open chord shapes (G, C, D, Em, Am, F, A, E, Bm, D7 etc)
4. Keep 4-6 chords total for the whole song
5. Verses repeat the same pattern, choruses repeat their own pattern

CHORD FORMAT — put chords on a line above the lyrics, spaced to align where they change:
G                    D
Headed down south to the land of the pines
Em                   C
And I'm thumbin' my way into North Carolina

FIRST LINE of your response must be:
KEY: [key] | CAPO: [capo fret or "No capo"]

Then the full chord chart with [Section Name] headers.
Return ONLY the key line + chord chart, no other explanation.

{lyrics_text}"""

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        print(f"Claude API status: {response.status_code}")
        data = response.json()

        if response.status_code != 200:
            print(f"Claude API error: {data}")
            return None

        raw = data["content"][0]["text"].strip()
        print(f"Claude response length: {len(raw)}")

        # Extract KEY and CAPO from first line if present
        lines_raw = raw.split("\n")
        detected_key = None
        detected_capo = None
        if lines_raw and lines_raw[0].startswith("KEY:"):
            first_line = lines_raw[0]
            try:
                key_part = first_line.split("|")[0].replace("KEY:", "").strip()
                capo_part = first_line.split("|")[1].replace("CAPO:", "").strip() if "|" in first_line else "No capo"
                detected_key = key_part
                detected_capo = capo_part
            except:
                pass
            raw = "\n".join(lines_raw[1:]).strip()

        # Store for returning in result
        if detected_key:
            add_chords_with_claude._last_key = detected_key
            add_chords_with_claude._last_capo = detected_capo

        # Parse response into sections
        result_sections = []
        current_section = None
        current_lines = []
        pending_chord = ""

        for line in raw.split("\n"):
            stripped = line.strip()

            # Section header
            if stripped.startswith("[") and stripped.endswith("]"):
                if current_section is not None:
                    result_sections.append({"name": current_section, "lines": current_lines})
                current_section = stripped[1:-1]
                current_lines = []
                pending_chord = ""
                continue

            if current_section is None:
                continue

            if not stripped:
                continue

            # Is this a chord line? — all tokens are chord-like (start with uppercase, short, no lowercase words)
            tokens = stripped.split()
            chord_chars = set("ABCDEFGabcdefg#mb/0123456789dimaugsusMmaj")
            is_chord_line = (
                1 <= len(tokens) <= 6 and
                all(
                    len(t) <= 5 and
                    t[0].isupper() and
                    all(c in chord_chars for c in t)
                    for t in tokens
                )
            )

            if is_chord_line:
                pending_chord = stripped
            else:
                current_lines.append({"chords": pending_chord, "lyrics": stripped})
                pending_chord = ""

        if current_section is not None and current_lines:
            result_sections.append({"name": current_section, "lines": current_lines})

        print(f"Parsed {len(result_sections)} sections")
        return result_sections if result_sections else None

    except Exception as e:
        print(f"Claude chord error: {e}")
        return None


def detect_strumming(tempo_bpm: str) -> dict:
    try:
        bpm = int(tempo_bpm.replace(" BPM", ""))
    except:
        bpm = 90
    if bpm < 70:
        return {"pattern": "D D D D", "description": "Slow, deliberate downstrokes"}
    elif bpm < 100:
        return {"pattern": "D DU UDU", "description": "Gentle folk strum"}
    elif bpm < 130:
        return {"pattern": "D D DU UD", "description": "Medium driving strum"}
    else:
        return {"pattern": "D D UD UD", "description": "Upbeat rhythmic strum"}


@app.get("/")
def root():
    return {"status": "TabCraft API running"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-claude")
async def test_claude():
    """Test endpoint to verify Claude API connectivity."""
    import httpx
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "No ANTHROPIC_API_KEY set in environment"}
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "Say OK"}]
            },
            timeout=15.0
        )
        data = response.json()
        if response.status_code == 200:
            return {"status": "ok", "message": "Claude API working", "response": data["content"][0]["text"]}
        else:
            return {"status": "error", "code": response.status_code, "detail": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        song_title = os.path.splitext(file.filename)[0]\
            .replace("-", " ").replace("_", " ")\
            .replace("(MP3)", "").replace("(WAV)", "").strip()

        print("Transcribing...")
        transcription = transcribe_audio(tmp_path)
        segments = transcription.get("segments", [])
        print(f"{len(segments)} segments")

        print("Detecting key/tempo...")
        key_tempo = detect_key_and_tempo(tmp_path)
        print(f"{key_tempo}")

        print("Grouping lyrics...")
        sections_lyrics = group_lyrics_into_sections(segments)

        print("Adding chords with Claude...")
        sections = add_chords_with_claude(song_title, key_tempo["key"], key_tempo["tempo"], sections_lyrics)

        # Fallback: no chords, just lyrics in sections
        if not sections:
            section_names = ["Verse 1","Chorus","Verse 2","Chorus","Bridge","Outro"]
            sections = [
                {
                    "name": section_names[i] if i < len(section_names) else f"Section {i+1}",
                    "lines": [{"chords": "", "lyrics": line} for line in lines]
                }
                for i, lines in enumerate(sections_lyrics)
            ]

        strum = detect_strumming(key_tempo["tempo"])

        final_key = getattr(add_chords_with_claude, "_last_key", key_tempo["key"])
        final_capo = getattr(add_chords_with_claude, "_last_capo", "No capo")

        return {
            "title": song_title,
            "key": final_key,
            "tempo": key_tempo["tempo"],
            "timeSignature": "4/4",
            "capo": final_capo,
            "strummingPattern": strum["pattern"],
            "strummingDescription": strum["description"],
            "sections": sections,
            "notes": "Chords suggested by AI based on key — verify by ear and adjust as needed."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
