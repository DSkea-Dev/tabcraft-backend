from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, httpx

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(audio_path, beam_size=5)
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
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_idx = int(chroma_mean.argmax())
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    major_corr = float(np.corrcoef(chroma_mean, np.roll(major_profile, key_idx))[0, 1])
    minor_corr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, key_idx))[0, 1])
    mode = "Major" if major_corr > minor_corr else "Minor"
    return {
        "key": f"{note_names[key_idx]} {mode}",
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
    if not ANTHROPIC_KEY:
        # No API key — return lyrics without chords
        return [
            {
                "name": ["Verse 1","Chorus","Verse 2","Chorus","Bridge","Outro"][min(i,5)],
                "lines": [{"chords": "", "lyrics": line} for line in lines]
            }
            for i, lines in enumerate(sections_lyrics)
        ]

    # Build lyrics text for Claude
    lyrics_text = ""
    section_names = ["Verse 1","Chorus","Verse 2","Chorus","Bridge","Outro"]
    for i, lines in enumerate(sections_lyrics):
        name = section_names[i] if i < len(section_names) else f"Section {i+1}"
        lyrics_text += f"[{name}]\n"
        for line in lines:
            lyrics_text += f"{line}\n"
        lyrics_text += "\n"

    prompt = f"""You are a guitarist. Add guitar chords to this song's lyrics.

Song: "{title}"
Key: {key}
Tempo: {tempo}

Rules:
- Use only 3-5 common chords that fit the key (e.g. for G Major use G, Em, C, D)
- Place chord names on a line ABOVE the lyric line they apply to
- Chords should change every 1-2 lines typically, not every word
- Keep it simple and playable — like a real singer-songwriter tab
- Return ONLY the tab with chords and lyrics, no explanation

{lyrics_text}"""

    try:
        import httpx
        response = httpx.post(
            ANTHROPIC_API,
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        data = response.json()
        raw = data["content"][0]["text"].strip()

        # Parse the response back into sections/lines
        result_sections = []
        current_section = None
        current_lines = []
        pending_chord = ""

        for line in raw.split("\n"):
            line = line.rstrip()
            if line.startswith("[") and line.endswith("]"):
                if current_section and current_lines:
                    result_sections.append({"name": current_section, "lines": current_lines})
                current_section = line[1:-1]
                current_lines = []
                pending_chord = ""
            elif not current_section:
                continue
            else:
                # Detect if line is mostly chords (short words, no common words)
                words = line.split()
                common_words = {"the","and","i","a","to","of","in","is","it","you","that","was","for","on","are","with","he","as","at","be","by","from","or","an","but","not","this","his","they","have","had","what","were","when","we","there","can","if","no","do","my","so","up","out","about","who","get","which","go","me","she","her","him","them","their","all","said","she'd","he'd","they'd","we'd","i'd","i'm","you're","it's"}
                is_chord_line = len(words) > 0 and len(words) <= 8 and all(
                    len(w) <= 4 and w[0].isupper() and w.lower() not in common_words
                    for w in words if w
                )
                if is_chord_line and line.strip():
                    pending_chord = line.strip()
                elif line.strip():
                    current_lines.append({"chords": pending_chord, "lyrics": line.strip()})
                    pending_chord = ""

        if current_section and current_lines:
            result_sections.append({"name": current_section, "lines": current_lines})

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
        return {"pattern": "D   D   D   D", "description": "Slow, deliberate downstrokes"}
    elif bpm < 100:
        return {"pattern": "D DU UDU", "description": "Gentle folk strum"}
    elif bpm < 130:
        return {"pattern": "D DU DU", "description": "Medium driving strum"}
    else:
        return {"pattern": "D D DU DU", "description": "Upbeat rhythmic strum"}


@app.get("/")
def root():
    return {"status": "TabCraft API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

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

        return {
            "title": song_title,
            "key": key_tempo["key"],
            "tempo": key_tempo["tempo"],
            "timeSignature": "4/4",
            "capo": "No capo",
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
