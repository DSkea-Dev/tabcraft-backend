from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, httpx

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
    segments_iter, info = model.transcribe(
        audio_path, beam_size=5, language="en",
        condition_on_previous_text=True, no_speech_threshold=0.6
    )
    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return {"segments": segments, "language": info.language}


def detect_tempo(audio_path: str) -> str:
    import librosa
    import numpy as np
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return f"{int(round(float(tempo)))} BPM"


def detect_chord_at_time(y, sr, start: float, end: float) -> str:
    """Detect the most likely chord during a time segment using chroma analysis."""
    import librosa
    import numpy as np

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Chord templates: major and minor triads
    def chord_template(root, intervals):
        t = np.zeros(12)
        for i in intervals:
            t[(root + i) % 12] = 1.0
        return t

    chord_templates = {}
    for r in range(12):
        chord_templates[note_names[r]] = chord_template(r, [0, 4, 7])        # major
        chord_templates[note_names[r] + 'm'] = chord_template(r, [0, 3, 7])  # minor
        chord_templates[note_names[r] + '7'] = chord_template(r, [0, 4, 7, 10])  # dominant 7th

    # Slice audio to segment
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = y[start_sample:end_sample]

    if len(segment) < sr * 0.1:  # too short
        return ""

    # Get chroma for this segment
    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    if chroma_mean.max() < 0.01:  # silence
        return ""

    # Normalise
    chroma_norm = chroma_mean / (chroma_mean.max() + 1e-6)

    # Find best matching chord
    best_chord = ""
    best_score = -1.0
    for chord_name, template in chord_templates.items():
        score = float(np.dot(chroma_norm, template) / (np.linalg.norm(template) + 1e-6))
        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord


def detect_key(y, sr) -> str:
    """Detect musical key using Krumhansl-Schmuckler key profiles."""
    import librosa
    import numpy as np

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_key = "G Major"
    best_score = -999.0
    for i in range(12):
        maj = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
        mn = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])
        if maj > best_score:
            best_score = maj
            best_key = f"{note_names[i]} Major"
        if mn > best_score:
            best_score = mn
            best_key = f"{note_names[i]} Minor"

    return best_key


def build_sections_with_chords(segments, y, sr) -> list:
    """Build song sections with real detected chords per lyric line."""
    section_names = ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Outro"]
    sections = []
    current_lines = []
    prev_end = 0.0
    section_idx = 0

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        lyrics = seg["text"].strip()
        if not lyrics:
            continue

        # New section on gap > 2 seconds
        if start - prev_end > 2.0 and current_lines:
            name = section_names[section_idx] if section_idx < len(section_names) else f"Section {section_idx+1}"
            sections.append({"name": name, "lines": current_lines})
            current_lines = []
            section_idx += 1

        # Detect chord for this lyric's time range
        chord = detect_chord_at_time(y, sr, start, end)

        # Also check halfway point for a second chord change
        mid = (start + end) / 2
        chord2 = detect_chord_at_time(y, sr, mid, end)

        if chord2 and chord2 != chord:
            chord_line = f"{chord:<20}{chord2}"
        else:
            chord_line = chord

        current_lines.append({"chords": chord_line, "lyrics": lyrics})
        prev_end = end

    if current_lines:
        name = section_names[section_idx] if section_idx < len(section_names) else f"Section {section_idx+1}"
        sections.append({"name": name, "lines": current_lines})

    return sections


def polish_with_claude(title, key, tempo, capo, sections) -> dict:
    """Use Claude to clean up chord names and make them musically consistent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"key": key, "capo": capo, "sections": sections}

    # Build tab text for Claude to review
    tab_text = ""
    for s in sections:
        tab_text += f"[{s['name']}]\n"
        for line in s["lines"]:
            if line["chords"]:
                tab_text += f"{line['chords']}\n"
            tab_text += f"{line['lyrics']}\n"
        tab_text += "\n"

    prompt = f"""You are an expert guitarist. I have auto-detected these chords from an audio recording. 
Please review and correct them to make a musically consistent guitar tab.

Song: "{title}"
Detected key: {key}
Tempo: {tempo}

AUTO-DETECTED TAB (chords may have errors):
{tab_text}

YOUR JOB:
1. Fix any wrong chords — use your musical knowledge to make the chord progression make sense
2. Keep chords that sound right, fix ones that don't fit the key
3. Make sure verses repeat the same chord pattern and choruses repeat their pattern
4. Use common guitar chords (G, C, D, Em, Am, F, A, E, Bm, D7 etc)
5. Determine the correct capo position if needed
6. Keep 2 chords per line where they genuinely change, 1 chord if it holds the whole line
7. First line of response: KEY: [key] | CAPO: [capo or "No capo"]
8. Then return the full corrected tab with [Section Name] headers
9. Chord above lyric format — chord on its own line above the lyric it applies to, spaced where changes occur:
G                    D
Headed down south to the land of the pines
Em                   C  
And I'm thumbin' my way into North Carolina

Return ONLY the key line + corrected tab. No explanation."""

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
        data = response.json()
        if response.status_code != 200:
            print(f"Claude error: {data}")
            return {"key": key, "capo": capo, "sections": sections}

        raw = data["content"][0]["text"].strip()
        lines = raw.split("\n")

        # Parse KEY/CAPO from first line
        final_key = key
        final_capo = capo
        start_idx = 0
        if lines and lines[0].startswith("KEY:"):
            try:
                parts = lines[0].split("|")
                final_key = parts[0].replace("KEY:", "").strip()
                final_capo = parts[1].replace("CAPO:", "").strip() if len(parts) > 1 else capo
            except:
                pass
            start_idx = 1

        # Parse corrected sections
        result_sections = []
        current_section = None
        current_lines = []
        pending_chord = ""

        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if current_section is not None:
                    result_sections.append({"name": current_section, "lines": current_lines})
                current_section = stripped[1:-1]
                current_lines = []
                pending_chord = ""
            elif not stripped:
                continue
            elif current_section is None:
                continue
            else:
                tokens = stripped.split()
                chord_chars = set("ABCDEFGabcdefg#mb/0123456789dimaug7susMmaj ")
                is_chord_line = (
                    1 <= len(tokens) <= 6 and
                    len(stripped) < 30 and
                    tokens[0][0].isupper() and
                    all(c in chord_chars for c in stripped)
                )
                if is_chord_line:
                    pending_chord = stripped
                else:
                    current_lines.append({"chords": pending_chord, "lyrics": stripped})
                    pending_chord = ""

        if current_section and current_lines:
            result_sections.append({"name": current_section, "lines": current_lines})

        return {
            "key": final_key,
            "capo": final_capo,
            "sections": result_sections if result_sections else sections
        }

    except Exception as e:
        print(f"Claude polish error: {e}")
        return {"key": key, "capo": capo, "sections": sections}


def detect_strumming(tempo_bpm: str) -> dict:
    try:
        bpm = int(tempo_bpm.replace(" BPM", ""))
    except:
        bpm = 90
    if bpm < 70:
        return {"pattern": "D D D D", "description": "Slow downstrokes"}
    elif bpm < 95:
        return {"pattern": "D DU UDU", "description": "Gentle folk strum"}
    elif bpm < 120:
        return {"pattern": "D DU UD", "description": "Medium folk strum"}
    else:
        return {"pattern": "D D UD UD", "description": "Upbeat driving strum"}


@app.get("/")
def root():
    return {"status": "TabCraft API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-claude")
async def test_claude():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "No ANTHROPIC_API_KEY set"}
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 20, "messages": [{"role": "user", "content": "Say OK"}]},
            timeout=15.0
        )
        data = response.json()
        if response.status_code == 200:
            return {"status": "ok", "response": data["content"][0]["text"]}
        return {"status": "error", "code": response.status_code, "detail": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    import librosa

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

        print("Loading audio for analysis...")
        y, sr = librosa.load(tmp_path)

        print("Detecting key...")
        key = detect_key(y, sr)
        print(f"Key: {key}")

        print("Detecting tempo...")
        tempo = detect_tempo(tmp_path)
        print(f"Tempo: {tempo}")

        print("Detecting chords per lyric line...")
        sections = build_sections_with_chords(segments, y, sr)
        print(f"{len(sections)} sections built")

        print("Polishing with Claude...")
        polished = polish_with_claude(song_title, key, tempo, "No capo", sections)

        strum = detect_strumming(tempo)

        return {
            "title": song_title,
            "key": polished["key"],
            "tempo": tempo,
            "timeSignature": "4/4",
            "capo": polished["capo"],
            "strummingPattern": strum["pattern"],
            "strummingDescription": strum["description"],
            "sections": polished["sections"],
            "notes": "Chords detected from audio and refined by AI — verify by ear."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
