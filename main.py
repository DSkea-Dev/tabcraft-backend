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


def split_long_segment(text, start, end):
    """Split a long Whisper segment into ~8-word lyric lines."""
    words = text.split()
    if len(words) <= 10:
        return [{"start": start, "end": end, "text": text.strip()}]
    chunk_size = 8
    parts = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]).strip()
        if chunk:
            parts.append(chunk)
    if not parts:
        return [{"start": start, "end": end, "text": text.strip()}]
    seg_dur = (end - start) / len(parts)
    return [
        {"start": start + i * seg_dur, "end": start + (i+1) * seg_dur, "text": p}
        for i, p in enumerate(parts)
    ]


def transcribe_audio(audio_path):
    import librosa
    import soundfile as sf
    import tempfile
    from faster_whisper import WhisperModel

    # Load full audio to get duration
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_duration = len(y) / sr
    print(f"Audio duration: {total_duration:.1f}s")

    model = WhisperModel("small", device="cpu", compute_type="int8")

    # Process in 60-second chunks to avoid memory issues
    chunk_size = 60
    all_segments = []

    if total_duration <= chunk_size:
        # Short song — process all at once
        chunks = [(0, total_duration, audio_path)]
    else:
        # Split into chunks
        chunks = []
        offset = 0
        while offset < total_duration:
            end = min(offset + chunk_size, total_duration)
            start_sample = int(offset * sr)
            end_sample = int(end * sr)
            chunk_audio = y[start_sample:end_sample]

            # Save chunk to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmp.name, chunk_audio, sr)
            tmp.close()
            chunks.append((offset, end, tmp.name))
            offset += chunk_size

    for chunk_start, chunk_end, chunk_path in chunks:
        is_temp = chunk_path != audio_path
        try:
            segs_iter, info = model.transcribe(
                chunk_path,
                beam_size=3,
                language="en",
                condition_on_previous_text=False,
                no_speech_threshold=0.8,
                vad_filter=False,
            )
            for seg in segs_iter:
                text = seg.text.strip()
                if text:
                    all_segments.append({
                        "start": float(seg.start) + chunk_start,
                        "end": float(seg.end) + chunk_start,
                        "text": text
                    })
        finally:
            if is_temp:
                try:
                    os.unlink(chunk_path)
                except Exception:
                    pass

    print(f"Raw segments: {len(all_segments)}")
    for s in all_segments:
        print(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text'][:60]}")

    # Split long segments into lyric lines
    segments = []
    for seg in all_segments:
        if len(seg["text"].split()) > 10:
            segments.extend(split_long_segment(seg["text"], seg["start"], seg["end"]))
        else:
            segments.append(seg)

    print(f"Final segments after splitting: {len(segments)}")
    return {"segments": segments, "language": "en"}


def detect_tempo(audio_path):
    import librosa, numpy as np
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return f"{int(round(float(tempo)))} BPM", y, sr


def detect_key(y, sr):
    import librosa, numpy as np
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    best_key = "G Major"
    best_score = -999.0
    for i in range(12):
        maj = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0,1])
        mn  = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0,1])
        if maj > best_score:
            best_score = maj; best_key = f"{note_names[i]} Major"
        if mn > best_score:
            best_score = mn;  best_key = f"{note_names[i]} Minor"
    return best_key


def detect_chord_at_time(y, sr, start, end):
    import librosa, numpy as np
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    def tmpl(root, intervals):
        t = np.zeros(12)
        for iv in intervals: t[(root+iv)%12] = 1.0
        return t

    templates = {}
    for r in range(12):
        templates[note_names[r]]       = tmpl(r, [0,4,7])
        templates[note_names[r]+'m']   = tmpl(r, [0,3,7])
        templates[note_names[r]+'7']   = tmpl(r, [0,4,7,10])

    s0 = int(start * sr); s1 = int(end * sr)
    segment = y[s0:s1]
    if len(segment) < sr * 0.1:
        return ""
    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    if chroma_mean.max() < 0.01:
        return ""
    cn = chroma_mean / (chroma_mean.max() + 1e-6)
    best, best_score = "", -1.0
    for name, t in templates.items():
        score = float(np.dot(cn, t) / (np.linalg.norm(t) + 1e-6))
        if score > best_score:
            best_score = score; best = name
    return best


def build_sections_with_chords(segments, y, sr):
    section_names = ["Verse 1","Chorus","Verse 2","Chorus","Bridge","Outro"]
    sections = []
    current_lines = []
    prev_end = 0.0
    section_idx = 0

    for seg in segments:
        start, end = seg["start"], seg["end"]
        lyrics = seg["text"].strip()
        if not lyrics:
            continue

        # Only split into new section on a gap > 4 seconds (more lenient)
        if start - prev_end > 4.0 and current_lines:
            name = section_names[section_idx] if section_idx < len(section_names) else f"Section {section_idx+1}"
            sections.append({"name": name, "lines": current_lines})
            current_lines = []
            section_idx += 1

        # Split each line into two halves for chord detection
        mid = start + (end - start) * 0.5
        chord1 = detect_chord_at_time(y, sr, start, mid)
        chord2 = detect_chord_at_time(y, sr, mid, end)

        if chord2 and chord2 != chord1:
            chord_line = f"{chord1:<20}{chord2}"
        else:
            chord_line = chord1

        current_lines.append({"chords": chord_line, "lyrics": lyrics})
        prev_end = end

    # Always save remaining lines even if no gap was found
    if current_lines:
        name = section_names[section_idx] if section_idx < len(section_names) else f"Section {section_idx+1}"
        sections.append({"name": name, "lines": current_lines})

    print(f"Sections detected by gap analysis: {len(sections)}")

    # If only 1 section, split evenly into verse/chorus/verse/chorus
    if len(sections) == 1:
        all_lines = sections[0]["lines"]
        n = len(all_lines)
        if n >= 8:
            q = n // 4
            sections = [
                {"name": "Verse 1",  "lines": all_lines[0:q]},
                {"name": "Chorus",   "lines": all_lines[q:q*2]},
                {"name": "Verse 2",  "lines": all_lines[q*2:q*3]},
                {"name": "Chorus",   "lines": all_lines[q*3:]},
            ]

    return sections


def polish_with_claude(title, key, tempo, sections):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"key": key, "capo": "No capo", "sections": sections}

    tab_text = ""
    for s in sections:
        tab_text += f"[{s['name']}]\n"
        for line in s["lines"]:
            if line["chords"]:
                tab_text += f"{line['chords']}\n"
            tab_text += f"{line['lyrics']}\n"
        tab_text += "\n"

    prompt = (
        f'You are an expert guitarist. I auto-detected these chords from audio. '
        f'Review and correct them into a proper guitar tab.\n\n'
        f'Song: "{title}"\nDetected key: {key}\nTempo: {tempo}\n\n'
        f'AUTO-DETECTED TAB:\n{tab_text}\n'
        f'RULES:\n'
        f'1. Fix wrong chords so the progression makes musical sense\n'
        f'2. Each lyric line should have 1-2 chords above it\n'
        f'3. Use common open chords: G C D Em Am F A E Bm D7 etc\n'
        f'4. 4-6 total chords for the whole song\n'
        f'5. Verses repeat same pattern, choruses repeat their own pattern\n'
        f'6. Determine correct key and capo\n'
        f'7. First line MUST be: KEY: [key] | CAPO: [capo fret or "No capo"]\n'
        f'8. Then full tab with [Section Name] headers\n'
        f'9. Chord format - chord names on line above lyric, spaced where they change:\n'
        f'G                    D\n'
        f'Headed down south to the land of the pines\n'
        f'Em                   C\n'
        f'Thumbin my way into North Carolina\n\n'
        f'Return ONLY the key line + tab. No explanation.'
    )

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
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        data = response.json()
        if response.status_code != 200:
            print(f"Claude error: {data}")
            return {"key": key, "capo": "No capo", "sections": sections}

        raw = data["content"][0]["text"].strip()
        lines = raw.split("\n")

        final_key = key
        final_capo = "No capo"
        start_idx = 0
        if lines and lines[0].startswith("KEY:"):
            try:
                parts = lines[0].split("|")
                final_key = parts[0].replace("KEY:", "").strip()
                final_capo = parts[1].replace("CAPO:", "").strip() if len(parts) > 1 else "No capo"
            except Exception:
                pass
            start_idx = 1

        result_sections = []
        current_section = None
        current_lines = []
        pending_chord = ""
        chord_chars = set("ABCDEFGabcdefg#mb/0123456789dimaug7susMmaj ")

        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if current_section is not None:
                    result_sections.append({"name": current_section, "lines": current_lines})
                current_section = stripped[1:-1]
                current_lines = []
                pending_chord = ""
            elif not stripped or current_section is None:
                continue
            else:
                tokens = stripped.split()
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
        return {"key": key, "capo": "No capo", "sections": sections}


def detect_strumming(tempo_bpm):
    try:
        bpm = int(tempo_bpm.replace(" BPM", ""))
    except Exception:
        bpm = 90
    if bpm < 70:
        return {"pattern": "D D D D", "description": "Slow downstrokes"}
    elif bpm < 95:
        return {"pattern": "D DU UDU", "description": "Gentle folk strum"}
    elif bpm < 120:
        return {"pattern": "D D UD UDU", "description": "Medium folk strum"}
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
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        song_title = (
            os.path.splitext(file.filename)[0]
            .replace("-", " ").replace("_", " ")
            .replace("(MP3)", "").replace("(WAV)", "").strip()
        )

        print("Transcribing...")
        transcription = transcribe_audio(tmp_path)
        segments = transcription.get("segments", [])
        print(f"{len(segments)} segments after splitting")

        print("Loading audio...")
        import librosa
        y, sr = librosa.load(tmp_path)

        print("Detecting key...")
        key = detect_key(y, sr)
        print(f"Key: {key}")

        print("Detecting tempo...")
        import numpy as np
        # Use onset strength for more accurate tempo on sung audio
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_val = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        bpm = int(round(float(tempo_val[0] if hasattr(tempo_val, "__len__") else tempo_val)))
        # If detected BPM seems wrong, try half/double
        if bpm < 60:
            bpm = bpm * 2
        elif bpm > 200:
            bpm = bpm // 2
        tempo_str = f"{bpm} BPM"
        print(f"Tempo: {tempo_str}")

        print("Detecting chords per line...")
        sections = build_sections_with_chords(segments, y, sr)
        print(f"{len(sections)} sections")

        print("Polishing with Claude...")
        polished = polish_with_claude(song_title, key, tempo_str, sections)

        strum = detect_strumming(tempo_str)

        return {
            "title": song_title,
            "key": polished["key"],
            "tempo": tempo_str,
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
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
