from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os

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


def detect_chords_librosa(audio_path: str) -> list:
    """
    Detect chords using librosa chroma features.
    Returns list of (time_seconds, chord_name) tuples.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, mono=True)
    
    # Use hop_length for ~0.5 second resolution
    hop_length = sr // 2
    
    # Chroma features — energy per pitch class over time
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Chord templates: major and minor triads for all 12 roots
    def make_templates():
        templates = {}
        for root in range(12):
            # Major: root, major third (+4), fifth (+7)
            major = [0] * 12
            major[root % 12] = 1
            major[(root + 4) % 12] = 1
            major[(root + 7) % 12] = 1
            templates[note_names[root]] = np.array(major, dtype=float)
            
            # Minor: root, minor third (+3), fifth (+7)
            minor = [0] * 12
            minor[root % 12] = 1
            minor[(root + 3) % 12] = 1
            minor[(root + 7) % 12] = 1
            templates[note_names[root] + 'm'] = np.array(minor, dtype=float)
        return templates

    templates = make_templates()
    
    # Match each chroma frame to best chord
    n_frames = chroma.shape[1]
    chord_sequence = []
    prev_chord = None

    for i in range(n_frames):
        frame = chroma[:, i]
        if frame.max() < 0.1:  # silence
            continue
        
        frame_norm = frame / (frame.max() + 1e-6)
        
        best_chord = None
        best_score = -1
        for chord_name, template in templates.items():
            score = float(np.dot(frame_norm, template))
            if score > best_score:
                best_score = score
                best_chord = chord_name
        
        time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        
        # Only add if chord changed
        if best_chord != prev_chord:
            chord_sequence.append((float(time), best_chord))
            prev_chord = best_chord

    return chord_sequence


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using faster-whisper."""
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
    """Detect key and tempo using librosa."""
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Key detection
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


def align_chords_to_lyrics(segments, chord_sequence):
    """Place the most relevant chord above each lyric line."""
    sections = []
    current_lines = []
    prev_end = 0

    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        lyrics = segment["text"].strip()
        if not lyrics:
            continue

        # Find all chords active during this segment
        seg_chords = []
        for (chord_time, chord_name) in chord_sequence:
            if seg_start <= chord_time < seg_end:
                seg_chords.append(chord_name)
        
        # Also include last chord before segment start (still ringing)
        if not seg_chords:
            before = [(t, c) for (t, c) in chord_sequence if t <= seg_start]
            if before:
                seg_chords = [before[-1][1]]

        # Deduplicate consecutive
        unique = []
        for c in seg_chords:
            if not unique or unique[-1] != c:
                unique.append(c)

        # Build chord line spaced over lyric
        if unique:
            lyric_len = max(len(lyrics) + 4, 40)
            if len(unique) == 1:
                chord_line = unique[0]
            else:
                spacing = max(lyric_len // len(unique), 6)
                chord_line = "".join(c.ljust(spacing) for c in unique).rstrip()
        else:
            chord_line = ""

        # New section if gap > 1.5s
        if seg_start - prev_end > 1.5 and current_lines:
            sections.append(current_lines)
            current_lines = []

        current_lines.append({"chords": chord_line, "lyrics": lyrics})
        prev_end = seg_end

    if current_lines:
        sections.append(current_lines)

    # Name sections
    section_names = ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Outro"]
    return [
        {
            "name": section_names[i] if i < len(section_names) else f"Section {i+1}",
            "lines": lines
        }
        for i, lines in enumerate(sections)
    ]


def detect_strumming(tempo_bpm: str) -> dict:
    """Suggest a strumming pattern based on tempo."""
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
            .replace("(MP3)", "").replace("(WAV)", "")\
            .strip()

        print(f"Transcribing: {file.filename}")
        transcription = transcribe_audio(tmp_path)
        segments = transcription.get("segments", [])
        print(f"Got {len(segments)} segments")

        print("Detecting chords...")
        chord_sequence = detect_chords_librosa(tmp_path)
        print(f"Got {len(chord_sequence)} chord changes")

        print("Detecting key and tempo...")
        key_tempo = detect_key_and_tempo(tmp_path)
        print(f"Key: {key_tempo['key']}, Tempo: {key_tempo['tempo']}")

        print("Aligning chords to lyrics...")
        sections = align_chords_to_lyrics(segments, chord_sequence)

        strum = detect_strumming(key_tempo["tempo"])

        return {
            "title": song_title,
            "key": key_tempo["key"],
            "tempo": key_tempo["tempo"],
            "timeSignature": "4/4",
            "capo": "No capo",
            "strummingPattern": strum["pattern"],
            "strummingDescription": strum["description"],
            "sections": sections if sections else [{
                "name": "Verse 1",
                "lines": [{"chords": "", "lyrics": "(No lyrics detected — check audio quality)"}]
            }],
            "notes": f"Chords detected automatically from audio — verify by ear and adjust as needed."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
