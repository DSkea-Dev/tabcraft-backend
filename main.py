from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, json, subprocess, sys

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

def get_chords_from_audio(audio_path: str) -> list:
    """Use basic-pitch to detect notes/chords from audio."""
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import numpy as np

        model_output, midi_data, note_events = predict(audio_path)

        # Group notes into rough chord regions (every 2 seconds)
        chords_by_time = {}
        for note in note_events:
            start_time = note[0]
            bucket = int(start_time / 2) * 2  # 2-second buckets
            if bucket not in chords_by_time:
                chords_by_time[bucket] = []
            chords_by_time[bucket].append(note[2])  # MIDI pitch

        # Convert MIDI pitches to chord names (simplified)
        chord_sequence = []
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for time_bucket in sorted(chords_by_time.keys()):
            pitches = chords_by_time[time_bucket]
            if pitches:
                # Get the root note (lowest pitch)
                root_midi = min(pitches) % 12
                root_name = note_names[root_midi]
                
                # Detect major/minor from intervals
                intervals = set([(p - min(pitches)) % 12 for p in pitches])
                if 3 in intervals:
                    chord = root_name + "m"
                else:
                    chord = root_name
                chord_sequence.append((time_bucket, chord))

        return chord_sequence
    except Exception as e:
        print(f"Chord detection error: {e}")
        return []


def transcribe_audio(audio_path: str) -> dict:
    """Use faster-whisper to transcribe audio with segment timestamps."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(audio_path, beam_size=5)

        # Convert to same format as openai-whisper output
        segments = []
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

        return {"segments": segments, "language": info.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def align_chords_to_lyrics(segments, chord_sequence):
    """Align detected chords to lyric lines."""
    sections = []
    current_section_lines = []
    prev_end = 0

    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        lyrics = segment["text"].strip()

        if not lyrics:
            continue

        # Find chords that fall within this segment's time range
        seg_chords = []
        for (chord_time, chord_name) in chord_sequence:
            if seg_start <= chord_time < seg_end:
                seg_chords.append(chord_name)

        # Remove duplicate consecutive chords
        unique_chords = []
        for c in seg_chords:
            if not unique_chords or unique_chords[-1] != c:
                unique_chords.append(c)

        # Format chords spaced above lyrics
        if unique_chords:
            # Space chords evenly across the lyric line length
            lyric_len = max(len(lyrics), 40)
            if len(unique_chords) == 1:
                chord_line = unique_chords[0].ljust(lyric_len)
            else:
                spacing = lyric_len // len(unique_chords)
                chord_line = "".join(c.ljust(spacing) for c in unique_chords)
        else:
            chord_line = ""

        # Detect verse/chorus breaks (gap > 1.5 seconds)
        if seg_start - prev_end > 1.5 and current_section_lines:
            sections.append(current_section_lines)
            current_section_lines = []

        current_section_lines.append({
            "chords": chord_line,
            "lyrics": lyrics,
            "start": seg_start,
        })
        prev_end = seg_end

    if current_section_lines:
        sections.append(current_section_lines)

    # Name sections
    named_sections = []
    section_names = ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Outro"]
    for i, lines in enumerate(sections):
        name = section_names[i] if i < len(section_names) else f"Section {i+1}"
        named_sections.append({
            "name": name,
            "lines": [{"chords": l["chords"], "lyrics": l["lyrics"]} for l in lines]
        })

    return named_sections


def detect_key_and_tempo(audio_path: str) -> dict:
    """Detect key and tempo using librosa."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Key detection using chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_idx = chroma_mean.argmax()
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Simple major/minor detection
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, key_idx))[0, 1]
        minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, key_idx))[0, 1]
        
        mode = "Major" if major_corr > minor_corr else "Minor"
        key = f"{note_names[key_idx]} {mode}"
        
        return {
            "key": key,
            "tempo": f"{int(round(float(tempo)))} BPM"
        }
    except Exception as e:
        print(f"Key/tempo detection error: {e}")
        return {"key": "Unknown", "tempo": "Unknown"}


@app.get("/")
def root():
    return {"status": "TabCraft API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save upload to temp file
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        song_title = os.path.splitext(file.filename)[0].replace("-", " ").replace("_", " ").strip()

        # Run all analysis
        print("Transcribing audio...")
        transcription = transcribe_audio(tmp_path)

        print("Detecting chords...")
        chord_sequence = get_chords_from_audio(tmp_path)

        print("Detecting key and tempo...")
        key_tempo = detect_key_and_tempo(tmp_path)

        print("Aligning chords to lyrics...")
        sections = align_chords_to_lyrics(
            transcription.get("segments", []),
            chord_sequence
        )

        # Detect time signature (simplified — most songs are 4/4)
        time_sig = "4/4"

        result = {
            "title": song_title,
            "key": key_tempo["key"],
            "tempo": key_tempo["tempo"],
            "timeSignature": time_sig,
            "capo": "No capo",
            "strummingPattern": "D DU UDU",
            "strummingDescription": "Adjust to match your playing style",
            "sections": sections if sections else [{
                "name": "Verse 1",
                "lines": [{"chords": "", "lyrics": "(No lyrics detected — check audio quality)"}]
            }],
            "notes": f"Transcribed from {file.filename}. Chords detected automatically — verify by ear."
        }

        return result

    finally:
        os.unlink(tmp_path)
