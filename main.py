import dotenv
import os

dotenv.load_dotenv()

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("AUTH_TOKEN"))

diarization = pipeline("audio.mp3")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    print(f"speaker_{speaker}:", turn)

import whisper_timestamped as whisper

audio = whisper.load_audio('audio.mp3')
model = whisper.load_model("base")

result = whisper.transcribe(model, audio, language="en")

# Parsing results
speaker_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

transcriptions = []
for segment in result['segments']:
    transcriptions.append({
        "start": segment['start'],
        "end": segment['end'],
        "text": segment['text'].strip()
    })

# Matching
def get_speaker_for_segment(segment, speaker_segments):
    matched_speakers = []
    for sp_segment in speaker_segments:
        if (sp_segment["start"] <= segment["start"] < sp_segment["end"]) or (sp_segment["start"] < segment["end"] <= sp_segment["end"]):
            matched_speakers.append(sp_segment["speaker"])
    # Return the most frequently occurring speaker in the matched speakers
    if matched_speakers:
        return max(set(matched_speakers), key=matched_speakers.count)
    return None

with open("output.txt" , "w") as f:
    for segment in transcriptions:
        speaker = get_speaker_for_segment(segment, speaker_segments)
        start_time = f"00:00:{int(segment['start']):02}:{int((segment['start']%1)*1000):03}"
        end_time = f"00:00:{int(segment['end']):02}:{int((segment['end']%1)*1000):03}"
        if speaker is not None:
            f.write(f"Speaker {speaker}: [{start_time} --> {end_time}] {segment['text']} \n")
        else:
            f.write(f"Speaker unknown: [{start_time} --> {end_time}] {segment['text']} \n")