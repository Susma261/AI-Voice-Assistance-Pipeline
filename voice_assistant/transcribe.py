#transcribe.py
import whisper
import sounddevice as sd
import numpy as np
import soundfile as sf
import webrtcvad
import time

# Load Whisper model
model = whisper.load_model("tiny.en")

def record_audio(duration=5, fs=16000):
    """Records audio from the microphone."""
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return recording.flatten()

def save_audio_as_wav(audio, filename="temp.wav", fs=16000):
    """Saves audio as a WAV file."""
    sf.write(filename, audio, fs)

def apply_vad(audio, fs=16000, vad_mode=3):
    """Applies VAD to the audio signal."""
    vad = webrtcvad.Vad(vad_mode)
    audio_int16 = (audio * 32767).astype(np.int16)
    frame_duration_ms = 20  # 20 ms frame length
    frame_length = int(fs * frame_duration_ms / 1000)

    if len(audio_int16) < frame_length:
        raise ValueError("Audio is too short to process with VAD")

    voiced_frames = []
    for start in range(0, len(audio_int16), frame_length):
        end = start + frame_length
        frame = audio_int16[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant', constant_values=0)
        try:
            if vad.is_speech(frame.tobytes(), fs):
                voiced_frames.append(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")

    if voiced_frames:
        voiced_audio = np.concatenate(voiced_frames)
    else:
        voiced_audio = np.array([], dtype=np.float32)

    voiced_audio = voiced_audio.astype(np.float32) / 32767
    return voiced_audio

def transcribe_audio(file_path):
    """Transcribes audio file to text using Whisper."""
    try:
        audio, _ = sf.read(file_path, dtype='float32')
        audio = apply_vad(audio)
        result = model.transcribe(audio)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""
