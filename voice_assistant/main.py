#main.py
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transcribe import record_audio, save_audio_as_wav, transcribe_audio
from text_to_response import generate_response
from text_to_speech import text_to_speech
import time

def main():
    try:
        # Step 1: Record Audio
        audio = record_audio()
        save_audio_as_wav(audio)

        # Step 2: Transcribe Audio
        text_output = transcribe_audio("temp.wav")
        print("Transcribed Text:", text_output)

        with open("transcribed_text.txt", "w") as f:
            f.write(text_output)

        # Step 3: Generate Response
        response = generate_response(text_output)
        print("Response:", response)

        with open("response_text.txt", "w") as f:
            f.write(response)

        # Step 4: Convert Text to Speech
        text_to_speech(response, "response.wav")

    except Exception as e:
        print(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken (latency): {end_time - start_time} seconds")
