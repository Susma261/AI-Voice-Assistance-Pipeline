#text_to_speech.py
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

def text_to_speech(text, output_file="output.wav", pitch=1.0, speed=1.0, gender='female'):
    try:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        inputs = processor(text=text, return_tensors="pt")

        if gender == 'male':
            speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)

        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)

        audio = speech.squeeze().cpu().numpy()
        audio = audio * pitch
        audio = np.interp(np.arange(0, len(audio), speed), np.arange(0, len(audio)), audio)

        sf.write(output_file, audio, 22050)
        print(f"Audio saved to {output_file}")

    except Exception as e:
        print(f"An error occurred during TTS conversion: {e}")
