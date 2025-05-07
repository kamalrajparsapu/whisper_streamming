import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel , BatchedInferencePipeline
import queue
import threading
import time
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps  # Correct Import


# Load Faster-Whisper Model
model = WhisperModel("large", device="cpu", compute_type="int8")
batched_model = BatchedInferencePipeline(model=model)
# Audio config
SAMPLERATE = 16000
CHANNELS = 1
FRAME_DURATION = 0.02  # seconds
FRAMES_PER_BUFFER = int(SAMPLERATE * FRAME_DURATION)

audio_queue = queue.Queue()
buffer_audio = []

# Initialize Silero VAD
vad_model = load_silero_vad()  # Initialize the VAD model

# Callback to collect mic data
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Worker to process and transcribe live
def transcribe_worker():
    global buffer_audio
    full_transcript = ""
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=1)
            buffer_audio.append(audio_chunk)

            # Combine small chunks into bigger ones (~1.5s)
            total_audio = np.concatenate(buffer_audio, axis=0)
            if total_audio.shape[0] >= SAMPLERATE * 1.5:  # 0.2 seconds of audio
                buffer_audio = []  # reset buffer after using

                total_audio = total_audio.flatten().astype(np.float32)

                # 👇 Apply VAD here
                speech_timestamps = get_speech_timestamps(total_audio, vad_model, return_seconds=True)
                if len(speech_timestamps) > 0:  # If speech detected
                    # segments, _ = model.transcribe(total_audio, language="en", beam_size=1, vad_filter=True,temperature=0.5)
                    segments, _ = batched_model.transcribe(total_audio, language="en", beam_size=5,batch_size=16, vad_filter=True,temperature=0.5)
                    for segment in segments:
                        if segment.text.strip() != "":
                            full_transcript += " " + segment.text.strip()

                    # Clear screen and reprint the full transcription
                    print("\033c", end="")  # ANSI code to clear terminal
                    print("🎙️ Live Transcription:\n")
                    print(full_transcript)
                else:
                    # No real speech detected, skip transcription
                    pass

        except queue.Empty:
            continue

def main():
    # Start transcribing in a separate thread
    threading.Thread(target=transcribe_worker, daemon=True).start()

    # Start microphone stream
    with sd.InputStream(
        samplerate=SAMPLERATE,
        blocksize=FRAMES_PER_BUFFER,
        dtype='float32',
        channels=CHANNELS,
        callback=audio_callback
    ):
        print("🎙️ Speak into your microphone. Live transcription starts...\n")
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
