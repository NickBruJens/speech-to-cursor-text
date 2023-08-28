import pyaudio
import numpy as np
from pydub import AudioSegment
import tempfile
import os
import threading
import time
import whisper
from pynput.keyboard import Controller
keyboard1 = Controller()
# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

fs = 44100
print(f"Using sample rate of {fs} Hz")

model = whisper.load_model("base")

def process_audio(file_path):
    print(f"Processing {file_path}")
    result = model.transcribe(file_path, fp16=False)
    audio_string = result["text"]
    keyboard1.type(audio_string)
    #print(audio_string)
    os.remove(file_path)
    print('File removed')

def record_audio(threshold=1000, pre_buffer_length=fs * 3, post_buffer_length=fs * 3, fs=fs):
    buffer = np.array([], dtype=np.int16)
    post_buffer = np.array([], dtype=np.int16)
    last_recorded_time = None

    while True:
        current_time = time.time()
        indata = np.frombuffer(stream.read(1024), dtype=np.int16)
        buffer = np.append(buffer, indata)

        if len(buffer) > pre_buffer_length:
            buffer = buffer[-pre_buffer_length:]

        level = np.max(buffer)


        if level > threshold:
            post_buffer = np.append(post_buffer, buffer)
            last_recorded_time = current_time

        # If audio continues to be above threshold, extend recording
        if last_recorded_time and (current_time - last_recorded_time <= 1):
            post_buffer = np.append(post_buffer, buffer)

        # Save and process audio if 3 seconds have passed since last above-threshold sound
        if last_recorded_time and (current_time - last_recorded_time > 3):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                audio_segment = AudioSegment(data=post_buffer.tobytes(), sample_width=2, channels=1, frame_rate=fs)
                audio_segment.export(temp_file.name, format="mp3")

                process_thread = threading.Thread(target=process_audio, args=(temp_file.name,))
                process_thread.start()

            post_buffer = np.array([], dtype=np.int16)
            last_recorded_time = None

        buffer = np.array([], dtype=np.int16)


if __name__ == "__main__":
    record_audio()

