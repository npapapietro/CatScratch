import wave
import pyaudio

CHUNK = 512
FORMAT = pyaudio.paInt16
RATE = 48000
RECORD_SECONDS = 5
FILE_NAME = "audio_out.wav"

p = pyaudio.PyAudio()

stream  = p.open(
    format = FORMAT,
    channels = 1,
    rate = RATE,
    input=True,
    frames_per_buffer=CHUNK)

frames = []
for i in range(0, int(RATE/CHUNK * 5)):
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

wf=wave.open(FILE_NAME,'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()
