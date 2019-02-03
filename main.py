from time import sleep
import RPi.GPIO as GPIO #pylint: disable=import-error
import pyaudio
import wave
import collections
from keras.models import load_model
import numpy as np
import scipy.signal as sig
import librosa_mod as lm

# Pi gpio setup and run
PIN = 11
_ON = GPIO.HIGH
_OFF = GPIO.LOW
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(PIN,GPIO.OUT)

def strobe_light(t = 50):
    for _ in range(t):
        GPIO.output(PIN,_ON)
        sleep(0.1)
        GPIO.output(PIN,_OFF)
        sleep(0.1)

# Load keras model
model = load_model('model_.h5')

# setup audio stream
CHUNK = 1024
RATE = 44100
RECORD_SECONDS = 5
SAMPLING_RATE = 22050
MAX_BUFFER = 4*SAMPLING_RATE


p = pyaudio.PyAudio()
Ringbuffer = collections.deque(maxlen=MAX_BUFFER)

def callback(in_data, frame_count, time_info, status):
    """ Callback function used during PyAudio capture loop
    
    Arguments:
        in_data {buffer like} -- main audio data
        frame_count {[type]} -- unused for this purpose
        time_info {[type]} -- unused for this purpose
        status {[type]} -- unused for this purpose
    
    Returns:
        (buffer, status code) - returns the processed audio and status code
    """

    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_num = int(audio_data.shape[-1] * RATE/float(SAMPLING_RATE))
    audio_data = sig.resample(audio_data, audio_num, axis=-1)

    Ringbuffer.extend(audio_data)

    if len(Ringbuffer) >= MAX_BUFFER:        
        data_live = np.array(np.mean(lm.mfcc(y=np.array(list(Ringbuffer)), sr=SAMPLING_RATE).T,axis=0)) 
        guess = model.predict(np.reshape(data_live,(1,40)), verbose=1).argmax(1)[0]
        if guess == 0:
            strobe_light()

    return (audio_data, pyaudio.paContinue)

try:
    stream = p.open(format = pyaudio.paFloat32,
                    channels=1,
                    rate = RATE,
                    input=True,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        sleep(0.1)
finally:
    stream.stop_stream()
    stream.close()

    p.terminate()
        

