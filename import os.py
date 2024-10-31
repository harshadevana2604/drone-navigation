import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import wave
import serial
import pyaudio
import librosa
import numpy as np
import serial.tools.list_ports
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

"""ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
serialInst.baudrate = 9600
serialInst.port = 'COM6'
serialInst.open()"""

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

def record_audio(filename, duration=1.6, fs=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs,input=True,frames_per_buffer=1024)
    print("Speak...")
    frames = []
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Done.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def preprocess_audio(filename, sr=16000):
    y, sr = librosa.load(filename, sr=sr, duration=1.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    return mfccs

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

while True:
    record_audio('C:/Users/Sudarshana/Music/EL/test.wav', duration=2)

    mfcc_features = preprocess_audio('C:/Users/Sudarshana/Music/EL/test.wav')

    model1 = create_model(input_shape=(40, 47, 1))
    model1.built = True
    model1.load_weights('C:/Users/Sudarshana/Desktop/TFfile/MODEL/uprecog.weights.h5')
    up = model1.predict(mfcc_features)
    print("UP :", up)

    model2 = create_model(input_shape=(40, 47, 1))
    model2.built = True
    model2.load_weights('C:/Users/Sudarshana/Desktop/TFfile/MODEL/downrecog.weights.h5')
    down = model2.predict(mfcc_features)
    print("DOWN :", down)

    if up > 0.95:
        decision = "UP"
        print("Model predicts :", decision)
    elif down > 0.9:
        decision = "DOWN"
        print("Model predicts :", decision)
    else:
        decision = "NONE"
        print("Model predicts :", decision)

    #serialInst.write(decision.encode('utf-8'))

    time.sleep(3);
