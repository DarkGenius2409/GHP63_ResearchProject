# Imports
import os
import glob
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Audio / MIDI Processing
from midi2audio import FluidSynth
from music21 import chord, converter, corpus, duration, instrument, note, stream

# TensorFlow / Keras 
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    LSTM, Activation, Dense, Dropout, Embedding, Flatten, Input, 
    Lambda, Multiply, Permute, RepeatVector, Reshape, Softmax, 
    TimeDistributed, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model, to_categorical  # Modern Keras replacements
import tensorflow.keras.backend as K

import subprocess
import os

def midi_to_wav(midi_path, wav_path):
    soundfont = os.path.expanduser('~/.fluidsynth/default_sound_font.sf2')
    subprocess.run([
        'fluidsynth', '-ni',
        '-F', wav_path,
        '-r', '44100',
        soundfont,
        midi_path,
    ], check=True)

# Load and Process Data
dataset_name = "input/debussy"
file_name = "debussy_cc_1.mid"
file = os.path.join(dataset_name, file_name)

original_score = converter.parse(file).chordify()

# Extract Data
notes = []
durations = []

for element in original_score.flatten():
    if isinstance(element, chord.Chord):
        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
        durations.append(element.duration.quarterLength)
    if isinstance(element, note.Note):
        if element.isRest:
            notes.append(str(element.name))
            durations.append(element.duration.quarterLength)
        else:
            notes.append(str(element.nameWithOctave))
            durations.append(element.duration.quarterLength) 


