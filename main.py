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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

from collections import Counter

# Clear dependencies
# files_to_delete = ['store/notes', 'store/durations', 'store/distincts', 'store/lookups']
# for f in files_to_delete:
#     if os.path.exists(f):
#         os.remove(f)
#         print(f'Deleted: {f}')
#     else:
#         print(f'Not found: {f}')

# Helper Functions
def midi_to_wav(midi_path, wav_path):
    soundfont = os.path.expanduser('~/.fluidsynth/default_sound_font.sf2')
    subprocess.run([
        'fluidsynth', '-ni',
        '-F', wav_path,
        '-r', '44100',
        soundfont,
        midi_path,
    ], check=True) 

def get_music_list(data_folder):
    return glob.glob(os.path.join(data_folder, "*.mid")), converter

def create_network(n_notes, n_durations, embed_size=100, rnn_units=256, use_attention=False):
    # Input Layers
    note_input = Input(shape=(None,), name='note_input')
    duration_input = Input(shape=(None,), name='duration_input')

    # Embedding Layers
    note_embedding = Embedding(input_dim=n_notes, output_dim=embed_size)(note_input)
    duration_embedding = Embedding(input_dim=n_durations, output_dim=embed_size)(duration_input)

    x = Concatenate()([note_embedding, duration_embedding])
    x = LSTM(rnn_units, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    if use_attention:
        x = LSTM(rnn_units, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Softmax()(e)
        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))
        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)
    else:
        c = LSTM(rnn_units)(x)
        x = Dropout(0.5)(x)

    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)

    model = Model([note_input, duration_input], [notes_out, durations_out])

    if use_attention:
        att_model = Model([note_input, duration_input], alpha)
    else:
        att_model = None

    opti = RMSprop(learning_rate=0.002)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, att_model

def get_distinct(elements):
    element_names = sorted(set(elements))
    n_element_names = len(element_names)
    return (element_names, n_element_names)

def create_mappings(element_names):
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))
    return (element_to_int, int_to_element)

def prepare_sequences(notes, durations, lookups, distincts, sequence_length=32):
    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_note_names, duration_names, n_duration_names = distincts

    notes_network_input = []
    durations_network_input = []
    notes_network_output = []
    durations_network_output = []

    for i in range(len(notes) - sequence_length):
        note_seq_in = notes[i:i + sequence_length]
        note_seq_out = notes[i + sequence_length]
        duration_seq_in = durations[i:i + sequence_length]
        duration_seq_out = durations[i + sequence_length]

        notes_network_input.append([note_to_int[note] for note in note_seq_in])
        notes_network_output.append(note_to_int[note_seq_out])
        durations_network_input.append([duration_to_int[duration] for duration in duration_seq_in])
        durations_network_output.append(duration_to_int[duration_seq_out])

    n_patterns = len(notes_network_input)

    # Reshaping input lists into a format compatible with LSTM layers
    notes_network_input = np.reshape(notes_network_input, (n_patterns, sequence_length))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, sequence_length))
    network_input = [notes_network_input, durations_network_input]

    # Reshaping output lists into a format compatible with LSTM layers
    notes_network_output = to_categorical(notes_network_output, num_classes=n_note_names)
    durations_network_output = to_categorical(durations_network_output, num_classes=n_duration_names)
    network_output = [notes_network_output, durations_network_output]

    return (network_input, network_output)

def sample_with_temp(preds, temp=1.0):
    if temp == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)
    
def quantize_duration(d):
    standard = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    return min(standard, key=lambda x: abs(x - d))
    

# Load and Process Data
run_folder = '.'

store_folder = os.path.join(run_folder, 'store')
data_folder = "input/chopin"

if not os.path.exists('store'):
    os.mkdir(os.path.join(run_folder, 'store'))
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))
    os.mkdir(os.path.join(run_folder, 'viz'))

mode = 'load'

# data params
intervals = range(-3, 3)
seq_len = 32

# model params
embed_size = 100
rnn_units = 256
use_attention = True

# Parse MIDI files and extract notes and durations
if mode == "build":
    music_list, parser = get_music_list(data_folder)
    music_list = []
    for folder in ['input/chopin', 'input/schubert', 'input/schumann', 
               'input/brahms', 'input/liszt', 'input/mozart', 
               'input/haydn', 'input/beeth']:
        music_list.extend(glob.glob(os.path.join(folder, "*.mid")))
    notes, durations = [], []

    for i, music_file in enumerate(music_list):
        print(i+1, "Parsing %s" % music_file)
        original_score = parser.parse(music_file)

        for interval in intervals:
            score = original_score.transpose(interval)

            notes.extend(['Start'] * seq_len)
            durations.extend([0] * seq_len)

            for element in score.flatten(): 
                if isinstance(element, note.Note):
                    if element.isRest:
                        notes.append(str(element.name))
                        durations.append(quantize_duration(float(element.duration.quarterLength)))
                    else:
                        notes.append(str(element.nameWithOctave))
                        durations.append(quantize_duration(float(element.duration.quarterLength)))

                # if isinstance(element, chord.Chord):  # note: 'if' not 'elif'
                #     notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                #     durations.append(quantize_duration(float(element.duration.quarterLength)))

    note_counts = Counter(notes)
    print('Most common notes:')
    for note_name, count in note_counts.most_common(20):
        print(f'  {count:4d}x  {note_name}')

    print('\nLeast common notes:')
    for note_name, count in note_counts.most_common()[-20:]:
        print(f'  {count:4d}x  {note_name}')

    print('\nSample of all unique notes:')
    print(sorted(set(notes))[:50])

    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pickle.dump(notes, f) 
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pickle.dump(durations, f) 
# Load preprocessed notes and durations if they already exist
else:
    with open(os.path.join(store_folder, 'notes'), 'rb') as f:
        notes = pickle.load(f)
    with open(os.path.join(store_folder, 'durations'), 'rb') as f:
        durations = pickle.load(f) 

# Creating embeddings for notes and durations

## Get distinct notes and durations
note_names, n_note_names = get_distinct(notes)
durations_names, n_duration_names = get_distinct(durations)
distincts = [note_names, n_note_names, durations_names, n_duration_names]

print(f'Unique notes: {n_note_names}')
print(f'Unique durations: {n_duration_names}')

with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
    pickle.dump(distincts, f)

## Create mappings for notes and durations
note_to_int, int_to_note = create_mappings(note_names)
duration_to_int, int_to_duration = create_mappings(durations_names)
lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]

with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
    pickle.dump(lookups, f)

network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, sequence_length=seq_len)

model, att_model = create_network(n_note_names, n_duration_names, embed_size=embed_size, rnn_units=rnn_units, use_attention=use_attention)

# Training the model
weights_folder = os.path.join(run_folder, 'weights')

## Save the weights every time it improves
checkpoint1 = ModelCheckpoint(
    os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5"),
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

## Save only the best weights
checkpoint2 = ModelCheckpoint(
    os.path.join(weights_folder, "best.weights.h5"),
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

## Stop training if the loss doesn't improve for 10 epochs and restore the best weights
early_stopping = EarlyStopping(
    monitor='val_loss'
    , restore_best_weights=True
    , patience = 30
)

## Reduce learning rate if the loss doesn't improve for 5 epochs
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      # halve the learning rate
    patience=5,      # after 5 epochs of no improvement
    min_lr=0.00001
)


callbacks_list = [
    checkpoint1
    , checkpoint2
    , early_stopping
 ]

model.save_weights(os.path.join(weights_folder, "initial.weights.h5"))
model.fit(network_input, network_output
          , epochs=2000000, batch_size=256
          , validation_split = 0.2
          , callbacks=callbacks_list
          , shuffle=True
         )

# Prediction

## Prediction params
notes_temp=0.7
duration_temp = 0.5
max_extra_notes = 100
max_seq_len = 32
seq_len = 32

notes = ['Start']
durations = [0]

if seq_len is not None:
    notes = ['Start'] * (seq_len - len(notes)) + notes
    durations = [0] * (seq_len - len(durations)) + durations

sequence_length = len(notes)

prediction_output = []
notes_input_seq = []
durations_input_seq = []
overall_preds = []

for n, d in zip(notes, durations):
    notes_input_seq.append(note_to_int[n])
    durations_input_seq.append(duration_to_int[d])

    prediction_output.append([n, d])

    if n != 'Start':
        midi_note = note.Note(n)
        new_note = np.zeros(128)
        new_note[midi_note.pitch.midi] = 1
        overall_preds.append(new_note)

att_matrix = np.zeros(shape = (max_extra_notes+sequence_length, max_extra_notes))

for note_idx in range(max_extra_notes):
    prediction_input = [
        np.array([notes_input_seq]),
        np.array([durations_input_seq])
    ]

    notes_pred, durations_pred = model.predict(prediction_input, verbose=0)

    if use_attention:
        att_pred = att_model.predict(prediction_input, verbose=0)[0]
        att_matrix[(note_idx - len(att_pred) + sequence_length):(note_idx + sequence_length), note_idx] = att_pred

    new_note = np.zeros(128)

    for idx, n_i in enumerate(notes_pred[0]):
        try:
            note_name = int_to_note[idx]
            midi_note = note.Note(note_name)
            new_note[midi_note.pitch.midi] = n_i
        except:
            pass

    overall_preds.append(new_note)

    i1 = sample_with_temp(notes_pred[0], notes_temp)
    i2 = sample_with_temp(durations_pred[0], duration_temp)

    note_result = int_to_note[i1]
    duration_result = int_to_duration[i2]

    prediction_output.append([note_result, duration_result])

    notes_input_seq.append(i1)
    durations_input_seq.append(i2)

    if len(notes_input_seq) > max_seq_len:
        notes_input_seq = notes_input_seq[1:]
        durations_input_seq = durations_input_seq[1:]

    # if note_result == "Start":
    #     break

overall_preds = np.transpose(np.array(overall_preds))
print(f"Generated sequence of {len(prediction_output)} notes")

# Visualizations:

## Heatmap
fig, ax = plt.subplots(figsize=(15,15))
ax.set_yticks([int(j) for j in range(35,70)])
plt.imshow(overall_preds[35:70,:], origin="lower", cmap='coolwarm', vmin = -0.5, vmax = 0.5, extent=[0, max_extra_notes, 35,70])
plt.xlabel("Note number",fontsize=20)
plt.ylabel("Pitch value (MIDI number)",fontsize=20)
plt.title("Probability distribution of the next possible note over time",fontsize=20)

## Attention
if use_attention:
    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(att_matrix[(seq_len-2):,], cmap='coolwarm', interpolation='nearest')    

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(prediction_output)- seq_len, 1), minor=True);
    ax.set_yticks(np.arange(-.5, len(prediction_output)- seq_len, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)    
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(prediction_output) - seq_len))
    ax.set_yticks(np.arange(len(prediction_output)- seq_len+2))
    # ... and label them with the respective list entries
    ax.set_xticklabels([n[0] for n in prediction_output[(seq_len):]])
    ax.set_yticklabels([n[0] for n in prediction_output[(seq_len - 2):]])
    ax.xaxis.tick_top()    
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va = "center", rotation_mode="anchor")
    plt.xlabel("sequence of generated notes",fontsize=20)
    plt.ylabel("The point of attention",fontsize=20)
    plt.title("The amount of attention given to the network hidden state",fontsize=30)
    plt.show()

# Generate MIDI file + corresponding WAV file
output_folder = os.path.join(run_folder, "output")

midi_stream = stream.Stream()

## Create note and chord objects depending on the values generated by the model
for pattern in prediction_output:
    note_pattern, duration_pattern = pattern

    ### Chord
    if ('.' in note_pattern):
        notes_in_chord = note_pattern.split(".")
        chord_notes = []

        for curr_note in notes_in_chord:
            new_note = note.Note(curr_note)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violin()
            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        midi_stream.append(new_chord)
    ### Rest
    elif note_pattern == 'rest':
        new_note = note.Rest()
        new_note.duration = duration.Duration(duration_pattern)
        new_note.stored_instrument = instrument.Violin()
        midi_stream.append(new_note)
    ### Note
    elif note_pattern != "Start":
        new_note = note.Note(note_pattern)
        new_note.duration = duration.Duration(duration_pattern)
        new_note.storedInstrument = instrument.Violin()
        midi_stream.append(new_note)

midi_stream = midi_stream.chordify()
timestr = time.strftime("%Y%m%d-%H%M%S")
new_file = 'output-'+timestr+'.mid'
midi_stream.write('midi', fp=os.path.join(output_folder, new_file))

## Convert MIDI file to audio
new_path = 'output/' + new_file
midi_to_wav(new_path, 'output/output.wav')

new_score = converter.parse(new_path).chordify()
new_score.show()
