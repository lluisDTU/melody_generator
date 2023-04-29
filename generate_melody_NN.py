import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Model
from keras.layers import LSTM, Dense, Activation, Input
from music21 import converter, note, stream
from tqdm import tqdm
from keras.layers import Dropout

# Load the sample melody
sample_melody = converter.parse("Melody.mid")

# Extract the notes and durations from the sample melody
notes = []
durations = []
for element in sample_melody.flat:
    if isinstance(element, note.Note):
        pitch_name = element.pitch.nameWithOctave
        duration = element.duration.quarterLength
        notes.append(pitch_name)
        durations.append(duration)

# Encode notes and durations as integers
note_encoder = LabelEncoder()
duration_encoder = LabelEncoder()
encoded_notes = note_encoder.fit_transform(notes)
encoded_durations = duration_encoder.fit_transform(durations)

# Prepare input-output pairs
sequence_length = 50  # Adjust this value to change the sequence length used for training
n_sequences = len(encoded_notes) - sequence_length

X = np.zeros((n_sequences, sequence_length), dtype=int)
y_notes = np.zeros(n_sequences, dtype=int)
y_durations = np.zeros(n_sequences, dtype=int)

for i in range(n_sequences):
    X[i, :] = encoded_notes[i:i + sequence_length]
    y_notes[i] = encoded_notes[i + sequence_length]
    y_durations[i] = encoded_durations[i + sequence_length]

# One-hot encode the output
y_notes = np_utils.to_categorical(y_notes)
y_durations = np_utils.to_categorical(y_durations)

input_layer = Input(shape=(sequence_length, 1))
lstm_layer1 = LSTM(256, return_sequences=True)(input_layer)
dropout1 = Dropout(0.3)(lstm_layer1)
lstm_layer2 = LSTM(256)(dropout1)
dropout2 = Dropout(0.3)(lstm_layer2)
output_notes = Dense(y_notes.shape[1], activation='softmax')(dropout2)
output_durations = Dense(y_durations.shape[1], activation='softmax')(dropout2)

model = Model(inputs=input_layer, outputs=[output_notes, output_durations])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Reshape the input to match the expected input shape for LSTM
X = X.reshape((*X.shape, 1))

model.fit(X, [y_notes, y_durations], epochs=200, batch_size=64)  # Adjust epochs and batch_size as needed

new_melody = stream.Stream()
current_sequence = encoded_notes[:sequence_length].tolist()

for _ in tqdm(range(60)):  # Add tqdm and set the range to tqdm(range(60))
    input_sequence = np.array(current_sequence[-sequence_length:]).reshape(1, sequence_length, 1)
    prediction_notes, prediction_durations = model.predict(input_sequence)
    predicted_note = np.argmax(prediction_notes)
    predicted_duration = np.argmax(prediction_durations)
    current_sequence.append(predicted_note)

    note_pitch = note_encoder.inverse_transform([predicted_note])[0]
    duration = duration_encoder.inverse_transform([predicted_duration])[0]
    new_melody.append(note.Note(note_pitch, quarterLength=duration))

# Save the new melody as a MIDI file
new_melody.write('midi', fp='new_melody_nn.mid')



