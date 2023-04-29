import re
from music21 import *
from collections import defaultdict
import random

# Load the sample melody
sample_melody = converter.parse("Melody.mid")

# Extract the notes and durations from the sample melody
notes_and_durations = []
for element in sample_melody.flat:
    if isinstance(element, note.Note):
        pitch_name = element.pitch.nameWithOctave  # extract pitch name with octave
        duration = element.duration.quarterLength  # extract note duration
        notes_and_durations.append((pitch_name, duration))

# Generate transition probabilities for higher-order Markov chains
order = 2  # Change this to increase or decrease the Markov chain order
transitions = defaultdict(list)
for i in range(len(notes_and_durations) - order):
    current_state = tuple(notes_and_durations[i:i + order])
    next_state = notes_and_durations[i + order]
    transitions[current_state].append(next_state)

# Generate a new melody using Markov Chain algorithm
new_melody = stream.Stream()
current_state = random.choice(list(transitions.keys()))
new_melody.append([note.Note(pitch_name, quarterLength=duration) for pitch_name, duration in current_state])

for _ in range(60):  # Generate 60 notes
    next_states = transitions[current_state]
    if next_states:
        next_state = random.choice(next_states)
        pitch_name, duration = next_state
        new_melody.append(note.Note(pitch_name, quarterLength=duration))
        current_state = current_state[1:] + (next_state,)
    else:
        current_state = random.choice(list(transitions.keys()))
        new_melody.append([note.Note(pitch_name, quarterLength=duration) for pitch_name, duration in current_state])

# Save the new melody as a MIDI file
new_melody.write('midi', fp='new_melody.mid')






