"""
Methods for returning sheet music from a piano roll. Much of this code is
inspired by http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/,
which is a tutorial for converting piano music into sheet music using lilypond.
author: Nakul Iyer
"""
import numpy as np
import os

from config import *

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** np.ceil(np.log2(x))

def fourier_transform(signal, period, tt):
    """One-Dimensional Fourier Transform simple implementation"""
    f = lambda function: (signal * function(2*np.pi*tt / period)).sum()
    return f(np.cos) + 1j * f(np.sin)

def quantize(notes, quarter_duration):
    """
    Quantize notes into lilypond format. Adapted directly from
    http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/
    """
    # the result is initialized with one 'empty' note.
    result = [ {'notes':[], 'duration':None, 't_strike':0} ]

    for note in notes:
        # the next line quantizes that time in eights.
        delay_q = 0.5*int((4.0*note.duration/quarter_duration+1)/2)

        if (delay_q == 0):
            # put note in previous chord
            if note.pitch not in result[-1]['notes']:
                result[-1]['notes'].append(note.pitch)

        else:
            # this is a 'new' note/chord
            result[-1]['duration'] = delay_q
            result.append( {'notes': [note.pitch],
                            'duration': None,
                            't_strike': note.onset_frame} )

     # give duration to last note
    result[-1]['duration'] = 4

    if result[0]['notes'] == []:
        # first note will surely be empty
        result.pop(0)

    return result

# Lilypond uses German note-naming schemes
lily_notes = ["c", "cis", "d", "ees", "e", "f",
              "fis", "g", "gis", "a", "bes", "b"]
lily_octaves = [",,,", ",,", ",", "", "'", "''", "'''", "''''", "'''''"]
lily_durations = {
    0.0625: "64",
    0.125: "32",
    0.25: "16",
    0.5: "8",
    1: "4",
    1.5: "4.",
    2: "2",
    2.5: "2",
    3: "2.",
    3.5: "2.",
    4: "1",
}

def get_duration(duration):
    if duration in lily_durations:
        return lily_durations[duration]
    elif duration < 0.0625:
        return "64"
    elif duration > 4:
        return "1"
    else:
        return "4"

def midi_to_lily(midi_pitch):
    """Converts midi_pitch to a lily note"""
    lily_octave = int(midi_pitch / 12) - 1
    lily_note = midi_pitch % 12
    return lily_notes[lily_note] + lily_octaves[lily_octave]

def strike_to_lily(strike):
    """Converts chord strikes to lily notes"""
    notes = strike['notes']
    duration = strike['duration']

    if len(notes) > 1:
        # chord
        chord = ' '.join(map(midi_to_lily, sorted(notes)))
        return "< {} >{}".format(chord, get_duration(duration))
    else:
        return midi_to_lily(notes[0]) + get_duration(duration)

def lily_score(strikes):
    """Converts a python list of srikes into Lilypond"""
    return "\n".join(map(strike_to_lily, strikes))

class Note:
    def __init__(self, pitch, onset_frame, offset_frame):
        self.pitch = pitch
        self.onset_frame = onset_frame
        self.offset_frame = offset_frame
        self.duration = offset_frame - onset_frame

    def to_dict(self, quarter_note_duration):
        pass

    def __repr__(self):
        note_map = {
            "pitch": self.pitch,
            "duration": self.duration
        }
        return "{}\n".format(note_map)


def piano_roll_to_sheet_music(piano_roll, save_location):
    num_midi_freqs, num_time_frames = piano_roll.shape
    notes = []
    for freq_idx in range(num_midi_freqs):
        pitch = freq_idx + midi_min

        note_playing = False
        current_onset = 0
        for time_idx in range(num_time_frames):
            current = piano_roll[freq_idx][time_idx]
            if not note_playing and current == 1:
                # Key pressed
                note_playing = True
                current_onset = time_idx
            elif note_playing and current == 0:
                # Key released, and write it into notes
                note_playing = False
                offset_frame = time_idx
                notes.append(Note(pitch, current_onset, offset_frame))
    if len(notes) == 0:
        SystemError("No Notes Detected")
    strike_times = piano_roll.sum(axis=0)
    tt = np.arange(len(strike_times))
    durations = np.arange(1.1, 30, 0.02)
    transform = np.array([fourier_transform(strike_times, d, tt) for d in durations])
    optimal_i = np.argmax(abs(transform))
    quarter_note_duration = durations[optimal_i]
    fps = sample_rate / time_step
    tempo = int(fps * 60.0 / quarter_note_duration)
    print("Tempo {}".format(tempo))

    # Left hand covers all notes under middle C
    left_hand = [note for note in notes if note.pitch < 60]
    right_hand = [note for note in notes if note.pitch >= 60]

    left_hand_quantized = quantize(left_hand, quarter_note_duration)
    right_hand_quantized = quantize(right_hand, quarter_note_duration)

    left_hand_lily = lily_score(left_hand_quantized)
    right_hand_lily = lily_score(right_hand_quantized)

    ly_location = "{}.ly".format(save_location)

    with open(ly_location, "w+") as f:
        lily_sequence = "\\score {\\new StaffGroup{ <<\\set StaffGroup.systemStartDelimiter = #'SystemStartSquare \\new Staff {\\clef treble " + right_hand_lily + " } \\new Staff {\\clef bass " + left_hand_lily + " }>>}}"
        f.write(lily_sequence)

    os.system("lilypond {}".format(ly_location))

    # Remove the original ly file
    os.remove(ly_location)

    # Move the file to the save location
    os.rename("{}.pdf".format(os.path.basename(save_location)), "{}.pdf".format(save_location))
