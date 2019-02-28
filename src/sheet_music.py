"""
Methods for returning sheet music from a piano roll. Much of this code is
inspired by http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/,
which is a tutorial for converting piano music into sheet music using lilypond.

author: Nakul Iyer
date: 2/28/19
"""
import numpy as np
import os
import subprocess

from config import *

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** np.ceil(np.log2(x))

def fourier_transform(signal, period, tt):
    """One-Dimensional Fourier Transform simple implementation"""
    f = lambda function: (signal * function(2*np.pi*tt / period)).sum()
    return f(np.cos) + 1j * f(np.sin)

def reg_time_to_note(duration, quarter_duration):
    """Converts regular time intervals into note intervals"""
    return 0.5*int((4.0*duration/quarter_duration+1)/2)

def quantize(notes, quarter_duration):
    """
    Quantize notes into lilypond format. Adapted directly from
    http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/
    """
    # the result is initialized with one 'empty' note.
    result = [ {'notes':[], 'duration':None, 't_strike':0} ]

    for note in notes:
        # the next line quantizes that time in eights.
        delay_q = reg_time_to_note(note.duration, quarter_duration)

        if (delay_q == 0) and not no_chords:
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
        return "1 " + get_duration(duration - 4)
    else:
        # This catch-all is not really great, but the best we can really do
        # if we don't know the note duration
        return "4"

def midi_to_lily(midi_pitch):
    """Converts midi_pitch to a lily note"""
    lily_octave = int(midi_pitch / 12) - 1
    lily_note = midi_pitch % 12
    return lily_notes[lily_note] + lily_octaves[lily_octave]

def strike_to_lily(strike):
    """Converts chord strikes to lily notes"""
    notes = strike["notes"]
    duration = strike["duration"]
    onset = strike["t_strike"]

    if len(notes) > 1:
        # chord
        chord = ' '.join(map(midi_to_lily, sorted(notes)))
        result = ""
        for dur in get_duration(duration).split(' '):
            result += "< {} >{}\n".format(chord, dur)
        return result, onset
    else:
        result = ""
        for dur in get_duration(duration).split(' '):
            result += "{}{}\n".format(midi_to_lily(notes[0]), dur)
        return result, onset

def lily_score(strikes, quarter_note_duration):
    """Converts a python list of srikes into Lilypond"""
    strike_list = []
    last_strike_time = 0
    for strike in strikes:
        ls = strike_to_lily(strike)
        frame_interval = ls[1] - last_strike_time
        note_time_interval = reg_time_to_note(frame_interval, quarter_note_duration)
        # Add in rests
        result = ""
        for dur in get_duration(note_time_interval).split(' '):
            result += "r{}\n".format(dur)
        strike_list.append(result)
        last_strike_time = ls[1]
        strike_list.append(ls[0])
    return "\n".join(strike_list)

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
        raise SystemError("No Notes Detected")
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

    left_hand_lily = lily_score(left_hand_quantized, quarter_note_duration)
    right_hand_lily = lily_score(right_hand_quantized, quarter_note_duration)

    ly_location = "{}.ly".format(save_location)

    with open(ly_location, "w+") as f:
        lily_sequence = "\\score {\\new StaffGroup{ <<\\set StaffGroup.systemStartDelimiter = #'SystemStartSquare \\new Staff {\\clef treble " + right_hand_lily + " } \\new Staff {\\clef bass " + left_hand_lily + " }>>}}"
        f.write(lily_sequence)

    cmd = "lilypond {}".format(ly_location)
    saving_pdf = False
    try:
        subprocess.call("lilypond")
        saving_pdf = True
    except OSError:
        print("LilyPond is not properly installed. Saving instead to seperate txt files.")
        txt_location = "{}.txt".format(save_location)
        with open(txt_location, "w+") as f:
            f.write("Right Hand: {} \n Left Hand: {}".format(right_hand_lily, left_hand_lily))
        print("Successfully saved as text file")

    if saving_pdf:
        os.system(cmd)

        # Remove the original ly file
        os.remove(ly_location)

        # Move the file to the save location
        os.rename("{}.pdf".format(os.path.basename(save_location)), "{}.pdf".format(save_location))
