"""
Various useful utilities
"""
import numpy as np
import sys

A4 = 440 # standard tuning
C0 = A4*(2**(-4.75))
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def HzToNote(freq):
    """
    Parameters
    ----------
    freq : float
            note as a frequency in Hz
    """
    # Compute the number of half-steps above C0
    half_steps = int(round(12 * np.log2(freq/C0)))

    # Get the octave number by diving out by 12
    # and the note by finding the remainder
    octave = half_steps // 12
    note_index = half_steps % 12
    return notes[note_index] + str(octave)

def noteToHz(s):
    """
    Parameters
    ----------
    s : string
        note as a string (i.e. C0, D#1)
        NOTE: pre-convert all flat note to sharps (i.e Bb --> A#)
    """
    # Find note index and octave number in string
    note_index = 0
    for note in notes:
        if not s.find(note) == -1:
            if s.find(note + "#") == -1:
                break
        elif note == "B":
            raise Exception("Note unrecognized")
        note_index += 1
    octave = int(s[s.find(note)+len(note):])

    # Compute half-steps above C0
    half_steps = octave * 12 + note_index
    return C0*(2**(half_steps/12))
