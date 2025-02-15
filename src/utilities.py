"""
Various useful utilities

author: Nakul Iyer
date: 2/28/19
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

A4 = 440 # standard tuning
C0 = A4*(2**(-4.75))
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def Hz_to_note(freq):
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

def note_to_Hz(s):
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

def in_group(song_root, group_name, pianos=None):
    """
    Returns whether the song is in a group, and its useful
    For getting specific pianos or specific song types (isolated notes,
    usual chords, etc)
    Parameters
    ----------
    song_root : string
                name of song without postfix (i.e. `.wav`)
    group_name : string
                name of the group to search for within the song_root
    """
    group_find = song_root.find(group_name) > -1
    piano_find = False
    if pianos:
        for piano_name in pianos:
            if song_root.find(piano_name) > -1:
                piano_find = True
    else:
        piano_find = True
    if group_find and piano_find:
        return True
    return False

def reduce_spec(specgram):
    """Reduces spectrogram from frequency bins to midinote bins"""
    result = np.nan_to_num(specgram)
    return 20*np.log10(result)

def disp_spec(specgram, data_len, sample_rate, time_step, xticks=True, yticks=True, reduce=True, show=True, dloc=None):
    """Displays spectrogram"""
    if reduce:
        plt.imshow(reduce_spec(specgram), aspect='auto', cmap='inferno', origin='lower')
    else:
        plt.imshow(specgram, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title('CQT spectrogram')

    if xticks:
        time_ticks = int(np.ceil(data_len/sample_rate))

        plt.xticks(np.round(np.arange(1, time_ticks)*sample_rate/time_step),
                   np.arange(1, time_ticks))
    plt.xlabel('Time (s)')

    if yticks:
        plt.yticks(np.arange(0, 24 * 8, 24),
                   ('A0','A1','A2','A3','A4','A5', 'A6', 'A7'))
    plt.ylabel('Frequency (semitones/log(Hz))')
    if dloc:
        plt.savefig(dloc + ".png")
    if show:
        plt.show()

def disp_roll(piano_roll, data_len, sample_rate, time_step, xticks=True, yticks=True, show=True, dloc=None):
    """Displays Piano-Roll"""
    plt.imshow(piano_roll, aspect='auto', cmap='inferno', origin='lower')
    plt.title('Piano Roll')
    if xticks:
        time_ticks = int(np.ceil(data_len/sample_rate))

        plt.xticks(np.round(np.arange(1, time_ticks)*sample_rate/time_step),
                   np.arange(1, time_ticks))
    plt.xlabel('Time (s)')

    if yticks:
        plt.yticks(np.arange(0, 88, 12),
                   ('A0','A1','A2','A3','A4','A5', 'A6', 'A7'))
    plt.ylabel('Midi Note')
    if dloc:
        plt.savefig(dloc + ".png")
    if show:
        plt.show()
