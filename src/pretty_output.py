import abjad

duration = abjad.Duration(1, 4)
notes = [abjad.Note(pitch, duration) for pitch in range(8)]
staff = abjad.Staff(notes)
abjad.show(staff)