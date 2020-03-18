import magenta.music as mm
import magenta
import tensorflow
from bokeh.io import export_png



from magenta.music.protobuf import music_pb2

twinkle_twinkle = music_pb2.NoteSequence()

# Add the notes to the sequence.
twinkle_twinkle.notes.add(pitch=60, start_time=0.0, end_time=0, velocity=80)
twinkle_twinkle.notes.add(pitch=60, start_time=0.5, end_time=0.5, velocity=80)
twinkle_twinkle.notes.add(pitch=67, start_time=1.0, end_time=1.0, velocity=80)
twinkle_twinkle.notes.add(pitch=67, start_time=1.5, end_time=1.5, velocity=80)
twinkle_twinkle.notes.add(pitch=69, start_time=2.0, end_time=2.0, velocity=80)
twinkle_twinkle.notes.add(pitch=69, start_time=2.5, end_time=2.5, velocity=80)
twinkle_twinkle.notes.add(pitch=67, start_time=3.0, end_time=3.0, velocity=80)
twinkle_twinkle.notes.add(pitch=65, start_time=4.0, end_time=4.0, velocity=80)
twinkle_twinkle.notes.add(pitch=65, start_time=4.5, end_time=4.5, velocity=80)
twinkle_twinkle.notes.add(pitch=64, start_time=5.0, end_time=5.0, velocity=80)
twinkle_twinkle.notes.add(pitch=64, start_time=5.5, end_time=5.5, velocity=80)
twinkle_twinkle.notes.add(pitch=62, start_time=6.0, end_time=6.0, velocity=80)
twinkle_twinkle.notes.add(pitch=62, start_time=6.5, end_time=6.5, velocity=80)
twinkle_twinkle.notes.add(pitch=60, start_time=7.0, end_time=7.0, velocity=80) 
twinkle_twinkle.total_time = 8

twinkle_twinkle.tempos.add(qpm=960)

# This is a colab utility method that visualizes a NoteSequence.
fig = mm.plot_sequence(twinkle_twinkle, show_figure=False)
export_png(fig, filename="plot.png")

mm.sequence_proto_to_midi_file(twinkle_twinkle, 'twinkle_twinkle.mid')



for qpm in range(60, 120, 60):
    drums = music_pb2.NoteSequence()

    drums.notes.add(pitch=36, start_time=0, end_time=0, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=38, start_time=0, end_time=0, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=0, end_time=0, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=46, start_time=0, end_time=0, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=0.25, end_time=0.25, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=0.375, end_time=0.375, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=0.5, end_time=0.5, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=50, start_time=0.5, end_time=0.5, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=36, start_time=0.75, end_time=0.75, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=38, start_time=0.75, end_time=0.75, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=0.75, end_time=0.75, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=45, start_time=0.75, end_time=0.75, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=36, start_time=1, end_time=1, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=1, end_time=1, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=46, start_time=1, end_time=1, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=42, start_time=1.25, end_time=1.25, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=48, start_time=1.25, end_time=1.25, is_drum=True, instrument=10, velocity=80)
    drums.notes.add(pitch=50, start_time=1.25, end_time=1.25, is_drum=True, instrument=10, velocity=80)
    drums.total_time = 1.375

    drums.tempos.add(qpm=qpm)

    mm.sequence_proto_to_midi_file(drums, 'drum%i.mid' % qpm)

drums = music_pb2.NoteSequence()

drums.notes.add(pitch=36, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=38, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=46, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=0.25, end_time=0.375, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=0.375, end_time=0.5, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=0.5, end_time=0.625, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=50, start_time=0.5, end_time=0.625, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=36, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=38, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=45, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=36, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=46, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=42, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=48, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
drums.notes.add(pitch=50, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
drums.total_time = 1.375

drums.tempos.add(qpm=60)

mm.sequence_proto_to_midi_file(drums, 'drum.mid')