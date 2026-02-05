from math import ceil
import numpy as np
np.set_printoptions(suppress=True, precision=4)

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def get_note_frequencies():
	# A4 = 440 Hz
	A4 = 440.0
	# The Bb clarinet transposed down 1 tone i think? so everything sounds a note lower than written?
	note_frequencies = {}

	# Calculate frequencies for octaves 1-8
	for octave in range(3, 6):
		for i, note in enumerate(notes):
			# Calculate semitone offset from A4
			semitone_offset = i - 9 + (octave - 4) * 12 - 2 #( The minus 2 is from the Bb clarinet transposed 2 semitones lower than written)
			frequency = A4 * (2 ** (semitone_offset / 12.0))
			note_name = f"{note}{octave}"
			note_frequencies[note_name] = frequency

	#print(len(note_frequencies.keys()))
	return note_frequencies

def filter_frequencies(magnitudes, frequencies, min_freq = 0, max_freq = 10000):
	magnitudes = magnitudes[(frequencies < max_freq) & (frequencies > min_freq)]
	frequencies = frequencies[(frequencies < max_freq) & (frequencies > min_freq)]
	magnitudes = np.ascontiguousarray(magnitudes)
	frequencies = np.ascontiguousarray(frequencies)
	
	return magnitudes, frequencies

def STFT(raw_signal, samplerate, chunk_size, chunk_hop, pad_final_chunk = False, gain=50):
	"""
	if the signal length does not evenly fit into chunks, pad_final_chunk determines
	whether the final chunk will be padded out with zeros, or if the final chunk will be discarded.
	"""
	assert chunk_hop <= chunk_size
	assert pad_final_chunk == False # We will add functionality for this later if needed

	# Chunk signal
	num_chunks = (len(raw_signal)-chunk_size)//chunk_hop + 1
	chunked_signal = np.ndarray(shape=(num_chunks, chunk_size), dtype=float)
	for i in range(num_chunks):
		chunked_signal[i][:] = raw_signal[i*chunk_hop : i*chunk_hop+chunk_size]

	# Apply Hann window
	# We divide by avg value of hann window to avoid reducing the overall amplitude
	for chunk in chunked_signal:
		chunk *= np.hanning(chunk_size) / 0.5 
	
	# Get frequency bucket frequencies
	frequencies = np.fft.rfftfreq(chunk_size, d=1./samplerate)
	magnitudes = np.ndarray(shape=(len(frequencies), num_chunks), dtype=float)

	# Calculate frequency bucket magnitudes with the Discrete Fourier Transform
	for i, chunk in enumerate(chunked_signal):
		magnitudes[:,i] = np.abs(np.fft.rfft(chunk)) / chunk_size * gain

	return magnitudes, frequencies


if __name__ == "__main__":
	test_signal = list(range(20))
	print(STFT(test_signal, 1, 5, 4))
