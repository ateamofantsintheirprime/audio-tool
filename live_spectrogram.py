import arcade
import sounddevice as sd
import numpy as np
from helper import *
from collections import deque
import threading

SCREEN_WIDTH = 1024 
SCREEN_HEIGHT = 800

class RealtimeSpectrogram(arcade.Window):
	def __init__(self, samplerate=44100, chunk_size=4096, chunk_hop=256, 
				 freq_range=(0, 5000), buffer_duration=5.0):
		super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Real-time Shader Spectrogram", gl_version=(3, 3))
		self.background_color = arcade.color.ALMOND

		# Audio parameters
		self.samplerate = samplerate
		self.chunk_size = chunk_size
		self.chunk_hop = chunk_hop
		self.freq_min, self.freq_max = freq_range

		# Calculate how many spectral columns we can fit in buffer_duration seconds
		self.buffer_duration = buffer_duration
		self.max_columns = int((buffer_duration * samplerate) / chunk_hop)

		# Audio buffer to accumulate incoming samples
		self.audio_buffer = deque(maxlen=chunk_size)
		self.lock = threading.Lock()

		# Spectrogram data buffer (scrolling window)
		self.spectrogram_buffer = None
		self.texture_height = None
		self.current_column = 0

		# GL geometry for rendering
		self.quad_fs = arcade.gl.geometry.quad_2d_fs()

		# Texture will be created after first FFT to know dimensions
		self.tex = None
		self.prog = None

		# Track time for shader effects
		self.total_time = 0

		# Flag to indicate first FFT has been computed
		self.initialized = False

		# Start audio stream
		self.stream = sd.InputStream(
			channels=1,
			samplerate=self.samplerate,
			callback=self.audio_callback,
			blocksize=chunk_hop
		)
		self.stream.start()

	def audio_callback(self, indata, frames, time, status):
		"""Called by sounddevice for each audio block"""
		if status:
			print(f"Audio status: {status}")

		with self.lock:
			# Add new samples to buffer
			self.audio_buffer.extend(indata[:, 0])

	def process_audio(self):
		"""Process accumulated audio samples and update spectrogram"""
		with self.lock:
			if len(self.audio_buffer) < self.chunk_size:
				return  # Not enough data yet

			# Get chunk_size samples for FFT
			audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_size:])
			audio_chunk *= np.hanning(self.chunk_size) / 0.5 

		# Compute FFT
		magnitudes = np.abs(np.fft.rfft(audio_chunk))
		frequencies = np.fft.rfftfreq(self.chunk_size, 1/self.samplerate)

		# Filter to desired frequency range
		freq_mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
		magnitudes_filtered = magnitudes[freq_mask]

		if not self.initialized:
			# First time: initialize texture and buffer
			self.texture_height = len(magnitudes_filtered)
			self.spectrogram_buffer = np.zeros((self.texture_height, self.max_columns), dtype='float32')

			self.tex = self.ctx.texture(
				size=(self.max_columns, self.texture_height),
				components=1,
				dtype='f4',
			)

			self.prog = self.ctx.load_program(
				vertex_shader="vertex_shader.glsl",
				fragment_shader="fragment_shader.glsl"
			)
			self.prog['t0'] = 0

			print(f"Initialized: texture size = {self.tex.size}, freq bins = {self.texture_height}")
			self.initialized = True

		# Add new column to spectrogram buffer (scroll left)
		self.spectrogram_buffer[:, :-1] = self.spectrogram_buffer[:, 1:]
		self.spectrogram_buffer[:, -1] = magnitudes_filtered.astype('float32')

		# Update texture
		self.tex.write(self.spectrogram_buffer)

	def detect_note(self, chunk):
		pass

	def detect_note_change(self, chunk):
		pass

	def spectral_delta(self, prev, curr):
		return curr-prev


	def on_update(self, delta_time):
		self.total_time += delta_time / 10.0
		self.process_audio()

	def on_draw(self):
		self.clear()

		if self.initialized:
			#self.prog['time'] = self.total_time
			self.tex.use(0)
			self.quad_fs.render(self.prog)

	def on_key_press(self, key, modifiers):
		"""Handle keyboard input"""
		if key == arcade.key.ESCAPE or key == arcade.key.Q:
			self.close()

	def on_close(self):
		"""Clean up audio stream on window close"""
		self.stream.stop()
		self.stream.close()
		super().on_close()


if __name__ == "__main__":
	# Configuration
	SAMPLERATE = 44100
	CHUNK_SIZE = 8192
	CHUNK_HOP = 256
	FREQ_RANGE = (0, 5000)
	BUFFER_DURATION = 5.0  # seconds of spectrogram history

	window = RealtimeSpectrogram(
		samplerate=SAMPLERATE,
		chunk_size=CHUNK_SIZE,
		chunk_hop=CHUNK_HOP,
		freq_range=FREQ_RANGE,
		buffer_duration=BUFFER_DURATION
	)
	arcade.run()
