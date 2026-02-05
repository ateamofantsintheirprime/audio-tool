import arcade
import soundfile as sf
import sounddevice as sd
from helper import *

SCREEN_WIDTH = 1024 
SCREEN_HEIGHT = 800

class InteractiveSpectrogram(arcade.Window):
	def __init__(self, raw_signal, samplerate):
		super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Shader Spectrogram", gl_version=(3, 3))
		self.background_color = arcade.color.ALMOND

		self.raw_signal = np.ascontiguousarray(raw_signal)
		self.samplerate = samplerate
		
		# Produce spectral data from raw signal
		magnitudes, frequencies = STFT(raw_signal=raw_signal, samplerate=samplerate, chunk_size=4096, chunk_hop=256)
		spectrogram, frequencies = filter_frequencies(magnitudes, frequencies, 0, 5000)
		spectrogram = spectrogram.astype("float32")
		t_height, t_width= spectrogram.shape

		# GL geometry that will be used to pass pixel coordinates to the shader
		# It has the same dimensions as the screen
		self.quad_fs = arcade.gl.geometry.quad_2d_fs()

		self.tex = self.ctx.texture(
			size=(t_width, t_height),
			components=1,
			dtype='f4',
		)

		print("spectrogram min/max/mean:", spectrogram.min(), spectrogram.max(), spectrogram.mean())
		print("Texture size:", self.tex.size)
		#print("Frequencies:", frequencies)

		#self.tex.filter = (arcade.gl.LINEAR, arcade.gl.LINEAR)
		self.tex.write(spectrogram)

		# Create a simple shader program

		self.prog = self.ctx.load_program(
			vertex_shader = "vertex_shader.glsl",
			fragment_shader = "fragment_shader.glsl"
		)

		# Register the texture uniforms in the shader program
		self.prog['t0'] = 0

		# Create a variable to track program run time
		self.total_time = 0
		self.current_frame = 0
		self.is_playing = False

		# Setup audio playback stream 
		self.stream = sd.OutputStream(
			samplerate=samplerate,
			channels=1
 		)

		self.stream.start()
		self.playback_position = 0
		self.total_duration = len(raw_signal) / samplerate

	def on_update(self, delta_time):
		if self.is_playing:
			self.total_time == delta_time
			samples_to_play = int(delta_time * self.samplerate)
			end_pos = min(self.playback_position + samples_to_play, len(self.raw_signal))

			if self.playback_position < len(self.raw_signal):
				chunk = self.raw_signal[self.playback_position:end_pos]
				self.stream.write(chunk)
				self.playback_position = end_pos
			else:
				self.is_playing = False

	def on_draw(self):
		self.clear()
		self.tex.use(0)
		self.quad_fs.render(self.prog)

		progress = self.playback_position / len(self.raw_signal)
		x_pos = progress * SCREEN_WIDTH
		arcade.draw_line(x_pos, 0, x_pos, SCREEN_HEIGHT, arcade.color.RED, 3)

	def on_key_press(self, key, modifiers):
		if key == arcade.key.SPACE:
			self.toggle_play()
		elif key == arcade.key.R:
			self.restart()
	
	def on_mouse_press(self, x, y, button, modifiers):
		self.scrub(x / SCREEN_WIDTH)
	
	def toggle_play(self):
		self.is_playing = not self.is_playing

	def restart(self):
		self.total_time = 0
		self.playback_position = 0
		self.is_playing = False

	def scrub(self, position):
		self.playback_position = int(position * len(self.raw_signal))
		self.total_time = position * self.total_duration
		self.is_playing = False


if __name__ == "__main__":
	sound_dir = "sounds/"
	filename = 'etude2.mp3'
	# filename = 'blake.mp3'
	filename = 'chromatic.mp3'
	# filename = 'e3f3.mp3'
	# filename = 'e3f3_fast.mp3'
	# filename = 'sin_test.mp3' 

	data, samplerate = sf.read(sound_dir + filename, dtype='float32')
	if len(data.shape) > 1:
		data = data[:,0]

	window = InteractiveSpectrogram(data, samplerate)

	arcade.run()

