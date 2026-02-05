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

		# Setup audio playback stream 
# 		stream = sd.OutputStream(
# 			samplerate=samplerate
# 		)

	def on_update(self, delta_time):
		self.total_time += delta_time/10.0

	def on_draw(self):
		# Draw a simple circle
		self.clear()
		self.prog['time'] = self.total_time
		self.tex.use(0)
		self.quad_fs.render(self.prog)

	def on_key_press(self, key, modifiers):
		if key == arcade.key.SPACE:
			self.toggle_play()
		elif key == arcade.key.R:
			self.restart()
	
	def start_stream(self):
		pass
	
	def togggle_play(self):
		pass

	def restart(self):
		pass

	def scrub(self):
		pass

filename = 'etude2.mp3'
# filename = 'blake.mp3'
filename = 'chromatic.mp3'
# filename = 'e3f3.mp3'
# filename = 'e3f3_fast.mp3'
# filename = 'sin_test.mp3' 

data, samplerate = sf.read(filename, dtype='float32')
if len(data.shape) > 1:
	data = data[:,0]

window = InteractiveSpectrogram(data, samplerate)

arcade.run()

