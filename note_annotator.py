from interactive_spectrogram import InteractiveSpectrogram
import soundfile as sf
import arcade 
import tkinter as tk

from tkinter import Toplevel, Frame, Label, Entry, Button, Scrollbar, Canvas, simpledialog

SCREEN_WIDTH = 1024 
SCREEN_HEIGHT = 800

class NoteAnnotatingSpectrogram(InteractiveSpectrogram):
	def __init__(self, raw_signal, samplerate):
		super().__init__(raw_signal, samplerate)
		self.annotations = []
		self.selecting = False
		self.selection_start = None
		self.selection_end = None

	def show_annotation_dialog(self, start_pos, end_pos):
		root = tk.Tk()
		root.withdraw()

		note = simpledialog.askstring("Annotate selection", "note name:", parent=root)

		root.destroy

		if note:
			self.annotations.append((start_pos, end_pos, note))

	def on_key_press(self, key, modifiers):
		super().on_key_press(key, modifiers)
		if key == arcade.key.M:
			self.show_annotation_manager()

	def on_mouse_press(self, x, y, button, modifiers):
		if button == arcade.MOUSE_BUTTON_LEFT:
			if 10 <= x <= 190 and 10 <= y <= 50:
				self.show_annotation_manager()
				return
			self.scrub(x/SCREEN_WIDTH)
		elif button == arcade.MOUSE_BUTTON_RIGHT:
			self.selecting = True
			self.selection_start = x / SCREEN_WIDTH
			self.selection_end = x / SCREEN_WIDTH

	def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
		if self.selecting:
			self.selection_end = x / SCREEN_WIDTH

	def on_mouse_release(self, x, y, button, modifiers):
		if button == arcade.MOUSE_BUTTON_RIGHT and self.selecting:
			self.selecting = False
			self.selection_end = x / SCREEN_WIDTH
			start = min(self.selection_start, self.selection_end)
			end = max(self.selection_start, self.selection_end)

			self.show_annotation_dialog(start, end)

			self.selection_start = None
			self.selection_end = None
			self.manage_button = None
			self.setup_ui()

	def setup_ui(self):
		pass

	def show_annotation_manager(self):
		"""Show window with editable list of annotations"""
		manager = tk.Tk()
		manager.title("Manage Annotations")
		manager.geometry("600x400")

		# Create scrollable frame
		canvas = Canvas(manager)
		scrollbar = Scrollbar(manager, orient="vertical", command=canvas.yview)
		scrollable_frame = Frame(canvas)

		scrollable_frame.bind(
			"<Configure>",
			lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
		)

		canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
		canvas.configure(yscrollcommand=scrollbar.set)

		# Store entry widgets for later retrieval
		entry_widgets = []

		# Create row for each annotation
		for i, (start_pos, end_pos, note) in enumerate(self.annotations):
			row_frame = Frame(scrollable_frame)
			row_frame.pack(fill="x", padx=5, pady=2)

			# Start time
			Label(row_frame, text="Start:").pack(side="left", padx=2)
			start_entry = Entry(row_frame, width=10)
			start_entry.insert(0, f"{start_pos:.4f}")
			start_entry.pack(side="left", padx=2)

			# End time
			Label(row_frame, text="End:").pack(side="left", padx=2)
			end_entry = Entry(row_frame, width=10)
			end_entry.insert(0, f"{end_pos:.4f}")
			end_entry.pack(side="left", padx=2)

			# Note label
			Label(row_frame, text="Note:").pack(side="left", padx=2)
			note_entry = Entry(row_frame, width=15)
			note_entry.insert(0, note)
			note_entry.pack(side="left", padx=2)

			# Delete button
			def delete_row(idx=i, frame=row_frame):
				frame.destroy()
				entry_widgets[idx] = None  # Mark as deleted

			delete_btn = Button(row_frame, text="X", command=delete_row, 
				 bg="red", fg="white", width=2)
			delete_btn.pack(side="left", padx=2)

			entry_widgets.append((start_entry, end_entry, note_entry))

		# Save button
		def save_changes():
			new_annotations = []
			for entries in entry_widgets:
				if entries is not None:  # Not deleted
					start_entry, end_entry, note_entry = entries
					try:
						start = float(start_entry.get())
						end = float(end_entry.get())
						note = note_entry.get()
						if note:  # Only save if note is not empty
							new_annotations.append((start, end, note))
					except ValueError:
						pass  # Skip invalid entries

			self.annotations = new_annotations
			manager.destroy()

		button_frame = Frame(manager)
		button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

		Button(button_frame, text="Save Changes", command=save_changes,
			bg="green", fg="white").pack(side="left", padx=5)
		Button(button_frame, text="Cancel", command=manager.destroy,
			bg="gray", fg="white").pack(side="left", padx=5)

		canvas.pack(side="left", fill="both", expand=True)
		scrollbar.pack(side="right", fill="y")

		manager.mainloop()

	def on_draw(self):
		self.clear()
		#self.prog['time'] = self.total_time
		self.tex.use(0)
		self.quad_fs.render(self.prog)

		# Draw saved annotations
		for start_pos, end_pos, note in self.annotations:
			x_start = start_pos * SCREEN_WIDTH
			x_end = end_pos * SCREEN_WIDTH
			width = x_end - x_start

			# Draw semi-transparent box
			arcade.draw_lrbt_rectangle_filled(
				x_start, x_end, 0, SCREEN_HEIGHT,
				(100, 150, 255, 80)  # Light blue with transparency
			)

			# Draw border
			arcade.draw_lrbt_rectangle_outline(
				x_start, x_end, 0, SCREEN_HEIGHT,
				(100, 150, 255, 200), 2
			)

			# Draw note label
			arcade.draw_text(note, x_start + 5, SCREEN_HEIGHT - 30,
				arcade.color.BLACK, 14, bold=True)

		# Draw current selection (while dragging)
		if self.selecting and self.selection_start is not None:
			start = min(self.selection_start, self.selection_end)
			end = max(self.selection_start, self.selection_end)
			x_start = start * SCREEN_WIDTH
			x_end = end * SCREEN_WIDTH

			arcade.draw_lrbt_rectangle_filled(
				x_start, x_end, 0, SCREEN_HEIGHT,
				(255, 255, 0, 100)  # Yellow highlight while selecting
			)

		# Draw playback position indicator
		progress = self.playback_position / len(self.raw_signal)
		x_pos = progress * SCREEN_WIDTH
		arcade.draw_line(x_pos, 0, x_pos, SCREEN_HEIGHT, arcade.color.RED, 3)
		button_x, button_y = 10, 10
		button_width, button_height = 180, 40

		arcade.draw_lrbt_rectangle_filled(
			button_x, button_x + button_width, 
			button_y, button_y + button_height,
			(50, 50, 200, 200)
		)
		arcade.draw_text("Manage (M)", button_x + 10, button_y + 12,
						arcade.color.WHITE, 14, bold=True)

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

	window = NoteAnnotatingSpectrogram(data, samplerate)

	arcade.run()


