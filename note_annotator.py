from interactive_spectrogram import InteractiveSpectrogram
import soundfile as sf
import arcade
from helper import note_label_to_midi, midi_to_note_label
from arcade.gui import (
    UIManager,
    UIAnchorLayout,
    UIBoxLayout,
    UIFlatButton,
    UILabel,
    UIInputText,
    UIOnClickEvent,
    UIOnChangeEvent,
)

SCREEN_WIDTH = 1024 
SCREEN_HEIGHT = 800

class NoteAnnotatingSpectrogram(InteractiveSpectrogram):
	def __init__(self, raw_signal, samplerate):
		super().__init__(raw_signal, samplerate)
		self.annotations = []
		self.selecting = False
		self.selection_start = None
		self.selection_end = None

		# Setup UI Manager
		self.ui = UIManager()
		self.ui.enable()

		# Create manage button
		self.setup_ui()

		# Current overlay (for dialogs)
		self.current_overlay = None

	def setup_ui(self):
		"""Create the manage button"""
		anchor = UIAnchorLayout()
		self.ui.add(anchor)

		manage_btn = UIFlatButton(
			text="Manage (M)",
			width=180,
			height=40,
			style=UIFlatButton.STYLE_BLUE
		)
		anchor.add(manage_btn, anchor_x="left", anchor_y="bottom", align_x=10, align_y=10)

		@manage_btn.event("on_click")
		def on_manage_click(event):
			self.show_annotation_manager()

	def show_annotation_dialog(self, start_pos, end_pos):
		"""Show dialog to input note name"""
		# Create overlay
		overlay = UIAnchorLayout(size_hint=(1, 1))
		overlay.with_background(color=(0, 0, 0, 128))

		# Create dialog box
		dialog_box = UIBoxLayout(
			vertical=True,
			width=400,
			height=200,
			space_between=10
		)
		dialog_box.with_padding(all=20)
		dialog_box.with_background(color=arcade.color.WHITE)

		dialog_box.add(UILabel(text="Enter Note Name:", font_size=16, text_color=arcade.color.BLACK))

		note_input = dialog_box.add(
			UIInputText(
				width=360,
				height=40,
				font_size=14,
                text_color=arcade.color.BLACK
			)
		)

		button_row = UIBoxLayout(vertical=False, space_between=10)
		dialog_box.add(button_row)

		ok_btn = button_row.add(UIFlatButton(text="OK", width=100, style=UIFlatButton.STYLE_BLUE))
		cancel_btn = button_row.add(UIFlatButton(text="Cancel", width=100))

		@ok_btn.event("on_click")
		def on_ok(event):
			note = note_input.text.strip().upper()
			if note and note_label_to_midi(note):
				self.annotations.append((start_pos, end_pos, note))
			self.ui.remove(overlay)
			self.current_overlay = None

		@cancel_btn.event("on_click")
		def on_cancel(event):
			self.ui.remove(overlay)
			self.current_overlay = None

		overlay.add(dialog_box, anchor_x="center", anchor_y="center")
		self.ui.add(overlay)
		self.current_overlay = overlay

	def show_annotation_manager(self):
		"""Show window with editable list of annotations"""
		# Create overlay
		overlay = UIAnchorLayout(size_hint=(1, 1))
		overlay.with_background(color=(0, 0, 0, 128))

		# Create dialog box
		dialog_width = 700
		dialog_height = 500
		dialog_box = UIBoxLayout(
			vertical=True,
			width=dialog_width,
			height=dialog_height,
			space_between=10
		)
		dialog_box.with_padding(all=20)
		dialog_box.with_background(color=arcade.color.WHITE)

		# Title
		dialog_box.add(
			UILabel(
				text="Manage Annotations",
				font_size=20,
				text_color=arcade.color.BLACK
			)
		)

		# Scrollable area for annotations
		scroll_area = UIBoxLayout(vertical=True, space_between=5, size_hint=(1, 1))

		# Store references to entry widgets
		entry_widgets = []

		# Create row for each annotation
		for i, (start_pos, end_pos, note) in enumerate(self.annotations):
			row = UIBoxLayout(vertical=False, space_between=5)

			row.add(UILabel(text="Start:", width=50, text_color=arcade.color.BLACK))
			start_entry = row.add(UIInputText(text=f"{start_pos:.4f}", width=80, height=30, text_color=arcade.color.BLACK))

			row.add(UILabel(text="End:", width=40, text_color=arcade.color.BLACK))
			end_entry = row.add(UIInputText(text=f"{end_pos:.4f}", width=80, height=30, text_color=arcade.color.BLACK))

			row.add(UILabel(text="Note:", width=50, text_color=arcade.color.BLACK))
			note_entry = row.add(UIInputText(text=note, width=150, height=30, text_color=arcade.color.BLACK))

			delete_btn = row.add(
				UIFlatButton(text="X", width=30, height=30, style=UIFlatButton.STYLE_RED)
			)

			# Mark for deletion
			def make_delete_handler(idx, row_widget):
				def on_delete(event):
					entry_widgets[idx] = None
					row_widget.visible = False
				return on_delete

			delete_btn.event("on_click")(make_delete_handler(i, row))

			scroll_area.add(row)
			entry_widgets.append((start_entry, end_entry, note_entry))

		dialog_box.add(scroll_area)

		# Button row
		button_row = UIBoxLayout(vertical=False, space_between=10)

		save_btn = button_row.add(
			UIFlatButton(text="Save Changes", width=150, style=UIFlatButton.STYLE_BLUE)
		)
		cancel_btn = button_row.add(
			UIFlatButton(text="Cancel", width=150)
		)

		@save_btn.event("on_click")
		def on_save(event):
			new_annotations = []
			for entries in entry_widgets:
				if entries is not None:  # Not deleted
					start_entry, end_entry, note_entry = entries
					try:
						start = float(start_entry.text)
						end = float(end_entry.text)
						note = note_entry.text.strip()
						if note:
							new_annotations.append((start, end, note))
					except ValueError:
						pass  # Skip invalid entries

			self.annotations = new_annotations
			self.ui.remove(overlay)
			self.current_overlay = None

		@cancel_btn.event("on_click")
		def on_cancel(event):
			self.ui.remove(overlay)
			self.current_overlay = None

		dialog_box.add(button_row)

		overlay.add(dialog_box, anchor_x="center", anchor_y="center")
		self.ui.add(overlay)
		self.current_overlay = overlay

	def on_key_press(self, key, modifiers):
		super().on_key_press(key, modifiers)
		if key == arcade.key.M:
			self.show_annotation_manager()

	def on_mouse_press(self, x, y, button, modifiers):
		# UIManager handles button clicks automatically, so just handle scrubbing and selection
		if button == arcade.MOUSE_BUTTON_LEFT:
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

	def on_draw(self):
		self.clear()
		self.tex.use(0)
		self.quad_fs.render(self.prog)

		# Draw saved annotations
		for start_pos, end_pos, note in self.annotations:
			x_start = start_pos * SCREEN_WIDTH
			x_end = end_pos * SCREEN_WIDTH

			# Draw semi-transparent box
			arcade.draw_lrbt_rectangle_filled(
				x_start, x_end, 0, SCREEN_HEIGHT,
				(100, 150, 255, 80)
			)

			# Draw border
			arcade.draw_lrbt_rectangle_outline(
				x_start, x_end, 0, SCREEN_HEIGHT,
				(100, 150, 255, 200), 2
			)

			# Draw note label
			arcade.draw_text(note, x_start + 5, SCREEN_HEIGHT - 30,
				arcade.color.WHITE, 14, bold=True)

		# Draw current selection (while dragging)
		if self.selecting and self.selection_start is not None:
			start = min(self.selection_start, self.selection_end)
			end = max(self.selection_start, self.selection_end)
			x_start = start * SCREEN_WIDTH
			x_end = end * SCREEN_WIDTH

			arcade.draw_lrbt_rectangle_filled(
				x_start, x_end, 0, SCREEN_HEIGHT,
				(255, 255, 0, 100)
			)

		# Draw playback position indicator
		progress = self.playback_position / len(self.raw_signal)
		x_pos = progress * SCREEN_WIDTH
		arcade.draw_line(x_pos, 0, x_pos, SCREEN_HEIGHT, arcade.color.RED, 3)

		# Draw UI (button is now handled by UIManager)
		self.ui.draw()

if __name__ == "__main__":
	sound_dir = "sounds/"
	filename = 'chromatic.mp3'

	data, samplerate = sf.read(sound_dir + filename, dtype='float32')
	if len(data.shape) > 1:
		data = data[:,0]

	window = NoteAnnotatingSpectrogram(data, samplerate)
	arcade.run()
