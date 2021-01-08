import threading
import tkinter as tk

import sounddevice as sd

from calculator.computer import Computer
from calculator.recognizers import create_single_operation_recognizer
from data.convert import wav2mfcc
from hmm.model import Model

RATE = 16000
sd.default.samplerate = RATE
sd.default.channels = 1


class App(tk.Frame):
    def __init__(self, parent, model: Model):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self._model = model
        self._init_ui()
        self._is_recording = False
        self._update_state('Ready')

    def _init_ui(self):
        self._state_label = tk.Label(self, font=('Verdana', 15))
        self._state_label.pack(side=tk.TOP, pady=10)

        self._mic_photo = tk.PhotoImage(file='../resources/microphone.png')
        self._button = tk.Button(self, image=self._mic_photo, command=self._on_click)
        self._button.pack(side=tk.TOP)

        self._result_label = tk.Label(self, font=('Verdana', 15), height=2, bg='white')
        self._result_label.pack(side=tk.TOP, fill=tk.BOTH)

    def _update_state(self, state):
        self._state_label['text'] = state

    def _display_result(self, result):
        self._result_label['text'] = result

    def _freeze(self):
        self._button['state'] = 'disabled'

    def _unfreeze(self):
        self._button['state'] = 'active'

    def _on_start_recording(self):
        self._is_recording = True
        self._update_state('Recording...')
        self._recording = sd.rec(20 * RATE)

    def _truncate_recording(self):
        border = next((i for i, x in enumerate(reversed(self._recording)) if x), None)
        self._recording = self._recording[:len(self._recording) - border]

    def _display_prediction(self, prediction):
        self._display_result(Computer().compute(prediction))

    def _process(self):
        self._update_state('Processing...')
        mfcc = wav2mfcc(RATE, self._recording)
        prediction = self._model.predict(mfcc)
        self._display_prediction(prediction)
        self._update_state('Ready')

    def _on_stop_recording(self):
        self._freeze()
        sd.stop()
        self._is_recording = False
        self._truncate_recording()
        self._process()
        self._unfreeze()

    def _on_click(self):
        if self._is_recording:
            fred = threading.Thread(target=self._on_stop_recording)
            fred.daemon = True
            fred.start()
        else:
            self._on_start_recording()


if __name__ == "__main__":
    model = create_single_operation_recognizer(version='viterbi-')
    root = tk.Tk()
    root.title('ASR Calculator')
    App(root, model).pack(side="top", fill="both", expand=True)
    root.mainloop()
