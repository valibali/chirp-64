# type: ignore
import logging
import queue
from typing import Optional, Tuple

import numpy as np
import pyaudio


class AudioCapture:
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 4410):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)

    def start_recording(self) -> None:
        """Start recording from the default input device."""
        self.is_recording = True
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()
        logging.info("Started audio capture")

    def stop_recording(self) -> None:
        """Stop recording and clean up."""
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = None
        self.audio.terminate()
        logging.info("Stopped audio capture")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream, puts data in queue."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def get_audio_chunk(self) -> Tuple[bool, np.ndarray]:
        """Get a chunk of audio data from the queue."""
        try:
            chunk = self.audio_queue.get_nowait()
            return True, chunk
        except queue.Empty:
            return False, np.array([], dtype=np.float32)

    def clear_queue(self) -> None:
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
