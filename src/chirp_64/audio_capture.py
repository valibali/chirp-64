import pyaudio
import numpy as np
import queue
import logging
import wave
from datetime import datetime
from typing import Optional, Tuple

class AudioCapture:
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 4410):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.int16)
        self.recorded_data = []
        
        self.device_index = self._find_input_device()
        if self.device_index is None:
            raise RuntimeError("No input audio device found")

    def _find_input_device(self) -> Optional[int]:
        info = self.audio.get_default_input_device_info()
        if info is not None:
            logging.info(f"Using input device: {info['name']}")
            return info['index']
        
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                logging.info(f"Using input device: {info['name']}")
                return i
        return None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_queue.put(audio_data)
        self.recorded_data.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self) -> None:
        self.is_recording = True
        self.recorded_data = []
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            logging.info("Started audio capture")
        except Exception as e:
            logging.error(f"Failed to start audio capture: {e}")
            self.is_recording = False
            raise

    def stop_recording(self) -> None:
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = None
        self.save_recording()
        self.audio.terminate()
        logging.info("Stopped audio capture")

    def get_audio_chunk(self) -> Tuple[bool, np.ndarray]:
        try:
            chunk = self.audio_queue.get_nowait()
            # Convert int16 to float32 normalized between -1 and 1
            chunk_float = chunk.astype(np.float32) / 32768.0
            return True, chunk_float
        except queue.Empty:
            return False, np.array([], dtype=np.float32)

    def clear_queue(self) -> None:
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def save_recording(self) -> str:
        if not self.recorded_data:
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_audio_{timestamp}.wav"
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.recorded_data))
        
        logging.info(f"Recording saved to {filename}")
        return filename