# type: ignore
import logging
import time
from typing import Dict, Optional

import numpy as np

from .config import Config
from .modem import Receiver
from .signal_processor import SignalProcessor


class ContinuousReceiver:
    def __init__(self, config: Config, receiver: Receiver):
        self.config = config
        self.receiver = receiver
        self.buffer_size = int(config.Fs * 0.5)  # 0.5 second buffer
        self.signal_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.minimum_frame_samples = int(config.Fs * 0.1)  # Minimum samples for a valid frame

    def process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Process a new chunk of audio data."""
        # Roll the buffer and add new data
        self.signal_buffer = np.roll(self.signal_buffer, -len(chunk))
        self.signal_buffer[-len(chunk) :] = chunk

        # Try to detect and process frame
        detection = self.receiver.process_received_signal(
            self.signal_buffer, self.receiver.config.preamble_bits, self.receiver.config.sfd_bits
        )

        if detection:
            self._handle_detection(detection)
            # Clear the buffer after successful detection
            self.signal_buffer.fill(0)

    def _handle_detection(self, detection: Dict) -> None:
        """Handle a successful frame detection."""
        payload_hex = SignalProcessor.bits_to_hex(detection["payload_bits"])
        crc_hex = SignalProcessor.bits_to_hex(detection["received_crc"])

        print("\nFrame Detected!")
        print(f"Payload (HEX): {payload_hex}")
        print(f"CRC (HEX): {crc_hex}")

        # Convert payload to ASCII if possible
        try:
            payload_bytes = bytes.fromhex(payload_hex)
            ascii_text = payload_bytes.decode("ascii", errors="ignore")
            print(f"Payload (ASCII): {ascii_text}")
        except:
            pass
