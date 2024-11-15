# type: ignore
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import correlate

from .config import Config
from .signal_processor import SignalProcessor


class Transmitter:
    def __init__(self, config: Config):
        self.config = config
        self.T = 1 / config.desired_baud_rate
        self.N = int(config.Fs / config.desired_baud_rate)
        self.t = np.linspace(0, self.T, self.N, endpoint=False)

        # Convert bits to arrays
        self.payload_bits = np.random.randint(0, 2, config.payload_length, dtype=int)
        self.payload_size_field_bits = np.array(
            [int(x) for x in format(config.payload_length, "016b")], dtype=int
        )
        self.crc_bits = SignalProcessor.crc32(self.payload_bits)
        self.data_bits = np.concatenate(
            [
                self.config.preamble_bits,
                self.config.sfd_bits,
                self.payload_size_field_bits,
                self.payload_bits,
                self.crc_bits,
            ]
        )

        # Log transmitted data
        logging.info("=== Transmission Data ===")
        logging.info(f"Payload (Hex): {SignalProcessor.bits_to_hex(self.payload_bits)}")
        logging.info(f"CRC (Hex): {SignalProcessor.bits_to_hex(self.crc_bits)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                f"Preamble (Hex): {SignalProcessor.bits_to_hex(self.config.preamble_bits)}"
            )
            logging.debug(f"SFD (Hex): {SignalProcessor.bits_to_hex(self.config.sfd_bits)}")
            logging.debug(
                f"Payload Size Field (Hex): {SignalProcessor.bits_to_hex(self.payload_size_field_bits)}"
            )

    def generate_frame(self) -> Tuple[np.ndarray, List[Tuple[float, int]]]:
        frame_signal = np.array([], dtype=float)
        bit_times = []
        current_time = 0
        for bit in self.data_bits:
            chirp_signal = (SignalProcessor.up_chirp if bit == 1 else SignalProcessor.down_chirp)(
                self.t, self.config.f_start, self.config.f_end, self.T, self.config.A
            )
            frame_signal = np.concatenate((frame_signal, chirp_signal))
            bit_times.append((current_time, bit))
            current_time += self.T
        return frame_signal, bit_times


class Receiver:
    def __init__(self, config: Config):
        self.config = config
        self.T = 1 / config.desired_baud_rate
        self.N = int(config.Fs / config.desired_baud_rate)
        self.t = np.linspace(0, self.T, self.N, endpoint=False)

        # Precompute chirp signals
        self.h_up = SignalProcessor.up_chirp(self.t, config.f_start, config.f_end, self.T, config.A)
        self.h_down = SignalProcessor.down_chirp(
            self.t, config.f_start, config.f_end, self.T, config.A
        )

        self.preamb_threshold = 0.95
        self.sfd_threshold = 0.95

    def detect_preamble(
        self, received_signal: np.ndarray, preamble_bits: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], np.ndarray]:
        preamble_sequence = self._bits_to_signal(preamble_bits)
        corr = correlate(received_signal, preamble_sequence, mode="valid")
        peak_index = np.argmax(corr)
        corr_normalized = corr / np.max(corr) if np.max(corr) != 0 else corr

        if corr_normalized[peak_index] > self.preamb_threshold:
            peak_time = peak_index / self.config.Fs
            logging.debug(f"Preamble detected at index: {peak_index}, time: {peak_time:.6f}s")
            return peak_index, peak_time, corr_normalized
        else:
            logging.warning("Preamble not detected")
            return None, None, corr_normalized

    def detect_sfd(
        self, received_signal: np.ndarray, sfd_bits: np.ndarray, search_start: int
    ) -> Tuple[Optional[int], Optional[float], np.ndarray]:
        sfd_sequence = self._bits_to_signal(sfd_bits)
        search_window_length = len(sfd_sequence) * 2
        search_window = received_signal[search_start : search_start + search_window_length]
        corr = correlate(search_window, sfd_sequence, mode="valid")
        peak_index = np.argmax(corr)
        corr_normalized = corr / np.max(corr) if np.max(corr) != 0 else corr

        if corr_normalized[peak_index] > self.sfd_threshold:
            absolute_peak_index = search_start + peak_index
            peak_time = absolute_peak_index / self.config.Fs
            logging.debug(f"SFD detected at index: {absolute_peak_index}, time: {peak_time:.6f}s")
            return absolute_peak_index, peak_time, corr_normalized
        else:
            logging.warning("SFD not detected")
            return None, None, corr_normalized

    def _bits_to_signal(self, bits: np.ndarray) -> np.ndarray:
        signal = np.array([], dtype=float)
        for bit in bits:
            chirp_signal = (SignalProcessor.up_chirp if bit == 1 else SignalProcessor.down_chirp)(
                self.t, self.config.f_start, self.config.f_end, self.T, self.config.A
            )
            signal = np.concatenate((signal, chirp_signal))
        return signal

    def extract_bits(
        self, received_signal: np.ndarray, start_index: int, num_bits: int
    ) -> np.ndarray:
        extracted_bits = []
        for i in range(num_bits):
            idx_start = start_index + i * self.N
            idx_end = idx_start + self.N
            if idx_end > len(received_signal):
                logging.warning("Incomplete bit segment detected")
                break
            symbol = received_signal[idx_start:idx_end]
            corr_u = np.correlate(symbol, self.h_up, mode="valid")[0]
            corr_d = np.correlate(symbol, self.h_down, mode="valid")[0]
            extracted_bits.append(1 if corr_u > corr_d else 0)
        return np.array(extracted_bits, dtype=int)

    def process_received_signal(
        self, received_signal: np.ndarray, preamble: np.ndarray, sfd: np.ndarray
    ) -> Optional[Dict]:
        preamble_index, preamble_peak_time, corr_preamble_normalized = self.detect_preamble(
            received_signal, preamble
        )
        if preamble_index is None:
            return None

        sfd_result = self.detect_sfd(
            received_signal, sfd, preamble_index + len(self._bits_to_signal(preamble))
        )
        if sfd_result[0] is None:
            return None
        sfd_index, sfd_peak_time, corr_sfd_normalized = sfd_result

        frame_start = sfd_index + len(self._bits_to_signal(sfd))
        payload_size_bits = self.extract_bits(received_signal, frame_start, 16)
        if len(payload_size_bits) < 16:
            logging.warning("Incomplete payload size field")
            return None

        payload_size = int("".join(payload_size_bits.astype(str)), 2)
        logging.info(f"Detected payload size: {payload_size} bits")

        total_message_bits = payload_size + 32
        message_bits = self.extract_bits(
            received_signal, frame_start + 16 * self.N, total_message_bits
        )
        if len(message_bits) < total_message_bits:
            logging.warning("Incomplete payload/CRC received")
            return None

        payload_bits_received = message_bits[:payload_size]
        received_crc = message_bits[payload_size:]

        logging.info("=== Reception Data ===")
        logging.info(
            f"Received Payload (Hex): {SignalProcessor.bits_to_hex(payload_bits_received)}"
        )
        logging.info(f"Received CRC (Hex): {SignalProcessor.bits_to_hex(received_crc)}")

        return {
            "preamble_peak_time": preamble_peak_time,
            "sfd_peak_time": sfd_peak_time,
            "payload_bits": payload_bits_received,
            "received_crc": received_crc,
            "payload_size": payload_size,
            "frame_start": frame_start,
            "preamble_length": len(preamble),
            "sfd_length": len(sfd),
        }


class Channel:
    def __init__(self, config: Config):
        self.config = config

    def transmit(self, signal: np.ndarray) -> np.ndarray:
        noise = np.sqrt(self.config.noise_power) * np.random.randn(len(signal))
        return signal + noise


class TextTransmitter(Transmitter):
    """Extended Transmitter class that supports custom text payloads."""

    def __init__(self, config: Config, custom_payload: Optional[np.ndarray] = None):
        super().__init__(config)
        if custom_payload is not None:
            self.payload_bits = custom_payload
            self.crc_bits = SignalProcessor.crc32(self.payload_bits)
            # Rebuild data_bits with new payload
            self.data_bits = np.concatenate(
                [
                    self.config.preamble_bits,
                    self.config.sfd_bits,
                    self.payload_size_field_bits,
                    self.payload_bits,
                    self.crc_bits,
                ]
            )
