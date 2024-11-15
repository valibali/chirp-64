import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import correlate


@dataclass
class Config:
    """Configuration parameters for the simulation."""

    Fs: int = 44100  # Sampling frequency (Hz)
    desired_baud_rate: int = 100  # Symbols per second
    f_start: int = 1000  # Start frequency for chirps (Hz)
    f_end: int = 1200  # End frequency for chirps (Hz)
    A: float = 1.0  # Amplitude of chirps
    noise_power: float = 1  # Noise power
    preamble: str = "10101010"  # 8 bits
    sfd: str = "11110000"  # 8 bits
    payload_length: int = 64  # Number of payload bits
    max_bits: int = 500  # Max bits to annotate in plots

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.Fs > 0, "Sampling frequency must be positive."
        assert self.desired_baud_rate > 0, "Baud rate must be positive."
        assert self.f_end > self.f_start, "End frequency must be greater than start frequency."
        assert self.payload_length > 0, "Payload length must be positive."
        assert 0 <= self.noise_power, "Noise power cannot be negative."
        assert isinstance(self.preamble, str), "Preamble must be a string of bits."
        assert isinstance(self.sfd, str), "SFD must be a string of bits."


class SignalProcessor:
    @staticmethod
    def up_chirp(t, f_start, f_end, T, A):
        k = (f_end - f_start) / T
        return A * np.cos(2 * np.pi * (f_start * t + 0.5 * k * t**2))

    @staticmethod
    def down_chirp(t, f_start, f_end, T, A):
        k = (f_end - f_start) / T
        return A * np.cos(2 * np.pi * (f_end * t - 0.5 * k * t**2))

    @staticmethod
    def crc32(data_bits):
        poly = 0x04C11DB7
        crc = 0xFFFFFFFF
        for bit in data_bits:
            crc ^= bit << 31
            for _ in range(8):
                if crc & 0x80000000:
                    crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
                else:
                    crc = (crc << 1) & 0xFFFFFFFF
        return np.array([int(x) for x in format(crc ^ 0xFFFFFFFF, "032b")], dtype=int)

    @staticmethod
    def bits_to_hex(bits):
        padding_length = (8 - len(bits) % 8) % 8
        if padding_length > 0:
            bits = np.concatenate([bits, np.zeros(padding_length, dtype=int)])
        byte_bits = bits.reshape(-1, 8)
        return "".join(format(int("".join(byte.astype(str)), 2), "02X") for byte in byte_bits)

    @staticmethod
    def normalize_signal(signal):
        max_abs = np.max(np.abs(signal))
        if max_abs == 0:
            max_abs = 1
        normalized = signal / max_abs
        return np.int16(normalized * 32767)


class Transmitter:
    def __init__(self, config: Config):
        self.config = config
        self.T = 1 / config.desired_baud_rate
        self.N = int(config.Fs / config.desired_baud_rate)
        self.t = np.linspace(0, self.T, self.N, endpoint=False)

        # Convert bits to arrays
        self.preamble_bits = np.array([int(bit) for bit in config.preamble], dtype=int)
        self.sfd_bits = np.array([int(bit) for bit in config.sfd], dtype=int)
        self.payload_bits = np.random.randint(0, 2, config.payload_length, dtype=int)
        self.payload_size_field_bits = np.array(
            [int(x) for x in format(config.payload_length, "016b")], dtype=int
        )
        self.crc_bits = SignalProcessor.crc32(self.payload_bits)
        self.data_bits = np.concatenate(
            [
                self.preamble_bits,
                self.sfd_bits,
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
            logging.debug(f"Preamble (Hex): {SignalProcessor.bits_to_hex(self.preamble_bits)}")
            logging.debug(f"SFD (Hex): {SignalProcessor.bits_to_hex(self.sfd_bits)}")
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
            if logging.getLogger().isEnabledFor(logging.DEBUG):
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
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    f"SFD detected at index: {absolute_peak_index}, time: {peak_time:.6f}s"
                )
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


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chirp Modem Simulation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-i", "--input", type=str, help="Input WAV file (optional)")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output WAV file")
    return parser.parse_args()


class Plotter:
    """Handles plotting of signals and correlation results."""

    def __init__(self, config):
        self.config = config

    def plot_correlation(self, corr_normalized, peak_time, threshold, title):
        """Plot normalized cross-correlation."""
        lags = np.arange(len(corr_normalized))
        corr_time = lags / self.config.Fs

        plt.figure(figsize=(15, 4))
        plt.plot(corr_time, corr_normalized, label="Normalized Correlation")
        plt.axvline(x=peak_time, color="red", linestyle="--", label="Peak")
        plt.axhline(y=threshold, color="green", linestyle="--", label=f"Threshold ({threshold})")
        plt.title(title)
        plt.xlabel("Time Lag (s)")
        plt.ylabel("Normalized Correlation")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_signal_with_bits(self, time_axis, signal, bit_times, title, bits=None):
        """Plot a signal with annotated bits."""
        plt.figure(figsize=(15, 6))
        plt.plot(time_axis, signal, label="Signal")

        for start_time, bit in bit_times:
            plt.axvspan(start_time, start_time + self.config.T, color="yellow", alpha=0.1)

        if bits is not None and len(bits) <= self.config.max_bits:
            for (start_time, _), bit in zip(bit_times, bits):
                plt.text(
                    start_time + self.config.T / 2,
                    self.config.A * 1.1,
                    str(bit),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    clip_on=True,
                )

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.ylim(min(signal) * 1.2, max(signal) * 1.2)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_received_with_markers(
        self, time_axis, received_signal, detection_info, bits=None
    ) -> None:
        """Plot the received signal with markers and annotated bits."""
        plt.figure(figsize=(15, 6))
        plt.plot(time_axis, received_signal, label="Received Signal")

        if detection_info and bits is not None and len(bits) <= self.config.max_bits:
            # Calculate timing boundaries
            preamble_start = detection_info["preamble_peak_time"]
            preamble_duration = detection_info["preamble_length"] * self.config.T
            preamble_end = preamble_start + preamble_duration

            sfd_start = detection_info["sfd_peak_time"]
            sfd_duration = detection_info["sfd_length"] * self.config.T
            sfd_end = sfd_start + sfd_duration

            size_start = sfd_end
            size_duration = 16 * self.config.T
            size_end = size_start + size_duration

            payload_start = size_end
            payload_duration = detection_info["payload_size"] * self.config.T
            payload_end = payload_start + payload_duration

            crc_start = payload_end
            crc_duration = 32 * self.config.T
            crc_end = crc_start + crc_duration

            # Highlight regions
            plt.axvspan(preamble_start, preamble_end, color="green", alpha=0.2, label="Preamble")
            plt.axvspan(sfd_start, sfd_end, color="blue", alpha=0.2, label="SFD")
            plt.axvspan(size_start, size_end, color="purple", alpha=0.2, label="Payload Size")
            plt.axvspan(payload_start, payload_end, color="orange", alpha=0.2, label="Payload")
            plt.axvspan(crc_start, crc_end, color="red", alpha=0.2, label="CRC")

            # Annotate bits if within limit
            bit_start_times = np.linspace(preamble_start, crc_end, len(bits))
            for bit_time, bit in zip(bit_start_times, bits):
                plt.text(
                    bit_time + self.config.T / 2,
                    self.config.A * 1.1,
                    str(bit),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    clip_on=True,
                )

        plt.title("Received Signal with Markers")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.ylim(min(received_signal) * 1.2, max(received_signal) * 1.2)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()


def main() -> None:
    args = parse_arguments()
    setup_logging(args.verbose)

    config = Config()
    config.__post_init__()

    # Initialize components
    transmitter = Transmitter(config)
    channel = Channel(config)
    receiver = Receiver(config)
    plotter = Plotter(config) if args.verbose else None

    if args.input:
        # Read from WAV file
        sample_rate, received_signal = read(args.input)
        received_signal = received_signal.astype(float) / 32767.0
        logging.info(f"Reading signal from {args.input}")
        frame_signal = None
        bit_times = None
    else:
        # Generate and transmit new frame
        frame_signal, bit_times = transmitter.generate_frame()
        signal_power = np.mean(frame_signal**2)
        received_signal = channel.transmit(frame_signal)

        # Calculate and log SNR
        snr = signal_power / config.noise_power
        snr_db = 10 * np.log10(snr)
        logging.info(f"Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")

        # Save transmitted signal
        s_tx_int = SignalProcessor.normalize_signal(frame_signal)
        write(args.output, config.Fs, s_tx_int)
        logging.info(f"Transmitted signal saved to {args.output}")

    # Process received signal
    detection = receiver.process_received_signal(
        received_signal, transmitter.preamble_bits, transmitter.sfd_bits
    )

    if detection:
        # Calculate and log BER
        if not args.input:  # Only if we have the original signal
            original_payload = transmitter.payload_bits
            received_payload = detection["payload_bits"]
            num_errors = np.sum(original_payload != received_payload)
            ber = num_errors / len(original_payload)
            logging.info(
                f"Bit Error Rate (BER): {ber:.6f} ({num_errors}/{len(original_payload)} bits)"
            )

        # Generate plots if in verbose mode
        if args.verbose and plotter:
            # Combine all bits for plotting
            if not args.input:
                bits_to_plot = np.concatenate(
                    [
                        transmitter.preamble_bits,
                        transmitter.sfd_bits,
                        transmitter.payload_size_field_bits,
                        detection["payload_bits"],
                        detection["received_crc"],
                    ]
                )
                if len(bits_to_plot) > config.max_bits:
                    bits_to_plot = bits_to_plot[: config.max_bits]
                    logging.debug(f"Truncated plot to first {config.max_bits} bits")

                # Plot transmitted signal
                time_axis = np.arange(len(frame_signal)) / config.Fs
                plotter.plot_signal_with_bits(
                    time_axis,
                    frame_signal,
                    bit_times,
                    "Transmitted Signal with Bits",
                    bits=bits_to_plot,
                )

            # Plot received signal
            plotter.plot_received_with_markers(
                np.arange(len(received_signal)) / config.Fs,
                received_signal,
                detection,
                bits=bits_to_plot if not args.input else None,
            )

            # Plot correlations
            preamble_corr = correlate(
                received_signal, receiver._bits_to_signal(transmitter.preamble_bits), mode="valid"
            )
            sfd_corr = correlate(
                received_signal, receiver._bits_to_signal(transmitter.sfd_bits), mode="valid"
            )

            # Normalize correlations
            preamble_corr_norm = (
                preamble_corr / np.max(preamble_corr)
                if np.max(preamble_corr) != 0
                else preamble_corr
            )
            sfd_corr_norm = sfd_corr / np.max(sfd_corr) if np.max(sfd_corr) != 0 else sfd_corr

            plotter.plot_correlation(
                preamble_corr_norm,
                detection["preamble_peak_time"],
                receiver.preamb_threshold,
                "Normalized Cross-Correlation for Preamble Detection",
            )

            plotter.plot_correlation(
                sfd_corr_norm,
                detection["sfd_peak_time"],
                receiver.sfd_threshold,
                "Normalized Cross-Correlation for SFD Detection",
            )
    else:
        logging.error("Failed to detect a complete frame in the received signal")


if __name__ == "__main__":
    main()
