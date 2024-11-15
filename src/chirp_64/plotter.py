# type: ignore
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


class Plotter:
    """Handles plotting of signals and correlation results."""

    def __init__(self, config: Config):
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

    def plot_received_with_markers(self, time_axis, received_signal, detection_info, bits=None):
        """Plot the received signal with markers and annotated bits."""
        plt.figure(figsize=(15, 6))
        plt.plot(time_axis, received_signal, label="Received Signal")

        if detection_info and bits is not None and len(bits) <= self.config.max_bits:
            T = 1 / self.config.desired_baud_rate  # Get symbol period from baud rate

            # Calculate timing boundaries based on actual baud rate
            preamble_start = detection_info["preamble_peak_time"]
            preamble_duration = detection_info["preamble_length"] * T
            preamble_end = preamble_start + preamble_duration

            sfd_start = detection_info["sfd_peak_time"]
            sfd_duration = detection_info["sfd_length"] * T
            sfd_end = sfd_start + sfd_duration

            size_start = sfd_end
            size_duration = 16 * T  # 16 bits for payload size
            size_end = size_start + size_duration

            payload_start = size_end
            payload_duration = detection_info["payload_size"] * T
            payload_end = payload_start + payload_duration

            crc_start = payload_end
            crc_duration = 32 * T  # 32 bits for CRC
            crc_end = crc_start + crc_duration

            # Calculate total frame duration for plot limits
            total_duration = crc_end - preamble_start

            # Adjust x-axis limits to show complete frame
            plt.xlim(max(0, preamble_start - total_duration * 0.1), crc_end + total_duration * 0.1)

            # Highlight regions
            plt.axvspan(preamble_start, preamble_end, color="green", alpha=0.2, label="Preamble")
            plt.axvspan(sfd_start, sfd_end, color="blue", alpha=0.2, label="SFD")
            plt.axvspan(size_start, size_end, color="purple", alpha=0.2, label="Payload Size")
            plt.axvspan(payload_start, payload_end, color="orange", alpha=0.2, label="Payload")
            plt.axvspan(crc_start, crc_end, color="red", alpha=0.2, label="CRC")

            # Annotate bits if within limit
            if bits is not None:
                # Calculate bit positions based on actual frame timing
                frame_duration = crc_end - preamble_start
                bits_per_frame = len(bits)
                bit_duration = frame_duration / bits_per_frame

                for i, bit in enumerate(bits):
                    bit_time = preamble_start + i * bit_duration
                    plt.text(
                        bit_time + bit_duration / 2,
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
        plt.show()
