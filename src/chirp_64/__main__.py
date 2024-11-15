# type: ignore
import argparse
import logging
from typing import Optional

import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import correlate

from .config import Config
from .modem import Channel, Receiver, Transmitter
from .plotter import Plotter
from .signal_processor import SignalProcessor


def setup_logging(verbosity: int) -> bool:
    """
    Set up logging with different verbosity levels:
    0 = WARNING and above
    1 (-v) = INFO and above
    2 (-vv) = Detailed INFO
    3 (-vvv) = DEBUG and above
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity == 2:
        level = logging.INFO
    else:  # verbosity >= 3
        level = logging.DEBUG

    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    return verbosity >= 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chirp Modem Simulation")
    # Verbosity and I/O arguments
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v, -vv, or -vvv)",
    )
    parser.add_argument("-i", "--input", type=str, help="Input WAV file (optional)")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output WAV file")

    # Signal parameters
    parser.add_argument(
        "--fs", type=int, default=44100, help="Sampling frequency in Hz (default: 44100)"
    )
    parser.add_argument(
        "--baud-rate", type=int, default=100, help="Symbols per second (default: 100)"
    )
    parser.add_argument(
        "--f-start", type=int, default=1000, help="Start frequency for chirps in Hz (default: 1000)"
    )
    parser.add_argument(
        "--f-end", type=int, default=1200, help="End frequency for chirps in Hz (default: 1200)"
    )
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="Amplitude of chirps (default: 1.0)"
    )
    parser.add_argument("--noise-power", type=float, default=1.0, help="Noise power (default: 1.0)")
    parser.add_argument(
        "--preamble", type=str, default="10101010", help="Preamble bit sequence (default: 10101010)"
    )
    parser.add_argument(
        "--sfd",
        type=str,
        default="11110000",
        help="Start Frame Delimiter bit sequence (default: 11110000)",
    )
    parser.add_argument(
        "--payload-length", type=int, default=64, help="Number of payload bits (default: 64)"
    )
    parser.add_argument(
        "--max-bits", type=int, default=500, help="Maximum bits to annotate in plots (default: 500)"
    )

    args = parser.parse_args()

    # Validate bit sequences
    for field, value in [("preamble", args.preamble), ("sfd", args.sfd)]:
        if not all(bit in "01" for bit in value):
            parser.error(f"{field} must be a binary string (only 0s and 1s)")

    # Validate numeric parameters
    if args.fs <= 0:
        parser.error("Sampling frequency must be positive")
    if args.baud_rate <= 0:
        parser.error("Baud rate must be positive")
    if args.f_end <= args.f_start:
        parser.error("End frequency must be greater than start frequency")
    if args.payload_length <= 0:
        parser.error("Payload length must be positive")
    if args.noise_power < 0:
        parser.error("Noise power cannot be negative")
    if args.max_bits <= 0:
        parser.error("Max bits must be positive")

    return args


def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create a Config object from parsed command line arguments."""
    return Config(
        Fs=args.fs,
        desired_baud_rate=args.baud_rate,
        f_start=args.f_start,
        f_end=args.f_end,
        A=args.amplitude,
        noise_power=args.noise_power,
        preamble=args.preamble,
        sfd=args.sfd,
        payload_length=args.payload_length,
        max_bits=args.max_bits,
    )


def process_input_signal(config: Config, args: argparse.Namespace):
    """Process input WAV file."""
    sample_rate, received_signal = read(args.input)
    received_signal = received_signal.astype(float) / 32767.0
    logging.info(f"Reading signal from {args.input}")
    return received_signal, None, None


def generate_new_signal(
    config: Config, transmitter: Transmitter, channel: Channel, args: argparse.Namespace
):
    """Generate and transmit new signal."""
    frame_signal, bit_times = transmitter.generate_frame()
    signal_power = np.mean(frame_signal**2)
    noisy_signal = channel.transmit(frame_signal)

    # Calculate and log SNR
    snr = signal_power / config.noise_power
    snr_db = 10 * np.log10(snr)
    logging.info(f"Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")

    # Save generated signal
    s_tx_int = SignalProcessor.normalize_signal(frame_signal)
    write(args.output, config.Fs, s_tx_int)
    logging.info(f"Transmitted signal saved to {args.output}")

    # Save generated noisy signal
    s_tx_int = SignalProcessor.normalize_signal(noisy_signal)
    write("noisy_" + args.output, config.Fs, s_tx_int)
    logging.info(f"Transmitted signal saved to {args.output}")

    return noisy_signal, frame_signal, bit_times


def process_detection_results(
    detection,
    transmitter,
    frame_signal,
    bit_times,
    received_signal,
    config: Config,
    plotter: Optional[Plotter],
    args: argparse.Namespace,
):
    """Process and visualize detection results."""
    if not detection:
        logging.error("Failed to detect a complete frame in the received signal")
        return

    # Calculate and log BER if we have original signal
    if not args.input:
        original_payload = transmitter.payload_bits
        received_payload = detection["payload_bits"]
        num_errors = np.sum(original_payload != received_payload)
        ber = num_errors / len(original_payload)
        logging.info(f"Bit Error Rate (BER): {ber:.6f} ({num_errors}/{len(original_payload)} bits)")

    # Generate plots if in verbose mode (-vv or higher)
    if plotter and args.verbose >= 2:
        plot_signals(
            plotter, detection, transmitter, frame_signal, bit_times, received_signal, config, args
        )


def plot_signals(
    plotter,
    detection,
    transmitter,
    frame_signal,
    bit_times,
    received_signal,
    config: Config,
    args: argparse.Namespace,
):
    """Generate all plots based on verbosity level."""
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
            time_axis, frame_signal, bit_times, "Transmitted Signal with Bits", bits=bits_to_plot
        )

    # Plot received signal
    plotter.plot_received_with_markers(
        np.arange(len(received_signal)) / config.Fs,
        received_signal,
        detection,
        bits=bits_to_plot if not args.input else None,
    )

    # Plot correlations (only for -vvv)
    if args.verbose >= 3:
        plot_correlations(plotter, received_signal, transmitter, detection)


def plot_correlations(plotter, received_signal, transmitter, detection):
    """Generate correlation plots for highest verbosity level."""
    receiver = Receiver(plotter.config)
    preamble_corr = correlate(
        received_signal, receiver._bits_to_signal(transmitter.preamble_bits), mode="valid"
    )
    sfd_corr = correlate(
        received_signal, receiver._bits_to_signal(transmitter.sfd_bits), mode="valid"
    )

    # Normalize correlations
    preamble_corr_norm = (
        preamble_corr / np.max(preamble_corr) if np.max(preamble_corr) != 0 else preamble_corr
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


def main():
    args = parse_arguments()
    show_plots = setup_logging(args.verbose)

    # Create config from command line arguments
    config = create_config_from_args(args)

    # Initialize components
    transmitter = Transmitter(config)
    channel = Channel(config)
    receiver = Receiver(config)
    plotter = Plotter(config) if show_plots else None

    # Process input or generate new signal
    if args.input:
        received_signal, frame_signal, bit_times = process_input_signal(config, args)
    else:
        received_signal, frame_signal, bit_times = generate_new_signal(
            config, transmitter, channel, args
        )

    # Process received signal
    detection = receiver.process_received_signal(
        received_signal, transmitter.preamble_bits, transmitter.sfd_bits
    )

    # Process and visualize results
    process_detection_results(
        detection, transmitter, frame_signal, bit_times, received_signal, config, plotter, args
    )


if __name__ == "__main__":
    main()
    main()
