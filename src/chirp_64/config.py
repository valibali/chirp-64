# type: ignore
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    """Configuration parameters for the simulation."""

    Fs: int = 44100  # Sampling frequency (Hz)
    desired_baud_rate: int = 100  # Symbols per second
    T = 1 / desired_baud_rate
    f_start: int = 1000  # Start frequency for chirps (Hz)
    f_end: int = 1200  # End frequency for chirps (Hz)
    A: float = 1.0  # Amplitude of chirps
    noise_power: float = 1  # Noise power
    preamble: str = "10101010"  # 8 bits
    preamble_bits = np.array([int(bit) for bit in preamble], dtype=int)
    sfd: str = "11110000"  # 8 bits
    sfd_bits = np.array([int(bit) for bit in sfd], dtype=int)
    payload_length: int = 64  # Number of payload bits
    max_bits: int = 500  # Max bits to annotate in plots

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.Fs > 0, "Sampling frequency must be positive."
        assert self.desired_baud_rate > 0, "Baud rate must be positive."
        assert self.f_end > self.f_start, "End frequency must be greater than start frequency."
        assert self.payload_length > 0, "Payload length must be positive."
        assert 0 <= self.noise_power, "Noise power cannot be negative."
        assert isinstance(self.preamble, str), "Preamble must be a string of bits."
        assert isinstance(self.sfd, str), "SFD must be a string of bits."
        assert isinstance(self.sfd, str), "SFD must be a string of bits."
