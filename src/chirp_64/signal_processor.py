# type: ignore
import numpy as np


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
