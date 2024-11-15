# Chirp Modem Simulator

A Python-based simulator for chirp modulation communication systems. This project implements a complete modem system using chirp signals for binary data transmission, including frame synchronization, payload handling, and error detection.

## Features

- Configurable chirp-based modulation
- Frame synchronization using preamble and Start Frame Delimiter (SFD)
- CRC32 error detection
- Configurable noise simulation
- Detailed signal visualization and analysis tools
- Support for WAV file input/output
- Multiple verbosity levels for debugging

## Project Structure

```
chirp-64/
├── pyproject.toml    # Poetry configuration
├── src/
|   ├── chirp_64/      # Main package directory
|   │   ├── __init__.py
|   │   ├── __main__.py   # Entry point
|   │   ├── config.py     # Configuration parameters and validation
|   │   ├── signal_processor.py # Signal processing utilities
|   │   ├── modem.py      # Transmitter, Receiver, and Channel classes
|   │   └── plotter.py    # Visualization tools
├── tests/            # Test directory
│   └── __init__.py
└── README.md         # This file
```

## Requirements

- Python 3.8+
- Poetry for dependency management

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/chirp-64.git
cd chirp-64
```

3. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Basic Usage

With Poetry, run the module using:

Generate and transmit a new signal:
```bash
poetry run chirp-64 -o output.wav
```

Process an existing WAV file:
```bash
poetry run chirp-64 -i input.wav
```

### Running Directly from Source

Alternatively, you can run using Python module syntax:
```bash
poetry run python -m chirp_64 -o output.wav
```

### Verbosity Levels

The program supports multiple verbosity levels:
- Default: Only warnings and errors
- `-v`: Basic info (SNR, BER, status messages)
- `-vv`: Info + signal plots
- `-vvv`: Debug info + all plots including correlations

### Configuration Parameters

All major parameters can be configured via command-line arguments:

```bash
poetry run chirp-64 -o output.wav \
    --fs 48000 \            # Sampling frequency (Hz)
    --baud-rate 200 \       # Symbols per second
    --f-start 2000 \        # Start frequency for chirps (Hz)
    --f-end 2400 \          # End frequency for chirps (Hz)
    --amplitude 0.8 \       # Chirp amplitude
    --noise-power 0.5 \     # Noise power
    --preamble "11001100" \ # Preamble bit sequence
    --sfd "11111111" \      # Start Frame Delimiter sequence
    --payload-length 128 \  # Number of payload bits
    --max-bits 1000        # Maximum bits to annotate in plots
```

### Example Commands

1. Generate signal with default parameters:
```bash
poetry run chirp-64 -o output.wav
```

2. Generate and visualize signal:
```bash
poetry run chirp-64 -vv -o output.wav
```

3. Process input file with custom parameters:
```bash
poetry run chirp-64 -v -i input.wav --baud-rate 200 --f-start 2000 --f-end 2400
```

4. Full debug mode with all plots:
```bash
poetry run chirp-64 -vvv -o output.wav
```

## Development

### Setting up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chirp-64.git
cd chirp-64
```

2. Install development dependencies:
```bash
poetry install --with dev
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

Format code using black:
```bash
poetry run black chirp_64
```

## Signal Structure

The transmitted frame consists of:
1. Preamble (8 bits): For frame detection
2. Start Frame Delimiter (8 bits): For frame synchronization
3. Payload Size Field (16 bits): Indicates payload length
4. Payload (variable length): User data
5. CRC32 (32 bits): Error detection

## Output Files

- WAV files containing the modulated signal
- Plot visualizations (when using -vv or -vvv):
  - Transmitted signal with bit annotations
  - Received signal with frame structure markers
  - Correlation plots for frame detection

## Logging Output

The program provides different levels of logging:

- Transmission data (payload, CRC)
- Reception data (received payload, CRC)
- Signal-to-Noise Ratio (SNR)
- Bit Error Rate (BER)
- Frame detection status
- Various debug information at higher verbosity levels

## Advanced Usage

### Custom Frame Parameters

```bash
poetry run chirp-64 -o output.wav \
    --preamble "10101010" \
    --sfd "11110000" \
    --payload-length 128
```

### Noise Testing

```bash
poetry run chirp-64 -o output.wav --noise-power 2.0
```

### High-Speed Communication

```bash
poetry run chirp-64 -o output.wav --baud-rate 200 --fs 48000
```

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`poetry install --with dev`)
4. Make your changes
5. Run tests (`poetry run pytest`)
6. Format code (`poetry run black chirp_64`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

MIT License