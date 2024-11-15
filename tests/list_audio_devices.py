import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("\nAvailable Audio Devices:")
    print("-" * 60)
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"Index {i}: {dev['name']}")
            print(f"    Input channels: {dev['maxInputChannels']}")
            print(f"    Sample rates: {int(dev['defaultSampleRate'])}") 
            print()
    
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()