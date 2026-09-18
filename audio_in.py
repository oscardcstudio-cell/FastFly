"""The fly hears what the computer plays: WASAPI loopback of the default output, no browser, no mic.

Same route as the beatgrid dashboard (pyaudiowpatch, like OBS). Band levels follow the pages'
formula (average power per band, -80..-20 dB -> 0..1) so the brain reacts the same either way.
    python audio_in.py      # prints the four band levels for 5 s: play something and watch
"""
import threading, time
import numpy as np

# Artistic mapping, not physiology: bass -> JO-B, mids -> JO-A, highs -> JO-C, loudness -> JO-E
BANDS = {"JO-B": (20, 150), "JO-A": (150, 2000), "JO-C": (2000, 16000), "JO-E": (20, 16000)}
N = 2048


def band_levels(mag, samplerate):
    """mag: |rfft| / N of N Blackman-windowed samples (the Web Audio analyser's scaling)."""
    hz = samplerate / N
    out = {}
    for name, (lo, hi) in BANDS.items():
        power = mag[int(lo / hz): int(np.ceil(hi / hz)) + 1] ** 2
        out[name] = float(np.clip((10 * np.log10(power.mean() + 1e-12) + 80) / 60, 0, 1))
    return out


def start(on_levels):
    """Calls on_levels({group: 0..1}) ~23 times/s from the audio thread. Returns a stop function."""
    import pyaudiowpatch as pa
    audio = pa.PyAudio()
    api = audio.get_host_api_info_by_type(pa.paWASAPI)
    speakers = audio.get_device_info_by_index(api["defaultOutputDevice"])
    device = next(d for d in audio.get_loopback_device_info_generator() if speakers["name"] in d["name"])
    sr, ch = int(device["defaultSampleRate"]), int(device["maxInputChannels"])
    window = np.blackman(N).astype(np.float32)
    smooth = np.zeros(N // 2 + 1, dtype=np.float32)
    # WASAPI loopback sends nothing at all while the computer is silent: without this the ear would stay stuck on its last level
    last = [time.monotonic()]

    def callback(data, frames, time_info, status):
        x = np.frombuffer(data, dtype=np.float32).reshape(-1, ch).mean(1)[-N:]
        if len(x) == N:
            mag = np.abs(np.fft.rfft(x * window)) / N  # the browser analyser's scaling, so both routes read the same level
            smooth[:] = 0.6 * smooth + 0.4 * mag  # same smoothing as the browser analyser
            last[0] = time.monotonic()
            on_levels(band_levels(smooth, sr))
        return (None, pa.paContinue)

    stream = audio.open(format=pa.paFloat32, channels=ch, rate=sr, frames_per_buffer=N, input=True,
                        input_device_index=device["index"], stream_callback=callback)
    alive = threading.Event()

    def watchdog():
        while not alive.wait(0.2):
            if time.monotonic() - last[0] > 0.3:
                smooth[:] = 0
                on_levels({k: 0.0 for k in BANDS})
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"audio in: {device['name']} @ {sr} Hz", flush=True)

    def stop():
        alive.set(); stream.close(); audio.terminate()
    return stop


if __name__ == "__main__":
    t = np.arange(N) / 48000
    lv = band_levels(np.abs(np.fft.rfft(np.sin(2 * np.pi * 60 * t) * np.blackman(N))) / N, 48000)
    assert lv["JO-B"] > 0.6 and lv["JO-C"] < 0.1, lv   # a full-scale 60 Hz sine is bass, not highs
    stop = start(lambda l: print(" ".join(f"{k} {v:.2f}" for k, v in l.items()), end="\r", flush=True))
    time.sleep(5); stop(); print()
