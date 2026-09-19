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
# The climate of the sound (mixing vocabulary, artistic): a "warm" sound has its weight in the low mids,
# a "cold" one in the highs. What counts is the SHARE of the band in the whole, not its level: loud is not warm.
WARM_HZ, COLD_HZ = (200, 500), (4000, 16000)
EQ_EDGES = 30 * (16000 / 30) ** (np.arange(33) / 32)  # 32 log bands, 30 Hz - 16 kHz: what the pages draw


def band_levels(mag, samplerate):
    """mag: |rfft| / N of N Blackman-windowed samples (the Web Audio analyser's scaling)."""
    hz = samplerate / N
    out = {}
    for name, (lo, hi) in BANDS.items():
        power = mag[int(lo / hz): int(np.ceil(hi / hz)) + 1] ** 2
        out[name] = float(np.clip((10 * np.log10(power.mean() + 1e-12) + 80) / 60, 0, 1))
    return out


def share_db(mag, samplerate, band):
    """Power of a band over the power of everything, in dB (0 = the band is all there is)."""
    hz, power = samplerate / N, mag ** 2
    lo, hi = band
    return float(10 * np.log10((power[int(lo / hz): int(np.ceil(hi / hz)) + 1].sum() + 1e-15) / (power[1:].sum() + 1e-12)))


class Climate:
    """HEAT and COLD 0..1 from ONE balance, warm share minus cold share (dB): the two are compared, never both on.
    Read separately (first version) each sat around 0.5 on any music, so the heat was always firing (Oscar, live set).
    The balance is read against ITS recent low and high (they creep back ~0.5 dB/s): every track has its own range.
    The middle of the range is neutral: neither. Warm = warmer than this music usually is."""
    SPAN, CREEP, DEAD = 6.0, 0.02, 0.1  # dB: least range worth stretching (a steady sound has no climate); creep per push; neutral half-width

    def __init__(self):
        self.range = None

    def push(self, warm_db, cold_db, loudness):
        gate = float(np.clip((loudness - 0.25) / 0.15, 0, 1))  # silence has no climate (loudness = the JO-E level)
        if gate == 0:
            return {"HEAT": 0.0, "COLD": 0.0}
        v = warm_db - cold_db
        lo, hi = self.range or (v, v)
        lo, hi = self.range = min(v, lo + self.CREEP), max(v, hi - self.CREEP)
        t = (v - (lo + hi) / 2) / max(hi - lo, self.SPAN)  # -0.5 (coldest lately) .. +0.5 (warmest lately)
        side = lambda x: float(np.clip((x - self.DEAD) / (0.5 - self.DEAD), 0, 1)) * gate
        return {"HEAT": side(t), "COLD": side(-t)}


def spectrum(mag, samplerate):
    """32 log-band levels 0..1, same dB scale as band_levels."""
    hz = samplerate / N
    edges = np.clip((EQ_EDGES / hz).astype(int), 1, len(mag) - 1)
    power = mag ** 2
    return [float(np.clip((10 * np.log10(power[a:max(b, a + 1)].mean() + 1e-12) + 80) / 60, 0, 1)) for a, b in zip(edges[:-1], edges[1:])]


def start(on_levels):
    """Calls on_levels({group: 0..1, 'HEAT':, 'COLD':}, [32 band levels]) ~23 times/s from the audio thread. Returns a stop function."""
    import pyaudiowpatch as pa
    audio = pa.PyAudio()
    api = audio.get_host_api_info_by_type(pa.paWASAPI)
    speakers = audio.get_device_info_by_index(api["defaultOutputDevice"])
    device = next(d for d in audio.get_loopback_device_info_generator() if speakers["name"] in d["name"])
    sr, ch = int(device["defaultSampleRate"]), int(device["maxInputChannels"])
    window = np.blackman(N).astype(np.float32)
    smooth = np.zeros(N // 2 + 1, dtype=np.float32)
    weather = Climate()
    # WASAPI loopback sends nothing at all while the computer is silent: without this the ear would stay stuck on its last level
    last = [time.monotonic()]

    def callback(data, frames, time_info, status):
        x = np.frombuffer(data, dtype=np.float32).reshape(-1, ch).mean(1)[-N:]
        if len(x) == N:
            mag = np.abs(np.fft.rfft(x * window)) / N  # the browser analyser's scaling, so both routes read the same level
            smooth[:] = 0.6 * smooth + 0.4 * mag  # same smoothing as the browser analyser
            last[0] = time.monotonic()
            levels = band_levels(smooth, sr)
            levels.update(weather.push(share_db(smooth, sr, WARM_HZ), share_db(smooth, sr, COLD_HZ), levels["JO-E"]))
            on_levels(levels, spectrum(smooth, sr))
        return (None, pa.paContinue)

    stream = audio.open(format=pa.paFloat32, channels=ch, rate=sr, frames_per_buffer=N, input=True,
                        input_device_index=device["index"], stream_callback=callback)
    alive = threading.Event()

    def watchdog():
        while not alive.wait(0.2):
            if time.monotonic() - last[0] > 0.3:
                smooth[:] = 0
                on_levels({k: 0.0 for k in [*BANDS, "HEAT", "COLD"]}, [0.0] * 32)
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"audio in: {device['name']} @ {sr} Hz", flush=True)

    def stop():
        alive.set(); stream.close(); audio.terminate()
    return stop


if __name__ == "__main__":
    t = np.arange(N) / 48000
    lv = band_levels(np.abs(np.fft.rfft(np.sin(2 * np.pi * 60 * t) * np.blackman(N))) / N, 48000)
    assert lv["JO-B"] > 0.6 and lv["JO-C"] < 0.1, lv   # a full-scale 60 Hz sine is bass, not highs
    assert share_db(np.abs(np.fft.rfft(np.sin(2 * np.pi * 300 * t) * np.blackman(N))) / N, 48000, WARM_HZ) > -0.1  # a 300 Hz sine is all warmth
    w = Climate()
    for v in [-10.0] * 50 + [-3.0]:
        heat = w.push(v, -30.0, 1.0)["HEAT"]
    assert heat > 0.9 and w.push(-3.0, -30.0, 0.0)["HEAT"] == 0.0  # warmer than usual = hot; silence = nothing
    w = Climate()
    for v in [-10.0] * 50:
        calm = w.push(v, -20.0, 1.0)
    assert calm == {"HEAT": 0.0, "COLD": 0.0}, calm  # a steady balance is neutral: the heat no longer fires all the time
    cold = w.push(-10.0, -12.0, 1.0)
    assert cold["COLD"] > 0.9 and cold["HEAT"] == 0.0, cold  # highs rising = cold, and never both at once
    assert len(spectrum(np.ones(N // 2 + 1, dtype=np.float32), 48000)) == 32
    stop = start(lambda l, s: print(" ".join(f"{k} {v:.2f}" for k, v in l.items()), end="\r", flush=True))
    time.sleep(5); stop(); print()
