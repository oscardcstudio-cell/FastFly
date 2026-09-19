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


def channel_level(x):
    """RMS loudness of one channel's block, -80..-20 dB -> 0..1 (same scale as band_levels)."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)
    return float(np.clip((20 * np.log10(rms) + 80) / 60, 0, 1))


def loom_onset(level, baseline, follow=0.08, gain=3.0):
    """Positive jump of `level` over its own slow-following baseline: something looming in fast.
    A steady tone settles to onset 0 (baseline catches up); a transient onsets near 1 first.
    ponytail: gain=3.0 is a first guess for "moderate", tune from /params once heard."""
    if baseline is None:
        return 0.0, level
    onset = float(np.clip((level - baseline) * gain, 0, 1))
    return onset, baseline + (level - baseline) * follow


def band_energy(bands01):
    """Mean level (0..1, same dB scale) of a slice of the 32-band spectrum."""
    return float(np.mean(bands01)) if len(bands01) else 0.0


def spectrum(mag, samplerate):
    """32 log-band levels 0..1, same dB scale as band_levels."""
    hz = samplerate / N
    edges = np.clip((EQ_EDGES / hz).astype(int), 1, len(mag) - 1)
    power = mag ** 2
    return [float(np.clip((10 * np.log10(power[a:max(b, a + 1)].mean() + 1e-12) + 80) / 60, 0, 1)) for a, b in zip(edges[:-1], edges[1:])]


def start(on_levels):
    """Calls on_levels({group: 0..1, 'HEAT':, 'COLD':}, [32 band levels]) ~23 times/s from the audio thread. Returns a stop function."""
    import pyaudiowpatch as pa
    window = np.blackman(N).astype(np.float32)
    smooth = np.zeros(N // 2 + 1, dtype=np.float32)
    weather = Climate()
    loom_base = {"L": None, "R": None}  # each channel's own slow-following floor, for loom_onset
    # WASAPI loopback sends nothing at all while the computer is silent: without this the ear would stay stuck on its last level
    last = [time.monotonic()]

    now = {}  # the PyAudio, stream and device being listened to: replaced when Windows' default output changes

    def callback(data, frames, time_info, status):
        sr = now["sr"]
        buf = np.frombuffer(data, dtype=np.float32).reshape(-1, now["ch"])
        x = buf.mean(1)[-N:]
        if len(x) == N:
            mag = np.abs(np.fft.rfft(x * window)) / N  # the browser analyser's scaling, so both routes read the same level
            smooth[:] = 0.6 * smooth + 0.4 * mag  # same smoothing as the browser analyser
            last[0] = time.monotonic()
            levels = band_levels(smooth, sr)
            levels.update(weather.push(share_db(smooth, sr, WARM_HZ), share_db(smooth, sr, COLD_HZ), levels["JO-E"]))
            # Looming test (Oscar, 2026-09-19; EQ split 2026-09-19): "vision approche" in EQ mode, not stereo —
            # a bass peak looms at the left eye, a treble peak looms at the right eye (Oscar: "je voudrais à
            # gauche les eq de bass et à droite les eq de treble"). Same 32-band spectrum the pages already draw,
            # split in half: no channel to read for a mono source, but the two halves still differ.
            spec = spectrum(smooth, sr)
            bass_lvl, treble_lvl = band_energy(spec[:16]), band_energy(spec[16:])
            on_l, loom_base["L"] = loom_onset(bass_lvl, loom_base["L"])
            on_r, loom_base["R"] = loom_onset(treble_lvl, loom_base["R"])
            levels["LOOM_L"], levels["LOOM_R"] = on_l, on_r
            on_levels(levels, spec)
        return (None, pa.paContinue)

    def default_output(audio):
        return audio.get_device_info_by_index(audio.get_host_api_info_by_type(pa.paWASAPI)["defaultOutputDevice"])["name"]

    def listen():
        audio = pa.PyAudio()  # a fresh one each time: PortAudio reads the device list once, at init
        name = default_output(audio)
        device = next(d for d in audio.get_loopback_device_info_generator() if name in d["name"])
        now.update(audio=audio, name=name, sr=int(device["defaultSampleRate"]), ch=int(device["maxInputChannels"]))
        now["stream"] = audio.open(format=pa.paFloat32, channels=now["ch"], rate=now["sr"], frames_per_buffer=N, input=True,
                                   input_device_index=device["index"], stream_callback=callback)
        print(f"audio in: {device['name']} @ {now['sr']} Hz", flush=True)

    def close():
        try:
            now["stream"].close(); now["audio"].terminate()
        except Exception:
            pass  # a device that vanished may refuse to close: the next listen() starts clean anyway
        now["name"] = None  # so a failed listen() is tried again at the next look

    listen()
    alive = threading.Event()

    def watchdog():
        checked = time.monotonic()
        while not alive.wait(0.2):
            if time.monotonic() - last[0] > 0.3:
                smooth[:] = 0
                on_levels({k: 0.0 for k in [*BANDS, "HEAT", "COLD", "LOOM_L", "LOOM_R"]}, [0.0] * 32)
            # Headphones plugged in mid-set: Windows moves the sound, the fly kept listening to the silent speakers
            # (Oscar, 2026-09-19). Only looked at during silence, every 3 s: a playing output is the right one.
            if time.monotonic() - last[0] > 3 and time.monotonic() - checked > 3:
                checked = time.monotonic()
                try:
                    probe = pa.PyAudio(); name = default_output(probe); probe.terminate()
                    if name != now["name"]:
                        close(); listen()
                except Exception as e:
                    print(f"audio in: {e}", flush=True)
    threading.Thread(target=watchdog, daemon=True).start()

    def stop():
        alive.set(); close()
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
    assert channel_level(np.zeros(N, dtype=np.float32)) < 0.05 and channel_level(np.ones(N, dtype=np.float32)) > 0.9
    on0, base0 = loom_onset(0.5, None)
    assert on0 == 0.0 and base0 == 0.5  # first call only seeds the baseline
    on1, base1 = loom_onset(0.9, base0)
    assert on1 > 0.9 and base0 < base1 < 0.9  # a sudden jump onsets near-max, baseline creeps toward it
    on2, _ = loom_onset(0.5, 0.5)
    assert on2 == 0.0  # steady level after settling = no onset
    stop = start(lambda l, s: print(" ".join(f"{k} {v:.2f}" for k, v in l.items()), end="\r", flush=True))
    time.sleep(5); stop(); print()
