"""Measure which neurons each musical input lights up, for the explanatory board.

Offline (own engine, not the live server: open tabs rewrite the shared engine's params).
    .venv/Scripts/python docs/planche/make_data.py   ->  static/planche/data.json
Each input = one tap (200 steps on), then 400 steps free. noise 0, gain 0.8, fatigue default.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
import cupy as cp
from sim_engine import SimEngine

INPUTS = [  # (id, what triggers it in the music, how it reaches the fly, source, amplitude)
    ("basses",  "Basses, kick (20-150 Hz)",       "oreille",    "JO-B", 1.0),
    ("mediums", "Médiums (150-2000 Hz)",          "oreille",    "JO-A", 1.0),
    ("aigus",   "Aigus, hi-hats (2-16 kHz)",      "oreille",    "JO-C", 1.0),
    ("volume",  "Volume global",                  "oreille",    "JO-E", 1.0),
    ("chaud",   "Loop de couleur chaude",         "séquenceur", "Temperature change", 0.5),
    ("froid",   "Loop de couleur froide",         "séquenceur", "Humidity change", 0.5),
    ("clair_g", "Loop claire, écran 1",           "séquenceur", "Light (left eye)", 0.15),
    ("clair_d", "Loop claire, écran 2",           "séquenceur", "Light (right eye)", 0.15),
    ("drop",    "Drop",                           "séquenceur", "Sugar (proboscis)", 0.8),
    ("break",   "Break",                          "séquenceur", "Odor (both antennae)", 0.4),
]

e = SimEngine(data_file="flywire_v783.bin")
e.set_noise_amp(0); e.set_weight_gain(0.8); e.frame_every = 5
e.send_active_indices = e.send_group_rates = e.send_motor_rates = False
ann = np.load("neuron_annotations.npz", allow_pickle=True)
cls = {k: np.asarray(ann[k]).astype(str) for k in ("super_class", "cell_class", "cell_type")}

# same view as /stage: X mirrored, centred, scaled by the largest half-extent; front view = (x, y)
P = e._positions.copy()
c = (P.min(0) + P.max(0)) / 2; s = ((P.max(0) - P.min(0)) / 2).max()
xy = np.stack([-(P[:, 0] - c[0]) / s, (P[:, 1] - c[1]) / s], 1)
pts = lambda idx: np.round(xy[idx], 3).tolist()

out = {"n_neurons": int(e.n_neurons), "background": pts(np.arange(0, e.n_neurons, 6)), "inputs": []}
for key, label, via, source, amp in INPUTS:
    src = e._audio[source][0].get() if source in e._audio else e._stimuli[source]
    e.reset_state(); e.inject_stimulus(src.tolist(), amp)
    seen = collections.Counter()
    for batch in range(3):
        if batch == 1: e.clear_stimulus()
        for f in e.step(200)["frames"]: seen.update(f["s"])
    down = np.array(sorted(set(seen) - set(src.tolist())), dtype=int)
    top = lambda col: collections.Counter(x for x in cls[col][down] if x).most_common(6)
    out["inputs"].append({"id": key, "label": label, "via": via, "source_name": source,
                          "n_source": int(len(src)), "n_downstream": int(len(down)),
                          "source": pts(src[:: max(1, len(src) // 1500)]), "downstream": pts(down),
                          "top_super_class": top("super_class"), "top_cell_class": top("cell_class"), "top_cell_type": top("cell_type")})
    print(f"{key:8s} source {len(src):6d} -> {len(down):6d}  {top('cell_class')[:4]}", flush=True)

json.dump(out, open("static/planche/data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("written", os.path.getsize("static/planche/data.json") // 1024, "KB")
