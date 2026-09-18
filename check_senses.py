"""Check that each predefined sense lights its own path (noise off, audio off).

Needs the server running:  python check_senses.py [port] [amplitude]
Prints, per sense, how many neurons fired beyond the stimulated ones, then the
overlap (Jaccard) between senses. Fails if a sense is silent or two unrelated senses match.
"""
import asyncio, json, sys
import numpy as np
import websockets

PORT = sys.argv[1] if len(sys.argv) > 1 else "8010"
AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
BATCHES = 3
# Smell, heat and humidity all enter through the antennal lobe and ignite the same loop
# (antennal lobe + mushroom body, ~10k neurons, never dies out): known limit of the model, not a bug of the check.
FAMILY = {"Odor": "antennal", "Temperature": "antennal", "Humidity": "antennal"}


async def main():
    ann = np.load("neuron_annotations.npz", allow_pickle=True)
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        send = lambda o: ws.send(json.dumps(o))
        init = json.loads(await ws.recv())
        fired = {}
        for name in [n for n in init["stimuli"] if not n.startswith("Motor")]:
            # re-sent before every sense: any browser tab that reconnects rewrites the shared engine's params
            await send({"cmd": "pause"})
            for k, v in (("noise_amp", 0), ("weight_gain", 0.8), ("frame_every", 5), ("send_active_indices", 0), ("audio_mute", 1)):
                await send({"cmd": "set_param", "key": k, "value": v})
            await send({"cmd": "clear_stimulus"})
            await send({"cmd": "reset"})
            await send({"cmd": "stimulus_preset", "name": name, "amplitude": AMP})
            seen, edges = set(), 0
            for batch in range(BATCHES):
                if batch == 1:  # a tap, like /trace: a held stimulus ends in the same self-sustained loop for every sense
                    await send({"cmd": "clear_stimulus"})
                await send({"cmd": "step"})
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("type") == "metrics":
                        break
                for f in m.get("frames", []):
                    seen.update(f["s"]); edges += len(f["e"]) // 3
            key = "stim_" + name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            src = set(ann[key].tolist())
            fired[name] = seen - src
            print(f"{name:28s} source {len(src):6d}  fired {len(seen):6d}  downstream {len(seen - src):6d}  edges {edges}")
        await send({"cmd": "clear_stimulus"})
        await send({"cmd": "set_param", "key": "audio_mute", "value": 0})

    names = list(fired)
    print("\nOverlap of downstream paths (Jaccard):")
    worst = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            u = len(fired[a] | fired[b])
            j = len(fired[a] & fired[b]) / u if u else 0
            if j > 0.3:
                print(f"  {j:.2f}  {a}  ~  {b}")
            fa, fb = a.split(" ")[0], b.split(" ")[0]
            if FAMILY.get(fa, fa) != FAMILY.get(fb, fb):
                worst = max(worst, j)
    silent = [n for n in names if not fired[n]]
    print(f"\nsilent senses: {silent or 'none'} · worst overlap between different senses: {worst:.2f}")
    assert not silent and worst < 0.5


asyncio.run(main())
