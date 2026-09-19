"""
FlyWire Connectome Simulator — Web Server

Usage:
    pip install fastapi uvicorn[standard]
    python app_server.py                        # synthetic data
    python app_server.py --data flywire_v783.bin
"""

import argparse
import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sim_engine import SimEngine
from fly_says import FlySays
from audio_in import Climate

parser = argparse.ArgumentParser(description="FlyWire Simulator Web Server")
parser.add_argument("--data", help="Binary connectome file")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--audio", action="store_true", help="The fly hears what the computer plays (WASAPI loopback), no browser needed")
parser.add_argument("--beatgrid-url", default="http://127.0.0.1:8765", help="Where the beatgrid dashboard listens")
parser.add_argument("--replay", help="With --beatgrid: play this past session journal (.jsonl) instead of the live one")
parser.add_argument("--beatgrid", help="Path to the vj-rien repo: its sequencer feeds the fly senses")
args = parser.parse_args()

print("\nInitializing simulation engine...")
engine = SimEngine(data_file=args.data)
print(f"Engine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses\n")

app = FastAPI(title="FlyWire Simulator")

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

sim_running = False
batch_size = 50  # the ear's level and the fly's reading change once a batch: 200 was two a second under load, slower than the beat (measured 2026-09-19)
fly_says = FlySays()
mic_weather = Climate()
clients: list[WebSocket] = []


audio_gain = 1.5  # 1.0 only lit the brain on the big kicks (Oscar, 2026-09-19); the pages' fader goes to 4
climate_gain = {"HEAT": 0.5, "COLD": 0.5}  # how hard the sound's warmth and coldness push their sense (the pages' faders)
audio_levels = {}
audio_spectrum = []


def hear(levels):
    """Band and climate levels 0..1 -> the engine, each with its own gain."""
    engine.set_audio({k: v * climate_gain.get(k, audio_gain) for k, v in levels.items()})


@app.on_event("startup")
async def start_audio_in():
    if not args.audio:
        return
    import audio_in

    def on_levels(levels, spectrum):  # audio thread: set_audio only stores floats
        audio_levels.update(levels)
        audio_spectrum[:] = spectrum
        hear(levels)
    app.state.audio_stop = audio_in.start(on_levels)  # keep it: a dropped stream is garbage-collected and goes silent

    async def show():  # the pages' meters, so Oscar sees the sound arrive
        while True:
            await broadcast({"type": "audio_levels", "levels": audio_levels, "spectrum": audio_spectrum})
            await asyncio.sleep(0.1)
    asyncio.create_task(show())


@app.on_event("startup")
async def start_beatgrid_bridge():
    if args.beatgrid:
        from beatgrid_bridge import follow, replay, watch_state
        import time
        last_section = {}

        def on_tap(name, amplitude, steps, event=None):
            if event != "pattern":  # a section comes both from the dashboard's reading and from its journal: once is enough
                if time.monotonic() - last_section.get(name, 0) < 2:
                    return
                last_section[name] = time.monotonic()
            print(f"beatgrid {event} -> {name}", flush=True)
            engine.tap(name, amplitude, steps)
            # the pages show it, so Oscar can check what the fly was told against what he hears
            asyncio.create_task(broadcast({"type": "beatgrid", "event": event, "sense": name}))
        if not args.replay:
            def on_state(st):
                if st.get("bpm"):  # the fly reads its bars in beats: it needs the length of one
                    fly_says.period = 60 / st["bpm"]
                asyncio.create_task(broadcast({"type": "beatgrid_state", "state": st}))
            asyncio.create_task(watch_state(args.beatgrid_url + "/etat", on_state, on_tap))
        asyncio.create_task(replay(args.replay, args.beatgrid, on_tap) if args.replay
                            else follow(args.beatgrid, on_tap))


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/stage")
async def stage():
    return FileResponse(os.path.join(static_dir, "stage.html"))


@app.get("/trace")
async def trace():
    return FileResponse(os.path.join(static_dir, "trace.html"))


@app.get("/params")
async def params_page():
    return FileResponse(os.path.join(static_dir, "params.html"))


weight_gain = 0.8  # the engine only keeps the scaled weights, not the factor
engine.set_weight_gain(weight_gain)  # calm at rest from the start, before any page sets it


@app.get("/api/params")
async def get_params():
    """Current knobs, so the settings window opens on the real values."""
    return JSONResponse({"audio_gain": audio_gain, "weight_gain": weight_gain,
                         "fly_step": fly_says.step, "adapt": float(engine.adapt_inc), "noise_amp": float(engine.noise_amp),
                         "heat_gain": climate_gain["HEAT"], "cold_gain": climate_gain["COLD"]})


def _anchors(idx):
    """One real neuron per side of the brain, the closest to where this group sits: where /stage points its label.
    A side holding under a fifth of the group is ignored (a stray cell is not a zone)."""
    import numpy as np
    pos = engine._positions
    if pos is None or not len(idx):
        return []
    idx = np.asarray(idx, dtype=np.int64)
    left = pos[idx, 0] < (pos[:, 0].min() + pos[:, 0].max()) / 2
    out = []
    for side in (idx[left], idx[~left]):
        if len(side) >= len(idx) / 5:
            out.append(int(side[np.argmin(((pos[side] - pos[side].mean(0)) ** 2).sum(1))]))
    return out


@app.get("/api/senses")
async def get_senses():
    """The sequencer-to-sense table with where each sense enters the brain, for /params, /stage and the dashboard."""
    from beatgrid_bridge import senses_table
    table = senses_table()
    for sense in table:
        for tap in sense["taps"]:
            tap["anchors"] = _anchors(engine._stimuli.get(tap["name"], []))
    ear = [i for k, v in engine._audio.items() if k.startswith("JO-") for i in v[0].get().tolist()]
    table.append({"label": "oreille", "when": "son", "taps": [{"name": "ear", "anchors": _anchors(ear)}]})
    return JSONResponse(table)


@app.get("/api/positions")
async def get_positions():
    """Return neuron 3D positions + class info for the brain visualizer."""
    pos_b64 = engine.get_positions_b64()
    classes = engine.get_neuron_classes()
    return JSONResponse({
        "n_neurons": engine.n_neurons,
        "positions_b64": pos_b64,
        "classes": classes,
    })


async def broadcast(msg: dict):
    data = json.dumps(msg)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


async def sim_loop():
    global sim_running
    while sim_running:
        try:
            metrics = await asyncio.get_event_loop().run_in_executor(
                None, engine.step, batch_size
            )
            metrics["type"] = "metrics"
            if "bass_readout" in metrics:
                metrics["fly_says"] = fly_says.push(metrics["bass_readout"])
            # Cap active_indices to limit WebSocket payload size
            ai = metrics.get("active_indices", [])
            if len(ai) > 5000:
                # even stride, so capping doesn't bias the viz toward low indices
                metrics["active_indices"] = ai[::-(-len(ai) // 5000)]
            await broadcast(metrics)
            await asyncio.sleep(0)
        except Exception as e:
            print(f"sim_loop error: {e}", flush=True)
            sim_running = False
            await broadcast({"type": "state", "running": False})
            break


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global sim_running, batch_size, audio_gain, weight_gain

    await ws.accept()
    clients.append(ws)

    await ws.send_text(json.dumps({
        "type": "init",
        "n_neurons": engine.n_neurons,
        "n_synapses": engine.n_synapses,
        "stimuli": engine.get_predefined_stimuli(),
        "group_labels": engine.group_labels,
        "body_info": engine.get_body_info(),
        "audio_groups": engine.get_audio_groups(),
    }))
    await ws.send_text(json.dumps({
        "type": "state",
        "running": sim_running,
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "start":
                if not sim_running:
                    sim_running = True
                    await broadcast({"type": "state", "running": True})
                    asyncio.create_task(sim_loop())

            elif cmd == "pause":
                sim_running = False
                await broadcast({"type": "state", "running": False})

            elif cmd == "step":
                sim_running = False
                metrics = await asyncio.get_event_loop().run_in_executor(
                    None, engine.step, batch_size
                )
                metrics["type"] = "metrics"
                await broadcast(metrics)

            elif cmd == "stimulus":
                indices = msg.get("indices", [])
                amplitude = float(msg.get("amplitude", 0.5))
                engine.inject_stimulus(indices, amplitude)

            elif cmd == "stimulus_preset":
                name = msg.get("name", "")
                amplitude = float(msg.get("amplitude", 0.5))
                engine.apply_predefined_stimulus(name, amplitude)

            elif cmd == "audio":
                engine.set_audio(msg.get("amps", {}))
                if "climate" in msg:  # a page listening through a microphone sends its two shares: same reading as the loopback
                    levels = mic_weather.push(*(float(v) for v in msg["climate"][:3]))
                    hear(levels)
                    await broadcast({"type": "audio_levels", "levels": levels})

            elif cmd == "reset":
                engine.reset_state()

            elif cmd == "tap":
                if engine.tap(msg.get("name", ""), float(msg.get("amplitude", 0.5)), int(msg.get("steps", 150))):
                    await broadcast({"type": "beatgrid", "event": "manual", "sense": msg.get("name")})  # every page shows a sense fired by hand too

            elif cmd == "clear_stimulus":
                engine.clear_stimulus()

            elif cmd == "set_param":
                key = msg.get("key")
                value = msg.get("value")
                if key == "noise_amp":
                    engine.set_noise_amp(float(value))
                elif key == "frame_every":
                    engine.frame_every = max(0, int(value))
                elif key == "adapt":
                    engine.set_adapt(float(value))
                elif key == "audio_gain":
                    audio_gain = max(0.0, min(4.0, float(value)))
                elif key in ("heat_gain", "cold_gain"):
                    climate_gain[key[:4].upper()] = max(0.0, min(1.5, float(value)))
                elif key == "fly_step":
                    fly_says.step = max(0.5, min(12.0, float(value)))
                elif key == "audio_mute":
                    engine.audio_mute = bool(value)
                elif key == "weight_gain":
                    weight_gain = float(value)
                    engine.set_weight_gain(weight_gain)
                elif key == "batch_size":
                    batch_size = max(1, min(500, int(value)))
                elif key == "send_active_indices":
                    engine.send_active_indices = bool(value)
                elif key == "send_group_rates":
                    engine.send_group_rates = bool(value)
                elif key == "send_motor_rates":
                    engine.send_motor_rates = bool(value)

            elif cmd == "motor_detail":
                group_name = msg.get("group", "")
                detail = engine.get_motor_detail(group_name)
                if detail:
                    await ws.send_text(json.dumps({
                        "type": "motor_detail",
                        **detail,
                    }))

    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
