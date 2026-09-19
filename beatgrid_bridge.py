"""Feed the fly's senses from the beatgrid sequencer (vj-rien/scripts/beatgrid).

Reads beatgrid's live session journal (sessions/*.jsonl) and its loop tags (tags.json).
Read-only on purpose: beatgrid's /ws hands each state to ONE reader and drains its energy
curve doing so, a second client would steal lines from Oscar's dashboard.

The kick is not here: the fly already hears the music through its ear (/stage audio).
The sequencer brings what sound cannot say: which loops play, and where the track is.

    python app_server.py --data flywire_v783.bin --port 8010 --beatgrid ../vj-rien
    python app_server.py ... --beatgrid ../vj-rien --replay ../vj-rien/sessions/<set>.jsonl   # a past set, no music needed
    python beatgrid_bridge.py        # self-check of the mapping, no GPU needed
"""
import asyncio, glob, json, os, time, urllib.request

# ---- The mapping. Artistic, not physiology: edit freely. (sense, amplitude, steps) ----
# Heat and humidity are NOT here any more (Oscar, 2026-09-19): they come from the sound, all the time
# (audio_in.Climate: low mids = warm, highs = cold). These two taps only serve the pages' by-hand buttons.
WARM = ("Temperature change", 0.5, 150)
COLD = ("Humidity change", 0.5, 150)
BRIGHT = [("Light (left eye)", 0.15, 60), ("Light (right eye)", 0.15, 60)]  # bright loop -> the eye of its screen slot
MOVING = [("Touch (left leg/body)", 0.4, 100), ("Touch (right leg/body)", 0.4, 100)]    # loop with a lot of motion -> touch, side of its slot
FAST = [("Tickle (left antenna)", 0.4, 100), ("Tickle (right antenna)", 0.4, 100)]      # loop tagged "rapide" -> wind on the antenna of its slot
SECTIONS = {
    "drop": [("Sugar (proboscis)", 0.8, 200)],       # the reward
    "break": [("Odor (both antennae)", 0.4, 200)],   # something in the air
    "suspension": FAST,                               # breath held: wind on both antennae
}
# What the pages show for each sense: (taps, French label, what fires it). /params lists them as buttons; two taps = left, right.
LABELS = [([WARM], "chaleur", "bas-médiums 200-500 Hz"), ([COLD], "humidité", "aigus 4-16 kHz"),
          (BRIGHT, "œil", "loop claire"), (MOVING, "toucher", "loop mobile"),
          (FAST, "vent antenne", "loop rapide · suspension"),
          (SECTIONS["drop"], "sucre", "drop"), (SECTIONS["break"], "odeur", "break")]


def senses_table():
    return [{"label": fr, "when": when, "taps": [{"name": n, "amplitude": a, "steps": st} for n, a, st in taps]}
            for taps, fr, when in LABELS]


# thresholds read on Oscar's 240 tagged loops: brightest quarter (max 0.57), most moving quarter
MIN_BRIGHT, MIN_MOTION = 0.31, 4.16


def senses_for(tags, slot=0):
    """Taps for one loop starting, from its tags (clarte 0-1, mouvement, vitesse)."""
    taps = []
    if (tags.get("clarte") or 0) >= MIN_BRIGHT:
        taps.append(BRIGHT[slot % 2])
    if (tags.get("mouvement") or 0) >= MIN_MOTION:
        taps.append(MOVING[slot % 2])
    if tags.get("vitesse") == "rapide":
        taps.append(FAST[slot % 2])
    return taps


def taps_for(event, tags_by_column):
    kind = event.get("kind")
    if kind in SECTIONS:
        return list(SECTIONS[kind])
    if kind == "pattern":
        return [t for slot, col in enumerate(event.get("colonnes") or [])
                for t in senses_for(tags_by_column.get(col, {}), slot)]
    return []


def load_tags(path):
    with open(path, encoding="utf-8") as fh:
        clips = json.load(fh).get("clips", {})
    return {c["column"]: c for c in clips.values() if isinstance(c.get("column"), int)}


async def follow(beatgrid_root, on_tap, poll=0.03):
    """Tail the newest session journal forever; history is skipped, only what happens now counts."""
    tags_path = os.path.join(beatgrid_root, "scripts", "beatgrid", "tags.json")
    tags, tags_mtime, path, fh, last_columns, newest, next_scan = {}, 0, None, None, None, None, 0
    while True:
        try:
            if os.path.getmtime(tags_path) != tags_mtime:
                tags_mtime, tags = os.path.getmtime(tags_path), load_tags(tags_path)
            if time.monotonic() >= next_scan:  # thousands of journals pile up there: never stat them all at every poll
                next_scan = time.monotonic() + 2
                newest = max(glob.glob(os.path.join(beatgrid_root, "sessions", "*.jsonl")), key=os.path.getmtime, default=None)
            if newest != path:  # beatgrid restarted: new journal
                if fh: fh.close()
                path, fh = newest, open(newest, encoding="utf-8") if newest else None
                if fh: fh.seek(0, os.SEEK_END)
            while fh:
                pos = fh.tell()
                line = fh.readline()
                if not line.endswith("\n"):
                    fh.seek(pos); break  # nothing new, or a half-written line: next round
                event = json.loads(line)
                if event.get("kind") == "pattern":  # journaled at every re-trigger: tap only when the loops change
                    if event.get("colonnes") == last_columns: continue
                    last_columns = event.get("colonnes")
                for tap in taps_for(event, tags):
                    on_tap(*tap, event=event.get("kind"))
        except (OSError, ValueError) as exc:
            print(f"beatgrid bridge: {exc}", flush=True)
        await asyncio.sleep(poll)


async def watch_state(url, on_state, on_tap, poll=0.1):
    """What the dashboard READS in the music (its break/drop advice, BPM), from beatgrid's /etat.
    The journal only has the sections actually fired; the reading exists all the time, like on Oscar's screen."""
    fetch = lambda: json.load(urllib.request.urlopen(url, timeout=1))
    last = None
    while True:
        try:
            state = await asyncio.get_running_loop().run_in_executor(None, fetch)
        except (OSError, ValueError):
            state = {}
        if state.get("t", 0) < time.time() - 3:
            state = {}  # beatgrid only beats while its dashboard is open: a stale state is no state
        on_state(state)
        advice = state.get("advice")
        if advice != last and advice in SECTIONS and last is not None:
            for tap in SECTIONS[advice]:
                on_tap(*tap, event=advice)
        last = advice if state else last
        await asyncio.sleep(poll)


async def replay(journal, beatgrid_root, on_tap, speed=1.0):
    """Play a past session journal again with its own timing, to watch the fly without a live set."""
    tags = load_tags(os.path.join(beatgrid_root, "scripts", "beatgrid", "tags.json"))
    while True:
        prev, last_columns = None, None
        with open(journal, encoding="utf-8") as fh:
            for line in fh:
                event = json.loads(line)
                if event.get("kind") == "pattern":
                    if event.get("colonnes") == last_columns: continue
                    last_columns = event.get("colonnes")
                taps = taps_for(event, tags)
                if not taps: continue
                t = event.get("t", 0)  # gaps clamped: the journal mixes two clocks and has long silences
                await asyncio.sleep(min(3.0, max(0.0, (t - prev) / speed)) if prev is not None else 0)
                prev = t
                for tap in taps:
                    on_tap(*tap, event=event.get("kind"))


if __name__ == "__main__":
    red = {"teinte": 11, "saturation": 0.4, "clarte": 0.3}
    blue = {"teinte": 210, "saturation": 0.5, "clarte": 0.5}
    grey = {"teinte": 11, "saturation": 0.05, "clarte": 0.2}
    assert senses_for(red) == []  # colour no longer warms the fly: the sound does
    assert senses_for(blue, slot=1) == [BRIGHT[1]]
    assert senses_for(grey) == [] and senses_for({}) == []
    assert taps_for({"kind": "drop", "bar": 3}, {}) == SECTIONS["drop"]
    assert taps_for({"kind": "suspension"}, {}) == FAST
    assert senses_for({"mouvement": 9.0, "vitesse": "rapide"}, slot=1) == [MOVING[1], FAST[1]]
    assert taps_for({"kind": "pattern", "colonnes": [5, 9, 404]}, {5: red, 9: blue}) == [BRIGHT[1]]
    assert taps_for({"kind": "tap"}, {}) == []
    print("mapping ok")
