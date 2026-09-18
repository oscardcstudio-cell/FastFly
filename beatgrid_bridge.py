"""Feed the fly's senses from the beatgrid sequencer (vj-rien/scripts/beatgrid).

Reads beatgrid's live session journal (sessions/*.jsonl) and its loop tags (tags.json).
Read-only on purpose: beatgrid's /ws hands each state to ONE reader and drains its energy
curve doing so, a second client would steal lines from Oscar's dashboard.

The kick is not here: the fly already hears the music through its ear (/stage audio).
The sequencer brings what sound cannot say: which loops play, and where the track is.

    python app_server.py --data flywire_v783.bin --port 8010 --beatgrid ../vj-rien
    python beatgrid_bridge.py        # self-check of the mapping, no GPU needed
"""
import asyncio, glob, json, os

# ---- The mapping. Artistic, not physiology: edit freely. (sense, amplitude, steps) ----
WARM = ("Temperature change", 0.5, 150)   # warm-coloured loop -> heat sensors
COLD = ("Humidity change", 0.5, 150)      # cold-coloured loop -> humidity sensors
BRIGHT = [("Light (left eye)", 0.15, 60), ("Light (right eye)", 0.15, 60)]  # bright loop -> the eye of its screen slot
SECTIONS = {
    "drop": ("Sugar (proboscis)", 0.8, 200),       # the reward
    "break": ("Odor (both antennae)", 0.4, 200),   # something in the air
}
MIN_SATURATION, MIN_BRIGHT = 0.15, 0.6


def senses_for(tags, slot=0):
    """Taps for one loop starting, from its colour tags (teinte 0-360, saturation, clarte 0-1)."""
    taps = []
    hue, sat = tags.get("teinte"), tags.get("saturation") or 0
    if hue is not None and sat >= MIN_SATURATION:
        if hue < 70 or hue >= 290:
            taps.append(WARM)
        elif 150 <= hue < 270:
            taps.append(COLD)
    if (tags.get("clarte") or 0) >= MIN_BRIGHT:
        taps.append(BRIGHT[slot % 2])
    return taps


def taps_for(event, tags_by_column):
    kind = event.get("kind")
    if kind in SECTIONS:
        return [SECTIONS[kind]]
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
    tags, tags_mtime, path, fh, last_columns = {}, 0, None, None, None
    while True:
        try:
            if os.path.getmtime(tags_path) != tags_mtime:
                tags_mtime, tags = os.path.getmtime(tags_path), load_tags(tags_path)
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
                    on_tap(*tap)
        except (OSError, ValueError) as exc:
            print(f"beatgrid bridge: {exc}", flush=True)
        await asyncio.sleep(poll)


if __name__ == "__main__":
    red = {"teinte": 11, "saturation": 0.4, "clarte": 0.3}
    blue = {"teinte": 210, "saturation": 0.5, "clarte": 0.8}
    grey = {"teinte": 11, "saturation": 0.05, "clarte": 0.2}
    assert senses_for(red) == [WARM]
    assert senses_for(blue, slot=1) == [COLD, BRIGHT[1]]
    assert senses_for(grey) == [] and senses_for({}) == []
    assert taps_for({"kind": "drop", "bar": 3}, {}) == [SECTIONS["drop"]]
    assert taps_for({"kind": "pattern", "colonnes": [5, 9, 404]}, {5: red, 9: blue}) == [WARM, COLD, BRIGHT[1]]
    assert taps_for({"kind": "tap"}, {}) == []
    print("mapping ok")
