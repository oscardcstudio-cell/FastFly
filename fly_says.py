"""What the fly makes of the music: break or drop, read from its own neurons.

The signal: the share of neurons wired directly to the bass ear (JO-B's targets, 611 of them) that
fired during a batch. Measured offline (noise 0.15, gain 0.8, fatigue 0.3):
    kick at 128 BPM   swings 0.09 <-> 0.23 about once a beat
    pads, no kick     flat around 0.09        sub bass, no kick   flat around 0.125
    silence           0
So a kick is not a LEVEL (a sub bass is as loud for these neurons), it is a PULSE: the readout
jumping above its recent floor. The rule is Oscar's (2026-09-19), counted in BEATS, not bars:
    two kicks in a row  -> drop          two beats with no kick -> break
The beat length comes from the beatgrid dashboard's BPM when it is there (app_server sets .period).
Thresholds are calibration knobs: real tracks will need tuning by ear.
    python fly_says.py      # self-check
"""
import collections, time

WINDOW = 9          # batches, about one second: where the floor is read
PULSE = 0.06        # a jump this far above the floor = the bass circuit just took a kick (the pages' « Seuil kick » fader)
SOUND_FLOOR = 0.01  # under this mean activity there is nothing to say
IN_A_ROW = 1.5      # beats: a kick this close to the last one is "d'affilée"
MISSING = 2.5       # beats since the last kick: the two next ones did not come
DEFAULT_PERIOD = 60 / 128  # ponytail: used only without beatgrid; half-time tracks keep the last word


class FlySays:
    def __init__(self):
        self.window = collections.deque(maxlen=WINDOW)
        self.verdict, self.period, self.pulse = None, DEFAULT_PERIOD, PULSE
        self._high, self._last_kick, self._run = False, None, 0

    def push(self, readout, now=None):
        now = time.monotonic() if now is None else now
        self.window.append(readout)
        w = self.window
        floor, mean = min(w), sum(w) / len(w)
        high = readout - floor > self.pulse
        if high and not self._high:  # rising edge: one kick, however many batches it lasts
            near = self._last_kick is not None and now - self._last_kick < IN_A_ROW * self.period
            self._run, self._last_kick = (self._run + 1 if near else 1), now
            if self._run >= 2: self.verdict = "drop"
        self._high = high
        if len(w) == WINDOW:
            if mean < SOUND_FLOOR: self.verdict, self._run = None, 0
            elif self._last_kick is None or now - self._last_kick > MISSING * self.period:
                self.verdict, self._run = "break", 0
        return {"verdict": self.verdict, "level": round(min(1.0, (max(w) - floor) / 0.15), 3)}


if __name__ == "__main__":
    f, clock = FlySays(), [0.0]
    dt = f.period / 4  # four batches a beat

    def say(seq):
        for x in seq:
            clock[0] += dt
            out = f.push(x, clock[0])
        return out["verdict"]
    kick, mute = [0.23, 0.11, 0.10, 0.10], [0.09, 0.10, 0.08, 0.09]
    say(mute)
    assert say(kick) != "drop"                                # one kick is not a drop yet
    assert say(kick) == "drop"                                # two in a row: drop, on the second
    assert say(kick * 4) == "drop"
    assert say(mute) == "drop"                                # one kick missing: not yet
    assert say(mute) == "break"                               # the second one did not come either
    assert say(kick) == "break" and say(mute) == "break"      # a lone kick does not make a drop
    assert say(kick * 2) == "drop"
    assert say([0.125, 0.11, 0.14, 0.12] * 4) == "break"      # a sub bass without kick is still a break
    assert say([0.0] * 9) is None
    print("fly_says ok")
