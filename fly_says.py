"""What the fly makes of the music: break or drop, read from its own neurons.

The signal: the share of neurons wired directly to the bass ear (JO-B's targets, 611 of them) that
fired during a batch. Measured offline (noise 0.15, gain 0.8, fatigue 0.3):
    kick at 128 BPM   swings 0.09 <-> 0.23 about once a beat
    pads, no kick     flat around 0.09        sub bass, no kick   flat around 0.125
    silence           0
So a drop is not a LEVEL (a sub bass is as loud for these neurons), it is a PULSE: the swing over
the last second. Thresholds are calibration knobs: real tracks will need tuning by ear.
    python fly_says.py      # self-check
"""
import collections

WINDOW = 9          # batches, about one second
DROP_SWING = 0.06   # above: the bass circuit is pulsing -> drop
BREAK_SWING = 0.04  # below (with sound present): it is flat -> break. In between: keep the last word
SOUND_FLOOR = 0.01  # under this mean activity there is nothing to say


class FlySays:
    def __init__(self):
        self.window = collections.deque(maxlen=WINDOW)
        self.verdict = None

    def push(self, readout):
        self.window.append(readout)
        w = self.window
        swing, mean = max(w) - min(w), sum(w) / len(w)
        if len(w) == WINDOW:
            if mean < SOUND_FLOOR: self.verdict = None
            elif swing > DROP_SWING: self.verdict = "drop"
            elif swing < BREAK_SWING: self.verdict = "break"
        return {"verdict": self.verdict, "level": round(min(1.0, swing / 0.15), 3)}


if __name__ == "__main__":
    f = FlySays()
    say = lambda seq: [f.push(x) for x in seq][-1]["verdict"]
    assert say([0.23, 0.11, 0.10, 0.10] * 5) == "drop"
    assert say([0.09, 0.10, 0.08] * 5) == "break"
    assert say([0.125, 0.11, 0.14] * 5) == "break"          # a sub bass without kick is still a break
    assert say([0.10, 0.145] * 5) == "break"                 # swing 0.045, in between: keeps its last word
    assert say([0.0] * 9) is None
    print("fly_says ok")
