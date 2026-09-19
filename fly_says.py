"""What the fly makes of the music: break or drop, read from its own neurons.

The signal: the share of neurons wired directly to the bass ear (JO-B's targets, 611 of them) that
fired during a batch. Measured offline (noise 0.15, gain 0.8, fatigue 0.3):
    kick at 128 BPM   swings 0.09 <-> 0.23 about once a beat
    pads, no kick     flat around 0.09        sub bass, no kick   flat around 0.125
    silence           0
Counting kicks beat by beat flipped break/drop every two bars (92 flips over 454 bars, 2026-09-19):
Oscar's verdict, « tu passe de break à drop toutes les 2 mesures c'est n'importe quoi ». So the
fly now reads the same thing as the beatgrid dashboard (sections.DetecteurEnergie): the STEP.
Each bar (4 beats, beat length from the dashboard's BPM) gets the mean of the readout; it is
compared to the four bars before. The bass circuit falls at once entering a break and jumps back
at the drop, while it barely moves inside a section.
    python fly_says.py      # self-check
"""
import collections, math, time

STEP_DB = 2.5       # a bar this far (dB) from the four before = the section changed (the pages' « Seuil marche » fader)
BARS_BEFORE = 4     # what a bar is compared to
BEATS_A_BAR = 4
SOUND_FLOOR = 0.01  # under this mean activity there is nothing to say
LEVEL_WINDOW = 9    # batches, about one second: only for the pulse gauge on the pages
DEFAULT_PERIOD = 60 / 128  # ponytail: used only without beatgrid; half-time tracks keep the last word


class FlySays:
    def __init__(self):
        self.window = collections.deque(maxlen=LEVEL_WINDOW)
        self.verdict, self.period, self.step = None, DEFAULT_PERIOD, STEP_DB
        self._bars = collections.deque(maxlen=BARS_BEFORE)
        self._bar_start, self._sum, self._n = None, 0.0, 0

    def push(self, readout, now=None):
        now = time.monotonic() if now is None else now
        self.window.append(readout)
        if self._bar_start is None:
            self._bar_start = now
        if now - self._bar_start >= BEATS_A_BAR * self.period:
            self._close_bar()
            self._bar_start = now
        self._sum += readout
        self._n += 1
        w = self.window
        return {"verdict": self.verdict, "level": round(min(1.0, (max(w) - min(w)) / 0.15), 3)}

    def _close_bar(self):
        level = self._sum / self._n if self._n else 0.0
        self._sum, self._n = 0.0, 0
        if level < SOUND_FLOOR:
            self.verdict = None
            self._bars.clear()
            return
        if len(self._bars) == BARS_BEFORE:
            gap = 20 * math.log10(level / (sum(self._bars) / BARS_BEFORE))
            if gap <= -self.step: self.verdict = "break"
            elif gap >= self.step: self.verdict = "drop"
        self._bars.append(level)


if __name__ == "__main__":
    f, clock = FlySays(), [0.0]
    f.period = 0.5
    dt = f.period / 4  # four batches a beat, sixteen a bar

    KICK, MUTE, SILENT = [0.23, 0.11, 0.10, 0.10] * 4, [0.09, 0.10, 0.08, 0.09] * 4, [0.0] * 16

    def bars(n, bar):
        for _ in range(n):
            for x in bar:
                clock[0] += dt
                out = f.push(x, clock[0])
        return out["verdict"]
    assert bars(6, KICK) is None                   # no step yet: nothing to say
    assert bars(1, MUTE) is None                   # the bar that falls is still being counted
    assert bars(1, MUTE) == "break"                # it closed: break, one bar late
    assert bars(6, MUTE) == "break"                # the break holds, bar after bar
    assert bars(2, KICK) == "drop"                 # the kick comes back: drop
    assert {bars(1, KICK) for _ in range(16)} == {"drop"}  # and holds: no flip every two bars
    assert bars(2, SILENT) is None                 # silence: nothing to say
    print("fly_says ok")
