"""
SimEngine — importable wrapper around the FlyWire CuPy simulator.

Loads neuron_annotations.npz (from download_metadata.py) for biologically
meaningful stimuli, heatmap groups, 3D positions, and motor neuron detail.

Usage (standalone test):
    python sim_engine.py                        # synthetic data
    python sim_engine.py --data flywire_v783.bin
"""

import base64
import os
import time
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not installed.  Run:  pip install cupy-cuda12x")
    sys.exit(1)

from flywire_sim import (CUDA_KERNELS, compile_kernels, load_connectome_binary,
                         generate_synthetic, quantize_weights_int8)

ANNOTATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "neuron_annotations.npz")


class SimEngine:
    """GPU-accelerated LIF simulator wrapping CuPy CUDA kernels."""

    def __init__(self, data_file=None, seed=42):
        if data_file:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                load_connectome_binary(data_file)
        else:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                generate_synthetic(seed=seed)

        self.kernels = compile_kernels()
        self.seed = seed
        self.current_step = 0

        # GPU arrays — connectivity
        self.d_offsets = cp.asarray(offsets)
        self.d_targets = cp.asarray(targets)
        w_int8, w_scales = quantize_weights_int8(weights, offsets, self.n_neurons)
        self.d_weights = cp.asarray(w_int8)
        self.d_weight_scales = cp.asarray(w_scales)
        self._base_weight_scales = self.d_weight_scales.copy()

        # GPU arrays — neuron state
        rng = cp.random.default_rng(seed)
        self.d_voltage = rng.uniform(0.0, 0.9, self.n_neurons).astype(cp.float32)
        self.d_current = cp.zeros(self.n_neurons, dtype=cp.float32)

        # Spike bookkeeping
        self.spike_words = (self.n_neurons + 31) // 32
        self.d_spike_bits = cp.zeros(self.spike_words, dtype=cp.uint32)
        self.d_spike_idx  = cp.zeros(self.n_neurons, dtype=cp.uint32)
        self.d_num_spikes = cp.zeros(1, dtype=cp.uint32)

        # LIF parameters
        self.tau_decay   = np.float32(0.9)
        self.v_threshold = np.float32(1.0)
        self.v_reset     = np.float32(0.0)
        self.noise_amp   = np.float32(0.4)

        # Launch config
        self.BLOCK = 256
        self.PROP_BLOCK = 128
        self.MAX_PROP_BLOCKS = 2048
        self.neuron_blocks  = (self.n_neurons + self.BLOCK - 1) // self.BLOCK
        self.compact_blocks = (self.spike_words + self.BLOCK - 1) // self.BLOCK

        # Stimulus state
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0

        # Last-step spike indices (for 3D viz)
        self._last_spike_indices = np.array([], dtype=np.int32)
        # OR of spike bits over a whole batch, so the viz sees every neuron that fired
        self.d_accum_bits = cp.zeros(self.spike_words, dtype=cp.uint32)

        # Live audio drive: group name -> (gpu indices, amplitude)
        self._audio = {}

        # Frame recording for slow-motion replay: every Nth substep (0 = off)
        self.frame_every = 0
        self.max_edges_per_frame = 400

        # GPU accumulators for sync-free counting
        self.d_total_spikes = cp.zeros(1, dtype=cp.uint64)

        # Feature toggles (can be set at runtime)
        self.send_active_indices = True
        self.send_group_rates = True
        self.send_motor_rates = True
        self.active_indices_interval = 1  # only transfer every Nth batch
        self._batch_counter = 0

        # Load annotations
        self._load_annotations()

        # Metrics history
        self.group_rates_history = []

    def _load_annotations(self):
        """Load neuron_annotations.npz for biological groups, stimuli, positions."""
        if os.path.exists(ANNOTATIONS_FILE):
            print(f"Loading neuron annotations from {ANNOTATIONS_FILE}...")
            data = np.load(ANNOTATIONS_FILE, allow_pickle=True)

            # Root IDs
            self._root_ids = data['root_ids'].astype(np.int64) if 'root_ids' in data else None

            # 3D positions (normalized to [-1,1])
            if 'pos_x' in data:
                self._positions = np.stack([
                    data['pos_x'], data['pos_y'], data['pos_z']
                ], axis=1).astype(np.float32)  # [N, 3]
                print(f"  3D positions loaded: {self._positions.shape}")
            else:
                self._positions = None

            # Super class per neuron (for coloring in 3D)
            self._super_class = data.get('super_class', None)

            # Stimuli
            self._stimuli = {}
            stim_names = list(data['stim_names'])
            for name in stim_names:
                safe = 'stim_' + name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                if safe in data:
                    self._stimuli[name] = data[safe].astype(np.int32)

            # Heatmap groups
            self.group_labels = list(data['group_names'])
            self.num_groups = len(self.group_labels)
            self._group_indices = []
            for name in self.group_labels:
                self._group_indices.append(data['group_' + name].astype(np.int32))

            self._neuron_to_group = cp.full(self.n_neurons, -1, dtype=cp.int32)
            for g, indices in enumerate(self._group_indices):
                self._neuron_to_group[cp.asarray(indices.astype(np.int64))] = g

            # Body sensory groups
            self._body_sensory = {}
            if 'body_sensory_names' in data:
                for name in data['body_sensory_names']:
                    key = 'bsens_' + name
                    if key in data:
                        self._body_sensory[str(name)] = data[key].astype(np.int32)

            # Body motor groups
            self._body_motor = {}
            if 'body_motor_names' in data:
                motor_names = [str(n) for n in data['body_motor_names']]
                for name in motor_names:
                    key = 'bmotor_' + name
                    if key in data:
                        self._body_motor[name] = data[key].astype(np.int32)

            self._motor_group_names = list(self._body_motor.keys())
            self._num_motor_groups = len(self._motor_group_names)
            self._neuron_to_motor = cp.full(self.n_neurons, -1, dtype=cp.int32)
            for g, name in enumerate(self._motor_group_names):
                indices = self._body_motor[name]
                self._neuron_to_motor[cp.asarray(indices.astype(np.int64))] = g

            self._build_audio_groups(data)
            self._use_annotations = True
            print(f"  {len(self._stimuli)} stimuli, {self.num_groups} heatmap groups")
            print(f"  {len(self._body_sensory)} body sensory, {len(self._body_motor)} body motor")

        else:
            print(f"No annotation file found ({ANNOTATIONS_FILE})")
            print("  Run download_metadata.py for biological annotations.")
            self._root_ids = None
            self._positions = None
            self._super_class = None
            self._setup_fallback_groups()
            self._use_annotations = False

    # Johnston's organ subtypes (FlyWire cell_type / cell_sub_class prefixes).
    # JO-A/B hear sound and vibration, JO-C/E sense wind and gravity.
    AUDIO_PREFIXES = ("JO-A", "JO-B", "JO-C", "JO-E")

    def _build_audio_groups(self, data):
        cols = [np.asarray(data[k]).astype(str) for k in ("cell_type", "cell_sub_class") if k in data]
        for prefix in self.AUDIO_PREFIXES:
            mask = np.zeros(self.n_neurons, dtype=bool)
            for col in cols:
                mask |= np.char.startswith(np.char.upper(col), prefix)
            idx = np.nonzero(mask)[0]
            if len(idx):
                self._audio[prefix] = [cp.asarray(idx), 0.0]
        print(f"  audio groups: { {k: len(v[0]) for k, v in self._audio.items()} }")

    def set_audio(self, amps):
        """amps: {group: amplitude}, e.g. {'JO-A': 0.8}. Unknown groups ignored."""
        for name, amp in amps.items():
            if name in self._audio:
                self._audio[name][1] = max(0.0, min(5.0, float(amp)))

    def get_audio_groups(self):
        return {k: int(len(v[0])) for k, v in self._audio.items()}

    def _setup_fallback_groups(self):
        """Fallback: equal-size index-range groups."""
        self.num_groups = 20
        group_size = self.n_neurons // self.num_groups
        self.group_labels = [f"Group {i}" for i in range(self.num_groups)]
        self._group_indices = []
        for g in range(self.num_groups):
            start = g * group_size
            end = start + group_size if g < self.num_groups - 1 else self.n_neurons
            self._group_indices.append(np.arange(start, end, dtype=np.int32))

        self._neuron_to_group = cp.full(self.n_neurons, -1, dtype=cp.int32)
        for g, indices in enumerate(self._group_indices):
            self._neuron_to_group[cp.asarray(indices.astype(np.int64))] = g

        self._stimuli = {
            "Neurons 0-1000": np.arange(0, min(1000, self.n_neurons), dtype=np.int32),
        }
        self._body_sensory = {}
        self._body_motor = {}
        self._motor_group_names = []
        self._num_motor_groups = 0
        self._neuron_to_motor = cp.full(self.n_neurons, -1, dtype=cp.int32)

    def inject_stimulus(self, neuron_indices, amplitude=0.5):
        self._stimulus_indices = cp.asarray(np.array(neuron_indices, dtype=np.int64))
        self._stimulus_amplitude = float(amplitude)

    def clear_stimulus(self):
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0

    def set_weight_gain(self, g):
        """Scale all synapses. Above ~0.85 the net latches into self-sustained firing."""
        self.d_weight_scales[:] = self._base_weight_scales * float(g)

    def reset_state(self):
        self.d_voltage.fill(0)
        self.d_current.fill(0)

    def set_noise_amp(self, value):
        self.noise_amp = np.float32(value)

    def step(self, n=50):
        """Run n timesteps and return a metrics dict.

        ZERO per-substep GPU→CPU syncs:
        - propagate_v2 reads d_num_spikes from device memory (no CPU readback)
        - count_spikes kernel counts groups/motors/total from spike_bits on GPU
        - compact kernel still runs (needed for propagate's spike_idx array)
        - Single sync at batch end to transfer results to CPU
        - active_indices only transferred when send_active_indices is True
        """
        t_start = time.perf_counter()

        # GPU accumulators — zeroed once, accumulated across all substeps (uint64 for atomicAdd)
        d_group_counts = cp.zeros(self.num_groups, dtype=cp.uint64)
        d_motor_counts = cp.zeros(max(self._num_motor_groups, 1), dtype=cp.uint64)
        d_total_spikes = self.d_total_spikes
        d_total_spikes.fill(0)

        # Local refs to avoid Python attribute lookups in inner loop
        d_current = self.d_current
        d_voltage = self.d_voltage
        d_spike_bits = self.d_spike_bits
        d_spike_idx = self.d_spike_idx
        d_num_spikes = self.d_num_spikes
        d_offsets = self.d_offsets
        d_targets = self.d_targets
        d_weights = self.d_weights
        d_weight_scales = self.d_weight_scales
        neuron_to_group = self._neuron_to_group
        neuron_to_motor = self._neuron_to_motor
        k_update_with_noise = self.kernels["update_with_noise"]
        k_compact = self.kernels["compact"]
        k_propagate_v2 = self.kernels["propagate_v2"]
        k_count = self.kernels["count_spikes"]
        neuron_blocks = self.neuron_blocks
        compact_blocks = self.compact_blocks
        BLOCK = self.BLOCK
        PROP_BLOCK = self.PROP_BLOCK
        MAX_PROP_BLOCKS = self.MAX_PROP_BLOCKS
        n_neurons_i32 = np.int32(self.n_neurons)
        spike_words_i32 = np.int32(self.spike_words)
        stim_indices = self._stimulus_indices
        stim_amp = self._stimulus_amplitude
        # audio_mute: any open /stage tab keeps feeding the shared ear; a sense test must silence it
        audio = [] if getattr(self, 'audio_mute', False) else [(idx, amp) for idx, amp in self._audio.values() if amp > 0]
        d_accum_bits = self.d_accum_bits
        d_accum_bits.fill(0)
        hist = cp.empty((n, self.spike_words), dtype=cp.uint32) if self.frame_every else None

        for sub in range(n):
            d_num_spikes.fill(0)

            if stim_indices is not None:
                d_current[stim_indices] += stim_amp
            for idx, amp in audio:
                d_current[idx] += amp

            k_update_with_noise(
                (neuron_blocks,), (BLOCK,),
                (d_voltage, d_current, d_spike_bits,
                 n_neurons_i32, spike_words_i32,
                 self.tau_decay, self.v_threshold, self.v_reset,
                 np.uint32(self.seed), np.uint32(self.current_step),
                 self.noise_amp))
            cp.bitwise_or(d_accum_bits, d_spike_bits, out=d_accum_bits)
            if hist is not None:
                hist[sub] = d_spike_bits

            k_compact(
                (compact_blocks,), (BLOCK,),
                (d_spike_bits, d_spike_idx, d_num_spikes,
                 spike_words_i32, n_neurons_i32))

            # Count group/motor/total spikes from spike_bits — pure GPU, no sync
            k_count(
                (compact_blocks,), (BLOCK,),
                (d_spike_bits, neuron_to_group, d_group_counts,
                 neuron_to_motor, d_motor_counts, d_total_spikes,
                 spike_words_i32, n_neurons_i32))

            # Propagate v2 — reads d_num_spikes from device memory, no CPU sync
            k_propagate_v2(
                (MAX_PROP_BLOCKS,), (PROP_BLOCK,),
                (d_spike_idx, d_num_spikes,
                 d_offsets, d_targets, d_weights, d_weight_scales, d_current))

            self.current_step += 1

        # === Single batch-end sync — all GPU work done ===
        cp.cuda.Stream.null.synchronize()

        total_spikes = int(d_total_spikes[0])
        t_elapsed = time.perf_counter() - t_start
        firing_rate = total_spikes / (n * self.n_neurons) if self.n_neurons > 0 else 0
        steps_per_sec = n / t_elapsed if t_elapsed > 0 else 0

        result = {
            "step": self.current_step,
            "spike_count": total_spikes,
            "firing_rate": round(firing_rate, 6),
            "mean_voltage": round(float(d_voltage.mean()), 4),
            "steps_per_sec": round(steps_per_sec, 1),
        }

        # Group rates (for heatmap) — only compute if enabled
        if self.send_group_rates:
            group_spike_counts = d_group_counts.get()
            group_rates = []
            for g in range(self.num_groups):
                group_n = len(self._group_indices[g])
                rate = float(group_spike_counts[g]) / (n * group_n) if group_n > 0 else 0
                group_rates.append(round(rate, 6))
            self.group_rates_history.append(group_rates)
            if len(self.group_rates_history) > 200:
                self.group_rates_history = self.group_rates_history[-200:]
            result["group_rates"] = group_rates

        # Motor rates — only compute if enabled
        if self.send_motor_rates:
            motor_spike_counts = d_motor_counts.get()
            motor_rates = {}
            for g, name in enumerate(self._motor_group_names):
                group_n = len(self._body_motor[name])
                rate = float(motor_spike_counts[g]) / (n * group_n) if group_n > 0 else 0
                motor_rates[name] = round(rate, 6)
            result["motor_rates"] = motor_rates

        # Active indices (for 3D viz) — only transfer every Nth batch
        self._batch_counter += 1
        if self.send_active_indices and self._batch_counter % self.active_indices_interval == 0:
            bits = cp.unpackbits(d_accum_bits.view(cp.uint8), bitorder='little')[:self.n_neurons]
            self._last_spike_indices = cp.nonzero(bits)[0].astype(cp.int32).get()
            result["active_indices"] = self._last_spike_indices.tolist()

        if hist is not None:
            result["frames"] = self._frames(hist)

        return result

    def _unpack(self, bits):
        return cp.nonzero(cp.unpackbits(bits.view(cp.uint8), bitorder='little')[:self.n_neurons])[0]

    def _frames(self, hist):
        """Every Nth substep: who fired, and which synapses carried it.

        An edge pre->post is kept when pre fired on the previous substep and
        post fires now: that spike was (partly) caused through this synapse.
        sign: +1 excitatory, -1 inhibitory.
        """
        frames = []
        rng = cp.random.default_rng(self.current_step)
        for t in range(max(1, self.frame_every), hist.shape[0], self.frame_every):
            now = self._unpack(hist[t])
            pre = self._unpack(hist[t - 1])
            edges = []
            if len(pre) and len(now):
                starts = self.d_offsets[pre]
                cnt = (self.d_offsets[pre + 1] - starts).astype(cp.int64)
                total = int(cnt.sum())
                if total:
                    first = cp.cumsum(cnt) - cnt
                    k = cp.arange(total, dtype=cp.int64) - cp.repeat(first, cnt)
                    syn = cp.repeat(starts.astype(cp.int64), cnt) + k
                    tgt = self.d_targets[syn].astype(cp.int64)
                    fired = (hist[t][tgt >> 5] >> (tgt & 31).astype(cp.uint32)) & 1
                    hit = cp.nonzero(fired)[0]
                    if len(hit) > self.max_edges_per_frame:
                        hit = hit[cp.argsort(rng.random(len(hit)))[:self.max_edges_per_frame]]
                    src = cp.repeat(pre, cnt)[hit]
                    sign = cp.sign(self.d_weights[syn[hit]].astype(cp.int32))
                    edges = cp.stack([src.astype(cp.int64), tgt[hit], sign.astype(cp.int64)], axis=1).get().ravel().tolist()
            # even stride, not [:3000]: a head cut shows the same low-index neurons whatever fired
            frames.append({"s": now[::-(-len(now) // 3000) or 1].get().tolist(), "e": edges})
        return frames

    # --- Data accessors ---

    def get_predefined_stimuli(self):
        return list(self._stimuli.keys())

    def get_body_info(self):
        sensory = {k: len(v) for k, v in self._body_sensory.items()}
        motor = {k: len(v) for k, v in self._body_motor.items()}
        return {"sensory": sensory, "motor": motor}

    def get_positions_b64(self):
        """Return neuron positions as base64-encoded float32 array [N*3]."""
        if self._positions is not None:
            return base64.b64encode(self._positions.tobytes()).decode('ascii')
        return None

    def get_neuron_classes(self):
        """Return super_class per neuron for 3D coloring."""
        if self._super_class is not None:
            # Encode as int: unique classes -> color indices
            unique = sorted(set(self._super_class))
            class_to_id = {c: i for i, c in enumerate(unique)}
            ids = np.array([class_to_id.get(c, 0) for c in self._super_class],
                           dtype=np.uint8)
            return {
                "labels": unique,
                "ids_b64": base64.b64encode(ids.tobytes()).decode('ascii')
            }
        return None

    def get_motor_detail(self, group_name):
        """Return detail for a motor group: neuron indices, root_ids, active status."""
        if group_name not in self._body_motor:
            return None
        indices = self._body_motor[group_name]
        active_set = set(self._last_spike_indices.tolist())
        neurons = []
        for idx in indices:
            idx = int(idx)
            rid = int(self._root_ids[idx]) if self._root_ids is not None else idx
            neurons.append({
                "index": idx,
                "root_id": rid,
                "active": idx in active_set,
            })
        return {"group": group_name, "neurons": neurons}

    def apply_predefined_stimulus(self, name, amplitude=None):
        if name not in self._stimuli:
            return False
        indices = self._stimuli[name]
        amp = amplitude if amplitude is not None else 0.5
        self.inject_stimulus(indices, amp)
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Binary connectome file")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    engine = SimEngine(data_file=args.data)
    print(f"\nSimEngine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses")
    print(f"Stimuli: {engine.get_predefined_stimuli()}")
    print(f"Groups:  {engine.group_labels}")
    print(f"Positions: {'yes' if engine._positions is not None else 'no'}")
    print(f"Running {args.steps} steps in batches of {args.batch}...\n")

    for i in range(0, args.steps, args.batch):
        metrics = engine.step(n=args.batch)
        print(f"  Step {metrics['step']:>6d}  "
              f"spikes={metrics['spike_count']:>6d}  "
              f"rate={metrics['firing_rate']*100:>5.2f}%  "
              f"V_mean={metrics['mean_voltage']:.3f}  "
              f"active_3d={len(metrics.get('active_indices', []))}  "
              f"steps/s={metrics['steps_per_sec']:.0f}")

    print("\nDone.")
