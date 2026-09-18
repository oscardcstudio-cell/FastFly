# FlyWire Connectome GPU Simulator

Custom CUDA simulator targeting real-time (or faster) simulation of the complete
Drosophila melanogaster brain connectome (139,255 neurons, 54.5M synapses).

## Architecture

- **Neuron model**: Leaky Integrate-and-Fire (LIF), validated as sufficient by arXiv:2404.17128
- **Spike detection**: Warp ballot intrinsics for bit-packed spike flags
- **Spike propagation**: Push model - only process synapses of neurons that actually fired
- **Connectivity**: CSR sparse format with FP16 weights, targets sorted per row
- **Load balancing**: Warp-per-spike with grid-stride loop

## Build

Requires NVIDIA CUDA Toolkit 12.x. Run `build.bat` to compile.

## Key optimization targets

1. Spike propagation kernel is the dominant cost (memory-bound, atomicAdd contention)
2. At 1% firing rate: ~549K active synapses/step, ~3.3 MB memory traffic
3. RTX 3080 Ti theoretical minimum: ~3.6 us/step at 912 GB/s bandwidth

## Next optimizations to try

- INT8 quantized weights (halve bandwidth vs FP16)
- Shared-memory accumulation buffers to reduce atomicAdd contention
- Connectivity pruning (remove weakest synapses)
- Warp-cooperative target sorting to batch atomicAdds
- Multi-stream pipelining
- Fused neuron update + noise kernel
- Pull model comparison at higher firing rates

## fly-vj (branche VJ) — pièges mesurés

- **Un seul moteur partagé entre tous les onglets.** Chaque page (/stage, /trace, /) réécrit les réglages à la connexion, et /stage envoie le son en continu. Avant toute mesure (`check_senses.py`) ou test d'un sens : fermer les autres onglets, y compris le navigateur de test de Claude. `audio_mute` coupe l'oreille, pas les autres réglages.
- **Fatigue (`adapt`, défaut 0.3)** : sans elle, toute stimulation finit dans la même boucle auto-entretenue (lobe antennaire + corps pédonculé, ~10 000 neurones) = centre blanc. Effet non monotone : toute retouche se valide par `python check_senses.py` (vert = recouvrement < 0.5 entre sens différents).
- Un sens dans /trace = une impulsion répétée (1 lot stimulé, 2 lots libres, effacement), pas un appui tenu.
- **Lancer branché sur beatgrid** : `.venv/Scripts/python -u app_server.py --data flywire_v783.bin --port 8010 --beatgrid ../vj-rien`. Le pont (`beatgrid_bridge.py`) lit le journal de session de beatgrid, jamais son `/ws` (un 2e client y vole les états du dashboard). La table sens ↔ événements est en tête du fichier ; `python beatgrid_bridge.py` la vérifie sans GPU.
