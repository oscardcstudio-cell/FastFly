// The sound as the fly's climate reads it, shared by /stage and /params: 32 log bands, the warm zone (low mids)
// and the cold zone (highs) tinted, each glowing with its level. Same bands and dB scale as audio_in.py.
const EQ_EDGES = Array.from({ length: 33 }, (_, i) => 30 * (16000 / 30) ** (i / 32));
const WARM_HZ = [200, 500], COLD_HZ = [4000, 16000], WARM_COLOR = '#ff7a3c', COLD_COLOR = '#9fdcff';

function drawEQ(canvas, spectrum, heat, cold) {
  const dpr = Math.min(devicePixelRatio, 2), w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const c = canvas.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  const bw = w / 32;
  const zone = (hz, color, level) => {  // the band the sense listens to, lit by how much it feels
    const x0 = Math.log(hz[0] / 30) / Math.log(16000 / 30) * w, x1 = Math.log(hz[1] / 30) / Math.log(16000 / 30) * w;
    c.globalAlpha = 0.12 + 0.45 * level; c.fillStyle = color; c.fillRect(x0, 0, x1 - x0, h); c.globalAlpha = 1;
  };
  zone(WARM_HZ, WARM_COLOR, heat || 0); zone(COLD_HZ, COLD_COLOR, cold || 0);
  (spectrum || []).forEach((v, i) => {
    const mid = Math.sqrt(EQ_EDGES[i] * EQ_EDGES[i + 1]);
    c.fillStyle = mid >= WARM_HZ[0] && mid < WARM_HZ[1] ? WARM_COLOR : mid >= COLD_HZ[0] ? COLD_COLOR : '#5a7890';
    c.fillRect(i * bw + 0.5, h - v * h, bw - 1, v * h);
  });
}

// From a Web Audio analyser (dB per bin): the 32 levels, and a band's share of the whole power in dB
function eqFromBins(bins, hz) {
  return EQ_EDGES.slice(0, 32).map((lo, i) => {
    const a = Math.max(1, Math.floor(lo / hz)), b = Math.max(a + 1, Math.floor(EQ_EDGES[i + 1] / hz));
    let sum = 0; for (let k = a; k < b && k < bins.length; k++) sum += 10 ** (bins[k] / 10);
    return Math.max(0, Math.min(1, (10 * Math.log10(sum / (b - a) + 1e-12) + 80) / 60));
  });
}
function shareDb(bins, hz, band) {
  let part = 0, all = 0;
  for (let k = 1; k < bins.length; k++) { const p = 10 ** (bins[k] / 10); all += p; if (k >= Math.floor(band[0] / hz) && k <= Math.ceil(band[1] / hz)) part += p; }
  return 10 * Math.log10((part + 1e-15) / (all + 1e-12));
}
