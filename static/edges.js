// Fading trail of the synapses that just carried a spike: cyan = excitation, pink = inhibition.
// Shared by /stage and /trace. e = flat [pre, post, sign, ...] from the engine's frames.
function makeEdgeTrail(scene, pos, maxSeg = 60000) {
  const EXC = [0.1, 0.6, 1.0], INH = [1.0, 0.2, 0.6];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxSeg * 6), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(maxSeg * 6), 3));
  g.setDrawRange(0, 0);
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })));
  const segI = new Float32Array(maxSeg), segRGB = new Float32Array(maxSeg * 3);
  let n = 0, head = 0;
  return {
    add(e) {
      const p = g.attributes.position.array;
      for (let k = 0; k < e.length; k += 3) {
        const a = e[k], b = e[k + 1], j = head;
        p.set(pos.subarray(a * 3, a * 3 + 3), j * 6); p.set(pos.subarray(b * 3, b * 3 + 3), j * 6 + 3);
        segI[j] = 1; segRGB.set(e[k + 2] > 0 ? EXC : INH, j * 3);
        head = (head + 1) % maxSeg; n = Math.min(maxSeg, n + 1);
      }
      g.attributes.position.needsUpdate = true;
    },
    draw(decay) {
      const col = g.attributes.color.array;
      for (let j = 0; j < n; j++) {
        const v = (segI[j] *= decay);
        for (let q = 0; q < 3; q++) col[j * 6 + q] = col[j * 6 + 3 + q] = segRGB[j * 3 + q] * v * 0.6;
      }
      g.setDrawRange(0, n * 2);
      g.attributes.color.needsUpdate = true;
    },
    clear() { segI.fill(0); },
  };
}
