import * as THREE from "three";
import type { PartGeometry } from "./protocol";

function b64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunk)));
  return btoa(s);
}

/** Per-part world-space triangle soup. A mesh belongs to the nearest ancestor whose name is a part id. */
export function exportParts(root: THREE.Object3D, partNames: Set<string>): PartGeometry[] {
  root.updateMatrixWorld(true);
  const acc = new Map<string, { pos: number[]; idx: number[] }>();
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    let p: THREE.Object3D | null = m;
    while (p && !partNames.has(p.name)) p = p.parent;
    if (!p) return;
    const entry = acc.get(p.name) ?? { pos: [], idx: [] };
    const g = m.geometry;
    const posAttr = g.getAttribute("position");
    const base = entry.pos.length / 3;
    const v = new THREE.Vector3();
    for (let i = 0; i < posAttr.count; i++) {
      v.fromBufferAttribute(posAttr, i).applyMatrix4(m.matrixWorld);
      entry.pos.push(v.x, v.y, v.z);
    }
    if (g.index) {
      for (let i = 0; i < g.index.count; i++) entry.idx.push(base + g.index.getX(i));
    } else {
      for (let i = 0; i < posAttr.count; i++) entry.idx.push(base + i);
    }
    acc.set(p.name, entry);
  });
  const out: PartGeometry[] = [];
  for (const [name, e] of acc) {
    const pos = new Float32Array(e.pos);
    const idx = new Uint32Array(e.idx);
    out.push({
      name,
      vertex_count: pos.length / 3,
      triangle_count: idx.length / 3,
      positions_b64: b64(pos.buffer),
      indices_b64: b64(idx.buffer),
    });
  }
  return out;
}
