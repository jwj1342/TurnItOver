import * as THREE from "three";
import type { HierarchyNode, StatsPayload } from "./protocol";

export function collectStats(renderer: THREE.WebGLRenderer, root: THREE.Object3D, lastRenderMs: number): StatsPayload {
  let trianglesScene = 0;
  let meshCount = 0;
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    meshCount += 1;
    const g = m.geometry;
    trianglesScene += g.index ? g.index.count / 3 : g.getAttribute("position").count / 3;
  });
  const info = renderer.info;
  return {
    property: "stats",
    triangles_rendered: info.render.triangles,
    draw_calls: info.render.calls,
    triangles_scene: trianglesScene,
    mesh_count: meshCount,
    geometries: info.memory.geometries,
    textures: info.memory.textures,
    programs: info.programs?.length ?? 0,
    last_render_ms: lastRenderMs,
  };
}

export function hierarchyOf(o: THREE.Object3D): HierarchyNode {
  return {
    name: o.name,
    type: o.type,
    position: [o.position.x, o.position.y, o.position.z],
    rotation: [o.rotation.x, o.rotation.y, o.rotation.z],
    scale: [o.scale.x, o.scale.y, o.scale.z],
    children: o.children.filter((c) => !(c as THREE.Mesh).isMesh).map(hierarchyOf),
  };
}

export type PartBoxes = Record<string, { min: [number, number, number]; max: [number, number, number] }>;

/** World-space AABB per part Group (a Group whose name is in partNames). */
export function partBoxes(root: THREE.Object3D, partNames: Set<string>): PartBoxes {
  root.updateMatrixWorld(true);
  const out: PartBoxes = {};
  root.traverse((o) => {
    if (!partNames.has(o.name)) return;
    const box = new THREE.Box3().setFromObject(o);
    out[o.name] = { min: [box.min.x, box.min.y, box.min.z], max: [box.max.x, box.max.y, box.max.z] };
  });
  return out;
}
