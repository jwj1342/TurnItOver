import * as THREE from "three";
import type { CameraInfo, Framing, ViewDef } from "./protocol";

const DEG = Math.PI / 180;

/** Camera for a view: orbit around framing.center at a distance derived from framing.radius. */
export function makeCamera(view: ViewDef, framing: Framing, aspect: number): { camera: THREE.Camera; info: CameraInfo } {
  const az = view.azimuth_deg * DEG;
  const el = view.elevation_deg * DEG;
  const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
  const center = new THREE.Vector3(...framing.center);
  const r = framing.radius;
  let camera: THREE.Camera;
  if (view.projection === "orthographic") {
    const h = r * 1.1;
    const cam = new THREE.OrthographicCamera(-h * aspect, h * aspect, h, -h, 0.01, r * 10);
    cam.position.copy(center).addScaledVector(dir, r * 3);
    camera = cam;
  } else {
    const fov = 40;
    const dist = (r * (view.distance_factor ?? 1.0)) / Math.sin((fov / 2) * DEG) * 1.05;
    const cam = new THREE.PerspectiveCamera(fov, aspect, 0.01, dist * 10);
    cam.position.copy(center).addScaledVector(dir, dist);
    camera = cam;
  }
  // Grazing views near the pole would degenerate lookAt's up vector.
  camera.up.set(0, 1, 0);
  if (Math.abs(view.elevation_deg) > 89) camera.up.set(0, 0, -1);
  camera.lookAt(center);
  camera.updateMatrixWorld(true);
  const p = camera.position;
  return { camera, info: { position: [p.x, p.y, p.z], target: [center.x, center.y, center.z], projection: view.projection } };
}
