// window.harness — the observation-action interface driven from Python via page.evaluate.
import * as THREE from "three";
import type { ArticulatedObject, JointHandle } from "./program_abi";
import type {
  ActuateResponse, Framing, GeometryExport, HarnessError, LoadRequest, LoadResponse,
  RuntimePayload, RuntimeProperty, ViewDef, ViewResponse,
} from "./protocol";
import { makeCamera } from "./camera";
import { collectStats, hierarchyOf, partBoxes, type PartBoxes } from "./stats";
import { exportParts } from "./export";

function fail(stage: HarnessError["stage"], message: string): never {
  const e: HarnessError = { stage, message };
  throw new Error(JSON.stringify(e));
}

class Harness {
  private canvas = document.getElementById("c") as HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private object: ArticulatedObject | null = null;
  private partNames = new Set<string>();
  private views = new Map<string, ViewDef>();
  private framing: Framing = { center: [0, 0, 0], radius: 1 };
  private camera: THREE.Camera | null = null;
  private cameraInfo: ViewResponse["camera"] | null = null;
  private jointValues: Record<string, number> = {};
  private snapshot: { since: string; boxes: PartBoxes } | null = null;
  private lastRenderMs = 0;

  constructor() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.scene.background = new THREE.Color(0x202024);
    const key = new THREE.DirectionalLight(0xffffff, 2.5); key.position.set(2, 4, 3);
    const fill = new THREE.DirectionalLight(0xffffff, 0.8); fill.position.set(-3, 1, -2);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6), key, fill);
  }

  gl_info(): { renderer: string; vendor: string } {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return {
      renderer: ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER)),
      vendor: ext ? String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL)) : String(gl.getParameter(gl.VENDOR)),
    };
  }

  async load(req: LoadRequest): Promise<LoadResponse> {
    this.dispose();
    this.renderer.setSize(req.width, req.height, false);
    this.views = new Map(req.views.map((v) => [v.id, v]));
    let mod: { default?: unknown };
    const url = URL.createObjectURL(new Blob([req.source_js], { type: "text/javascript" }));
    try {
      mod = await import(/* @vite-ignore */ url);
    } catch (e) {
      fail("import", String(e));
    } finally {
      URL.revokeObjectURL(url);
    }
    if (typeof mod.default !== "function") fail("validate", "default export is not a function");
    let obj: ArticulatedObject;
    try {
      obj = (mod.default as (t: typeof THREE) => ArticulatedObject)(THREE);
    } catch (e) {
      fail("create", String(e));
    }
    this.validate(obj);
    this.object = obj;
    this.scene.add(obj.root);
    this.partNames = new Set<string>();
    obj.root.traverse((o) => { if (o.name && !o.name.startsWith("joint:") && !o.name.endsWith("#mesh") && o !== obj.root) this.partNames.add(o.name); });
    this.jointValues = {};
    for (const id of Object.keys(obj.joints)) { obj.joints[id].set(0); this.jointValues[id] = 0; }
    obj.root.updateMatrixWorld(true);
    this.framing = req.framing === "auto" ? this.autoFraming(obj.root) : req.framing;
    this.snapshot = { since: "load", boxes: partBoxes(obj.root, this.partNames) };
    this.camera = null; this.cameraInfo = null;
    return {
      ok: true,
      parts: [...this.partNames],
      joints: Object.entries(obj.joints).map(([id, j]) => ({ id, type: j.type, part: j.part, limits: j.limits })),
      framing: this.framing,
    };
  }

  requestView(req: { view_id: string }): ViewResponse {
    const obj = this.require();
    const view = this.views.get(req.view_id);
    if (!view) fail("runtime", `unknown view_id ${req.view_id}`);
    const { camera, info } = makeCamera(view, this.framing, this.canvas.width / this.canvas.height);
    this.camera = camera; this.cameraInfo = info;
    const img = this.render(obj);
    return { view_id: req.view_id, camera: info, ...img };
  }

  actuateJoint(req: { joint_id: string; value: number }): ActuateResponse {
    const obj = this.require();
    const j: JointHandle | undefined = obj.joints[req.joint_id];
    if (!j) fail("runtime", `unknown joint_id ${req.joint_id}`);
    this.snapshot = { since: `actuate:${req.joint_id}:${req.value}`, boxes: partBoxes(obj.root, this.partNames) };
    j.set(req.value);
    this.jointValues[req.joint_id] = Math.min(j.limits[1], Math.max(j.limits[0], req.value));
    obj.root.updateMatrixWorld(true);
    if (!this.camera) this.requestView({ view_id: this.views.keys().next().value as string });
    const img = this.render(obj);
    return { joint_id: req.joint_id, value: this.jointValues[req.joint_id], ...img };
  }

  setJoints(values: Record<string, number>): { ok: true } {
    const obj = this.require();
    for (const [id, v] of Object.entries(values)) {
      const j = obj.joints[id]; if (!j) fail("runtime", `unknown joint_id ${id}`);
      j.set(v); this.jointValues[id] = Math.min(j.limits[1], Math.max(j.limits[0], v));
    }
    obj.root.updateMatrixWorld(true);
    return { ok: true };
  }

  queryRuntime(req: { property: RuntimeProperty }): RuntimePayload {
    const obj = this.require();
    switch (req.property) {
      case "stats":
        if (!this.camera) this.requestView({ view_id: this.views.keys().next().value as string });
        return collectStats(this.renderer, obj.root, this.lastRenderMs);
      case "hierarchy":
        return { property: "hierarchy", root: hierarchyOf(obj.root) };
      case "joint_state":
        return { property: "joint_state", joints: { ...this.jointValues } };
      case "state_delta": {
        const now = partBoxes(obj.root, this.partNames);
        const prev = this.snapshot?.boxes ?? now;
        const parts: Record<string, { aabb_min_delta: [number, number, number]; aabb_max_delta: [number, number, number] }> = {};
        for (const name of Object.keys(now)) {
          const a = prev[name] ?? now[name]; const b = now[name];
          parts[name] = {
            aabb_min_delta: [b.min[0] - a.min[0], b.min[1] - a.min[1], b.min[2] - a.min[2]],
            aabb_max_delta: [b.max[0] - a.max[0], b.max[1] - a.max[1], b.max[2] - a.max[2]],
          };
        }
        return { property: "state_delta", since: this.snapshot?.since ?? "load", parts };
      }
    }
    fail("runtime", `unknown property ${String(req.property)}`);
  }

  exportGeometry(): GeometryExport {
    const obj = this.require();
    return { joint_state: { ...this.jointValues }, parts: exportParts(obj.root, this.partNames) };
  }

  dispose(): { ok: true } {
    if (this.object) {
      this.scene.remove(this.object.root);
      try { this.object.dispose?.(); } catch { /* best effort */ }
      this.object = null;
    }
    this.renderer.renderLists.dispose();
    return { ok: true };
  }

  private require(): ArticulatedObject {
    if (!this.object) fail("runtime", "no program loaded");
    return this.object;
  }

  private validate(obj: ArticulatedObject): void {
    if (!obj || !(obj.root as THREE.Object3D)?.isObject3D) fail("validate", "root is not an Object3D");
    if (!obj.joints || typeof obj.joints !== "object") fail("validate", "joints missing");
    const names = new Set<string>();
    obj.root.traverse((o) => {
      if (!o.name || o === obj.root || (o as THREE.Mesh).isMesh || o.name.startsWith("joint:")) return;
      if (names.has(o.name)) fail("validate", `duplicate part name ${o.name}`);
      names.add(o.name);
    });
    for (const [id, j] of Object.entries(obj.joints)) {
      if (typeof j.set !== "function") fail("validate", `joint ${id} has no set()`);
      if (!names.has(j.part)) fail("validate", `joint ${id} drives unknown part ${j.part}`);
    }
  }

  private autoFraming(root: THREE.Object3D): Framing {
    const box = new THREE.Box3().setFromObject(root);
    const c = box.getCenter(new THREE.Vector3());
    const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
    return { center: [c.x, c.y, c.z], radius: r };
  }

  private render(obj: ArticulatedObject): { image_png_b64: string; width: number; height: number; render_ms: number } {
    if (!this.camera) fail("runtime", "no camera");
    obj.root.updateMatrixWorld(true);
    const t0 = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.lastRenderMs = performance.now() - t0;
    const dataUrl = this.canvas.toDataURL("image/png");
    return { image_png_b64: dataUrl.slice(dataUrl.indexOf(",") + 1), width: this.canvas.width, height: this.canvas.height, render_ms: this.lastRenderMs };
  }
}

declare global { interface Window { harness: Harness; harnessReady: boolean } }
window.harness = new Harness();
window.harnessReady = true;
