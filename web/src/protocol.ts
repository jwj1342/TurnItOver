// Request/response types of window.harness. Mirrored by turnitover/render/protocol.py.
export type Vec3 = [number, number, number];
export type Projection = "perspective" | "orthographic";

export interface Framing { center: Vec3; radius: number }

export interface ViewDef { id: string; azimuth_deg: number; elevation_deg: number; projection: Projection; distance_factor?: number }

export interface LoadRequest { source_js: string; framing: Framing | "auto"; width: number; height: number; views: ViewDef[] }
export interface LoadResponse {
  ok: true;
  parts: string[];
  joints: { id: string; type: string; part: string; limits: [number, number] }[];
  framing: Framing;
}

export interface CameraInfo { position: Vec3; target: Vec3; projection: Projection }
export interface ImageResponse { image_png_b64: string; width: number; height: number; render_ms: number }
export interface ViewResponse extends ImageResponse { view_id: string; camera: CameraInfo }
export interface ActuateResponse extends ImageResponse { joint_id: string; value: number }

export type RuntimeProperty = "stats" | "hierarchy" | "joint_state" | "state_delta";
export interface StatsPayload {
  property: "stats";
  triangles_rendered: number; draw_calls: number; triangles_scene: number; mesh_count: number;
  geometries: number; textures: number; programs: number; last_render_ms: number;
}
export interface HierarchyNode { name: string; type: string; position: Vec3; rotation: Vec3; scale: Vec3; children: HierarchyNode[] }
export interface HierarchyPayload { property: "hierarchy"; root: HierarchyNode }
export interface JointStatePayload { property: "joint_state"; joints: Record<string, number> }
export interface StateDeltaPayload {
  property: "state_delta"; since: string;
  parts: Record<string, { aabb_min_delta: Vec3; aabb_max_delta: Vec3 }>;
}
export type RuntimePayload = StatsPayload | HierarchyPayload | JointStatePayload | StateDeltaPayload;

export interface PartGeometry { name: string; vertex_count: number; triangle_count: number; positions_b64: string; indices_b64: string }
export interface GeometryExport { joint_state: Record<string, number>; parts: PartGeometry[] }

export interface HarnessError { stage: "import" | "create" | "validate" | "runtime"; message: string }
