# Harness protocol

Source of truth: `web/src/program_abi.ts`, `web/src/protocol.ts`, mirrored by `turnitover/render/protocol.py`.

## Program ABI v1

A program is an ES module (TypeScript is transpiled by esbuild before injection) whose default export is
`createObject(THREE)` returning `{ root, joints, dispose? }`.

- `root`: a `THREE.Object3D`. Every part is a `Group` named `<part_id>`; meshes live under it and are named
  `<part_id>#mesh`. Joint pivots are `Group`s named `joint:<joint_id>`. Part names must be unique.
- `joints[id] = { type: "revolute"|"prismatic", part, limits: [lo, hi], set(value) }`. `set` is absolute and
  clamps to limits. Rest pose is `set(0)`.
- Programs may `import * as THREE from "three"` (resolved through the page import map to the shared
  instance) but receive `THREE` as an argument and need not import anything.

## Transport

Python (Playwright, sync API) calls `page.evaluate("(arg) => window.harness.<method>(arg)", arg)`.
`web/dist` is served from disk through `page.route("https://harness.local/**")`; no HTTP server.
Errors thrown by the harness are JSON `{stage, message}` with `stage ∈ import|create|validate|runtime`,
surfaced in Python as `HarnessError(stage, message)`. esbuild failures raise `CompileError`.

## Methods

| method | request | response |
|---|---|---|
| `load` | `{source_js, framing: Framing \| "auto", width, height, views: ViewDef[]}` | `{ok, parts: string[], joints: [{id,type,part,limits}], framing}` |
| `requestView` | `{view_id}` | `{view_id, image_png_b64, width, height, render_ms, camera:{position,target,projection}}` |
| `actuateJoint` | `{joint_id, value}` | `{joint_id, value, image_png_b64, width, height, render_ms}` (re-renders from the current camera) |
| `setJoints` | `{joint_id: value, ...}` | `{ok}` (engine helper, not a judge action) |
| `queryRuntime` | `{property}` | see payloads below |
| `exportGeometry` | – | `{joint_state, parts:[{name, vertex_count, triangle_count, positions_b64, indices_b64}]}` |
| `dispose` | – | `{ok}` |
| `gl_info` | – | `{renderer, vendor}` |

`exportGeometry` is a checker privilege, not part of the judge's action space.

### Runtime payloads

```jsonc
{"property":"stats","triangles_rendered":108,"draw_calls":9,"triangles_scene":108,"mesh_count":9,"geometries":9,"textures":0,"programs":1,"last_render_ms":12.3}
{"property":"hierarchy","root":{"name":"toy_cabinet","type":"Group","position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1],"children":[...]}}
{"property":"joint_state","joints":{"drawer_0":0.0,"drawer_1":0.35,"door":0.0}}
{"property":"state_delta","since":"actuate:drawer_1:0.35","parts":{"drawer_1":{"aabb_min_delta":[0,0,0.35],"aabb_max_delta":[0,0,0.35]}}}
```

Geometry arrays are little-endian `Float32` (xyz interleaved, world space) and `Uint32` (triangle indices),
base64 in JSON. Decode with `turnitover.render.protocol.decode_positions / decode_indices`.

## Rules

- **Framing**: the engine loads the reference program with `framing: "auto"`, records the returned
  `{center, radius}`, and passes that explicit framing for every corrupted variant. Otherwise scale and offset
  corruptions would be partially normalized away.
- **Detents**: `zero -> limits[0]`, `mid -> midpoint`, `limit -> limits[1]`.
- **Action cost**: `request_view`, `actuate_joint`, `query_runtime` cost 1; `emit_diagnosis`, `stop` cost 0.
  The loop forces `stop` when the budget would be exceeded.
- **Action keys** (detectability matrix columns): `view:<id>`, `joint:<id>:<detent>`, `runtime:<property>`.
- **Evidence state keys**: `rest`, `joint:<id>:<detent>`.
- **Rendering determinism**: `antialias:false`, pixel ratio 1, fixed lights and background, image taken with
  `canvas.toDataURL` right after `renderer.render`; SwiftShader (`--use-angle=swiftshader`).
