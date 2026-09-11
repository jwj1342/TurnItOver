// Program ABI v1 — the only contract a generated program must satisfy.
// A program is an ES module whose default export is `createObject(THREE)`.
// Naming: every part is a Group named `<part_id>`; joint pivots are Groups named `joint:<joint_id>`.
import type * as THREE from "three";

export type JointKind = "revolute" | "prismatic";

export interface JointHandle {
  type: JointKind;
  part: string;                 // name of the driven part Group
  limits: [number, number];     // rad for revolute, m for prismatic; rest pose is 0
  set(value: number): void;     // absolute; implementations clamp to limits
}

export interface ArticulatedObject {
  root: THREE.Object3D;
  joints: Record<string, JointHandle>;
  dispose?(): void;
}

export type CreateObject = (three: typeof THREE) => ArticulatedObject;
