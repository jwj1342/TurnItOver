"""One authored procedural task; public requirements and private scoring source."""
from __future__ import annotations

import math
import re

from turnitover.core.program import ObjectProgram
from turnitover.core.spec import AssetSpec, JointSpec, JointType, MaterialSpec, PartSpec

TASK = """Reconstruct this simple articulated demonstration object in metres, Y up, +Z front.
Use exactly three named box parts (dimensions X,Y,Z, centres at rest):
base: (1,0.1,0.6) at (0,0.05,0), tan #b08a5a;
drawer: (0.4,0.2,0.4) at (-0.25,0.25,0), blue #5a7fb0;
door: (0.4,0.5,0.04) at (0.25,0.4,0.22), red #8a5a5a.
No additional visible components. Joint order is drawer_slide, then door_hinge.
drawer_slide is prismatic, drives drawer along world +Z from 0 to 0.3 m.
door_hinge is revolute, drives door about world Y through (0.05,0.4,0.22),
from 0 to pi/2 radians, negative rotation (opens towards the viewer).
Joint values are absolute, clamped, and zero is the rest state.
Use Groups named base, drawer, door and pivots joint:drawer_slide, joint:door_hinge.
All materials are matte. Compare rest and actuated geometry and runtime budgets.
Verification scope is geometry, joint motion and runtime only; material appearance is reconstruction
guidance, and material defects are outside the diagnostic scoring scope.
Only report defects supported by acquired evidence; unobserved motion is not established.
"""


def reference_program() -> ObjectProgram:
    spec = AssetSpec("api_demo", "procedural_demo", (
        PartSpec("base", None, (0, .05, 0), size=(1, .1, .6), material="base"),
        PartSpec("drawer", None, (-.25, .25, 0), size=(.4, .2, .4), material="drawer"),
        PartSpec("door", None, (.25, .4, .22), size=(.4, .5, .04), material="door"),
    ), (
        JointSpec("drawer_slide", "drawer", JointType.PRISMATIC, (0, 0, 1), (-.25, .25, 0), (0, .3)),
        JointSpec("door_hinge", "door", JointType.REVOLUTE, (0, 1, 0), (.05, .4, .22), (0, math.pi/2), sign=-1),
    ), tuple(MaterialSpec(pid, color, roughness=.8) for pid, color in (
        ("base", "#b08a5a"), ("drawer", "#5a7fb0"), ("door", "#8a5a5a"))))
    return ObjectProgram.from_spec(spec)


def generation_prompt(abi: str, task: str = TASK) -> str:
    geometry = ("Use one ordinary BoxGeometry per part " if task == TASK else
                "Use Three.js geometry appropriate to the public reference and task ")
    return ("Generate an offline deterministic Three.js TypeScript ES module from the public task "
            "and actual reference image. Output one TypeScript code block only. "
            "Use export default function createObject(THREE). No imports at runtime, external assets, "
            "randomness, time, animation loops, lights or cameras. " + geometry +
            "and MeshStandardMaterial, preserve all named groups and joint pivots.\nProgram ABI:\n"
            + abi + "\nPublic task:\n" + task)


def controlled_cases(base: ObjectProgram) -> dict[str, ObjectProgram]:
    """Inject through existing public ABI handles; never substitute reference code."""
    source, count = re.subn(r"export\s+default\s+function\s+createObject\b",
                            "function createInitialObject", base.source)
    if count != 1:
        raise ValueError("Controlled pilot requires the documented named default factory")
    wrapper = "\nexport default function createObject(THREE) {\n const obj = createInitialObject(THREE);\n"
    # Constants are visible to the repair generator in its candidate source, never to the verifier.
    offset = """ const part = obj.root.getObjectByName('drawer');
 const initialX = part.position.x;
 const set = obj.joints.drawer_slide.set.bind(obj.joints.drawer_slide);
 obj.joints.drawer_slide.set = (v) => { set(v); part.position.x = initialX + 0.15; };
"""
    axis = """ const pivot = obj.root.getObjectByName('joint:door_hinge');
 obj.joints.door_hinge.set = (v) => {
   v = Math.min(Math.PI / 2, Math.max(0, v));
   pivot.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -v);
 };
"""
    return {"c00": base, "c01": ObjectProgram(source + wrapper + offset + " return obj;\n}\n"),
            "c02": ObjectProgram(source + wrapper + axis + " return obj;\n}\n")}
