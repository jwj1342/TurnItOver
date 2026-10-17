You are repairing the current Three.js reconstruction in `candidate.html`.

The browser render check failed. Use the reported failure below to diagnose the HTML/JavaScript and edit `candidate.html` directly so it renders successfully.

Rules:
- Preserve the intended object reconstruction whenever possible.
- Fix the smallest relevant cause first.
- Keep a visible canvas and working Three.js render loop.
- Preserve the candidate's fixed neutral evaluation environment when it exists: restrained base fill, key/fill/rim lights, fixed exposure/tone mapping, and RoomEnvironment + PMREM for scene.environment. Do not replace it with a strong AmbientLight-only setup merely to pass the render check.
- Do not create unrelated files.
- Finish only after `candidate.html` contains the repaired complete HTML.

Render failure:
{repair_message}
