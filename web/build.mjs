// Builds web/dist: three.esm.js (shared Three instance), harness.js, index.html.
import { build } from "esbuild";
import { copyFileSync, mkdirSync } from "node:fs";

mkdirSync("dist", { recursive: true });

await build({
  entryPoints: ["src/three-entry.ts"],
  bundle: true,
  format: "esm",
  target: "es2022",
  outfile: "dist/three.esm.js",
  logLevel: "warning",
});

await build({
  entryPoints: ["src/harness.ts"],
  bundle: true,
  format: "esm",
  target: "es2022",
  external: ["three"],
  outfile: "dist/harness.js",
  logLevel: "warning",
});

copyFileSync("src/index.html", "dist/index.html");
console.log("built web/dist");
