// Audit adapter for the EXISTING native harness, not another scene executor.
// Adapted from the RP29 native material and owned-resource readbacks; see
// docs/native-scoring-provenance.json. Returned values are private audit data.
async arg => {
  const h = window.harness;
  if (arg.mode === 'disposed') {
    const t = window.__tioPrivateAudit;
    t.proto.dispatchEvent = t.original;
    const value = {events: t.events, resources: [...t.resources.values()],
      active_object_null: h.object === null, root_detached: t.root.parent === null};
    delete window.__tioPrivateAudit;
    return value;
  }
  const root = h.object.root;
  root.updateMatrixWorld(true);
  if (arg.mode === 'start') {
    let sample;
    root.traverse(o => { if (o.isMesh && !sample) sample = o.geometry; });
    if (!sample) throw Error('No mesh resource prototype to observe');
    let proto = sample;
    while (proto && !Object.hasOwn(proto, 'dispatchEvent')) proto = Object.getPrototypeOf(proto);
    if (!proto) throw Error('No native event-dispatch prototype');
    const t = window.__tioPrivateAudit = {root, proto, original: proto.dispatchEvent,
      events: [], resources: new Map()};
    proto.dispatchEvent = function(event) {
      if (event.type === 'dispose' && (this.isTexture || this.isMaterial || this.isBufferGeometry)) {
        t.events.push({uuid: this.uuid, kind: this.isTexture ? 'texture' : this.isMaterial ? 'material' : 'geometry',
          active_object_null: h.object === null, root_detached: root.parent === null});
      }
      return t.original.call(this, event);
    };
  }
  const t = window.__tioPrivateAudit;
  const textureCache = new WeakMap(), materialCache = new WeakMap();
  const hash = async array => {
    const data = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)))
      .map(x => x.toString(16).padStart(2, '0')).join('');
  };
  const own = (resource, kind) => {
    if (resource) t.resources.set(resource.uuid, {uuid: resource.uuid, kind});
  };
  const texture = async value => {
    if (!value) return null;
    if (textureCache.has(value)) return textureCache.get(value);
    own(value, 'texture');
    let data = value.image?.data;
    let width = value.image?.width, height = value.image?.height;
    if (!data && width && height) {
      const canvas = document.createElement('canvas');
      canvas.width = width; canvas.height = height;
      const context = canvas.getContext('2d');
      try { context.drawImage(value.image, 0, 0); data = context.getImageData(0, 0, width, height).data; }
      catch (_) { /* Unsupported image is explicit incomplete evidence. */ }
    }
    const result = {complete: !!data && ArrayBuffer.isView(data),
      rgba_sha256: data && ArrayBuffer.isView(data) ? await hash(data) : null,
      width: width ?? null, height: height ?? null, flipY: value.flipY,
      colorSpace: value.colorSpace, repeat: value.repeat.toArray(), offset: value.offset.toArray(),
      center: value.center.toArray(), rotation: value.rotation, matrix: value.matrix.toArray(),
      matrixAutoUpdate: value.matrixAutoUpdate, wrapS: value.wrapS, wrapT: value.wrapT,
      magFilter: value.magFilter, minFilter: value.minFilter,
      generateMipmaps: value.generateMipmaps, channel: value.channel,
      format: value.format, type: value.type};
    textureCache.set(value, result);
    return result;
  };
  const material = async m => {
    if (materialCache.has(m)) return materialCache.get(m);
    own(m, 'material');
    const properties = {};
    for (const key of ['color', 'specular', 'shininess', 'roughness', 'metalness', 'ior',
      'specularColor', 'specularIntensity', 'opacity', 'transparent', 'depthWrite', 'vertexColors',
      'emissive', 'emissiveIntensity', 'normalScale', 'normalMapType', 'side', 'flatShading']) {
      const value = m[key];
      properties[key] = value?.toArray ? value.toArray() : value ?? null;
    }
    const textures = {};
    for (const key of ['map', 'roughnessMap', 'metalnessMap', 'normalMap', 'emissiveMap',
      'aoMap', 'alphaMap', 'bumpMap', 'displacementMap', 'lightMap', 'envMap']) {
      textures[key] = await texture(m[key]);
    }
    // Observe the last real render; never init/compile a replacement program.
    const program = h.renderer.properties.get(m).currentProgram;
    const uniforms = program?.getUniforms().map;
    const gl = h.renderer.getContext();
    const gpu = {program_present: !!program};
    for (const key of ['roughness', 'metalness', 'emissive', 'normalScale']) {
      const u = uniforms?.[key];
      const value = u ? gl.getUniform(program.program, u.addr) : null;
      gpu[key] = ArrayBuffer.isView(value) ? Array.from(value) : value;
    }
    const shader = program ? gl.getShaderSource(program.fragmentShader) ?? '' : '';
    for (const key of ['ROUGHNESSMAP', 'METALNESSMAP', 'NORMALMAP', 'EMISSIVEMAP']) {
      gpu['uses_' + key.toLowerCase()] = new RegExp('^#define USE_' + key + '\\b', 'm').test(shader);
    }
    const result = {type: m.type, properties, textures, gpu,
      roughness_and_basecolor_share_texture: !!m.roughnessMap && m.roughnessMap === m.map};
    materialCache.set(m, result);
    return result;
  };
  const meshes = [];
  root.traverse(o => { if (o.isMesh) meshes.push(o); });
  const rows = [];
  // Area-integrated attribute moments survive subdivision and vertex
  // renumbering. Position correlations also catch flipped/shuffled UVs.
  const profile = (geometry, attribute) => {
    const position = geometry.getAttribute('position'), index = geometry.index;
    if (!ArrayBuffer.isView(position?.array) || position.itemSize !== 3 ||
        attribute.count !== position.count || attribute.normalized ||
        index && !ArrayBuffer.isView(index.array)) return null;
    const size = attribute.itemSize, count = index?.count ?? position.count;
    const mean = Array(size).fill(0), second = Array(size).fill(0), cross = Array(size * 3).fill(0);
    let total = 0;
    for (let i = 0; i + 2 < count; i += 3) {
      const ids = [0, 1, 2].map(j => index ? index.array[i + j] : i + j);
      const p = ids.map(id => Array.from(position.array.slice(id * 3, id * 3 + 3)));
      const u = p[1].map((x, j) => x - p[0][j]), v = p[2].map((x, j) => x - p[0][j]);
      const area = Math.hypot(u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]);
      if (!area) continue;
      total += area;
      for (let k = 0; k < size; k++) {
        const values = ids.map(id => attribute.array[id * size + k]);
        const sum = values.reduce((a, b) => a + b, 0);
        mean[k] += area * sum / 3;
        second[k] += area * (sum * sum + values.reduce((a, b) => a + b * b, 0)) / 12;
        for (let j = 0; j < 3; j++) cross[k * 3 + j] += area *
          (sum * p.reduce((a, b) => a + b[j], 0) + values.reduce((a, b, l) => a + b * p[l][j], 0)) / 12;
      }
    }
    return total > 0 ? {mean: mean.map(x => x / total), second: second.map(x => x / total),
      position_cross: cross.map(x => x / total)} : null;
  };
  for (const mesh of meshes) {
    own(mesh.geometry, 'geometry');
    const attributes = {};
    for (const key of ['uv', 'uv1', 'uv2', 'normal', 'tangent', 'color']) {
      const attribute = mesh.geometry.getAttribute(key);
      attributes[key] = attribute ? (ArrayBuffer.isView(attribute.array)
        ? {sha256: await hash(attribute.array), count: attribute.count, itemSize: attribute.itemSize,
           profile: profile(mesh.geometry, attribute)}
        : {complete: false, reason: 'Unsupported interleaved or GPU-only attribute'}) : null;
    }
    let owner = mesh;
    while (owner && !h.partNames.has(owner.name)) owner = owner.parent;
    const materials = [];
    for (const m of (Array.isArray(mesh.material) ? mesh.material : [mesh.material])) materials.push(await material(m));
    rows.push({part: owner?.name ?? null, mesh: mesh.name,
      supported_geometry: !mesh.isSkinnedMesh && !mesh.isInstancedMesh && !Object.keys(mesh.geometry.morphAttributes).length,
      vertices: mesh.geometry.getAttribute('position')?.count ?? 0,
      triangles: (mesh.geometry.index?.count ?? mesh.geometry.getAttribute('position')?.count ?? 0) / 3,
      attributes, materials});
  }
  return {schema_version: 1, meshes: rows,
    hierarchy: h.queryRuntime({property: 'hierarchy'}),
    tick_callback_present: typeof h.object.tick === 'function'};
}
