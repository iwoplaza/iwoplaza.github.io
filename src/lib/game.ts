import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { mat4 } from "wgpu-matrix";
import { sdBoxFrame3d, sdPlane, sdSphere } from "@typegpu/sdf";

const sdCylinder = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, r, h) => {
  const dd = d.vec2f(std.length(p.xz), p.y);
  const q = d.vec2f(dd.x - r, std.abs(dd.y) - h/2);
  return std.min(std.max(q.x, q.y), 0) + std.length(std.max(q, d.vec2f()));
});

const sdCone = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, h, r) => {
  const q = d.vec2f(std.length(p.xz), p.y);
  const c = d.vec2f(r, h);
  const a = std.mul(q, d.vec2f(c.y, -c.x));
  const b = std.mul(q, d.vec2f(c.x, c.y));
  const k = std.select(std.dot(q, c), std.length(q), c.y * q.x > c.x * q.y);
  return std.length(std.max(a, d.vec2f())) + std.min(k, 0);
});

const smoothstep = tgpu.fn([d.f32, d.f32, d.f32], d.f32)`(a, b, t) {
  return smoothstep(a, b, t);
}`;

export async function game(canvas: HTMLCanvasElement, signal: AbortSignal) {
  const root = await tgpu.init();
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  const uniforms = root.createUniform(
    d.struct({
      invView: d.mat4x4f,
      invViewProj: d.mat4x4f,
    }),
  );

  let invProj = mat4.identity(d.mat4x4f());
  let invView = mat4.identity(d.mat4x4f());

  function uploadUniforms() {
    const invViewProj = mat4.mul(invView, invProj, d.mat4x4f());
    uniforms.writePartial({ invView, invViewProj });
  }

  function resizeCanvas(canvas: HTMLCanvasElement) {
    const devicePixelRatio = window.devicePixelRatio;
    const width = window.innerWidth * devicePixelRatio;
    const height = window.innerHeight * devicePixelRatio;
    canvas.width = width;
    canvas.height = height;

    const aspect = canvas.width / canvas.height;
    const fov = (50 / 180) * Math.PI;
    mat4.identity(invProj);
    mat4.scale(invProj, d.vec3f(aspect, 1, 1/Math.tan(fov)), invProj);
    uploadUniforms();
  }

  resizeCanvas(canvas);
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas(canvas);
  });
  resizeObserver.observe(canvas);

  let isDragging = false;
  let prevX = 0;
  let prevY = 0;
  const orbitOrigin = d.vec3f(0, 2, 0);
  // Yaw and pitch angles facing the origin.
  let orbitRadius = 2;
  let orbitYaw = 0;
  let orbitPitch = 0.4;

  function updateCameraOrbit(dx: number, dy: number) {
    console.log({
      orbitRadius,
      orbitYaw,
      orbitPitch,
    });

    const orbitSensitivity = 0.005;
    orbitYaw += -dx * orbitSensitivity;
    orbitPitch += dy * orbitSensitivity;
    // if we didn't limit pitch, it would lead to flipping the camera which is disorienting.
    const maxPitch = Math.PI / 2 - 0.01;
    if (orbitPitch > maxPitch) orbitPitch = maxPitch;
    if (orbitPitch < -maxPitch) orbitPitch = -maxPitch;
    // basically converting spherical coordinates to cartesian.
    // like sampling points on a unit sphere and then scaling them by the radius.
    const logOrbitRadius = orbitRadius ** 2;
    const newCamX = logOrbitRadius * -Math.sin(orbitYaw) * Math.cos(orbitPitch);
    const newCamY = logOrbitRadius * Math.sin(orbitPitch);
    const newCamZ = logOrbitRadius * Math.cos(orbitYaw) * Math.cos(orbitPitch);
    const newCameraPos = std.add(d.vec3f(newCamX, newCamY, newCamZ), orbitOrigin);

    invView = mat4.aim(newCameraPos, orbitOrigin, d.vec3f(0, 1, 0), d.mat4x4f());
    uploadUniforms();
  }

  canvas.addEventListener("wheel", (event: WheelEvent) => {
    event.preventDefault();
    const zoomSensitivity = 0.005;
    orbitRadius = Math.max(1, orbitRadius + event.deltaY * zoomSensitivity);
    updateCameraOrbit(0, 0);
  });

  canvas.addEventListener("mousedown", (event) => {
    if (event.button === 0) {
      // Left Mouse Button controls Camera Orbit.
      isDragging = true;
    }
    prevX = event.clientX;
    prevY = event.clientY;
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  canvas.addEventListener("mousemove", (event) => {
    const dx = event.clientX - prevX;
    const dy = event.clientY - prevY;
    prevX = event.clientX;
    prevY = event.clientY;

    if (isDragging) {
      updateCameraOrbit(dx, dy);
    }
  });

  // Mobile touch support.
  canvas.addEventListener("touchstart", (event: TouchEvent) => {
    event.preventDefault();
    if (event.touches.length === 1) {
      // Single touch controls Camera Orbit.
      isDragging = true;
    }
    // Use the first touch for rotation.
    prevX = event.touches[0].clientX;
    prevY = event.touches[0].clientY;
  });

  canvas.addEventListener("touchmove", (event: TouchEvent) => {
    event.preventDefault();
    const touch = event.touches[0];
    const dx = touch.clientX - prevX;
    const dy = touch.clientY - prevY;
    prevX = touch.clientX;
    prevY = touch.clientY;

    if (isDragging && event.touches.length === 1) {
      updateCameraOrbit(dx, dy);
    }
  });

  canvas.addEventListener("touchend", (event: TouchEvent) => {
    event.preventDefault();
    if (event.touches.length === 0) {
      isDragging = false;
    }
  });

  const time = root.createUniform(d.f32);

  const MAX_STEPS = 1000;
  const MAX_DIST = 30;
  const SURF_DIST = 0.001;

  const skyColor = d.vec4f(0.7, 0.8, 0.9, 1);

  // Structure to hold both distance and color
  const Shape = d.struct({
    color: d.vec3f,
    dist: d.f32,
  });

  const checkerBoard = tgpu.fn([d.vec2f], d.f32)((uv) => {
    const fuv = std.floor(uv);
    return std.abs(fuv.x + fuv.y) % 2;
  });

  const smoothShapeUnion = tgpu.fn([Shape, Shape, d.f32], Shape)((a, b, k) => {
    const h = std.max(k - std.abs(a.dist - b.dist), 0) / k;
    const m = h * h;

    // Smooth min for distance
    const dist = std.min(a.dist, b.dist) - m * k * (1 / d.f32(4));

    // Blend colors based on relative distances and smoothing
    const weight = m + std.select(0, 1 - m, a.dist > b.dist);
    const color = std.mix(a.color, b.color, weight);

    return { dist, color };
  });

  const shapeUnion = tgpu.fn([Shape, Shape], Shape)((a, b) => ({
    color: std.select(a.color, b.color, a.dist > b.dist),
    dist: std.min(a.dist, b.dist),
  }));

  const getPineTree = tgpu.fn([d.vec3f], Shape)((p) => {
    const treePos = d.vec3f(-3, 0, 2);
    const localP = std.sub(p, treePos);
    
    // Trunk
    const trunkHeight = d.f32(2);
    const trunkRadius = d.f32(0.15);
    const trunk = Shape({
      dist: sdCylinder(localP, trunkRadius, trunkHeight),
      color: d.vec3f(0.4, 0.2, 0.1),
    });
    
    // Pine tree layers (cones stacked on top of each other)
    let tree = trunk;
    
    // Bottom layer
    const layer1Pos = std.sub(localP, d.vec3f(0, 1.2, 0));
    const layer1 = Shape({
      dist: sdCone(layer1Pos, 1.2, 0.8),
      color: d.vec3f(0.1, 0.4, 0.1),
    });
    tree = shapeUnion(tree, layer1);
    
    // Middle layer
    const layer2Pos = std.sub(localP, d.vec3f(0, 1.8, 0));
    const layer2 = Shape({
      dist: sdCone(layer2Pos, 1.0, 0.6),
      color: d.vec3f(0.15, 0.5, 0.15),
    });
    tree = shapeUnion(tree, layer2);
    
    // Top layer
    const layer3Pos = std.sub(localP, d.vec3f(0, 2.3, 0));
    const layer3 = Shape({
      dist: sdCone(layer3Pos, 0.8, 0.4),
      color: d.vec3f(0.2, 0.6, 0.2),
    });
    tree = shapeUnion(tree, layer3);
    
    return tree;
  });

  const getMorphingShape = tgpu.fn([d.vec3f, d.f32], Shape)((p, t) => {
    // Center position
    const center = d.vec3f(0, 2, 0);
    const localP = std.sub(p, center);
    const rotMatZ = d.mat4x4f.rotationZ(-t);
    const rotMatX = d.mat4x4f.rotationX(-t * 0.6);
    const rotatedP = std.mul(rotMatZ, std.mul(rotMatX, d.vec4f(localP, 1))).xyz;

    // Animate shapes
    const boxSize = d.vec3f(0.7);

    // Create two spheres that move in a circular pattern
    const sphere1Offset = d.vec3f(
      std.cos(t * 2) * 0.8,
      std.sin(t * 3) * 0.3,
      std.sin(t * 2) * 0.8,
    );
    const sphere2Offset = d.vec3f(
      std.cos(t * 2 + 3.14) * 0.8,
      std.sin(t * 3 + 1.57) * 0.3,
      std.sin(t * 2 + 3.14) * 0.8,
    );

    // Calculate distances and assign colors
    const sphere1 = Shape({
      dist: sdSphere(std.sub(localP, sphere1Offset), 0.5),
      color: d.vec3f(0.4, 0.5, 1),
    });
    const sphere2 = Shape({
      dist: sdSphere(std.sub(localP, sphere2Offset), 0.3),
      color: d.vec3f(1, 0.8, 0.2),
    });
    const box = Shape({
      dist: sdBoxFrame3d(rotatedP, boxSize, 0.1),
      color: d.vec3f(1.0, 0.3, 0.3),
    });

    // Smoothly blend shapes and colors
    const spheres = smoothShapeUnion(sphere1, sphere2, 0.1);
    return smoothShapeUnion(spheres, box, 0.2);
  });

  const getSceneDist = tgpu.fn([d.vec3f], Shape)((p) => {
    const shape = getMorphingShape(p, time.$);
    const tree = getPineTree(p);
    const floor = Shape({
      dist: sdPlane(p, d.vec3f(0, 1, 0), 0),
      color: std.mix(
        d.vec3f(1),
        d.vec3f(0.2),
        checkerBoard(std.mul(p.xz, 2)),
      ),
    });

    const sceneWithTree = shapeUnion(shape, tree);
    return shapeUnion(sceneWithTree, floor);
  });

  const rayMarch = tgpu.fn([d.vec3f, d.vec3f], Shape)((ro, rd) => {
    let dO = d.f32(0);
    const result = Shape({
      dist: d.f32(MAX_DIST),
      color: d.vec3f(0, 0, 0),
    });

    for (let i = 0; i < MAX_STEPS; i++) {
      const p = std.add(ro, std.mul(rd, dO));
      const scene = getSceneDist(p);
      dO += scene.dist;

      if (dO > MAX_DIST || scene.dist < SURF_DIST) {
        result.dist = dO;
        result.color = scene.color;
        break;
      }
    }

    return result;
  });

  const softShadow = tgpu.fn(
    [d.vec3f, d.vec3f, d.f32, d.f32, d.f32],
    d.f32,
  )((ro, rd, minT, maxT, k) => {
    let res = d.f32(1);
    let t = minT;

    for (let i = 0; i < 100; i++) {
      if (t >= maxT) break;
      const h = getSceneDist(std.add(ro, std.mul(rd, t))).dist;
      if (h < 0.001) return 0;
      res = std.min(res, k * h / t);
      t += std.max(h, 0.001);
    }

    return res;
  });

  const getNormal = tgpu.fn([d.vec3f], d.vec3f)((p) => {
    const dist = getSceneDist(p).dist;
    const e = 0.01;

    const n = d.vec3f(
      getSceneDist(std.add(p, d.vec3f(e, 0, 0))).dist - dist,
      getSceneDist(std.add(p, d.vec3f(0, e, 0))).dist - dist,
      getSceneDist(std.add(p, d.vec3f(0, 0, e))).dist - dist,
    );

    return std.normalize(n);
  });

  const getOrbitingLightPos = tgpu.fn([d.f32], d.vec3f)((t) => {
    const radius = d.f32(3);
    const height = d.f32(6);
    const speed = d.f32(1);

    return d.vec3f(
      std.cos(t * speed) * radius,
      height + std.sin(t * speed) * radius,
      4,
    );
  });

  const vertexMain = tgpu["~unstable"].vertexFn({
    in: { idx: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, uv: d.vec2f },
  })(({ idx }) => {
    const pos = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
    const uv = [d.vec2f(0, 0), d.vec2f(2, 0), d.vec2f(0, 2)];

    return {
      pos: d.vec4f(pos[idx], 0.0, 1.0),
      uv: uv[idx],
    };
  });

  const fragmentMain = tgpu["~unstable"].fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })((input) => {
    const uv = std.sub(std.mul(input.uv, 2), 1);

    // Ray origin and direction
    const ro = std.mul(uniforms.$.invView, d.vec4f(0, 0, 0, 1)).xyz;
    const rd = std.normalize(std.mul(uniforms.$.invViewProj, d.vec4f(uv.x, uv.y, 1, 0)).xyz);

    const march = rayMarch(ro, rd);

    const fog = std.pow(std.min(march.dist / MAX_DIST, 1), 0.7);

    const p = std.add(ro, std.mul(rd, march.dist));
    const n = getNormal(p);

    // Lighting with orbiting light
    const lightPos = getOrbitingLightPos(time.$);
    const l = std.normalize(std.sub(lightPos, p));
    const diff = std.max(std.dot(n, l), 0);

    // Soft shadows
    const shadowRo = p;
    const shadowRd = l;
    const shadowDist = std.length(std.sub(lightPos, p));
    const shadow = softShadow(shadowRo, shadowRd, 0.1, shadowDist, 16);

    // Combine lighting with shadows and color
    const litColor = std.mul(march.color, diff);
    const finalColor = std.mix(
      std.mul(litColor, 0.5), // Shadow color
      litColor, // Lit color
      shadow,
    );

    return std.mix(d.vec4f(finalColor, 1), skyColor, fog);
  });

  const renderPipeline = root["~unstable"]
    .withVertex(vertexMain, {})
    .withFragment(fragmentMain, { format: presentationFormat })
    .createPipeline();

  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  context.configure({
    device: root.device,
    format: presentationFormat,
    alphaMode: "premultiplied",
  });

  let animationFrame: number;
  function run(timestamp: number) {
    time.write(timestamp / 1000 % 1000);

    renderPipeline
      .withColorAttachment({
        view: context.getCurrentTexture().createView(),
        clearValue: [1, 1, 1, 1],
        loadOp: "clear",
        storeOp: "store",
      })
      .draw(3);

    animationFrame = requestAnimationFrame(run);
  }
  requestAnimationFrame(run);

  signal.addEventListener("abort", () => {
    cancelAnimationFrame(animationFrame);
    resizeObserver.disconnect();
    root.destroy();
  });

  updateCameraOrbit(0, 0);
}
