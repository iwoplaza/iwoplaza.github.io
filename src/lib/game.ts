import { sdPlane } from '@typegpu/sdf';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { createOrbitCamera } from './camera.ts';
import { createFrog } from './frog.ts';
import {
  AABB,
  AABBHit,
  intersectAABB,
  MAX_AABBS,
  Shape,
  sdCone,
  sdCylinder,
  shapeUnion,
  sortHits,
} from './sdf.ts';
import { createGizmoState } from './gizmo.ts';

const INSPECT = false;
const pixelation = INSPECT ? 1 : 4;
const gameCameraOptions = {
  radius: 6,
  yaw: -Math.PI / 4,
  pitch: 0.8,
};
const inspectCameraOptions = {
  radius: 3,
  yaw: 0,
  pitch: 0,
};

export async function game(canvas: HTMLCanvasElement, signal: AbortSignal) {
  const root = await tgpu.init();
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  const camera = createOrbitCamera(
    root,
    canvas,
    INSPECT ? inspectCameraOptions : gameCameraOptions,
  );

  // Create uniform for AABBs
  const sceneAABBs = root.createUniform(d.arrayOf(AABB, MAX_AABBS));
  const numAABBs = root.createUniform(d.u32);

  function resizeCanvas(canvas: HTMLCanvasElement) {
    const devicePixelRatio = window.devicePixelRatio;
    const width = (window.innerWidth * devicePixelRatio) / pixelation;
    const height = (window.innerHeight * devicePixelRatio) / pixelation;
    canvas.width = width;
    canvas.height = height;
    camera.updateProjection(width, height);
  }

  resizeCanvas(canvas);
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas(canvas);
  });
  resizeObserver.observe(canvas);

  const time = root.createUniform(d.f32);

  const MAX_STEPS = 200;
  const MAX_DIST = 100;
  const SURF_DIST = 0.02;

  const skyColor = d.vec4f(0.7, 0.8, 0.9, 1);

  const checkerBoard = tgpu.fn(
    [d.vec2f],
    d.f32,
  )((uv) => {
    const fuv = std.floor(uv);
    return std.abs(fuv.x + fuv.y) % 2;
  });

  const getPineTree = tgpu.fn(
    [d.vec3f],
    Shape,
  )((p) => {
    // Repeat in XZ plane with 16x16 grid spacing
    const cellSize = d.f32(16);
    const cellId = std.floor(std.div(p.xz, cellSize));
    const cellP = std.sub(p.xz, std.mul(cellId, cellSize));

    // Add slight offset for each cell using hash-like function
    const hash = std.fract(
      std.sin(std.dot(cellId, d.vec2f(12.9898, 78.233))) * 43758.5453,
    );
    const offset = std.mul(std.sub(hash, 0.5), 2);
    const offsetPos = std.add(cellP, offset);

    // Center the tree in each cell
    const localP = d.vec3f(
      offsetPos.x - cellSize * 0.5,
      p.y,
      offsetPos.y - cellSize * 0.5,
    );

    const scaledP = std.div(localP, 4);

    // Trunk
    const trunkHeight = d.f32(2);
    const trunkRadius = d.f32(0.15);
    const trunk = Shape({
      dist: std.mul(sdCylinder(scaledP, trunkRadius, trunkHeight), 4),
      color: d.vec3f(0.4, 0.2, 0.1),
    });

    // Pine tree layers (cones stacked on top of each other)
    let tree = trunk;

    // Bottom layer
    const layer1Pos = std.sub(scaledP, d.vec3f(0, 1.8, 0));
    const layer1 = Shape({
      dist: std.mul(sdCone(layer1Pos, d.vec2f(std.sin(1), std.cos(1)), 1), 4),
      color: d.vec3f(0.1, 0.4, 0.1),
    });
    tree = shapeUnion(tree, layer1);

    // Middle layer
    const layer2Pos = std.sub(scaledP, d.vec3f(0, 2.4, 0));
    const layer2 = Shape({
      dist: std.mul(
        sdCone(layer2Pos, d.vec2f(std.sin(1.1), std.cos(1.1)), 1),
        4,
      ),
      color: d.vec3f(0.15, 0.5, 0.15),
    });
    tree = shapeUnion(tree, layer2);

    // Top layer
    const layer3Pos = std.sub(scaledP, d.vec3f(0, 3, 0));
    const layer3 = Shape({
      dist: std.mul(
        sdCone(layer3Pos, d.vec2f(std.sin(1.2), std.cos(1.2)), 1),
        4,
      ),
      color: d.vec3f(0.2, 0.6, 0.2),
    });
    tree = shapeUnion(tree, layer3);

    return tree;
  });

  const frog = createFrog(root);
  const gizmoState = createGizmoState(root);

  const getSceneDist = tgpu.fn(
    [d.vec3f],
    Shape,
  )((p) => {
    const frogShape = frog.getFrog(p);
    const tree = getPineTree(p);
    const floor = Shape({
      dist: sdPlane(p, d.vec3f(0, 1, 0), 0),
      color: std.mix(
        d.vec3f(0.3, 0.8, 0.4),
        d.vec3f(0.2, 0.6, 0.3),
        checkerBoard(std.mul(p.xz, 0.5)),
      ),
    });

    let scene = floor;
    if (!INSPECT) {
      scene = shapeUnion(scene, tree);
    }
    scene = shapeUnion(scene, frogShape);
    return scene;
  });

  const createArray = tgpu.fn([], d.arrayOf(AABBHit, MAX_AABBS))`() {
    return array<AABBHit, MAX_AABBS>();
  }`.$uses({ AABBHit, MAX_AABBS });

  const rayMarch = tgpu.fn(
    [d.vec3f, d.vec3f],
    Shape,
  )((ro, rd) => {
    let dO = d.f32(0);
    const result = Shape({
      dist: d.f32(MAX_DIST),
      color: d.vec3f(0, 0, 0),
    });

    // Get all AABB intersections
    let step = d.u32(0);
    const hits = createArray();
    let numHits = d.u32(0);

    for (let i = d.u32(0); i < std.min(numAABBs.$, MAX_AABBS); i++) {
      const hit = intersectAABB(ro, rd, sceneAABBs.$[i]);
      if (hit.enter <= hit.exit && hit.exit >= 0) {
        // Valid intersection
        hits[numHits] = hit;
        numHits++;
      }
    }

    // Sort hits by enter distance
    sortHits(hits, d.i32(numHits));

    // March through each AABB
    for (let hitIndex = d.u32(0); hitIndex < numHits; hitIndex++) {
      const hit = hits[hitIndex];

      // Start marching from max (current distance, enter point)
      dO = std.max(dO, hit.enter);

      // March until we exit this AABB
      while (dO < hit.exit && dO < MAX_DIST) {
        const p = std.add(ro, std.mul(rd, dO));
        const scene = getSceneDist(p);

        if (scene.dist < SURF_DIST) {
          result.dist = dO;
          result.color = scene.color;
          return result;
        }

        dO += scene.dist;
        step++;
        if (step > MAX_STEPS) {
          result.dist = MAX_DIST;
          return result;
        }
      }
    }

    result.dist = MAX_DIST;
    return result;
  });

  const softShadow = tgpu.fn(
    [d.vec3f, d.vec3f, d.f32, d.f32, d.f32],
    d.f32,
  )((ro, rd, minT, maxT, k) => {
    let res = d.f32(1);
    let t = minT;

    for (let i = 0; i < 32; i++) {
      if (t >= maxT) break;
      const h = getSceneDist(std.add(ro, std.mul(rd, t))).dist;
      if (h < 0.0) return 0;
      res = std.min(res, (k * h) / t);
      t += std.max(h, 0.01);
    }

    return res;
  });

  const getNormal = tgpu.fn(
    [d.vec3f],
    d.vec3f,
  )((p) => {
    const dist = getSceneDist(p).dist;
    const e = 0.01;

    const n = d.vec3f(
      getSceneDist(std.add(p, d.vec3f(e, 0, 0))).dist - dist,
      getSceneDist(std.add(p, d.vec3f(0, e, 0))).dist - dist,
      getSceneDist(std.add(p, d.vec3f(0, 0, e))).dist - dist,
    );

    return std.normalize(n);
  });

  const vertexMain = tgpu['~unstable'].vertexFn({
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

  const fragmentMain = tgpu['~unstable'].fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })((input) => {
    const uv = std.sub(std.mul(input.uv, 2), 1);

    // Ray origin and direction
    const ro = std.mul(camera.pov.$.invView, d.vec4f(0, 0, 0, 1)).xyz;
    const rd = std.normalize(
      std.mul(camera.pov.$.invViewProj, d.vec4f(uv.x, uv.y, 1, 0)).xyz,
    );

    const march = rayMarch(ro, rd);

    const p = std.add(ro, std.mul(rd, march.dist));
    const n = getNormal(p);

    // Lighting with orbiting light
    const l = std.normalize(d.vec3f(0.2, 1, 1));
    const diff = std.max(std.dot(n, l), 0);

    // Soft shadows
    const shadowRo = p;
    const shadowRd = l;
    // Max dist when looking for the shadow caster.
    // Can be concrete with a point light, but with a directional
    // light, it's a limit
    const shadowDist = d.f32(32);
    const shadow = softShadow(shadowRo, shadowRd, 0.1, shadowDist, 16);

    // Combine lighting with shadows and color
    const litColor = std.mul(march.color, 0.3 + std.min(diff, shadow) * 0.7);

    if (march.dist >= MAX_DIST) {
      return skyColor;
    }

    return d.vec4f(litColor, 1);
  });

  const renderPipeline = root['~unstable']
    .withVertex(vertexMain, {})
    .withFragment(fragmentMain, { format: presentationFormat })
    .createPipeline();

  const context = canvas.getContext('webgpu') as GPUCanvasContext;

  context.configure({
    device: root.device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  // Function to update scene AABBs
  function updateSceneAABBs() {
    const aabbs = Array.from({ length: MAX_AABBS }, () =>
      AABB({
        min: d.vec3f(),
        max: d.vec3f(),
      }),
    );

    // AABB for the frog
    aabbs[0] = AABB({
      min: d.vec3f(-2, 0, -2), // Approximate bounds
      max: d.vec3f(2, 7, 2),
    });

    // AABB for the floor
    aabbs[1] = AABB({
      min: d.vec3f(-100, -1, -100), // Approximate bounds
      max: d.vec3f(100, 0, 100),
    });

    if (!INSPECT) {
      // AABB for the infinite repeating trees
      aabbs[2] = AABB({
        min: d.vec3f(-100, 0, -100),
        max: d.vec3f(100, 16, 100),
      });
    }

    // Update the uniforms
    sceneAABBs.write(aabbs);
    numAABBs.write(3);
  }

  let animationFrame: number;
  let lastTime: undefined | number;
  function run(timestamp: number) {
    if (lastTime === undefined) {
      lastTime = timestamp;
    }
    const dt = (timestamp - lastTime) * 0.001;
    lastTime = timestamp;
    time.write((timestamp / 1000) % 1000);

    gizmoState.enable();
    frog.update(dt);
    frog.uploadRig();
    updateSceneAABBs();

    const view = context.getCurrentTexture().createView();
    renderPipeline
      .withColorAttachment({
        view,
        clearValue: [1, 1, 1, 1],
        loadOp: 'clear',
        storeOp: 'store',
      })
      .draw(3);

    gizmoState.draw(view);
    gizmoState.disable();
    animationFrame = requestAnimationFrame(run);
  }
  requestAnimationFrame(run);

  signal.addEventListener('abort', () => {
    cancelAnimationFrame(animationFrame);
    resizeObserver.disconnect();
    root.destroy();
  });
}
