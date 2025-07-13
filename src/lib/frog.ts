import { opSmoothUnion, opUnion, sdRoundedBox3d, sdSphere } from '@typegpu/sdf';
import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat3, mat4 } from 'wgpu-matrix';
import {
  opElongate,
  opSubtraction,
  Shape,
  sdCappedTorus,
  sdOctahedron,
  shapeUnion,
  smoothShapeUnion,
  sdVerticalCapsule,
} from './sdf.ts';

// Palette
// const skinColor = d.vec3f(0.3, 0.8, 0.4);
// const backpackColor = d.vec3f(0.4, 0.4, 0.1);
const skinColor = d.vec3f(0.8, 0.6, 0.2);
const backpackColor = d.vec3f(0.2, 0.4, 0.6);

const getFrogHead = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  const center = d.vec3f(0, 0.6, 0);
  const localP = std.sub(p, center);
  // Symmetric along the X-axis
  localP.x = std.abs(localP.x);
  let head = sdRoundedBox3d(localP, d.vec3f(0.8, 0.7, 0.6), 0.6);
  const frownAngle = 2.95;
  const frownP = d.vec3f(localP.x, -localP.y - 2.6, localP.z - 0.8);
  const lipP = std.add(frownP, d.vec3f(0, 0, 0.2));
  // Lip
  head = opSmoothUnion(
    head,
    sdCappedTorus(
      lipP,
      d.vec2f(std.sin(frownAngle), std.cos(frownAngle)),
      2.5,
      0.2,
    ),
    0.3,
  );
  // Frown
  head = opSubtraction(
    sdCappedTorus(
      frownP,
      d.vec2f(std.sin(frownAngle), std.cos(frownAngle)),
      2.5,
      0.05,
    ),
    head,
  );
  // EyeBulge
  const eyeBulgeP = std.add(localP, d.vec3f(-0.5, -0.35, -0.2));
  head = opSmoothUnion(head, sdSphere(eyeBulgeP, 0.4), 0.2);
  // Cheek
  const cheekP = std.add(localP, d.vec3f(-0.5, 0.2, -0.2));
  head = opSmoothUnion(head, sdSphere(cheekP, 0.4), 0.2);
  // ...

  const headShape = Shape({
    dist: head,
    color: skinColor,
  });

  const eyeP = std.add(localP, d.vec3f(-0.45, -0.3, -0.5));
  const eyeShape = Shape({
    dist: sdSphere(eyeP, 0.2),
    color: d.vec3f(),
  });

  return shapeUnion(headShape, eyeShape);
});

const strapRot = tgpu['~unstable'].const(
  d.mat3x3f,
  (() => {
    const mat = mat3.identity(d.mat3x3f());
    mat3.rotateX(mat, 0.1, mat);
    mat3.rotateY(mat, -0.05, mat);
    return mat;
  })(),
);

const getBackpack = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  const center = d.vec3f(0, 0.8, 0);
  const localP = std.sub(p, center);
  // Symmetric along the X-axis
  localP.x = std.abs(localP.x);

  const backpackP = std.sub(localP, d.vec3f(0, 0, -0.8));
  let backpack = sdRoundedBox3d(backpackP, d.vec3f(0.7, 0.8, 0.4), 0.2);

  // Strap
  const strapAngle = d.f32(0.8);
  let strapP = std.sub(
    std.mul(localP.yzx, d.vec3f(1, -1, 1)),
    d.vec3f(0, 0.1, 0.5),
  );
  strapP = std.mul(strapRot.$, strapP);
  strapP = opElongate(strapP, d.vec3f(0.3, 0.1, 0.07));
  backpack = opUnion(
    backpack,
    sdCappedTorus(
      strapP,
      d.vec2f(std.sin(strapAngle), std.cos(strapAngle)),
      0.5,
      0.02,
    ),
  );

  return {
    dist: backpack,
    color: backpackColor,
  };
});

const getFrogBody = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  const center = d.vec3f();
  const localP = std.sub(p, center);
  // Symmetric along the X-axis
  localP.x = std.abs(localP.x);

  const torsoP = std.sub(localP, d.vec3f(0, 0.8, 0));
  let torso = sdOctahedron(opElongate(torsoP, d.vec3f(0.2, 0.3, 0)), 0.1) - 0.4;
  const shoulderP = std.sub(localP, d.vec3f(0.6, 1.25, 0));
  torso = opSmoothUnion(torso, sdSphere(shoulderP, 0.25), 0.1);
  // Neck
  const neckP = std.sub(localP, d.vec3f(0, 1.5, 0));
  torso = opSmoothUnion(torso, sdVerticalCapsule(neckP, 0.5, 0.2), 0.2);
  const torsoShape = Shape({
    dist: torso,
    color: skinColor,
  });

  return torsoShape;
});

const getArm = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  return {
    dist: sdVerticalCapsule(p, 0.7, 0.15),
    color: skinColor,
  };
});

const getForearm = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  return {
    dist: sdVerticalCapsule(p, 0.7, 0.15),
    color: skinColor,
  };
});

const getThigh = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  return {
    dist: sdVerticalCapsule(p, 0.7, 0.2),
    color: skinColor,
  };
});

const getShin = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  return {
    dist: sdVerticalCapsule(p, 0.8, 0.17),
    color: skinColor,
  };
});

export const FrogRig = d.struct({
  head: d.mat4x4f,
  leftArm: d.mat4x4f,
  leftForearm: d.mat4x4f,
  leftThigh: d.mat4x4f,
  leftShin: d.mat4x4f,
});

export function createFrog(root: TgpuRoot) {
  let progress = 0;
  let headPitch = 0;
  let headYaw = 0;
  const frogRigCpu = FrogRig({
    head: mat4.identity(d.mat4x4f()),
    leftArm: mat4.identity(d.mat4x4f()),
    leftForearm: mat4.identity(d.mat4x4f()),
    leftThigh: mat4.identity(d.mat4x4f()),
    leftShin: mat4.identity(d.mat4x4f()),
  });
  const frogRig = root.createUniform(FrogRig, frogRigCpu);
  function uploadRig() {
    frogRig.write(frogRigCpu);
  }

  const getFrog = tgpu.fn(
    [d.vec3f],
    Shape,
  )((p) => {
    const headOrigin = d.vec3f(0, 4.2, 0);
    const hp = std.sub(p, headOrigin);
    const thp = std.mul(frogRig.$.head, d.vec4f(hp, 1)).xyz;

    const bodyOrigin = d.vec3f(0, 2.3, 0);
    const bp = std.sub(p, bodyOrigin);
    const tbp = bp; // TODO: Transform

    const leftArmOrigin = d.vec3f(-0.7, 3.55, 0);
    const leftArmP = std.sub(p, leftArmOrigin);
    const leftArmTP = std.mul(frogRig.$.leftArm, d.vec4f(leftArmP, 1)).xyz;

    const leftForearmTP = std.mul(
      frogRig.$.leftForearm,
      d.vec4f(leftArmP, 1),
    ).xyz;

    const leftThighOrigin = d.vec3f(-0.3, 2.3, 0);
    const leftThighP = std.sub(p, leftThighOrigin);
    const leftThighTP = std.mul(
      frogRig.$.leftThigh,
      d.vec4f(leftThighP, 1),
    ).xyz;

    const leftShinTP = std.mul(frogRig.$.leftShin, d.vec4f(leftThighP, 1)).xyz;

    let skin = shapeUnion(getFrogHead(thp), getArm(leftArmTP));
    skin = smoothShapeUnion(skin, getForearm(leftForearmTP), 0.1);
    skin = smoothShapeUnion(skin, getThigh(leftThighTP), 0.1);
    skin = smoothShapeUnion(skin, getShin(leftShinTP), 0.1);
    skin = smoothShapeUnion(skin, getFrogBody(tbp), 0.1);
    const backpack = getBackpack(tbp);
    return shapeUnion(skin, backpack);
  });

  return {
    getFrog,
    update(dt: number) {
      progress += dt;
      headYaw = Math.cos(progress);
      headPitch = Math.sin(progress * 10) * 0.2;

      // All transformations are inverse, since it's actually the inverse
      // transformation matrix we're sending over to the GPU
      mat4.identity(frogRigCpu.head);
      mat4.rotateX(frogRigCpu.head, -headPitch, frogRigCpu.head);
      mat4.rotateY(frogRigCpu.head, -headYaw, frogRigCpu.head);

      // Left arm
      mat4.identity(frogRigCpu.leftArm);
      mat4.rotateZ(
        frogRigCpu.leftArm,
        -0.3 + Math.sin(progress) * 0.1,
        frogRigCpu.leftArm,
      );
      mat4.rotateX(frogRigCpu.leftArm, Math.PI - 0.2, frogRigCpu.leftArm);

      // Left forearm
      mat4.identity(frogRigCpu.leftForearm);
      mat4.rotateX(frogRigCpu.leftForearm, 0.4, frogRigCpu.leftForearm);
      mat4.translate(
        frogRigCpu.leftForearm,
        d.vec3f(0, -0.7, 0),
        frogRigCpu.leftForearm,
      );
      mat4.mul(
        frogRigCpu.leftForearm,
        frogRigCpu.leftArm,
        frogRigCpu.leftForearm,
      );

      // Left thigh
      mat4.identity(frogRigCpu.leftThigh);
      mat4.rotateZ(frogRigCpu.leftThigh, -0.1, frogRigCpu.leftThigh);
      mat4.rotateX(frogRigCpu.leftThigh, Math.PI + 0.2, frogRigCpu.leftThigh);

      // Left shin
      mat4.identity(frogRigCpu.leftShin);
      mat4.rotateX(frogRigCpu.leftShin, -0.4, frogRigCpu.leftShin);
      mat4.translate(
        frogRigCpu.leftShin,
        d.vec3f(0, -0.8, 0),
        frogRigCpu.leftShin,
      );
      mat4.mul(frogRigCpu.leftShin, frogRigCpu.leftThigh, frogRigCpu.leftShin);
    },
    uploadRig,
  };
}
