import { opSmoothUnion, opUnion, sdRoundedBox3d, sdSphere } from '@typegpu/sdf';
import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat3, mat4 } from 'wgpu-matrix';
import { extractAnglesBetweenPoints, solveIK } from './ik.ts';
import {
  opElongate,
  opSubtraction,
  Shape,
  sdCappedTorus,
  sdOctahedron,
  sdVerticalCapsule,
  shapeUnion,
  smoothShapeUnion,
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
  body: d.mat4x4f,
  leftArm: d.mat4x4f,
  leftForearm: d.mat4x4f,
  rightArm: d.mat4x4f,
  rightForearm: d.mat4x4f,
  leftThigh: d.mat4x4f,
  leftShin: d.mat4x4f,
  rightThigh: d.mat4x4f,
  rightShin: d.mat4x4f,
});

export function createFrog(root: TgpuRoot) {
  let progress = 0;
  let headPitch = 0;
  let headYaw = 0;
  const frogRigCpu = FrogRig({
    head: mat4.identity(d.mat4x4f()),
    body: mat4.identity(d.mat4x4f()),
    leftArm: mat4.identity(d.mat4x4f()),
    leftForearm: mat4.identity(d.mat4x4f()),
    rightArm: mat4.identity(d.mat4x4f()),
    rightForearm: mat4.identity(d.mat4x4f()),
    leftThigh: mat4.identity(d.mat4x4f()),
    leftShin: mat4.identity(d.mat4x4f()),
    rightThigh: mat4.identity(d.mat4x4f()),
    rightShin: mat4.identity(d.mat4x4f()),
  });
  const frogRig = root.createUniform(FrogRig, frogRigCpu);
  function uploadRig() {
    frogRig.write(frogRigCpu);
  }

  const getFrog = tgpu.fn(
    [d.vec3f],
    Shape,
  )((p) => {
    const headOrigin = d.vec3f(0, 1.9, 0);
    const hp = std.sub(p, headOrigin);
    const thp = std.mul(frogRig.$.head, d.vec4f(hp, 1)).xyz;

    const bodyOrigin = d.vec3f();
    const bp = std.sub(p, bodyOrigin);
    const tbp = std.mul(frogRig.$.body, d.vec4f(bp, 1)).xyz;

    const leftArmOrigin = d.vec3f(-0.7, 1.25, 0);
    const leftArmP = std.sub(p, leftArmOrigin);
    const leftArmTP = std.mul(frogRig.$.leftArm, d.vec4f(leftArmP, 1)).xyz;

    const leftForearmTP = std.mul(
      frogRig.$.leftForearm,
      d.vec4f(leftArmP, 1),
    ).xyz;

    const rightArmOrigin = d.vec3f(0.7, 1.25, 0);
    const rightArmP = std.sub(p, rightArmOrigin);
    const rightArmTP = std.mul(frogRig.$.rightArm, d.vec4f(rightArmP, 1)).xyz;

    const rightForearmTP = std.mul(
      frogRig.$.rightForearm,
      d.vec4f(rightArmP, 1),
    ).xyz;

    const leftThighOrigin = d.vec3f(-0.3, 0, 0);
    const leftThighP = std.sub(p, leftThighOrigin);
    const leftThighTP = std.mul(
      frogRig.$.leftThigh,
      d.vec4f(leftThighP, 1),
    ).xyz;

    const leftShinTP = std.mul(frogRig.$.leftShin, d.vec4f(leftThighP, 1)).xyz;

    const rightThighOrigin = d.vec3f(0.3, 0, 0);
    const rightThighP = std.sub(p, rightThighOrigin);
    const rightThighTP = std.mul(
      frogRig.$.rightThigh,
      d.vec4f(rightThighP, 1),
    ).xyz;

    const rightShinTP = std.mul(
      frogRig.$.rightShin,
      d.vec4f(rightThighP, 1),
    ).xyz;

    let skin = shapeUnion(getFrogHead(thp), getArm(leftArmTP));
    skin = smoothShapeUnion(skin, getForearm(leftForearmTP), 0.1);
    skin = smoothShapeUnion(skin, getArm(rightArmTP), 0.1);
    skin = smoothShapeUnion(skin, getForearm(rightForearmTP), 0.1);
    skin = smoothShapeUnion(skin, getThigh(leftThighTP), 0.1);
    skin = smoothShapeUnion(skin, getShin(leftShinTP), 0.1);
    skin = smoothShapeUnion(skin, getThigh(rightThighTP), 0.1);
    skin = smoothShapeUnion(skin, getShin(rightShinTP), 0.1);
    skin = smoothShapeUnion(skin, getFrogBody(tbp), 0.1);
    const backpack = getBackpack(tbp);
    return shapeUnion(skin, backpack);
  });

  const legChain = [0.8, 1];

  return {
    getFrog,
    update(dt: number) {
      const {
        body,
        head,
        leftArm,
        leftForearm,
        rightArm,
        rightForearm,
        leftThigh,
        leftShin,
        rightThigh,
        rightShin,
      } = frogRigCpu;

      progress += dt;
      headYaw = Math.cos(progress) * 0.1;
      headPitch = Math.sin(progress * 2) * 0.1;

      const hipPos = d.vec3f(
        Math.sin(progress * 1.5) * 0.1,
        -1.3 + Math.sin(progress * 3) * 0.1,
        0,
      );

      // All transformations are inverse, since it's actually the inverse
      // transformation matrix we're sending over to the GPU
      mat4.identity(head);
      mat4.rotateX(head, -headPitch, head);
      mat4.rotateY(head, -headYaw, head);
      mat4.translate(head, hipPos, head);

      // Body
      mat4.identity(body);
      mat4.translate(body, hipPos, body);

      // Left arm
      mat4.identity(leftArm);
      mat4.rotateZ(leftArm, -0.3 + Math.sin(progress) * 0.1, leftArm);
      mat4.rotateX(leftArm, Math.PI - 0.2, leftArm);
      mat4.translate(leftArm, hipPos, leftArm);

      // Left forearm
      mat4.identity(leftForearm);
      mat4.rotateX(leftForearm, 0.4, leftForearm);
      mat4.translate(leftForearm, d.vec3f(0, -0.7, 0), leftForearm);
      mat4.mul(leftForearm, leftArm, leftForearm);

      // Right arm
      mat4.identity(rightArm);
      mat4.rotateZ(rightArm, 0.3 - Math.sin(progress) * 0.1, rightArm);
      mat4.rotateX(rightArm, Math.PI - 0.2, rightArm);
      mat4.translate(rightArm, hipPos, rightArm);

      // Right forearm
      mat4.identity(rightForearm);
      mat4.rotateX(rightForearm, 0.4, rightForearm);
      mat4.translate(rightForearm, d.vec3f(0, -0.7, 0), rightForearm);
      mat4.mul(rightForearm, rightArm, rightForearm);

      const leftPull = d.vec3f(0, 0, 1);
      const leftLegTarget = std.sub(d.vec3f(0.2, 0, 0), hipPos);
      const leftLegPoints = solveIK(legChain, leftLegTarget, leftPull);
      const leftLegAngles = extractAnglesBetweenPoints(leftLegPoints);

      const rightPull = d.vec3f(0, 0, 1);
      const rightLegTarget = std.sub(d.vec3f(-0.2, 0, 0), hipPos);
      const rightLegPoints = solveIK(legChain, rightLegTarget, rightPull);
      const rightLegAngles = extractAnglesBetweenPoints(rightLegPoints);

      // Left thigh
      mat4.identity(leftThigh);
      mat4.rotateX(leftThigh, Math.PI + leftLegAngles[0].x, leftThigh);
      mat4.rotateZ(leftThigh, leftLegAngles[0].y, leftThigh);
      mat4.translate(leftThigh, hipPos, leftThigh);

      // Left shin
      mat4.identity(leftShin);
      // Undoing parent rotation
      mat4.rotateX(leftShin, -leftLegAngles[0].x, leftShin);
      mat4.rotateZ(leftShin, -leftLegAngles[0].y, leftShin);
      // Local rotation
      mat4.rotateX(leftShin, leftLegAngles[1].x, leftShin);
      mat4.rotateZ(leftShin, leftLegAngles[1].y, leftShin);
      mat4.translate(leftShin, d.vec3f(0, -0.8, 0), leftShin);
      mat4.mul(leftShin, leftThigh, leftShin);

      // Right thigh
      mat4.identity(rightThigh);
      mat4.rotateX(rightThigh, Math.PI + rightLegAngles[0].x, rightThigh);
      mat4.rotateZ(rightThigh, rightLegAngles[0].y, rightThigh);
      mat4.translate(rightThigh, hipPos, rightThigh);

      // Right shin
      mat4.identity(rightShin);
      // Undoing parent rotation
      mat4.rotateX(rightShin, -rightLegAngles[0].x, rightShin);
      mat4.rotateZ(rightShin, -rightLegAngles[0].y, rightShin);
      // Local rotation
      mat4.rotateX(rightShin, rightLegAngles[1].x, rightShin);
      mat4.rotateZ(rightShin, rightLegAngles[1].y, rightShin);
      mat4.translate(rightShin, d.vec3f(0, -0.8, 0), rightShin);
      mat4.mul(rightShin, rightThigh, rightShin);
    },
    uploadRig,
  };
}
