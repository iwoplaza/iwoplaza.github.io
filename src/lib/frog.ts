import {
  opSmoothUnion,
  opUnion,
  sdBox3d,
  sdRoundedBox3d,
  sdSphere,
} from '@typegpu/sdf';
import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat3, mat4, vec3 } from 'wgpu-matrix';
import { getRotationMatricesBetweenPoints, solveIK } from './ik.ts';
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
import { Gizmo } from './gizmo.ts';

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
    dist: sdBox3d(std.sub(p, d.vec3f(0, 0.45, 0)), d.vec3f(0.2, 0.45, 0.2)),
    color: skinColor,
  };
  // return {
  //   dist: sdVerticalCapsule(p, 0.7, 0.2),
  //   color: skinColor,
  // };
});

const getShin = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  return {
    dist: sdBox3d(std.sub(p, d.vec3f(0, 0.4, 0)), d.vec3f(0.2, 0.4, 0.2)),
    color: skinColor,
  };
  // return {
  //   dist: sdVerticalCapsule(p, 0.8, 0.17),
  //   color: skinColor,
  // };
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
  let bodyYaw = 0;
  let leftFootYaw = 0;
  let rightFootYaw = 0;
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

    const leftArmTP = std.mul(frogRig.$.leftArm, d.vec4f(p, 1)).xyz;
    const leftForearmTP = std.mul(frogRig.$.leftForearm, d.vec4f(p, 1)).xyz;

    const rightArmTP = std.mul(frogRig.$.rightArm, d.vec4f(p, 1)).xyz;
    const rightForearmTP = std.mul(frogRig.$.rightForearm, d.vec4f(p, 1)).xyz;

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

  const legChain = [0.8, 0.8];
  const armChain = [0.7, 0.7];

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
      bodyYaw = progress;
      leftFootYaw = progress;
      rightFootYaw = progress;

      const hipPos = d.vec3f(
        Math.sin(progress * 1.5) * 0.6,
        -1.3 + Math.sin(progress * 3) * 0.1,
        0,
      );

      // All transformations are inverse, since it's actually the inverse
      // transformation matrix we're sending over to the GPU
      mat4.identity(head);
      mat4.rotateX(head, -headPitch, head);
      mat4.rotateY(head, -headYaw, head);
      mat4.translate(head, hipPos, head);

      // BODY
      mat4.identity(body);
      mat4.rotateY(body, -bodyYaw, body);
      mat4.translate(body, hipPos, body);

      const leftArmLocalPos = d.vec3f(0.7, -1.25, 0);
      const leftArmGlobalPos = vec3.transformMat4(
        leftArmLocalPos,
        body,
        d.vec3f(),
      );

      // Draw gizmos for left arm position
      Gizmo.color(d.vec3f(1, 0, 0));
      Gizmo.sphere(leftArmGlobalPos, 0.1);

      const armPull = d.vec3f(Math.sin(bodyYaw), 0, Math.cos(bodyYaw));
      const leftArmTarget = std.sub(d.vec3f(0.2, 0, 0), hipPos);
      const leftArmPoints = solveIK(armChain, leftArmTarget, armPull);
      const leftArmMats = getRotationMatricesBetweenPoints(
        leftArmPoints,
        armPull,
      );

      const rightArmTarget = std.sub(d.vec3f(0, 0, 0), hipPos);
      const rightArmPoints = solveIK(armChain, rightArmTarget, armPull);
      const rightArmMats = getRotationMatricesBetweenPoints(
        rightArmPoints,
        armPull,
      );

      // LEFT ARM
      mat4.identity(leftArm);
      // Local transform
      mat4.rotateZ(leftArm, -0.3 + Math.sin(progress) * 0.1, leftArm);
      mat4.rotateX(leftArm, Math.PI - 0.2, leftArm);
      // Undoing parent rotation
      mat4.rotateY(leftArm, bodyYaw, leftArm);
      // Lock into place
      mat4.translate(leftArm, leftArmLocalPos, leftArm);
      // Parent transform
      mat4.mul(leftArm, body, leftArm);

      // Left forearm
      mat4.identity(leftForearm);
      mat4.rotateX(leftForearm, 0.4, leftForearm);
      mat4.translate(leftForearm, d.vec3f(0, -armChain[0], 0), leftForearm);
      mat4.mul(leftForearm, leftArm, leftForearm);

      // RIGHT ARM
      mat4.identity(rightArm);
      // Local transform
      mat4.rotateZ(rightArm, 0.3 - Math.sin(progress) * 0.1, rightArm);
      mat4.rotateX(rightArm, Math.PI - 0.2, rightArm);
      // Undoing parent rotation
      mat4.rotateY(rightArm, bodyYaw, rightArm);
      // Lock into place
      mat4.translate(rightArm, d.vec3f(-0.7, -1.25, 0), rightArm);
      // Parent transform
      mat4.mul(rightArm, body, rightArm);

      // Right forearm
      mat4.identity(rightForearm);
      mat4.rotateX(rightForearm, 0.4, rightForearm);
      mat4.translate(rightForearm, d.vec3f(0, -0.7, 0), rightForearm);
      mat4.mul(rightForearm, rightArm, rightForearm);

      const leftLegPull = d.vec3f(
        -Math.sin(leftFootYaw),
        0,
        -Math.cos(leftFootYaw),
      );
      const leftLegTarget = std.sub(d.vec3f(0.2, 0, 0), hipPos);
      const leftLegPoints = solveIK(legChain, leftLegTarget, leftLegPull);
      const leftLegMats = getRotationMatricesBetweenPoints(
        leftLegPoints,
        leftLegPull,
      );

      const rightLegPull = d.vec3f(
        -Math.sin(rightFootYaw),
        0,
        -Math.cos(rightFootYaw),
      );
      const rightLegTarget = std.sub(d.vec3f(0, 0, 0), hipPos);
      const rightLegPoints = solveIK(legChain, rightLegTarget, rightLegPull);
      const rightLegMats = getRotationMatricesBetweenPoints(
        rightLegPoints,
        rightLegPull,
      );

      // Left thigh
      const leftThighRot = leftLegMats[0];
      const leftThighInvRot = mat3.transpose(leftThighRot);
      mat4.identity(leftThigh);
      mat4.mul(leftThigh, mat4.fromMat3(leftThighInvRot), leftThigh);
      mat4.translate(leftThigh, hipPos, leftThigh);

      // Left shin
      const leftShinInvRot = mat3.transpose(leftLegMats[1]);
      mat4.identity(leftShin);
      // Local rotation
      mat4.mul(leftShin, mat4.fromMat3(leftShinInvRot), leftShin);
      // Undoing parent rotation
      mat4.mul(leftShin, mat4.fromMat3(leftThighRot), leftShin);
      mat4.translate(leftShin, d.vec3f(0, -0.8, 0), leftShin);
      mat4.mul(leftShin, leftThigh, leftShin);

      // Right thigh
      const rightThighRot = rightLegMats[0];
      const thighInvRot = mat3.transpose(rightThighRot);
      mat4.identity(rightThigh);
      mat4.mul(rightThigh, mat4.fromMat3(thighInvRot), rightThigh);
      mat4.translate(rightThigh, hipPos, rightThigh);

      // Right shin
      const rightShinInvRot = mat3.transpose(rightLegMats[1]);
      mat4.identity(rightShin);
      // Local rotation
      mat4.mul(rightShin, mat4.fromMat3(rightShinInvRot), rightShin);
      // Undoing parent rotation
      mat4.mul(rightShin, mat4.fromMat3(rightThighRot), rightShin);
      mat4.translate(rightShin, d.vec3f(0, -0.8, 0), rightShin);
      mat4.mul(rightShin, rightThigh, rightShin);
    },
    uploadRig,
  };
}
