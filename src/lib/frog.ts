import { opSmoothUnion, opUnion, sdRoundedBox3d, sdSphere } from '@typegpu/sdf';
import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat3, mat4, quatn, vec3 } from 'wgpu-matrix';
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
import { Bone } from './rig.ts';

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
  // return {
  //   dist: sdBox3d(std.sub(p, d.vec3f(0, 0.45, 0)), d.vec3f(0.2, 0.45, 0.2)),
  //   color: skinColor,
  // };
  return {
    dist: sdVerticalCapsule(p, 0.7, 0.2),
    color: skinColor,
  };
});

const getShin = tgpu.fn(
  [d.vec3f],
  Shape,
)((p) => {
  // return {
  //   dist: sdBox3d(std.sub(p, d.vec3f(0, 0.4, 0)), d.vec3f(0.2, 0.4, 0.2)),
  //   color: skinColor,
  // };
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
  const legChain = [0.8, 0.8];
  const armChain = [0.7, 0.7];

  let progress = 0;
  let headPitch = 0;
  let headYaw = 0;
  let bodyYaw = 0;
  let leftFootYaw = 0;
  let rightFootYaw = 0;
  const body = new Bone(d.vec3f(), d.vec4f(), {});
  const head = new Bone(d.vec3f(0, 1.9, 0), d.vec4f(), { parent: body });
  const leftThigh = new Bone(d.vec3f(-0.3, 0, 0), d.vec4f(), { parent: body });
  const rightThigh = new Bone(d.vec3f(0.3, 0, 0), d.vec4f(), { parent: body });
  const leftShin = new Bone(d.vec3f(0, legChain[0], 0), d.vec4f(), {
    parent: leftThigh,
  });
  const rightShin = new Bone(d.vec3f(0, legChain[0], 0), d.vec4f(), {
    parent: rightThigh,
  });
  const leftArm = new Bone(d.vec3f(-0.7, 1.25, 0), d.vec4f(), { parent: body });
  const rightArm = new Bone(d.vec3f(0.7, 1.25, 0), d.vec4f(), { parent: body });
  const leftForearm = new Bone(d.vec3f(0, armChain[0], 0), d.vec4f(), {
    parent: leftArm,
  });
  const rightForearm = new Bone(d.vec3f(0, armChain[0], 0), d.vec4f(), {
    parent: rightArm,
  });
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
  // Overriding the clones
  frogRigCpu.body = body.invMat;
  frogRigCpu.head = head.invMat;
  frogRigCpu.leftThigh = leftThigh.invMat;
  frogRigCpu.rightThigh = rightThigh.invMat;
  frogRigCpu.leftShin = leftShin.invMat;
  frogRigCpu.rightShin = rightShin.invMat;
  frogRigCpu.leftArm = leftArm.invMat;
  frogRigCpu.rightArm = rightArm.invMat;
  frogRigCpu.leftForearm = leftForearm.invMat;
  frogRigCpu.rightForearm = rightForearm.invMat;
  const frogRig = root.createUniform(FrogRig, frogRigCpu);
  function uploadRig() {
    frogRig.write(frogRigCpu);
  }

  const getFrog = tgpu.fn(
    [d.vec3f],
    Shape,
  )((p) => {
    const thp = std.mul(frogRig.$.head, d.vec4f(p, 1)).xyz;
    const tbp = std.mul(frogRig.$.body, d.vec4f(p, 1)).xyz;

    const leftArmTP = std.mul(frogRig.$.leftArm, d.vec4f(p, 1)).xyz;
    const leftForearmTP = std.mul(frogRig.$.leftForearm, d.vec4f(p, 1)).xyz;
    const rightArmTP = std.mul(frogRig.$.rightArm, d.vec4f(p, 1)).xyz;
    const rightForearmTP = std.mul(frogRig.$.rightForearm, d.vec4f(p, 1)).xyz;
    const leftThighTP = std.mul(frogRig.$.leftThigh, d.vec4f(p, 1)).xyz;
    const leftShinTP = std.mul(frogRig.$.leftShin, d.vec4f(p, 1)).xyz;
    const rightThighTP = std.mul(frogRig.$.rightThigh, d.vec4f(p, 1)).xyz;
    const rightShinTP = std.mul(frogRig.$.rightShin, d.vec4f(p, 1)).xyz;

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

  return {
    getFrog,
    update(dt: number) {
      progress += dt;
      headYaw = Math.cos(progress) * 0.1;
      headPitch = Math.sin(progress * 2) * 0.1;
      bodyYaw = Math.cos(progress * 1.5) * 0.5;
      leftFootYaw = bodyYaw;
      rightFootYaw = bodyYaw;

      // BODY
      body.pos.x = -Math.sin(progress * 1.5) * 0.6;
      body.pos.y = 1.3 - Math.sin(progress * 3) * 0.1;
      body.pos.z = 0;
      const chestPos = std.add(body.pos, d.vec3f(0, 1, 0));

      quatn.fromEuler(0, bodyYaw, 0, 'yxz', body.rot);
      body.compute();

      const hipDir = d.vec3f(Math.sin(bodyYaw), 0, Math.cos(bodyYaw));
      const armPull = std.neg(hipDir);

      // HEAD
      quatn.fromEuler(headPitch, headYaw, 0, 'yxz', head.rot);
      head.compute();

      // ARMS
      const leftArmGlobalPos = vec3.transformMat4(
        leftArm.pos,
        body.mat,
        d.vec3f(),
      );

      const rightArmGlobalPos = vec3.transformMat4(
        rightArm.pos,
        body.mat,
        d.vec3f(),
      );

      const leftArmTarget = d.vec3f(-1.8, 2, 1);
      const rightArmTarget = d.vec3f(1.8, 2, 1);

      const leftArmPoints = solveIK(
        armChain,
        std.sub(leftArmTarget, leftArmGlobalPos),
        armPull,
      );
      const rightArmPoints = solveIK(
        armChain,
        std.sub(rightArmTarget, rightArmGlobalPos),
        armPull,
      );

      const leftArmMats = getRotationMatricesBetweenPoints(
        leftArmPoints,
        armPull,
      );

      const rightArmMats = getRotationMatricesBetweenPoints(
        rightArmPoints,
        armPull,
      );

      // Left arm
      quatn.fromMat(leftArmMats[0], leftArm.rot);
      quatn.rotateX(leftArm.rot, -Math.PI, leftArm.rot);
      leftArm.compute();

      // Left forearm
      quatn.fromMat(leftArmMats[1], leftForearm.rot);
      quatn.rotateX(leftForearm.rot, -Math.PI, leftForearm.rot);
      leftForearm.compute();

      // Right arm
      quatn.fromMat(rightArmMats[0], rightArm.rot);
      quatn.rotateX(rightArm.rot, -Math.PI, rightArm.rot);
      rightArm.compute();

      // Right forearm
      quatn.fromMat(rightArmMats[1], rightForearm.rot);
      quatn.rotateX(rightForearm.rot, -Math.PI, rightForearm.rot);
      rightForearm.compute();

      // Legs
      const leftLegGlobalPos = vec3.transformMat4(
        leftThigh.pos,
        body.mat,
        d.vec3f(),
      );
      const rightLegGlobalPos = vec3.transformMat4(
        rightThigh.pos,
        body.mat,
        d.vec3f(),
      );
      const leftLegPull = d.vec3f(
        Math.sin(leftFootYaw),
        0,
        Math.cos(leftFootYaw),
      );
      const rightLegPull = d.vec3f(
        Math.sin(rightFootYaw),
        0,
        Math.cos(rightFootYaw),
      );

      const leftLegTarget = d.vec3f(-0.6, 0, 0);
      const rightLegTarget = d.vec3f(0.6, 0, 0);

      const leftLegPoints = solveIK(
        legChain,
        std.sub(leftLegTarget, leftLegGlobalPos),
        leftLegPull,
      );
      const rightLegPoints = solveIK(
        legChain,
        std.sub(rightLegTarget, rightLegGlobalPos),
        rightLegPull,
      );

      const leftLegMats = getRotationMatricesBetweenPoints(
        leftLegPoints,
        leftLegPull,
      );
      const rightLegMats = getRotationMatricesBetweenPoints(
        rightLegPoints,
        rightLegPull,
      );

      // Left thigh
      quatn.fromMat(leftLegMats[0], leftThigh.rot);
      quatn.rotateX(leftThigh.rot, -Math.PI, leftThigh.rot);
      leftThigh.compute();

      // Left shin
      quatn.fromMat(leftLegMats[1], leftShin.rot);
      quatn.rotateX(leftShin.rot, -Math.PI, leftShin.rot);
      leftShin.compute();

      // Right thigh
      quatn.fromMat(rightLegMats[0], rightThigh.rot);
      quatn.rotateX(rightThigh.rot, -Math.PI, rightThigh.rot);
      rightThigh.compute();

      // Right shin
      quatn.fromMat(rightLegMats[1], rightShin.rot);
      quatn.rotateX(rightShin.rot, -Math.PI, rightShin.rot);
      rightShin.compute();

      // Draw gizmo for joints
      Gizmo.color(d.vec3f(1));
      Gizmo.sphere(leftArmGlobalPos, 0.1);
      Gizmo.sphere(rightArmGlobalPos, 0.1);
      Gizmo.sphere(leftLegGlobalPos, 0.1);
      Gizmo.sphere(rightLegGlobalPos, 0.1);

      // Draw gizmos for IK targets
      Gizmo.color(d.vec3f(0, 0, 1));
      Gizmo.sphere(leftLegTarget, 0.1);
      Gizmo.color(d.vec3f(0, 1, 0));
      Gizmo.sphere(rightLegTarget, 0.1);

      // Draw gizmo for body direction
      Gizmo.color(d.vec3f(1, 0, 0));
      Gizmo.arrow(chestPos, std.add(chestPos, hipDir));
    },
    uploadRig,
  };
}
