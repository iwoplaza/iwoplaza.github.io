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
  
  // IK target positions
  const leftArmTarget = d.vec3f(-1.8, 2, 1);
  const rightArmTarget = d.vec3f(1.8, 2, 1);
  const leftLegTarget = d.vec3f(-0.6, 0, 0);
  const rightLegTarget = d.vec3f(0.6, 0, 0);
  
  // Maximum distance a target can be from the body before resetting
  const MAX_TARGET_DISTANCE = 2.0;

  // Movement tracking for rotation
  const prevRootPos = d.vec3f();
  const movementDirection = d.vec3f(0, 0, 1); // Default forward direction
  
  // Rotation parameters
  const HEAD_ROTATION_SPEED = 8.0;  // How quickly the head turns to face movement
  const BODY_ROTATION_SPEED = 2.5;  // How quickly the body follows the head (slower for follow-through)
  const MIN_MOVEMENT_THRESHOLD = 0.01; // Minimum movement required to change direction
  
  let progress = 0;
  let headPitch = 0;
  let headYaw = 0;
  let targetHeadYaw = 0;
  let bodyYaw = 0;
  let targetBodyYaw = 0;
  let leftFootYaw = 0;
  let rightFootYaw = 0;
  const rootPos = d.vec3f();
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
    get position() {
      return rootPos;
    },
    set position(v: d.v3f) {
      rootPos.x = v.x;
      rootPos.y = v.y;
      rootPos.z = v.z;
    },
    update(dt: number) {
      progress += dt;
      
      // Calculate movement direction
      const moveX = rootPos.x - prevRootPos.x;
      const moveZ = rootPos.z - prevRootPos.z;
      const moveMagnitude = Math.sqrt(moveX * moveX + moveZ * moveZ);
      
      // Update movement direction if there is significant movement
      if (moveMagnitude > MIN_MOVEMENT_THRESHOLD) {
        movementDirection.x = moveX / moveMagnitude;
        movementDirection.z = moveZ / moveMagnitude;
        
        // Calculate target yaw angle from movement direction (atan2 gives angle in radians)
        targetHeadYaw = Math.atan2(movementDirection.x, movementDirection.z);
        
        // Add a slight tilt to the head in the direction of movement
        headPitch = Math.sin(progress * 2) * 0.1 - moveMagnitude * 0.05;
      } else {
        // When not moving, return to a natural idle animation
        headPitch = Math.sin(progress * 2) * 0.1;
      }
      
      // Smoothly rotate head toward target direction
      const headYawDiff = targetHeadYaw - headYaw;
      
      // Normalize the angle difference to be between -PI and PI
      const normalizedHeadYawDiff = Math.atan2(Math.sin(headYawDiff), Math.cos(headYawDiff));
      
      // Apply smooth rotation to head
      headYaw += normalizedHeadYawDiff * HEAD_ROTATION_SPEED * dt;
      
      // Body follows head with delay (follow-through effect)
      targetBodyYaw = headYaw;
      const bodyYawDiff = targetBodyYaw - bodyYaw;
      
      // Normalize the angle difference
      const normalizedBodyYawDiff = Math.atan2(Math.sin(bodyYawDiff), Math.cos(bodyYawDiff));
      
      // Apply smooth rotation to body (slower than head)
      bodyYaw += normalizedBodyYawDiff * BODY_ROTATION_SPEED * dt;
      
      // Natural head movement is now handled above based on movement
      
      // Feet follow body rotation
      leftFootYaw = bodyYaw;
      rightFootYaw = bodyYaw;
      
      // Store current position for next frame's movement calculation
      prevRootPos.x = rootPos.x;
      prevRootPos.y = rootPos.y;
      prevRootPos.z = rootPos.z;

      // BODY
      body.pos.x = rootPos.x;
      body.pos.y = rootPos.y + 1.3 - Math.sin(progress * 2) * 0.05; // More subtle y-axis only animation
      body.pos.z = rootPos.z;
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
      
      // Check if arm targets are too far from body and reset if needed
      const bodyGlobalPos = vec3.transformMat4(
        d.vec3f(0, 0, 0),
        body.mat,
        d.vec3f(),
      );
      
      // Calculate distance from body to arm targets
      const leftArmTargetDist = vec3.distance(leftArmTarget, bodyGlobalPos);
      const rightArmTargetDist = vec3.distance(rightArmTarget, bodyGlobalPos);
      
      // Reset arm targets if they're too far from the body
      if (leftArmTargetDist > MAX_TARGET_DISTANCE) {
        leftArmTarget.x = bodyGlobalPos.x - 1.0;
        leftArmTarget.y = bodyGlobalPos.y + 1.0;
        leftArmTarget.z = bodyGlobalPos.z + 0.5;
      }
      
      if (rightArmTargetDist > MAX_TARGET_DISTANCE) {
        rightArmTarget.x = bodyGlobalPos.x + 1.0;
        rightArmTarget.y = bodyGlobalPos.y + 1.0;
        rightArmTarget.z = bodyGlobalPos.z + 0.5;
      }

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
      
      // Calculate distance from body to leg targets
      const leftLegTargetDist = vec3.distance(leftLegTarget, bodyGlobalPos);
      const rightLegTargetDist = vec3.distance(rightLegTarget, bodyGlobalPos);
      
      // Reset leg targets if they're too far from the body
      if (leftLegTargetDist > MAX_TARGET_DISTANCE) {
        leftLegTarget.x = bodyGlobalPos.x - 0.6;
        leftLegTarget.y = bodyGlobalPos.y - 1.0;
        leftLegTarget.z = bodyGlobalPos.z;
      }
      
      if (rightLegTargetDist > MAX_TARGET_DISTANCE) {
        rightLegTarget.x = bodyGlobalPos.x + 0.6;
        rightLegTarget.y = bodyGlobalPos.y - 1.0;
        rightLegTarget.z = bodyGlobalPos.z;
      }

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
