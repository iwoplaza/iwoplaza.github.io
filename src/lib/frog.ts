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
  const leftLegTarget = d.vec3f(-0.6, 0, 0.1);
  const rightLegTarget = d.vec3f(0.6, 0, 0.1);

  // Maximum distance a target can be from the body before resetting
  const MAX_TARGET_DISTANCE = 0.6;

  // Movement tracking for rotation
  const prevRootPos = d.vec3f();
  const movementDirection = d.vec3f(0, 0, 1); // Default forward direction
  const velocity = d.vec3f(); // Current velocity
  const movement = d.vec3f(); // Desired movement direction and magnitude

  // Rotation parameters
  const HEAD_ROTATION_SPEED = 8.0; // How quickly the head turns to face movement
  const BODY_ROTATION_SPEED = 7; // How quickly the body follows the head (slower for follow-through)
  const MIN_MOVEMENT_THRESHOLD = 0.001; // Minimum movement required to change direction

  // Movement parameters
  const MAX_SPEED = 8; // Maximum movement speed
  const ACCELERATION = 20; // How quickly velocity approaches target movement
  const DECELERATION = 15; // How quickly velocity decelerates when no input

  // Arm animation parameters
  const ARM_FIGURE8_BASE_AMPLITUDE = 0.02; // Base amplitude of the figure-8 pattern
  const ARM_MAX_AMPLITUDE = 1; // Maximum amplitude when moving at full speed
  const BASE_ARM_ANIMATION_SPEED = 2.5; // Base speed of the figure-8 animation
  const ARM_ANIMATION_SPEED = 1.3; // Speed of the figure-8 animation
  let armAnimationPhase = 0.0; // Current phase of the arm animation

  let progress = 0;
  let rootYaw = 0;
  let targetRootYaw = 0;
  let headPitch = 0;
  let headYaw = 0;
  let targetHeadYaw = 0;
  let bodyYaw = 0;
  let bodyPitch = 0;

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

  let rightLegPlaced = false;

  // Leg transition state
  const leftLegPrevTarget = d.vec3f();
  const rightLegPrevTarget = d.vec3f();
  let leftLegTransitionTime = 0;
  let rightLegTransitionTime = 0;
  let leftLegInTransition = false;
  let rightLegInTransition = false;
  const LEG_TRANSITION_DURATION = 0.3; // Duration of leg transition in seconds
  const FOOT_LIFT_HEIGHT = 0.4; // How high to lift the foot during transition

  // Initialize previous leg target positions
  vec3.copy(leftLegTarget, leftLegPrevTarget);
  vec3.copy(rightLegTarget, rightLegPrevTarget);

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
    get movement() {
      return movement;
    },
    set movement(v: d.v3f) {
      movement.x = v.x;
      movement.y = v.y;
      movement.z = v.z;
    },
    update(dt: number) {
      progress += dt;
      
      // Calculate target velocity from movement input
      const targetVelocity = d.vec3f(
        movement.x * MAX_SPEED,
        movement.y * MAX_SPEED,
        movement.z * MAX_SPEED,
      );
      
      // Smoothly interpolate velocity towards target
      const movementMagnitude = Math.sqrt(movement.x * movement.x + movement.y * movement.y + movement.z * movement.z);
      const accelerationRate = movementMagnitude > 0.001 ? ACCELERATION : DECELERATION;
      
      const velocityDiff = d.vec3f(
        targetVelocity.x - velocity.x,
        targetVelocity.y - velocity.y,
        targetVelocity.z - velocity.z,
      );
      
      velocity.x += velocityDiff.x * accelerationRate * dt;
      velocity.y += velocityDiff.y * accelerationRate * dt;
      velocity.z += velocityDiff.z * accelerationRate * dt;
      
      // Apply velocity to position
      rootPos.x += velocity.x * dt;
      rootPos.y += velocity.y * dt;
      rootPos.z += velocity.z * dt;
      
      // Calculate movement direction
      const moveX = rootPos.x - prevRootPos.x;
      const moveZ = rootPos.z - prevRootPos.z;
      const moveMagnitude =
        Math.sqrt(moveX * moveX + moveZ * moveZ) / Math.max(0.001, dt);

      // Progress the arm animation phase based on movement velocity
      // When moving faster, the arms should swing more rapidly
      // When stationary, they should still have a subtle movement
      const velocityPhaseBoost =
        BASE_ARM_ANIMATION_SPEED +
        moveMagnitude * ARM_ANIMATION_SPEED;
      armAnimationPhase += velocityPhaseBoost * dt;

      // Keep the phase within a reasonable range to avoid floating point issues
      if (armAnimationPhase > Math.PI * 2) {
        armAnimationPhase -= Math.PI * 2;
      }

      // Calculate the figure-8 pattern
      // Figure-8 is created using sin for horizontal and sin*cos for vertical movement
      const figure8X = Math.sin(armAnimationPhase);
      const figure8Y = Math.sin(armAnimationPhase * 2 - 0.5) * 0.5; // Vertical component of figure-8

      // Scale the amplitude based on movement velocity
      // Map movement magnitude to a 0-1 range for amplitude scaling
      // Use a minimum threshold to ensure some movement even when nearly stationary
      const velocityFactor = std.clamp(moveMagnitude * 0.2, 0.01, 1); // Scale up for better response
      const armAmplitude =
        ARM_FIGURE8_BASE_AMPLITUDE +
        (ARM_MAX_AMPLITUDE - ARM_FIGURE8_BASE_AMPLITUDE) * velocityFactor;

      // Update movement direction if there is significant movement
      if (moveMagnitude > MIN_MOVEMENT_THRESHOLD) {
        movementDirection.x = moveX / moveMagnitude;
        movementDirection.z = moveZ / moveMagnitude;

        // Calculate target yaw angle from movement direction (atan2 gives angle in radians)
        targetHeadYaw = Math.atan2(movementDirection.x, movementDirection.z);

        // Add a slight tilt to the body in the direction of movement
        bodyPitch = moveMagnitude * 0.03;
      } else {
        // When not moving, return to a natural idle animation
        headPitch = Math.sin(progress * 2) * 0.1;
      }

      // Smoothly rotate head toward target direction
      const headBodyYawDiff = targetHeadYaw - bodyYaw;

      // Normalize the angle difference to be between -PI and PI
      const normHeadHeadYawDiff = Math.atan2(
        Math.sin(headBodyYawDiff),
        Math.cos(headBodyYawDiff),
      );


      // Smoothly rotate head toward target direction
      const headYawDiff = targetHeadYaw - headYaw;

      // Normalize the angle difference to be between -PI and PI
      const normalizedHeadYawDiff = Math.atan2(
        Math.sin(headYawDiff),
        Math.cos(headYawDiff),
      );

      // Apply smooth rotation to head
      headYaw += normalizedHeadYawDiff * HEAD_ROTATION_SPEED * dt;
      const headRight = d.vec3f(Math.cos(headYaw), 0, -Math.sin(headYaw));
      const headForward = d.vec3f(Math.sin(headYaw), 0, Math.cos(headYaw));

      // Body follows head with delay (follow-through effect)
      targetRootYaw = headYaw;
      const rootYawDiff = targetRootYaw - rootYaw;

      // Normalize the angle difference
      const normalizedRootYawDiff = Math.atan2(
        Math.sin(rootYawDiff),
        Math.cos(rootYawDiff),
      );

      // Apply smooth rotation to body (slower than head)
      rootYaw += normalizedRootYawDiff * BODY_ROTATION_SPEED * dt;

      // Actual body rotation
      bodyYaw = rootYaw + figure8X * armAmplitude * 0.4;

      // Feet follow body rotation
      leftFootYaw = rootYaw;
      rightFootYaw = rootYaw;

      // Store current position for next frame's movement calculation
      prevRootPos.x = rootPos.x;
      prevRootPos.y = rootPos.y;
      prevRootPos.z = rootPos.z;

      // BODY
      body.pos.x = rootPos.x;
      body.pos.y =
        rootPos.y +
        1.5 -
        Math.sin(progress * 2) * 0.05 + // Raised torso position with subtle animation
        figure8Y * armAmplitude * 0.3;
      body.pos.z = rootPos.z;
      const chestPos = std.add(body.pos, d.vec3f(0, 1, 0));

      quatn.fromEuler(bodyPitch, bodyYaw, 0, 'yxz', body.rot);
      body.compute();

      const hipDir = d.vec3f(Math.sin(rootYaw), 0, Math.cos(rootYaw));
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

      const baseArmOffset = 0.1 + 0.4 * velocityFactor;

      // Left arm target - positioned to the left side of the body
      vec3.copy(
        std.add(
          bodyGlobalPos,
          std.add(
            std.mul(headForward, figure8X * armAmplitude + 0.1),
            std.add(
              std.mul(headRight, -1.3),
              d.vec3f(0, figure8Y * armAmplitude * 0.7 + baseArmOffset, 0),
            ),
          ),
        ),
        leftArmTarget,
      );

      // Right arm target - positioned to the right side of the body
      vec3.copy(
        std.add(
          bodyGlobalPos,
          std.add(
            std.mul(headForward, -figure8X * armAmplitude + 0.1),
            std.add(
              std.mul(headRight, 1.3),
              d.vec3f(0, figure8Y * armAmplitude * 0.7 + baseArmOffset, 0),
            ),
          ),
        ),
        rightArmTarget,
      );

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

      let prefersLeftLeg =
        (armAnimationPhase / (Math.PI * 2)) % 1 > 0.5 && rightLegPlaced;
      let prefersRightLeg =
        (armAnimationPhase / (Math.PI * 2)) % 1 < 0.5 && !rightLegPlaced;
      if (velocityFactor < 0.1) {
        // We care less about placing steps in synchronicity
        prefersLeftLeg = true;
        prefersRightLeg = true;
      }

      const leftPick = std.add(
        rootPos,
        std.add(
          std.mul(headForward, 0.2 + 1.6 * velocityFactor),
          std.mul(headRight, -0.6),
        ),
      );

      // Position in front of the body
      const rightPick = std.add(
        rootPos,
        std.add(
          std.mul(headForward, 0.2 + 1.6 * velocityFactor),
          std.mul(headRight, 0.6),
        ),
      );

      // Calculate distance from the current and the desired position
      const leftLegTargetDist = vec3.distance(leftLegTarget, leftPick);
      const rightLegTargetDist = vec3.distance(rightLegTarget, rightPick);

      // Update leg transition timers
      if (leftLegInTransition) {
        leftLegTransitionTime += dt;
        if (leftLegTransitionTime >= LEG_TRANSITION_DURATION) {
          leftLegInTransition = false;
          leftLegTransitionTime = 0;
        }
      }

      if (rightLegInTransition) {
        rightLegTransitionTime += dt;
        if (rightLegTransitionTime >= LEG_TRANSITION_DURATION) {
          rightLegInTransition = false;
          rightLegTransitionTime = 0;
        }
      }

      // Start leg transitions if they're too far from the body
      if (leftLegTargetDist > MAX_TARGET_DISTANCE && prefersLeftLeg && !leftLegInTransition) {
        rightLegPlaced = false;
        vec3.copy(leftLegTarget, leftLegPrevTarget);
        leftLegInTransition = true;
        leftLegTransitionTime = 0;
      }

      if (rightLegTargetDist > MAX_TARGET_DISTANCE && prefersRightLeg && !rightLegInTransition) {
        rightLegPlaced = true;
        vec3.copy(rightLegTarget, rightLegPrevTarget);
        rightLegInTransition = true;
        rightLegTransitionTime = 0;
      }

      // Interpolate leg targets during transitions
      if (leftLegInTransition) {
        const t = leftLegTransitionTime / LEG_TRANSITION_DURATION;
        const smoothT = t * t * (3 - 2 * t); // Smooth step interpolation
        
        // Linear interpolation between previous and new target
        leftLegTarget.x = leftLegPrevTarget.x + (leftPick.x - leftLegPrevTarget.x) * smoothT;
        leftLegTarget.z = leftLegPrevTarget.z + (leftPick.z - leftLegPrevTarget.z) * smoothT;
        
        // Add foot lift using sine curve
        const liftAmount = Math.sin(t * Math.PI) * FOOT_LIFT_HEIGHT;
        leftLegTarget.y = leftLegPrevTarget.y + (leftPick.y - leftLegPrevTarget.y) * smoothT + liftAmount;
      }

      if (rightLegInTransition) {
        const t = rightLegTransitionTime / LEG_TRANSITION_DURATION;
        const smoothT = t * t * (3 - 2 * t); // Smooth step interpolation
        
        // Linear interpolation between previous and new target
        rightLegTarget.x = rightLegPrevTarget.x + (rightPick.x - rightLegPrevTarget.x) * smoothT;
        rightLegTarget.z = rightLegPrevTarget.z + (rightPick.z - rightLegPrevTarget.z) * smoothT;
        
        // Add foot lift using sine curve
        const liftAmount = Math.sin(t * Math.PI) * FOOT_LIFT_HEIGHT;
        rightLegTarget.y = rightLegPrevTarget.y + (rightPick.y - rightLegPrevTarget.y) * smoothT + liftAmount;
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
