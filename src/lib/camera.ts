import type { TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat4 } from 'wgpu-matrix';

export interface OrbitOptions {
  radius: number;
  pitch: number;
  yaw: number;
}

export const POV = d.struct({
  invView: d.mat4x4f,
  aspect: d.vec2f,
  invViewProj: d.mat4x4f,
});

export function createOrbitCamera(
  root: TgpuRoot,
  touchSurface: HTMLElement,
  options: OrbitOptions,
) {
  let active = true;
  let isDragging = false;
  let prevX = 0;
  let prevY = 0;

  // Orbit center that will follow the target
  const orbitOrigin = d.vec3f(0, 3, 0);
  // Target position to follow (typically the frog)
  const targetPosition = d.vec3f(0, 0, 0);
  // Previous target position to calculate movement direction
  const prevTargetPosition = d.vec3f(0, 0, 0);
  // Direction of movement for overshooting
  const movementDirection = d.vec3f(0, 0, 0);
  // Current offset from target (includes overshooting)
  const currentOffset = d.vec3f(0, 0, 0);
  // Smoothing factor for camera movement (lower = smoother)
  const smoothingFactor = 0.1;
  // Maximum overshoot distance
  const maxOvershoot = 2.0;

  // Yaw and pitch angles facing the origin.
  let orbitRadius = options.radius;
  let orbitYaw = options.yaw;
  let orbitPitch = options.pitch;

  let aspect = 1;
  const invProj = mat4.identity(d.mat4x4f());
  const invView = mat4.identity(d.mat4x4f());

  function uploadUniforms() {
    const invViewProj = mat4.mul(invView, invProj, d.mat4x4f());
    pov.writePartial({ invView, invViewProj, aspect: d.vec2f(aspect, 1) });
  }

  const pov = root.createUniform(POV);

  function updateOrbit(dx: number, dy: number) {
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
    const newCameraPos = std.add(
      d.vec3f(newCamX, newCamY, newCamZ),
      orbitOrigin,
    );

    mat4.aim(newCameraPos, orbitOrigin, d.vec3f(0, 1, 0), invView);
    uploadUniforms();
  }
  updateOrbit(0, 0);

  touchSurface.addEventListener('wheel', (event: WheelEvent) => {
    if (!active) {
      return;
    }
    event.preventDefault();
    const zoomSensitivity = 0.005;
    orbitRadius = Math.max(1, orbitRadius + event.deltaY * zoomSensitivity);
    updateOrbit(0, 0);
  });

  touchSurface.addEventListener('mousedown', (event) => {
    if (!active) {
      return;
    }
    if (event.button === 0) {
      // Left Mouse Button controls Camera Orbit.
      isDragging = true;
    }
    prevX = event.clientX;
    prevY = event.clientY;
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  touchSurface.addEventListener('mousemove', (event) => {
    if (!active) {
      return;
    }
    const dx = event.clientX - prevX;
    const dy = event.clientY - prevY;
    prevX = event.clientX;
    prevY = event.clientY;

    if (isDragging) {
      updateOrbit(dx, dy);
    }
  });

  // Mobile touch support.
  touchSurface.addEventListener('touchstart', (event: TouchEvent) => {
    if (!active) {
      return;
    }

    event.preventDefault();
    if (event.touches.length === 1) {
      // Single touch controls Camera Orbit.
      isDragging = true;
    }
    // Use the first touch for rotation.
    prevX = event.touches[0].clientX;
    prevY = event.touches[0].clientY;
  });

  touchSurface.addEventListener('touchmove', (event: TouchEvent) => {
    if (!active) {
      return;
    }

    event.preventDefault();
    const touch = event.touches[0];
    const dx = touch.clientX - prevX;
    const dy = touch.clientY - prevY;
    prevX = touch.clientX;
    prevY = touch.clientY;

    if (isDragging && event.touches.length === 1) {
      updateOrbit(dx, dy);
    }
  });

  touchSurface.addEventListener('touchend', (event: TouchEvent) => {
    event.preventDefault();
    if (event.touches.length === 0) {
      isDragging = false;
    }
  });

  // Update the target position and calculate the orbit origin with overshooting
  function updateTargetPosition(newPosition: d.v3f, dt: number) {
    // Store previous position before updating
    prevTargetPosition.x = targetPosition.x;
    prevTargetPosition.y = targetPosition.y;
    prevTargetPosition.z = targetPosition.z;

    // Update target position
    targetPosition.x = newPosition.x;
    targetPosition.y = newPosition.y;
    targetPosition.z = newPosition.z;

    // Calculate movement direction and magnitude
    movementDirection.x = targetPosition.x - prevTargetPosition.x;
    movementDirection.y = targetPosition.y - prevTargetPosition.y;
    movementDirection.z = targetPosition.z - prevTargetPosition.z;

    const movementMagnitude = Math.sqrt(
      movementDirection.x * movementDirection.x +
        movementDirection.y * movementDirection.y +
        movementDirection.z * movementDirection.z,
    );

    // Normalize movement direction if there is movement
    if (movementMagnitude > 0.001) {
      movementDirection.x /= movementMagnitude;
      movementDirection.y /= movementMagnitude;
      movementDirection.z /= movementMagnitude;

      // Calculate desired offset with overshooting based on movement speed
      const desiredOffset = {
        x: movementDirection.x * Math.min(movementMagnitude * 2, maxOvershoot),
        y: 0, // Keep vertical position stable
        z: movementDirection.z * Math.min(movementMagnitude * 2, maxOvershoot),
      };

      // Smoothly interpolate current offset towards desired offset
      currentOffset.x +=
        (desiredOffset.x - currentOffset.x) * smoothingFactor * dt * 10;
      currentOffset.y +=
        (desiredOffset.y - currentOffset.y) * smoothingFactor * dt * 10;
      currentOffset.z +=
        (desiredOffset.z - currentOffset.z) * smoothingFactor * dt * 10;
    } else {
      // If not moving, gradually reduce the offset
      currentOffset.x *= 1 - smoothingFactor * dt * 5;
      currentOffset.y *= 1 - smoothingFactor * dt * 5;
      currentOffset.z *= 1 - smoothingFactor * dt * 5;
    }

    // Update orbit origin with target position plus offset
    orbitOrigin.x = targetPosition.x + currentOffset.x;
    orbitOrigin.y = targetPosition.y + 3; // Keep camera looking slightly above the target
    orbitOrigin.z = targetPosition.z + currentOffset.z;

    // Update the camera view
    updateOrbit(0, 0);
  }

  return {
    get active() {
      return active;
    },
    set active(v: boolean) {
      active = v;
    },
    get orbitRadius() {
      return orbitRadius;
    },
    get orbitYaw() {
      return orbitYaw;
    },
    get orbitPitch() {
      return orbitPitch;
    },
    pov,
    updateTargetPosition,
    updateProjection(width: number, height: number) {
      aspect = width / height;
      const fov = (24 / 180) * Math.PI;
      mat4.identity(invProj);
      mat4.scale(invProj, d.vec3f(aspect, 1, 1 / Math.tan(fov)), invProj);
      uploadUniforms();
    },
  };
}
