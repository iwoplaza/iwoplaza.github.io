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

  const orbitOrigin = d.vec3f(0, 3, 0);
  // Yaw and pitch angles facing the origin.
  let orbitRadius = options.radius;
  let orbitYaw = options.yaw;
  let orbitPitch = options.pitch;

  const invProj = mat4.identity(d.mat4x4f());
  const invView = mat4.identity(d.mat4x4f());

  function uploadUniforms() {
    const invViewProj = mat4.mul(invView, invProj, d.mat4x4f());
    pov.writePartial({ invView, invViewProj });
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
    updateProjection(width: number, height: number) {
      const aspect = width / height;
      const fov = (24 / 180) * Math.PI;
      mat4.identity(invProj);
      mat4.scale(invProj, d.vec3f(aspect, 1, 1 / Math.tan(fov)), invProj);
      uploadUniforms();
    },
  };
}
