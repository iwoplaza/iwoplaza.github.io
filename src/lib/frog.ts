import { opSmoothUnion, sdRoundedBox3d, sdSphere } from "@typegpu/sdf";
import tgpu, { type TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { mat4 } from "wgpu-matrix";
import { opSubtraction, sdCappedTorus, Shape, shapeUnion } from "./sdf.ts";

const getFrogHead = tgpu.fn([d.vec3f], Shape)((p) => {
  const center = d.vec3f(0, 2, 0);
  const localP = std.sub(p, center);
  // Symmetric along the X-axis
  localP.x = std.abs(localP.x);
  const skinColor = d.vec3f(0.3, 0.8, 0.4);
  // let head = sdRoundedCylinder(localP.xzy, 0.5, 0.5, 0);
  let head = sdRoundedBox3d(localP, d.vec3f(0.8, 0.7, 0.6), 0.6);
  // head = opSmoothUnion(head, sdCappedTorus(localP, d.vec2f(1), 0, 1), 0.1);
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

export const FrogRig = d.struct({
  head: d.mat4x4f,
});

export function createFrog(root: TgpuRoot) {
  const frogRigCpu = FrogRig({
    head: mat4.identity(d.mat4x4f()),
  });
  const frogRig = root.createUniform(FrogRig, frogRigCpu);
  function updateFrogRig() {
    frogRig.write(frogRigCpu);
  }

  const getFrog = tgpu.fn([d.vec3f], Shape)((p) => {
    const tp = std.mul(frogRig.$.head, d.vec4f(p, 1)).xyz;
    return getFrogHead(tp);
  });

  return {
    getFrog,
  };
}
