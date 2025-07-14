import { mat4x4f, type v3f, type v4f } from 'typegpu/data';
import { neg } from 'typegpu/std';
import { mat3n, mat4n, quatn } from 'wgpu-matrix';

interface BoneOptions {
  readonly parent?: Bone | undefined;
  readonly inheritRotation?: boolean;
}

export class Bone {
  readonly mat = mat4n.identity(mat4x4f());
  readonly invMat = mat4n.identity(mat4x4f());

  constructor(
    readonly pos: v3f,
    readonly rot: v4f,
    private readonly options: BoneOptions,
  ) {}

  compute() {
    // INVERSE
    // Local transform
    mat4n.fromQuat(quatn.inverse(this.rot), this.invMat);
    if (this.options.parent && !this.options.inheritRotation) {
      // Undoing parent rotation
      mat4n.mul(
        this.invMat,
        mat4n.fromMat3(mat3n.fromMat4(this.options.parent.mat)),
        this.invMat,
      );
    }
    // Lock into place
    mat4n.translate(this.invMat, neg(this.pos), this.invMat);
    if (this.options.parent) {
      // Parent transform
      mat4n.mul(this.invMat, this.options.parent.invMat, this.invMat);
    }

    // NORMAL
    mat4n.inverse(this.invMat, this.mat);
  }
}
