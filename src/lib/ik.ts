import { type v3f, vec3f } from 'typegpu/data';
import { add, mul, normalize, sub } from 'typegpu/std';

/**
 * Origin of the chain is assumed to be (0, 0, 0)
 * @param chain
 * @param target
 * @param pull
 * @param forward
 * @param right
 */
export function solveIK(
  chain: readonly number[],
  target: v3f,
  pull: v3f,
  forward: v3f,
  right: v3f,
) {
  const chainPoints: v3f[] = [vec3f()];
  const dir = normalize(target);
  let acc = vec3f();
  for (let i = 0; i < chain.length; ++i) {
    acc = add(acc, mul(dir, chain[i]));
    // Adding points along the direction of IK, nudged aside by 'pull'
    chainPoints.push(add(acc, pull));
  }

  const iterations = 10;
  for (let iter = 0; iter < iterations; iter++) {
    // Making sure the last point is AT the target at the beginning of the pass
    chainPoints[chainPoints.length - 1] = target;

    // Pull the chain towards the target
    for (let i = chainPoints.length - 2; i >= 0; i--) {
      const curr = chainPoints[i];
      const segmentLength = chain[i];

      const next = chainPoints[i + 1];
      const dir = normalize(sub(curr, next));
      chainPoints[i] = add(next, mul(dir, segmentLength));
    }

    // Ensure each segment maintains its length
    for (let i = 1; i < chainPoints.length; i++) {
      const prev = chainPoints[i - 1];
      const curr = chainPoints[i];
      const segmentLength = chain[i - 1];

      const dir = normalize(add(curr, mul(prev, -1)));
      chainPoints[i] = add(prev, mul(dir, segmentLength));
    }
  }

  return chainPoints;
}
