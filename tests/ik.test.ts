import { expect, test } from 'vitest';
import { solveIK } from '../src/lib/ik.ts';
import { vec3f } from 'typegpu/data';
import { distance, normalize, sub } from 'typegpu/std';

test('IK approximates well for 2 equal-length links', () => {
  const chain = [1, 1];
  const target = vec3f(1, 1, 0);
  const pull = vec3f(0, 0, 1);
  const forward = vec3f(0, 0, 1);
  const right = vec3f(1, 0, 0);

  const points = solveIK(chain, target, pull, forward, right);

  expect(points.length).toBe(3);
  // The last point should be AT the target
  expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

  // Distances between points should be kept intact
  expect(distance(points[0], points[1])).toBeCloseTo(1, 4);
  expect(distance(points[1], points[2])).toBeCloseTo(1, 4);
});

test('IK approximates well for 3 equal-length links', () => {
  const chain = [1, 1, 1];
  const target = vec3f(2, 1, 0);
  const pull = vec3f(0, 0, 1);
  const forward = vec3f(0, 0, 1);
  const right = vec3f(1, 0, 0);

  const points = solveIK(chain, target, pull, forward, right);

  expect(points.length).toBe(chain.length + 1);
  expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);
});

test('IK approximates well for 4 links', () => {
  const chain = [1, 1, 1, 1];
  const target = vec3f(2, 2, 0);
  const pull = vec3f(0, 0, 1);
  const forward = vec3f(0, 0, 1);
  const right = vec3f(1, 0, 0);

  const points = solveIK(chain, target, pull, forward, right);

  expect(points.length).toBe(chain.length + 1);
  expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);
});

test('IK preserves chain segment lengths', () => {
  const chain = [1, 1, 1];
  const target = vec3f(2, 1, 0);
  const pull = vec3f(0, 0, 1);
  const forward = vec3f(0, 0, 1);
  const right = vec3f(1, 0, 0);

  const points = solveIK(chain, target, pull, forward, right);

  for (let i = 1; i < points.length; i++) {
    const segmentLength = distance(points[i], points[i - 1]);
    expect(segmentLength).toBeCloseTo(chain[i - 1], 4);
  }
});

test('IK handles unreachable targets', () => {
  const chain = [1, 1];
  const target = vec3f(10, 10, 0); // Target too far to reach
  const pull = vec3f(0, 0, 1);
  const forward = vec3f(0, 0, 1);
  const right = vec3f(1, 0, 0);

  const points = solveIK(chain, target, pull, forward, right);

  expect(points.length).toBe(chain.length + 1);
  // Chain should stretch towards target
  const finalDir = normalize(sub(target, points[0]));
  const endDir = normalize(sub(points[points.length - 1], points[0]));
  expect(distance(finalDir, endDir)).toBeLessThan(0.1);
});
