import * as d from 'typegpu/data';
import { distance, normalize, sub } from 'typegpu/std';
import { describe, expect, test } from 'vitest';
import { extractAnglesBetweenPoints, solveIK } from '../src/lib/ik.ts';

describe('solveIK', () => {
  test('IK approximates well for 2 equal-length links', () => {
    const chain = [1, 1];
    const target = d.vec3f(1, 1, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(3);
    // The last point should be AT the target
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Distances between points should be kept intact
    expect(distance(points[0], points[1])).toBeCloseTo(1, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(1, 4);
  });

  test('IK approximates well for 3 equal-length links', () => {
    const chain = [1, 1, 1];
    const target = d.vec3f(2, 1, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(chain.length + 1);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);
  });

  test('IK approximates well for 4 links', () => {
    const chain = [1, 1, 1, 1];
    const target = d.vec3f(2, 2, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(chain.length + 1);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);
  });

  test('IK preserves chain segment lengths', () => {
    const chain = [1, 1, 1];
    const target = d.vec3f(2, 1, 0);

    const points = solveIK(chain, target);

    for (let i = 1; i < points.length; i++) {
      const segmentLength = distance(points[i], points[i - 1]);
      expect(segmentLength).toBeCloseTo(chain[i - 1], 4);
    }
  });

  test('IK handles unreachable targets', () => {
    const chain = [1, 1];
    const target = d.vec3f(10, 10, 0); // Target too far to reach

    const points = solveIK(chain, target);

    expect(points.length).toBe(chain.length + 1);
    // Chain should stretch towards target
    const finalDir = normalize(sub(target, points[0]));
    const endDir = normalize(sub(points[points.length - 1], points[0]));
    expect(distance(finalDir, endDir)).toBeLessThan(0.1);
  });
});

describe('extractAnglesBetweenPoints', () => {
  test('returns 0-angles for chain that goes straight down', () => {
    const points = [d.vec3f(), d.vec3f(0, 1, 0)];
    const angles = extractAnglesBetweenPoints(points);

    expect(angles).toMatchInlineSnapshot(`
      [
        e [
          0,
          0,
        ],
      ]
    `);
  });

  test('extracts angles between 2 points', () => {
    const points = [d.vec3f(), d.vec3f(1, 1, 1)];
    const angles = extractAnglesBetweenPoints(points);

    expect(angles).toMatchInlineSnapshot(`
      [
        e [
          1.5707963705062866,
          1.5707963705062866,
        ],
      ]
    `);
  });
});
