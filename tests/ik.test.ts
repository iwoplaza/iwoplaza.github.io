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

  test('IK works with 2 different-length segments (short-long)', () => {
    const chain = [0.5, 2];
    const target = d.vec3f(1.5, 1, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(3);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify segment lengths are preserved
    expect(distance(points[0], points[1])).toBeCloseTo(0.5, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(2, 4);
  });

  test('IK works with 2 different-length segments (long-short)', () => {
    const chain = [2, 0.5];
    const target = d.vec3f(1.5, 1, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(3);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify segment lengths are preserved
    expect(distance(points[0], points[1])).toBeCloseTo(2, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(0.5, 4);
  });

  test('IK works with 3 different-length segments (ascending)', () => {
    const chain = [0.5, 1, 1.5];
    const target = d.vec3f(2, 1.5, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(4);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify all segment lengths are preserved
    expect(distance(points[0], points[1])).toBeCloseTo(0.5, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(1, 4);
    expect(distance(points[2], points[3])).toBeCloseTo(1.5, 4);
  });

  test('IK works with 3 different-length segments (descending)', () => {
    const chain = [2, 1, 0.5];
    const target = d.vec3f(2, 1.5, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(4);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify all segment lengths are preserved
    expect(distance(points[0], points[1])).toBeCloseTo(2, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(1, 4);
    expect(distance(points[2], points[3])).toBeCloseTo(0.5, 4);
  });

  test('IK works with 4 mixed-length segments', () => {
    const chain = [0.8, 1.5, 0.3, 1.2];
    const target = d.vec3f(2.5, 2, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(5);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify all segment lengths are preserved
    for (let i = 1; i < points.length; i++) {
      const segmentLength = distance(points[i], points[i - 1]);
      expect(segmentLength).toBeCloseTo(chain[i - 1], 4);
    }
  });

  test('IK handles very short segments in mixed chain', () => {
    const chain = [0.1, 2, 0.05, 1.5];
    const target = d.vec3f(2, 2, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(5);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify all segment lengths are preserved, especially the tiny ones
    expect(distance(points[0], points[1])).toBeCloseTo(0.1, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(2, 4);
    expect(distance(points[2], points[3])).toBeCloseTo(0.05, 4);
    expect(distance(points[3], points[4])).toBeCloseTo(1.5, 4);
  });

  test('IK handles extreme length ratios', () => {
    const chain = [0.01, 5, 0.02];
    const target = d.vec3f(3, 3, 0);

    const points = solveIK(chain, target);

    expect(points.length).toBe(4);
    expect(distance(points[points.length - 1], target)).toBeLessThan(0.1);

    // Verify segment lengths are preserved despite extreme ratios
    expect(distance(points[0], points[1])).toBeCloseTo(0.01, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(5, 4);
    expect(distance(points[2], points[3])).toBeCloseTo(0.02, 4);
  });

  test('IK with different-length segments handles unreachable targets', () => {
    const chain = [0.5, 1.5, 0.8];
    const target = d.vec3f(20, 20, 0); // Target too far to reach

    const points = solveIK(chain, target);

    expect(points.length).toBe(4);

    // Chain should stretch towards target
    const finalDir = normalize(sub(target, points[0]));
    const endDir = normalize(sub(points[points.length - 1], points[0]));
    expect(distance(finalDir, endDir)).toBeLessThan(0.1);

    // Segment lengths should still be preserved
    expect(distance(points[0], points[1])).toBeCloseTo(0.5, 4);
    expect(distance(points[1], points[2])).toBeCloseTo(1.5, 4);
    expect(distance(points[2], points[3])).toBeCloseTo(0.8, 4);
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
          0.6154797077178955,
          0.6154797077178955,
        ],
      ]
    `);
  });

  test('returns correct pitch for positive Z movement', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(0, 1, 1)];
    const angles = extractAnglesBetweenPoints(points);

    // Normalized direction is (0, 1/√2, 1/√2), so pitch = asin(1/√2) = π/4
    expect(angles[0].x).toBeCloseTo(Math.PI / 4, 4); // pitch
    expect(angles[0].y).toBeCloseTo(0, 4); // roll
  });

  test('returns correct pitch for negative Z movement', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(0, 1, -1)];
    const angles = extractAnglesBetweenPoints(points);

    // Normalized direction is (0, 1/√2, -1/√2), so pitch = asin(-1/√2) = -π/4
    expect(angles[0].x).toBeCloseTo(-Math.PI / 4, 4); // pitch
    expect(angles[0].y).toBeCloseTo(0, 4); // roll
  });

  test('returns correct roll for positive X movement', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(1, 1, 0)];
    const angles = extractAnglesBetweenPoints(points);

    // Normalized direction is (1/√2, 1/√2, 0), so roll = asin(1/√2) = π/4
    expect(angles[0].x).toBeCloseTo(0, 4); // pitch
    expect(angles[0].y).toBeCloseTo(Math.PI / 4, 4); // roll
  });

  test('returns correct roll for negative X movement', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(-1, 1, 0)];
    const angles = extractAnglesBetweenPoints(points);

    // Normalized direction is (-1/√2, 1/√2, 0), so roll = asin(-1/√2) = -π/4
    expect(angles[0].x).toBeCloseTo(0, 4); // pitch
    expect(angles[0].y).toBeCloseTo(-Math.PI / 4, 4); // roll
  });

  test('handles pure X-axis movement', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(1, 0, 0)];
    const angles = extractAnglesBetweenPoints(points);

    // Direction is (1, 0, 0), so pitch = 0, roll = π/2
    expect(angles[0].x).toBeCloseTo(0, 4); // pitch
    expect(angles[0].y).toBeCloseTo(Math.PI / 2, 4); // roll
  });

  test('handles pure Z-axis movement', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(0, 0, 1)];
    const angles = extractAnglesBetweenPoints(points);

    // Direction is (0, 0, 1), so pitch = π/2, roll = 0
    expect(angles[0].x).toBeCloseTo(Math.PI / 2, 4); // pitch
    expect(angles[0].y).toBeCloseTo(0, 4); // roll
  });

  test('handles diagonal movement in XZ plane', () => {
    const points = [d.vec3f(0, 0, 0), d.vec3f(1, 0, 1)];
    const angles = extractAnglesBetweenPoints(points);

    // Normalized direction is (1/√2, 0, 1/√2), so pitch = π/4, roll = π/4
    expect(angles[0].x).toBeCloseTo(Math.PI / 4, 4); // pitch
    expect(angles[0].y).toBeCloseTo(Math.PI / 4, 4); // roll
  });

  test('extracts angles for 3-point chain', () => {
    const points = [
      d.vec3f(0, 0, 0),
      d.vec3f(1, 1, 0), // First segment: roll = π/4, pitch = 0
      d.vec3f(1, 2, 1), // Second segment: roll = 0, pitch = π/4
    ];
    const angles = extractAnglesBetweenPoints(points);

    expect(angles.length).toBe(2);

    // First segment angles
    expect(angles[0].x).toBeCloseTo(0, 4); // pitch
    expect(angles[0].y).toBeCloseTo(Math.PI / 4, 4); // roll

    // Second segment angles
    expect(angles[1].x).toBeCloseTo(Math.PI / 4, 4); // pitch
    expect(angles[1].y).toBeCloseTo(0, 4); // roll
  });

  test('extracts angles for 4-point chain with varying directions', () => {
    const points = [
      d.vec3f(0, 0, 0),
      d.vec3f(0, 1, 0), // Straight down: pitch = 0, roll = 0
      d.vec3f(1, 2, 0), // Right: pitch = 0, roll = π/4
      d.vec3f(1, 2, 1), // Forward: pitch = π/2, roll = 0
    ];
    const angles = extractAnglesBetweenPoints(points);

    expect(angles.length).toBe(3);

    // First segment: straight down
    expect(angles[0].x).toBeCloseTo(0, 4);
    expect(angles[0].y).toBeCloseTo(0, 4);

    // Second segment: diagonal right
    expect(angles[1].x).toBeCloseTo(0, 4);
    expect(angles[1].y).toBeCloseTo(Math.PI / 4, 4);

    // Third segment: straight forward
    expect(angles[2].x).toBeCloseTo(Math.PI / 2, 4);
    expect(angles[2].y).toBeCloseTo(0, 4);
  });

  test('handles segments with different lengths correctly', () => {
    const points = [
      d.vec3f(0, 0, 0),
      d.vec3f(2, 2, 0), // Long segment at 45° roll
      d.vec3f(2.5, 2.5, 0.5), // Short segment: direction (0.5, 0.5, 0.5) normalized to (1/√3, 1/√3, 1/√3)
    ];
    const angles = extractAnglesBetweenPoints(points);

    expect(angles.length).toBe(2);

    // First segment: long diagonal in XY plane
    expect(angles[0].x).toBeCloseTo(0, 4);
    expect(angles[0].y).toBeCloseTo(Math.PI / 4, 4);

    // Second segment: direction (0.5, 0.5, 0.5) normalized gives asin(1/√3) for both angles
    const expectedAngle = Math.asin(1 / Math.sqrt(3));
    expect(angles[1].x).toBeCloseTo(expectedAngle, 4);
    expect(angles[1].y).toBeCloseTo(expectedAngle, 4);
  });

  test('handles extreme angles near limits', () => {
    const points = [
      d.vec3f(0, 0, 0),
      d.vec3f(0, 0.1, 1), // Nearly vertical forward
    ];
    const angles = extractAnglesBetweenPoints(points);

    // Should be close to maximum pitch (π/2) with minimal roll
    expect(angles[0].x).toBeCloseTo(Math.asin(1 / Math.sqrt(1.01)), 4);
    expect(angles[0].y).toBeCloseTo(0, 4);
  });
});
