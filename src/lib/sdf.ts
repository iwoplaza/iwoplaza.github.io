import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

export const MAX_AABBS = 32;

// Structure to hold both distance and color
export const Shape = d.struct({
	color: d.vec3f,
	dist: d.f32,
});

// Structure to hold AABB information
export const AABB = d.struct({
	min: d.vec3f,
	max: d.vec3f,
});

// Structure to hold ray-AABB intersection result
export const AABBHit = d.struct({
	enter: d.f32,
	exit: d.f32,
});

// Ray-AABB intersection test
export const intersectAABB = tgpu.fn(
	[d.vec3f, d.vec3f, AABB],
	AABBHit,
)((ro, rd, box) => {
	const dirfrac = d.vec3f(d.f32(1) / rd.x, d.f32(1) / rd.y, d.f32(1) / rd.z);

	const t1 = std.mul(std.sub(box.min, ro), dirfrac);
	const t2 = std.mul(std.sub(box.max, ro), dirfrac);

	const tmin = std.max(
		std.max(std.min(t1.x, t2.x), std.min(t1.y, t2.y)),
		std.min(t1.z, t2.z),
	);
	const tmax = std.min(
		std.min(std.max(t1.x, t2.x), std.max(t1.y, t2.y)),
		std.max(t1.z, t2.z),
	);

	return AABBHit({
		enter: tmin,
		exit: tmax,
	});
});

// Function to sort AABBHits by enter distance
export const sortHits = tgpu.fn([
	d.ptrFn(d.arrayOf(AABBHit, MAX_AABBS)),
	d.i32,
])((hits, numHits) => {
	// Simple bubble sort since we expect small number of AABBs
	for (let i = d.i32(0); i < numHits - 1; i++) {
		for (let j = d.i32(0); j < numHits - i - 1; j++) {
			if (hits[j].enter > hits[j + 1].enter) {
				// Swap
				const temp = hits[j];
				hits[j] = hits[j + 1];
				hits[j + 1] = temp;
			}
		}
	}
});

export const opSubtraction = tgpu.fn(
	[d.f32, d.f32],
	d.f32,
)((d1, d2) => std.max(-d1, d2));

export const opElongate = tgpu.fn(
	[d.vec3f, d.vec3f],
	d.vec3f,
)((p, h) => std.sub(p, std.clamp(p, std.neg(h), h)));

export const sdOctahedron = tgpu.fn(
	[d.vec3f, d.f32],
	d.f32,
)((p, s) => {
	const pp = d.vec3f(std.abs(p.x), std.abs(p.y), std.abs(p.z));
	const m = pp.x + pp.y + pp.z - s;

	let q = pp;
	if (3.0 * pp.x < m) q = pp.xyz;
	else if (3.0 * pp.y < m) q = pp.yzx;
	else if (3.0 * pp.z < m) q = pp.zxy;
	else return m * 0.57735027;

	const k = std.clamp(0.5 * (q.z - q.y + s), 0.0, s);
	return std.length(d.vec3f(q.x, q.y - s + k, q.z - k));
});

// float opSmoothSubtraction( float d1, float d2, float k )
// {
//     float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
//     return mix( d2, -d1, h ) + k*h*(1.0-h);
// }

export const sdCylinder = tgpu.fn(
	[d.vec3f, d.f32, d.f32],
	d.f32,
)((p, r, h) => {
	const dd = d.vec2f(std.length(p.xz), p.y);
	const q = d.vec2f(dd.x - r, std.abs(dd.y) - h / 2);
	return std.min(std.max(q.x, q.y), 0) + std.length(std.max(q, d.vec2f()));
});

export const sdRoundedCylinder = tgpu.fn(
	[d.vec3f, d.f32, d.f32, d.f32],
	d.f32,
)((p, ra, rb, h) => {
	const dd = d.vec2f(std.length(p.xz) - 2.0 * ra + rb, std.abs(p.y) - h);
	return (
		std.min(std.max(dd.x, dd.y), 0.0) + std.length(std.max(dd, d.vec2f())) - rb
	);
});

export const sdCappedTorus = tgpu.fn(
	[d.vec3f, d.vec2f, d.f32, d.f32],
	d.f32,
)((p, sc, ra, rb) => {
	const px = d.vec3f(std.abs(p.x), p.y, p.z);
	const k = std.select(
		std.dot(px.xy, sc),
		std.length(px.xy),
		sc.y * px.x > sc.x * px.y,
	);
	return std.sqrt(std.dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
});

/**
 * c is the sin/cos of the angle, h is height
 */
export const sdCone = tgpu.fn(
	[d.vec3f, d.vec2f, d.f32],
	d.f32,
)((p, c, h) => {
	const q = std.length(p.xz);
	return std.max(std.dot(c.xy, d.vec2f(q, p.y)), -h - p.y);
});

export const smoothstep = tgpu.fn([d.f32, d.f32, d.f32], d.f32)`(a, b, t) {
  return smoothstep(a, b, t);
}`;

export const smoothShapeUnion = tgpu.fn(
	[Shape, Shape, d.f32],
	Shape,
)((a, b, k) => {
	const h = std.max(k - std.abs(a.dist - b.dist), 0) / k;
	const m = h * h;

	// Smooth min for distance
	const dist = std.min(a.dist, b.dist) - m * k * (1 / d.f32(4));

	// Blend colors based on relative distances and smoothing
	const weight = m + std.select(0, 1 - m, a.dist > b.dist);
	const color = std.mix(a.color, b.color, weight);

	return { dist, color };
});

export const shapeUnion = tgpu.fn(
	[Shape, Shape],
	Shape,
)((a, b) => ({
	color: std.select(a.color, b.color, a.dist > b.dist),
	dist: std.min(a.dist, b.dist),
}));
