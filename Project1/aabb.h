#pragma once
#include "vec3f.h"
#include <float.h>
#include <stdlib.h>
#include <cstdio>

FORCEINLINE void
vmin(vec3f& a, const vec3f& b)
{
	a.set_value(
		fmin(a[0], b[0]),
		fmin(a[1], b[1]),
		fmin(a[2], b[2]));
}

FORCEINLINE void
vmax(vec3f& a, const vec3f& b)
{
	a.set_value(
		fmax(a[0], b[0]),
		fmax(a[1], b[1]),
		fmax(a[2], b[2]));
}


class aabb {
	FORCEINLINE void init() {
		_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	}

public:
	vec3f _min;
	vec3f _max;

	FORCEINLINE aabb() {
		init();
	}

	FORCEINLINE aabb(const vec3f& v) {
		_min = _max = v;
	}

	FORCEINLINE aabb(const vec3f& a, const vec3f& b) {
		_min = a;
		_max = a;
		vmin(_min, b);
		vmax(_max, b);
	}

	FORCEINLINE bool overlaps(const aabb& b) const
	{
		if (_min[0] > b._max[0]) return false;
		if (_min[1] > b._max[1]) return false;
		if (_min[2] > b._max[2]) return false;

		if (_max[0] < b._min[0]) return false;
		if (_max[1] < b._min[1]) return false;
		if (_max[2] < b._min[2]) return false;

		return true;
	}

	FORCEINLINE bool overlaps(const aabb& b, aabb& ret) const
	{
		if (!overlaps(b))
			return false;

		ret._min.set_value(
			fmax(_min[0], b._min[0]),
			fmax(_min[1], b._min[1]),
			fmax(_min[2], b._min[2]));

		ret._max.set_value(
			fmin(_max[0], b._max[0]),
			fmin(_max[1], b._max[1]),
			fmin(_max[2], b._max[2]));

		return true;
	}

	FORCEINLINE bool inside(const vec3f& p) const
	{
		if (p[0] < _min[0] || p[0] > _max[0]) return false;
		if (p[1] < _min[1] || p[1] > _max[1]) return false;
		if (p[2] < _min[2] || p[2] > _max[2]) return false;

		return true;
	}

	FORCEINLINE aabb& operator += (const vec3f& p)
	{
		vmin(_min, p);
		vmax(_max, p);
		return *this;
	}

	FORCEINLINE aabb& operator += (const aabb& b)
	{
		vmin(_min, b._min);
		vmax(_max, b._max);
		return *this;
	}

	FORCEINLINE aabb operator + (const aabb& v) const
	{
		aabb rt(*this); return rt += v;
	}

	FORCEINLINE double width()  const { return _max[0] - _min[0]; }
	FORCEINLINE double height() const { return _max[1] - _min[1]; }
	FORCEINLINE double depth()  const { return _max[2] - _min[2]; }
	FORCEINLINE vec3f center() const { return (_min + _max) * 0.5; }
	FORCEINLINE double volume() const { return width() * height() * depth(); }


	// FORCEINLINE bool empty() const {
	// 	return _max.x < _min.x;
	// }

	FORCEINLINE void enlarge(double thickness) {
		_max += vec3f(thickness, thickness, thickness);
		_min -= vec3f(thickness, thickness, thickness);
	}

	vec3f getMax() { return _max; }
	vec3f getMin() { return _min; }

	// void print(FILE* fp) {
	// 	fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	// }

	void visualize();
};