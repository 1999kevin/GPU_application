#pragma once

class xyz2rgb {
	vec3f _xyz;
	vec3f _rgb;
	int _index; //position index
	int _pidx; //img index

public:
	xyz2rgb() {}

	xyz2rgb(vec3f& pos, vec3f& cr, int& i, int& p) {
		_xyz = pos;
		_rgb = cr;
		_index = i;
		_pidx = p;
	}

	vec3f xyz() const { return _xyz; }
	vec3f rgb() const { return _rgb; }
	int pos() const { return _pidx; }
	int index() const { return _index; }
};
