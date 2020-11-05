#include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <cuda.h>
// #include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "cuda_function.h"

#include "vec3f.h"
#include "timer.h"
#include "box.h"
#include "xyz-rgb.h"
#include "windows.h"
#include "wingdi.h"
#include "WTypesbase.h"

using namespace std;

std::vector<xyz2rgb> gHashTab;

class SceneData {
protected:
	int _cx, _cy;
	float* _xyzs;
	float* _rgbs;

	float* _rgbsNew;
	BOX _bound;

public:
	void loadRGB(char* fname)
	{
		FILE* fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);
		int sz = _cx * _cy * 3;
		_rgbs = new float[sz];
		fread(_rgbs, sizeof(float), sz, fp);
		fclose(fp);
	}

	void loadXYZ(char* fname)
	{
		FILE* fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx * _cy * 3;
		_xyzs = new float[sz];
		fread(_xyzs, sizeof(float), sz, fp);
		fclose(fp);
	}

	int width() const { return _cx; }
	int height() const { return _cy; }

	virtual void saveAsBmp(float* ptr, char* fn) {
		int sz = _cx * _cy * 3;

		BYTE* idx = new BYTE[sz];
		for (int i = 0; i < sz; ) {
			idx[i] = ptr[i + 2] * 255;
			idx[i + 1] = ptr[i + 1] * 255;
			idx[i + 2] = ptr[i] * 255;

			i += 3;
		}

		if (!idx)
			return;

		int colorTablesize = 0;

		int biBitCount = 24;

		if (biBitCount == 8)
			colorTablesize = 1024;

		//
		int lineByte = (_cx * biBitCount / 8 + 3) / 4 * 4;

		//
		FILE* fp = fopen(fn, "wb");
		if (fp == 0)
			return;

		//
		BITMAPFILEHEADER fileHead;
		fileHead.bfType = 0x4D42;

		fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * _cy;
		fileHead.bfReserved1 = 0;
		fileHead.bfReserved2 = 0;

		//
		fileHead.bfOffBits = 54 + colorTablesize;

		//
		fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

		//
		BITMAPINFOHEADER head;
		head.biBitCount = biBitCount;
		head.biClrImportant = 0;
		head.biClrUsed = 0;
		head.biCompression = 0;
		head.biHeight = _cy;
		head.biPlanes = 1;
		head.biSize = 40;
		head.biSizeImage = lineByte * _cy;
		head.biWidth = _cx;
		head.biXPelsPerMeter = 0;
		head.biYPelsPerMeter = 0;

		//
		fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

		//
		fwrite(idx, _cy * lineByte, 1, fp);

		//
		fclose(fp);

		delete[] idx;
	}

	void resetRGBnew() {
		if (_rgbsNew) delete[] _rgbsNew;
		_rgbsNew = new float[_cx * _cy * 3];

		for (int i = 0; i < _cx; i++) {
			for (int j = 0; j < _cy; j++) {
				float* p2 = _rgbsNew + (i * _cy + j) * 3;
				p2[0] = 1;
				p2[1] = 0;
				p2[2] = 0;
			}
		}
	}

public:
	SceneData() {
		_cx = _cy = 0;
		_xyzs = _rgbs = NULL;
		_rgbsNew = NULL;
	}

	~SceneData() {
		if (_xyzs) delete[] _xyzs;
		if (_rgbs) delete[] _rgbs;
		if (_rgbsNew) delete[] _rgbsNew;
	}

	float* rgbs() { return _rgbs; }

	float* xyzs() { return _xyzs; }

	void saveNewRGB(char* fn) {
		saveAsBmp(_rgbsNew, fn);
	}

	void load(char* fname, int numofImg, bool updateHash = true) {
		char rgbFile[512], xyzFile[512];
		sprintf(rgbFile, "%s%s", fname, ".rgb");
		sprintf(xyzFile, "%s%s", fname, ".xyz");

		if (updateHash)
			loadRGB(rgbFile);
		// printf("try to load XYZ\n");
		loadXYZ(xyzFile);
		// printf("in load\n");
		resetRGBnew();

		float* p1 = _xyzs;
		float* p2 = _rgbs;
		int num = _cx * _cy;

		for (int i = 0; i < num; i++) {
			if (updateHash)
				gHashTab.push_back(xyz2rgb(vec3f(p1), vec3f(p2), numofImg, i));

			_bound += vec3f(p1);
			p1 += 3;
			p2 += 3;
		}
	}
};

class ProgScene : public SceneData {
protected:

	float* _nearest;

	void resetNearest()
	{
		int sz = _cx * _cy;
		_nearest = new float[sz];

		for (int i = 0; i < sz; i++)
			_nearest[i] = 1000000;
	}

public:
	ProgScene() {
		_nearest = NULL;
	}

	~ProgScene() {
		if (_nearest)
			delete[] _nearest;
	}


	void load(char* fname) {
		printf("here\n");
		SceneData::load(fname, -1, false);
		resetNearest();
	}

	void save(char* fname) {
		saveNewRGB(fname);
	}


};


void loadSceneXyzs(float *scene_xyzs, int size){
	for (int i=0; i<size; i++){
		scene_xyzs[i*3] = gHashTag[i]._xyz.x;
		scene_xyzs[i*3+1] = gHashTag[i]._xyz.y;
		scene_xyzs[i*3+2] = gHashTag[i]._xyz.z;
	}
}


SceneData scene[18];
ProgScene target;
int main()
{
	printf("sizeof*xyz2rgb: %d\n", sizeof(xyz2rgb));
	TIMING_BEGIN("Start loading...")
	target.load("all.bmp");
	// printf("size1: %d\n",gHashTab.size());
	scene[0].load("0-55.bmp", 0);
	printf("size2: %d\n",gHashTab.size());
	TIMING_END("Loading done...")

	// TIMING_BEGIN("Start rescanning...")
	// // 	target.update();
	// TIMING_END("Rescaning done...")

	float *target_xyzs = target.xyzs();
	int scene_size = gHashTag.size();

	float *scene_xyzs = （float*)malloc(scene_size*3*sizeof(float));
	loadSceneXyzs(scene_xyzs,scene_size);

	

	target.save("output.bmp");
	float a = 0;
	test_wrapper(&a);
	printf("a: %f\n", a);
	return 0;
}