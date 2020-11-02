//**************************************************************************************
//  Copyright (C) 2019 - 2022, Min Tang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#include "stdafx.h"

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <vector>
#include <iostream>
using namespace std;

#include "vec3f.h"
#include "box.h"
#include "xyz-rgb.h"
#include "timer.h"

void build_bvh();
void destroy_bvh();
bool findNearest2(xyz2rgb &org, xyz2rgb &ret);
int maxPts();

std::vector<xyz2rgb> gHashTab;

bool getColor(vec3f &xyz, float *nearest, vec3f &cr, int &idx, int &pidx)
{
	xyz2rgb input(xyz, vec3f(), idx, pidx), ret;

	bool find = findNearest2(input, ret);
	if (!find)
		return false;

	vec3f now = ret.xyz();
	double dist = vdistance(xyz, now);
	if (dist < *nearest) {
		*nearest = dist;
		cr = ret.rgb();
		idx = ret.index();
		pidx = ret.pos();

		return true;
	}
	else
		return false;
}

class SceneData {
protected:
	int _cx, _cy;
	float *_xyzs;
	float *_rgbs;
	
	float *_rgbsNew;
	BOX _bound;

public:
	void loadRGB(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);
		int sz = _cx*_cy * 3;
		_rgbs = new float[sz];
		fread(_rgbs, sizeof(float), sz, fp);
		fclose(fp);
	}

	void loadXYZ(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx*_cy * 3;
		_xyzs = new float[sz];
		fread(_xyzs, sizeof(float), sz, fp);
		fclose(fp);
	}

	int width() const { return _cx; }
	int height() const { return _cy; }

	virtual void saveAsBmp(float *ptr, char *fn) {
		int sz = _cx * _cy * 3;

		BYTE *idx = new BYTE[sz];
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

		//待存储图像数据每行字节数为4的倍数
		int lineByte = (_cx * biBitCount / 8 + 3) / 4 * 4;

		//以二进制写的方式打开文件
		FILE *fp = fopen(fn, "wb");
		if (fp == 0)
			return;

		//申请位图文件头结构变量，填写文件头信息
		BITMAPFILEHEADER fileHead;
		fileHead.bfType = 0x4D42;//bmp类型
								 //bfSize是图像文件4个组成部分之和
		fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * _cy;
		fileHead.bfReserved1 = 0;
		fileHead.bfReserved2 = 0;

		//bfOffBits是图像文件前3个部分所需空间之和
		fileHead.bfOffBits = 54 + colorTablesize;

		//写文件头进文件
		fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

		//申请位图信息头结构变量，填写信息头信息
		BITMAPINFOHEADER head;
		head.biBitCount = biBitCount;
		head.biClrImportant = 0;
		head.biClrUsed = 0;
		head.biCompression = 0;
		head.biHeight = _cy;
		head.biPlanes = 1;
		head.biSize = 40;
		head.biSizeImage = lineByte*_cy;
		head.biWidth = _cx;
		head.biXPelsPerMeter = 0;
		head.biYPelsPerMeter = 0;

		//写位图信息头进内存
		fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

		//写位图数据进文件
		fwrite(idx, _cy*lineByte, 1, fp);

		//关闭文件
		fclose(fp);

		delete[] idx;
	}

	void resetRGBnew() {
		if (_rgbsNew) delete[] _rgbsNew;
		_rgbsNew = new float[_cx*_cy * 3];

		for (int i = 0; i <_cx; i++) {
			for (int j = 0; j < _cy; j++) {
				float *p2 = _rgbsNew + (i*_cy + j) * 3;
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

	float *rgbs() { return _rgbs; }

	void saveNewRGB(char *fn) {
		saveAsBmp(_rgbsNew, fn);
	}
	
	void load(char *fname, int numofImg, bool updateHash=true) {
		char rgbFile[512], xyzFile[512];
	   sprintf(rgbFile, "%s%s", fname, ".rgb");
	   sprintf(xyzFile, "%s%s", fname, ".xyz");

	   if (updateHash)
		loadRGB(rgbFile);

	   loadXYZ(xyzFile);
	   
	   resetRGBnew();

	   float *p1 = _xyzs;
	   float *p2 = _rgbs;
	   int num = _cx*_cy;

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

	float *_nearest;

	void resetNearest()
	{
		int sz = _cx*_cy;
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


	void load(char *fname) {
		SceneData::load(fname, -1, false);
		resetNearest();
	}

	void save(char *fname) {
		saveNewRGB(fname);
	}


	void update() {

#pragma omp parallel for schedule(dynamic, 5)
		for (int i = 0; i <_cx; i++) {
			printf("%d of %d done...\n", i, _cx);

			for (int j = 0; j < _cy; j++) {
				float *p1 = _xyzs + (i*_cy + j) * 3;
				float *p2 = _rgbsNew + (i*_cy + j) * 3;
				float *p3 = _nearest + (i*_cy + j);
				float *p4 = _rgbs + (i*_cy + j) * 3;

				int idx, pidx;

				vec3f cr;
				bool ret = getColor(vec3f(p1), p3, cr, idx, pidx);
				if (ret) {
					p2[0] = cr.x;
					p2[1] = cr.y;
					p2[2] = cr.z;
				}
			}
		}

	}
};

SceneData scene[18];
ProgScene target;

int main()
{
	TIMING_BEGIN("Start loading...")
		target.load("all.bmp");
		scene[0].load("0-55.bmp", 0);
	TIMING_END("Loading done...")
	
	TIMING_BEGIN("Start build_bvh...")
		build_bvh();
	TIMING_END("build_bvh done...")

	TIMING_BEGIN("Start rescanning...")
		target.update();
	TIMING_END("Rescaning done...")

	destroy_bvh();
	target.save("output.bmp");

	return 0;
}