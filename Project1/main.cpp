#include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <cuda.h>
// #include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <algorithm>
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

// #define THREAD_PER_BLOCK 1024



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

	void setRgbsNew(float *rgbs){
		memcpy(_rgbsNew, rgbs, _cx*_cy*sizeof(float)*3);
	}

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
		SceneData::load(fname, -1, false);
		resetNearest();
	}

	void save(char* fname) {
		saveNewRGB(fname);
	}


};



void setRedBackground(float *new_target_rgbs, int size){
	for (int i=0; i<size; i++){
		new_target_rgbs[i*3] = 1;
		new_target_rgbs[i*3+1] = 0;
		new_target_rgbs[i*3+2] = 0;
	}
}



class Location{
public:
	float x;
	float y;
	float z; 
	int idx;

	bool operator<(const Location& b) {
		if(this->x<b.x){
			return true;
		}else if(this->x == b.x){
			if(this->y < b.y){
				return true;
			}else if(this->y == b.y){
				return z < b.z;
			}else{
				return false;
			}
		}else{
			return false;
		}
	}

	bool operator==(const Location& b) {
		return this->x == b.x && this->y == b.y && this->z == b.z; 
	}

	bool operator!=(const Location& b) {
		return this->x != b.x || this->y != b.y || this->z != b.z; 
	}

 };



float *reduced_target_xyzs;
vector<Location> Locations;
size_t compressPoints(float *target_xyzs, size_t target_size){
	Location cur_loca;
	for (int i=0; i<target_size; i++){
		cur_loca.x =  target_xyzs[3*i];
		cur_loca.y =  target_xyzs[3*i+1];
		cur_loca.z =  target_xyzs[3*i+2];
		cur_loca.idx = i;
		Locations.push_back(cur_loca);
	}

	sort(Locations.begin(),Locations.end());
	vector<Location> reduced_Locations(Locations);
    reduced_Locations.erase(unique(reduced_Locations.begin(), reduced_Locations.end()), reduced_Locations.end());
	size_t reduced_target_size = reduced_Locations.size();

	reduced_target_xyzs = new float[reduced_target_size*3];

	for (int i=0; i<reduced_target_size; i++){
		reduced_target_xyzs[3*i] = reduced_Locations[i].x;
		reduced_target_xyzs[3*i+1] = reduced_Locations[i].y;
		reduced_target_xyzs[3*i+2] = reduced_Locations[i].z;
	}
	return reduced_target_size;
}


__global__ void findNearestNeibor(float *target_xyzs_dev, float *scene_xyzs_dev, int scene_size, 
								  float *min_distances_dev, int *min_neibor_idxs_dev)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float min_distance = 1e6, cur_distance;
	
	float target_x = target_xyzs_dev[index*3];
	float target_y = target_xyzs_dev[index*3+1];
	float target_z = target_xyzs_dev[index*3+2];

	int min_neibor_idx = -1;

	for (int i=0; i<scene_size; i++){
		float scene_x = scene_xyzs_dev[i*3];
		float scene_y = scene_xyzs_dev[i*3+1];
		float scene_z = scene_xyzs_dev[i*3+2];
		cur_distance = (target_x-scene_x)*(target_x-scene_x)+(target_y-scene_y)*(target_y-scene_y)+(target_z-scene_z)*(target_z-scene_z);
		if (cur_distance < min_distance){
			min_neibor_idx = i;
			min_distance = cur_distance;
		}
	}

	min_distances_dev[index] = min_distance;
	min_neibor_idxs_dev[index] = min_neibor_idx;
								  
}


__global__ void findNearestNeibor2(float *target_xyzs_dev, float *scene_xyzs_dev, int scene_size, 
								  float *min_distances_dev, int *min_neibor_idxs_dev)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int index = bx * blockDim.x + tx;
	int Element_Per_Thread = 4;
	int shared_xyzs_size = 512*Element_Per_Thread;
	__shared__ float shared_scene_xyzs[4*512*3];
	// int loop_total = scene_size / 512;

	min_distances_dev[index] = 1e6;
	min_neibor_idxs_dev[index] = -1;

	float target_x = target_xyzs_dev[index*3];
	float target_y = target_xyzs_dev[index*3+1];
	float target_z = target_xyzs_dev[index*3+2];
	float scene_x, scene_y, scene_z, cur_distance;

	for (int count = 0 ; count < scene_size / shared_xyzs_size; count++){
		for (int e = 0; e < Element_Per_Thread; e++){
			shared_scene_xyzs[3*(tx+e)] = scene_xyzs_dev[(count*shared_xyzs_size+tx+e)*3];
			shared_scene_xyzs[3*(tx+e)+1] = scene_xyzs_dev[(count*shared_xyzs_size+tx+e)*3+1];
			shared_scene_xyzs[3*(tx+e)+2] = scene_xyzs_dev[(count*shared_xyzs_size+tx+e)*3+2];
		}

		__syncthreads();
		float cur_min_distances = min_distances_dev[index];
		int cur_min_index = min_neibor_idxs_dev[index];
		for (int i = 0; i < shared_xyzs_size; i++){
			scene_x = shared_scene_xyzs[i*3];
			scene_y = shared_scene_xyzs[i*3+1];
			scene_z = shared_scene_xyzs[i*3+2];
			cur_distance = (target_x-scene_x)*(target_x-scene_x)+(target_y-scene_y)*(target_y-scene_y)+(target_z-scene_z)*(target_z-scene_z);
			if (cur_distance < cur_min_distances){
				cur_min_distances = cur_distance;
				cur_min_index = count*shared_xyzs_size+i;
			}
		}
		min_distances_dev[index] = cur_min_distances;
		min_neibor_idxs_dev[index] = cur_min_index;
		__syncthreads();
	}

								  
}



SceneData scene[18];
ProgScene target;
int main()
{
	// cudaDeviceProp properties;
	// cudaGetDeviceProperties(&properties, 0);
	// printf("sharedmemo  per mp is \t%ld\n", properties.sharedMemPerBlock);
	// printf("Max thread per block is \t%ld\n", properties.maxThreadsPerBlock);
	// printf("Max threads dimemsion is \t%ld;%ld;%ld\n", properties.maxThreadsDim[0],properties.maxThreadsDim[1], properties.maxThreadsDim[1]);
	
	
	TIMING_BEGIN("Start loading...")
	target.load("all.bmp");
	scene[0].load("0-55.bmp", 0);
	// printf("size2: %d\n",gHashTab.size());
	TIMING_END("Loading done...")

	bool compressData = false;
	float *target_xyzs = target.xyzs();
	size_t target_size = target.width() * target.height();

	size_t calculated_target_size, reduced_target_size;
	if (compressData){
		reduced_target_size = compressPoints(target_xyzs, target_size);
		printf("reduced size: %d, target_size: %d\n", reduced_target_size, target_size);
		calculated_target_size = reduced_target_size;
	}else{
		calculated_target_size = target_size;
	}


	float *scene_xyzs = scene[0].xyzs();
	size_t scene_size = scene[0].width() * scene[0].height();
	int* min_neibor_idxs = (int*)malloc(calculated_target_size*sizeof(int));
	float* min_distances = (float*)malloc(calculated_target_size*sizeof(float));

	/* prepare device memory */
	float *target_xyzs_dev, *scene_xyzs_dev, *min_distances_dev;
	int* min_neibor_idxs_dev;
	TIMING_BEGIN("Preparing data for cuda...")
	cudaMalloc((void**)&target_xyzs_dev, calculated_target_size*3*sizeof(float));
	cudaMalloc((void**)&scene_xyzs_dev, scene_size*3*sizeof(float));
	cudaMalloc((void**)&min_distances_dev, calculated_target_size*sizeof(float));
	cudaMalloc((void**)&min_neibor_idxs_dev, calculated_target_size*sizeof(int));
	if(compressData){
		cudaMemcpy(target_xyzs_dev, reduced_target_xyzs, calculated_target_size*3*sizeof(float), cudaMemcpyHostToDevice);
	}else{
		cudaMemcpy(target_xyzs_dev, target_xyzs, calculated_target_size*3*sizeof(float), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(scene_xyzs_dev, scene_xyzs, scene_size*3*sizeof(float), cudaMemcpyHostToDevice);
	TIMING_END("Prepare data for cuda done...")


	dim3 block_size(512);
	dim3 grid_size(calculated_target_size/512);
	/* apply kernel function */ 
	TIMING_BEGIN("Running kernel function ...")
	findNearestNeibor<<<grid_size, block_size>>>(target_xyzs_dev, scene_xyzs_dev, scene_size, min_distances_dev, min_neibor_idxs_dev);
	cudaDeviceSynchronize();
	TIMING_END("Kernel function done...")


	TIMING_BEGIN("Getting data from GPU...")
	cudaMemcpy(min_distances, min_distances_dev, calculated_target_size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(min_neibor_idxs, min_neibor_idxs_dev, calculated_target_size*sizeof(int), cudaMemcpyDeviceToHost);
	TIMING_END("Get data from GPU done...")


	float *new_target_rgbs = (float *)malloc(target_size*3*sizeof(float));
	float *scene_rgbs = scene[0].rgbs();
	float bound = 1.5*1.5;
	TIMING_BEGIN("Setting new rgbs ...")
	setRedBackground(new_target_rgbs, target_size);
	if(compressData){
		int count = 0;
		if(min_distances[count] < bound){
			new_target_rgbs[Locations[0].idx*3] = scene_rgbs[min_neibor_idxs[count]*3];
			new_target_rgbs[Locations[0].idx*3+1] = scene_rgbs[min_neibor_idxs[count]*3+1];
			new_target_rgbs[Locations[0].idx*3+2] = scene_rgbs[min_neibor_idxs[count]*3+2];
		}
		for (int i=1; i<target_size; i++){
			if(Locations[i] != Locations[i-1]){
				count++;
			}
			if(min_distances[count] < bound){
				new_target_rgbs[Locations[i].idx*3] = scene_rgbs[min_neibor_idxs[count]*3];
				new_target_rgbs[Locations[i].idx*3+1] = scene_rgbs[min_neibor_idxs[count]*3+1];
				new_target_rgbs[Locations[i].idx*3+2] = scene_rgbs[min_neibor_idxs[count]*3+2];
			}
		}
	}else{
		for (int i=0; i<target_size; i++){
			if(min_distances[i]<bound){
				int min_neibor_idx = min_neibor_idxs[i];
				// printf("i: %d, idx: %d\n", i, min_neibor_idx);
				if(min_neibor_idx>0){
					new_target_rgbs[i*3] = scene_rgbs[min_neibor_idx*3];
					new_target_rgbs[i*3+1] = scene_rgbs[min_neibor_idx*3+1];
					new_target_rgbs[i*3+2] = scene_rgbs[min_neibor_idx*3+2];
				}
			}
		}
	}

	TIMING_END("Set new rgbs done...")

	cudaFree(target_xyzs_dev);
	cudaFree(scene_xyzs_dev);
	cudaFree(min_distances_dev);
	cudaFree(min_neibor_idxs);

	// printf("try to save\n");
	TIMING_BEGIN("Writing output file ...")
	target.setRgbsNew(new_target_rgbs);
	target.save("share.bmp");
	TIMING_END("Write output file done ...")
	// printf("after save\n");
	free(min_neibor_idxs);
	free(min_distances);
	free(new_target_rgbs);
	if(compressData){
		delete[] reduced_target_xyzs;
		// delete[] indexs;
	}
	// printf("last line\n");

	return 0;
}