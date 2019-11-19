#ifndef __LANDMARK_NCNN_H__
#define __LANDMARK_NCNN_H__
#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <iostream>
#include "common.h"

class landmark
{

public:
    landmark();

	void get_landmark(const cv::Mat& image,std::vector<FaceBox> &faces);

    ~landmark();
private:
	/* data */
	ncnn::Net _landmark_net;
	int _input_size = 112;
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	
};
#endif //__LANDMARK_NCNN_H__
