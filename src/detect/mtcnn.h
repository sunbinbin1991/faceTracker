#ifndef __MTCNN_NCNN_H__
#define __MTCNN_NCNN_H__
#include "net.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include "common.h"
using namespace std;


class MTCNN {

public:
	MTCNN();
	void Initialize();
	~MTCNN();

	void SetMinFace(int minSize);
	void detect(ncnn::Mat& img_, std::vector<FaceBox>& finalFaceBox);
	void detectMaxFace(ncnn::Mat& img_, std::vector<FaceBox>& finalFaceBox);
	float rnet(ncnn::Mat& img);
	FaceBox onet(ncnn::Mat& img,int x, int y,int w,int h);
		
	//void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
private:
	void generateFaceBox(ncnn::Mat score, ncnn::Mat location, vector<FaceBox>& boundingBox_, float scale);
	void nmsTwoBoxs(vector<FaceBox> &boundingBox_, vector<FaceBox> &previousBox_, const float overlap_threshold, string modelname = "Union");
	void nms(vector<FaceBox> &boundingBox_, const float overlap_threshold, string modelname = "Union");
	void refine(vector<FaceBox> &vecFaceBox, const int &height, const int &width, bool square);
	void extractMaxFace(vector<FaceBox> &boundingBox_);
	float iou(FaceBox & b1, FaceBox & b2, string modelname = "Union");
	void SmoothFaceBox(std::vector<FaceBox>& winList);
	void PNet(float scale);
	void PNet();
	void RNet();
	void ONet();

	ncnn::Mat img;

	const float nms_threshold[3] = { 0.5f, 0.7f, 0.7f };
	const float mean_vals[3] = { 127.5, 127.5, 127.5 };
	const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
	const int MIN_DET_SIZE = 12;
	std::vector<FaceBox> firstPreviousFaceBox_, secondPreviousFaceBox_, thirdPrevioussFaceBox_;
	std::vector<FaceBox> firstFaceBox_, secondFaceBox_, thirdFaceBox_;
	int img_w, img_h;

private:
	ncnn::Net Pnet, Rnet, Onet;
	const float threshold[3] = { 0.8f, 0.8f, 0.9f };
	int minsize = 48;
	const float pre_facetor = 0.7090f;
	bool smooth = true;//�ȶ���
};


#endif //__MTCNN_NCNN_H__
