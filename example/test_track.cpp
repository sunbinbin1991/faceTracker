#include "iostream"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include "../src/detect/mtcnn.h"
#include "../src/landmark/landmark.h"
#include "../src/track/track.h"
#include "utils.hpp"


void test_video_cameral() {
	string model_path = "../../src/detect/models/ncnn";
	MTCNN  mt = MTCNN();
	mt.Initialize(model_path);
	cv::Mat frame;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return;
	}
	std::vector<FaceBox> faces;
	while (cap.isOpened()) {
		cap >> frame;
		if (frame.empty()) {
			printf("The End\n");
			break;
		}
		int64 ticbegin = cv::getTickCount();
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		mt.detect(ncnn_img, faces);

		drawFaces(frame, faces);
		int64 ticend = cv::getTickCount();
		printf("detection time used %f\n", (ticend - ticbegin) / cv::getTickFrequency());
		cv::imshow("image", frame);
		char key = cv::waitKey(1);
		if (key == 27) {
			break;
		}
	}
}

void test_image() {
	track tk;
	tk.init_detector();
	
	cv::Mat frame;
	frame = cv::imread("../../data/IU.jpg");
	if (frame.empty()) {
		printf("The End\n");
		return;
	}
	tk.tracking(frame);
	std::vector<FaceBox> faces;

}

int main(){
	//test_video_cameral();
	test_image();
	printf("hello world\n");
}