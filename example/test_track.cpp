#include "iostream"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include "../src/detect/mtcnn.h"
#include "../src/landmark/landmark.h"
#include "../src/track/track.h"
#include "utils.hpp"

void test_video_cameral() {
	tracker tk;
	cv::Mat frame;
	regions_t tracks;
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
		tk.TrackingSyncProcess(frame, tracks);
		tk.DrawTracks(frame);
		cv::imshow("image", frame);
		char key = cv::waitKey(1);
		if (key == 27) {
			break;
		}
	}
}

void test_image() {
	tracker tk;
	regions_t tracks;
	cv::Mat frame;
	frame = cv::imread("../../data/IU.jpg");
	if (frame.empty()) {
		printf("The End\n");
		return;
	}
	tk.TrackingSyncProcess(frame, tracks);
	std::vector<FaceBox> faces;

}


int main(){
	test_video_cameral();
	//test_image();
	printf("hello world\n");
}