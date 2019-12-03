#include "iostream"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include "../src/detect/mtcnn.h"
#include "../src/landmark/landmark.h"
#include "../src/track/track.h"
#include "utils.hpp"
#include "concurrentqueue.h"

using namespace moodycamel;

void test_video_cameral() {

	//std::unique_ptr<tracker> tk = std::unique_ptr<tracker>(new tracker());
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
		//  tracking with detect
		//tk.TrackingSyncProcess(frame, tracks);
		
		// tracking with new thread detect
		int64 t1 = cv::getTickCount();

		tk.TrackingAsyncProcess(frame, tracks);
		
		int64 t2 = cv::getTickCount();
		printf("total %gms\n", (t2 - t1) * 1000 / cv::getTickFrequency());
		DrawTracks(frame, tracks);
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

void test_concurrentqueue() {
	ConcurrentQueue<int> q;

	for (int i = 0; i != 123; ++i)
		q.enqueue(i);

	int item;
	for (int i = 0; i != 123; ++i) {
		q.try_dequeue(item);
		assert(item == i);
	}

}

int main(){
	test_video_cameral();
	//test_image();

	//test_concurrentqueue();
	printf("hello world\n");
}