#include "track.h"
#include "../example/utils.hpp"
track::track(/* args */)
{
	bool flagtemp = m_flag.load();
	printf("%d\n", flagtemp);
	m_detector = std::unique_ptr<MTCNN>(new MTCNN());
}

track::~track() = default;

void track::init_detector() {
	m_detector->Initialize("D:/git/track/faceTracker/src/detect/models/ncnn");	
	//m_detector->Initialize("../../src/detect/models/ncnn");
	//../../src/detect/models/ncnn
}

void track::tracking(const cv::Mat& frame) {
	m_curr_frame = frame.clone();
	if (m_curr_frame.empty()) {
		printf("The End\n");
		return;
	}
	std::vector<FaceBox> faces;
	int64 ticbegin = cv::getTickCount();
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels
							(m_curr_frame.data, ncnn::Mat::PIXEL_BGR2RGB, m_curr_frame.cols, m_curr_frame.rows);
	m_detector->detect(ncnn_img, faces);
	drawFaces(m_curr_frame, faces);
	int64 ticend = cv::getTickCount();
	printf("detection time used %f\n", (ticend - ticbegin) / cv::getTickFrequency());
	//cv::imwrite("../../data/IU_mark.jpg", frame);
	cv::imshow("image", m_curr_frame);
	cv::waitKey(0);
};
