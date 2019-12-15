#include <thread>

#include "track.h"
#include "../example/utils.hpp"

//构造函数中为什么不能初始化线程?
tracker::tracker()
{
	m_detector = std::unique_ptr<MTCNN>(new MTCNN());
	m_landmark = std::unique_ptr<landmark>(new landmark());
	InitDetector();

	//m_flag = true;

	queue_images = std::make_unique<ConcurrentQueue<cv::Mat>>(max_queue_num);
	printf("\thello world %d\n", m_flag);
	//detectionThread.();
}

tracker::~tracker() {
	m_flag = false;
	detectionThread.join();
};

void tracker::InitDetector() {
	if (!m_isDetectorInitialized) {
		m_detector->Initialize();
		m_isDetectorInitialized = true;
	}
}

void tracker::Detecting(const cv::Mat& frame, regions_t& regs) {
	m_curr_frame = frame.clone();
	if (m_curr_frame.empty()) {
		printf("The End\n");
		return;
	}
	if (m_isDetectorInitialized) {
		InitDetector();
	}
	std::vector<FaceBox> faces;
	int64 ticbegin = cv::getTickCount();
	m_detect_buffer = ncnn::Mat::from_pixels(m_curr_frame.data, ncnn::Mat::PIXEL_BGR2RGB, m_curr_frame.cols, m_curr_frame.rows);
	m_detector->detect(m_detect_buffer, faces);
	regs.resize(faces.size());
	for (size_t i = 0; i < faces.size(); i++)
	{
		regs[i].bbox_ = faces[i];
	}
	int64 ticend = cv::getTickCount();
	//printf("detection time used %f\n", (ticend - ticbegin) / cv::getTickFrequency());
};

void tracker::Detecting(const cv::Mat& frame, std::vector<FaceBox>& regs) {
	m_curr_frame = frame.clone();
	if (m_curr_frame.empty()) {
		printf("The End\n");
		return;
	}
	if (m_isDetectorInitialized) {
		InitDetector();
	}
	std::vector<FaceBox> faces;
	int64 ticbegin = cv::getTickCount();
	m_detect_buffer = ncnn::Mat::from_pixels(m_curr_frame.data, ncnn::Mat::PIXEL_BGR2RGB, m_curr_frame.cols, m_curr_frame.rows);
	m_detector->detect(m_detect_buffer, regs);
	int64 ticend = cv::getTickCount();
	//printf("detection time used %f\n", (ticend - ticbegin) / cv::getTickFrequency());
};

//simple tracking matching two rects by iou
void tracker::MatchingByIOU(const cv::Mat& frame, const regions_t& regs) {
	if (m_tracks.empty()) {
		for (const auto& reg : regs)
		{
			FaceTrack tempTrk = reg;
			tempTrk.id_ = m_trackID++;;
			m_tracks.push_back(tempTrk);
		}
	}
	else {
		if (!regs.empty()) {
			for (auto& reg:regs)
			{
				FaceTrack tempTrk = reg;
				std::pair<int, float> ious(0, 0.);
				c_tracker->CalculateIOUs(tempTrk.bbox_, m_tracks, ious);
				if (ious.second < 0.2) {					
					//iou<0.5 add new track
					tempTrk.id_ = m_trackID++; 
					c_tracker->AddNewTracks(tempTrk, m_tracks);
				}
				else {	
					//iou>0.5 merge with previous track
					c_tracker->UpdateTracks(tempTrk, m_tracks[ious.first]);
					m_tracks[ious.first].existsTimes_++;
				}
			}
		}
		
		//every trk age++
		for (auto &trk : m_tracks)
		{
			trk.age_++;
		}
		c_tracker->DeleteLostTracks(m_tracks);

		for (auto &trk : m_tracks)
		{
			//printf("age = %d time = %d", trk.age_, trk.existsTimes_);
		}
	}
}

void tracker::MatchingByHungarian(const cv::Mat& frame, const regions_t& regs) {

}

void tracker::TrackingSyncProcess(const cv::Mat& frame, regions_t& regs) {
	
	regions_t curr_detections;
	
	Detecting(frame, curr_detections);
	
	MatchingByIOU(frame, curr_detections);

	regs.assign(std::begin(m_tracks), std::end(m_tracks));
}

void tracker::TrackingAsyncProcess(const cv::Mat& frame, regions_t& regs) {
	if (!isFirstTimeRun) {
		m_flag = true;
		isFirstTimeRun = true;
		detectionThread = std::thread(&tracker::DetectThreading, this);
		printf("new thread %d\n", m_flag);
	}
	regions_t trackers;
	cv::Mat cloned = frame.clone();

	if (queue_images->size_approx() < 1) {
		//printf("enqueue size of queue %d\n", queue_images->size_approx());
		queue_images->enqueue(cloned);
	}

	std::vector<FaceBox> curr_dets;
	bool noNeedGetLandmark = false;
	if (m_det_ready) {
		std::vector<FaceBox> temp_dets = buffer_dets;
		m_landmark->get_landmark(cloned, temp_dets);
		c_tracker->Landmark2Box(temp_dets, curr_dets);
		m_det_ready = false;
		noNeedGetLandmark = true;
	}
	else {
		if (!m_tracks.empty()) {
			std::vector<FaceBox> prev_dets;
			std::vector<FaceBox> predcit_dets;
			prev_dets.resize(m_tracks.size());
			for (size_t i = 0; i < m_tracks.size(); i++)
			{
				prev_dets[i] = m_tracks[i].bbox_;
			}
			//1: use history trks to get origin kpts
			if (noNeedGetLandmark) {
				prev_dets = curr_dets;
			}
			else {
				m_landmark->get_landmark(cloned, prev_dets);
			}

			//2: use history kpts to get new kpts
			PredictKptsByOptflow(cloned, prev_dets, predcit_dets);

			c_tracker->Landmark2Box(predcit_dets, curr_dets);
		}
	}
	
	regions_t curr_tracks;
	curr_tracks.resize(curr_dets.size());
	for (size_t i = 0; i < curr_dets.size(); i++)
	{
		curr_tracks[i].bbox_ = curr_dets[i];
	}

	MatchingByIOU(cloned, curr_tracks);

	regs.assign(m_tracks.begin(), m_tracks.end());
}

void tracker::DetectThreading() {
	cv::Mat img(480, 640, CV_8UC3);
	while (m_flag) {	
		//printf("before size of queue %d,m_flag =%d\n", queue_images->size_approx(), m_flag);
		if (queue_images->try_dequeue(img)) {
			std::vector<FaceBox> temp_dets;
			Detecting(img, temp_dets);

			if (!temp_dets.empty()) {
				buffer_dets = temp_dets;
				m_det_ready = true;
				cv::cvtColor(img, m_prev_gray, cv::COLOR_BGR2GRAY);
				//m_flag = false;
			}
			else {
				buffer_dets.clear();
				m_det_ready = true;
			}
			//printf("try_dequeue size of queue %d\n", queue_images->size_approx());
		}
	}
}

void tracker::PredictKptsByOptflow(const cv::Mat & frame, const std::vector<FaceBox>& prev_boxes, std::vector<FaceBox>& opt_predict_box) {
	if (m_prev_gray.empty()) {
		cv::cvtColor(frame, m_prev_gray, cv::COLOR_BGR2GRAY);
	}
	cv::Mat curr_frame_gray;
	cv::cvtColor(frame, curr_frame_gray, cv::COLOR_BGR2GRAY);

	std::vector<cv::Point2f> prev_points;
	std::vector<cv::Point2f> predict_points;
	
	for (size_t i = 0; i < prev_boxes.size(); i++)
	{
		FaceBox temp = prev_boxes[i];

		for (size_t j = 0; j < temp.numpts; j++)
		{
			cv::Point2f tmpPoint;
			tmpPoint.x = temp.ppoint[2 * j];
			tmpPoint.y = temp.ppoint[2 * j+1];
			prev_points.push_back(tmpPoint);
		}
	}
	predict_points.reserve(prev_points.size());
	//cv::imshow("prev", m_prev_gray);
	//cv::imshow("curr", curr_frame_gray);
	std::vector<uchar> status;
	std::vector<float> err;
	int windowsize = 21;
	//cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	cv::TermCriteria termcrit(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 0, 0.001);
	cv::calcOpticalFlowPyrLK(m_prev_gray, curr_frame_gray, prev_points, predict_points, status, err,
		cv::Size(windowsize, windowsize),3,termcrit);
	
	//printf("res %d %d \n", status.size(), err.size());

	//get predict pts
	std::vector<int> delete_idx;
	for (size_t i = 0; i < prev_boxes.size(); i++)
	{
		FaceBox temp;
		int numpts_num = prev_boxes[i].numpts;
		temp.numpts = numpts_num;
		opt_predict_box.push_back(temp);
		int lose_count = 0;
		int lost_cencert_x = (prev_boxes[i].x2 + prev_boxes[i].x1) >> 1;
		int lost_cencert_y = (prev_boxes[i].y2 + prev_boxes[i].y1) >> 1;

		for (size_t j = 0; j < numpts_num; j++)
		{
			if (status[i*numpts_num + j]) {
				opt_predict_box[i].ppoint[2 * j] = predict_points[i*numpts_num + j].x;
				opt_predict_box[i].ppoint[2 * j + 1] = predict_points[i*numpts_num + j].y;
			}
			else {
				lose_count += 1;
				opt_predict_box[i].ppoint[2 * j] = lost_cencert_x;
				opt_predict_box[i].ppoint[2 * j+1] = lost_cencert_y;
			}
		}
		if (lose_count > 20) {
			//lost almost 20 pts, the pts is not useful
			delete_idx.push_back(i);
		}
		//printf("err_count %d \n", lose_count);
	}

	//if lost too much kpts then lose it
	if (delete_idx.empty())
		return;
	std::sort(delete_idx.begin(), delete_idx.end());
	// delete elements from high to low
	for (int i = delete_idx.size() - 1; i >= 0; i--) {
		opt_predict_box.erase(opt_predict_box.begin() + delete_idx[i]);
	}
};