#include <thread>

#include "track.h"
#include "../example/utils.hpp"

//构造函数中为什么不能初始化线程
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

void tracker::Tracking(const cv::Mat& frame, const regions_t& regs) {
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

void tracker::TrackingSyncProcess(const cv::Mat& frame, regions_t& regs) {
	
	regions_t curr_detections;
	
	Detecting(frame, curr_detections);
	
	Tracking(frame, curr_detections);

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
		printf("enqueue size of queue %d\n", queue_images->size_approx());
		queue_images->enqueue(cloned);
	}

	std::vector<FaceBox> curr_dets;
	if (m_det_ready) {
		curr_dets = buffer_dets;
		m_landmark->get_landmark(cloned, curr_dets);
		m_det_ready = false;
	}
	else {
		if (!m_tracks.empty()) {
			std::vector<FaceBox> prev_dets;
			prev_dets.resize(m_tracks.size());
			for (size_t i = 0; i < m_tracks.size(); i++)
			{
				prev_dets[i] = m_tracks[i].bbox_;
			}
			m_landmark->get_landmark(cloned, prev_dets);
			c_tracker->Landmark2Box(prev_dets, curr_dets);
			printf("hell0");
		} else{
			return;
		}	
	}

	regions_t curr_tracks;
	curr_tracks.resize(curr_dets.size());
	for (size_t i = 0; i < curr_dets.size(); i++)
	{
		curr_tracks[i].bbox_ = curr_dets[i];
	}
	if (!curr_tracks.empty()) {
		Tracking(cloned, curr_tracks);
	}

	regs.assign(m_tracks.begin(), m_tracks.end());
}

void tracker::DetectThreading() {
	while (m_flag) {
		cv::Mat img(480,640,CV_8UC3);
		printf("before size of queue %d,m_flag =%d\n", queue_images->size_approx(), m_flag);
		if (queue_images->try_dequeue(img)) {
			std::vector<FaceBox> temp_dets;
			Detecting(img, temp_dets);
			if (!temp_dets.empty()) {
				buffer_dets = temp_dets;
				m_det_ready = true;
			}
			else {
				buffer_dets.clear();
				m_det_ready = false;
			}
			printf("try_dequeue size of queue %d\n", queue_images->size_approx());
		}
	}
}