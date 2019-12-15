#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "opencv2/opencv.hpp"
#include "concurrentqueue.h"

#include "common.h"
#include "../detect/mtcnn.h"
#include "../landmark/landmark.h"
#include "ctracker.h"

typedef struct FrameInfo
{
	cv::Mat m_frame;
	FaceTrack m_trackInfo;

	std::condition_variable m_cond;
	std::mutex m_mutex;
	bool m_captured = false;
}FrameInfo;

using namespace moodycamel;

class tracker
{
public:
	tracker();
    ~tracker();
	
	void TrackingSyncProcess(const cv::Mat& frame, regions_t& regs);

	void TrackingAsyncProcess(const cv::Mat& frame, regions_t& regs);

private :
	//detect related init	
	void InitDetector();
	
	void Detecting(const cv::Mat& frame, regions_t& regs);

	void Detecting(const cv::Mat& frame, std::vector<FaceBox>& regs);

	void MatchingByIOU(const cv::Mat& frame, const regions_t& regs);
	
	void MatchingByHungarian(const cv::Mat& frame, const regions_t& regs);

	void DetectThreading();

	void PredictKptsByOptflow(const cv::Mat & frame,const std::vector<FaceBox>& prev_box1, std::vector<FaceBox>& predict_box);

	FrameInfo m_frameInfo[2];
	//draw related

	char angletext[256];
	char scoretext[256];

private:
	std::thread detectionThread;

	bool m_isTrackerInitialized = false;
	bool m_isDetectorInitialized = false;
	
	cv::Mat m_prev_gray;
	cv::Mat m_curr_frame;
	ncnn::Mat m_detect_buffer;

	std::unique_ptr<MTCNN> m_detector;
	std::unique_ptr<landmark> m_landmark;
	std::unique_ptr<CTracker> c_tracker;
	size_t m_trackID = 0;
	regions_t m_tracks;

	std::atomic<bool> m_flag = ATOMIC_VAR_INIT(false);
	std::atomic<bool> m_det_ready =  ATOMIC_VAR_INIT(false);
	int thread_num = 1;
	bool isFirstTimeRun = false;

	std::unique_ptr <ConcurrentQueue<cv::Mat>> queue_images;
	size_t max_queue_num = 10;

	std::vector<FaceBox> buffer_dets;

	


// tracking strategy 0 : KCF
	
// tracking match strategy
	float m_distThres = 0.5;
	size_t m_maximumAllowedSkippedFrames = 25;

};


