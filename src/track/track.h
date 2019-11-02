#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "opencv2/opencv.hpp"

#include "common.h"
#include "../detect/mtcnn.h"
#include "ctracker.h"

typedef struct FrameInfo
{
	cv::Mat m_frame;
	FaceTrack m_trackInfo;

	std::condition_variable m_cond;
	std::mutex m_mutex;
	bool m_captured = false;
}FrameInfo;

class tracker
{
public:
	tracker();
    ~tracker();
	
	void TrackingSyncProcess(const cv::Mat& frame, regions_t& regs);
	void DrawTracks(cv::Mat& frame);

private :
	//detect related init	
	void InitDetector();
	
	void Detecting(const cv::Mat& frame, regions_t& regs);
	void Tracking(const cv::Mat& frame, const regions_t& regs);
	
	FrameInfo m_frameInfo[2];
	//draw related

	char angletext[256];
	char scoretext[256];

private:
	std::thread detectionThread;

	bool m_isTrackerInitialized = false;
	bool m_isDetectorInitialized = false;
	
	cv::Mat m_curr_frame;
	ncnn::Mat m_detect_buffer;

	std::atomic<bool> m_flag = ATOMIC_VAR_INIT(false);
	std::unique_ptr<MTCNN> m_detector;
	std::unique_ptr<CTracker> m_tracker;
	size_t m_trackID = 0;
	regions_t m_tracks;

// tracking strategy 0 : KCF
	
// tracking match strategy


};


