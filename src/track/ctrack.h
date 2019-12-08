#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "opencv2/opencv.hpp"
#include "kalman.h"
#include "common.h"

class CTrack
{
public:
	CTrack(const CRegion& region,
		tracking::KalmanType kalmanType,
		track_t deltaTime,
		track_t accelNoiseMag,
		size_t trackID,
		bool filterObjectSize,
		tracking::LostTrackType externalTrackerForLost);
	
	///
	/// \brief CalcDist
	/// Euclidean distance in pixels between objects centres on two N and N+1 frames
	/// \param reg
	/// \return
	///
	track_t CalcDistCenter(const CRegion& reg) const;
	///
	/// \brief CalcDist
	/// Euclidean distance in pixels between object contours on two N and N+1 frames
	/// \param reg
	/// \return
	///
	track_t CalcDistRect(const CRegion& reg) const;
	///
	/// \brief CalcDistJaccard
	/// Jaccard distance from 0 to 1 between object bounding rectangles on two N and N+1 frames
	/// \param reg
	/// \return
	///
	track_t CalcDistJaccard(const CRegion& reg) const;
	///
	/// \brief CalcDistJaccard
	/// Distance from 0 to 1 between objects histogramms on two N and N+1 frames
	/// \param reg
	/// \param currFrame
	/// \return
	///
	track_t CalcDistHist(const CRegion& reg, cv::UMat currFrame) const;
	///
	/// \brief CalcDistHOG
	/// Euclidean distance from 0 to 1 between HOG descriptors on two N and N+1 frames
	/// \param reg
	/// \return
	///
	track_t CalcDistHOG(const CRegion& reg) const;

	bool CheckType(const std::string& type) const;

	void Update(const CRegion& region, bool dataCorrect, size_t max_trace_length, int width, int height);

	bool IsStatic() const;
	bool IsStaticTimeout(int framesTime) const;
	bool IsOutOfTheFrame() const;

	cv::RotatedRect GetLastRect() const;

	const CRegion& LastRegion() const;
	size_t SkippedFrames() const;
	size_t& SkippedFrames();

	//TrackingObject ConstructObject() const;

private:
	//Trace m_trace;
	size_t m_trackID = 0;
	size_t m_skippedFrames = 0;
	CRegion m_lastRegion;

	Point_t m_predictionPoint;
	cv::RotatedRect m_predictionRect;
	std::unique_ptr<TkalmanFilter> m_kalman;
	bool m_filterObjectSize = false;
	bool m_outOfTheFrame = false;

	tracking::LostTrackType m_externalTrackerForLost;
#ifdef USE_OCV_KCF
	cv::Ptr<cv::Tracker> m_tracker;
#endif
	//std::unique_ptr<VOTTracker> m_VOTTracker;

	//void RectUpdate(const CRegion& region, bool dataCorrect, cv::UMat prevFrame, cv::UMat currFrame);
	void RectUpdate(const CRegion& region, bool dataCorrect, int width, int height);

	//void CreateExternalTracker(int channels);

	//void PointUpdate(const Point_t& pt, const cv::Size& newObjSize, bool dataCorrect, const cv::Size& frameSize);

	bool CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region);
	bool m_isStatic = false;
	int m_staticFrames = 0;
	cv::UMat m_staticFrame;
	cv::Rect m_staticRect;
};

typedef std::vector<std::unique_ptr<CTrack>> CTracks_t;
