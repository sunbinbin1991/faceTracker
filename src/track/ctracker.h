#pragma once
#include "common.h"
#include "HungarianAlg.h"
#include "ctrack.h"
#include "track.h"


struct TrackerSettings
{
	//tracking::DistType m_distType = tracking::DistCenters;
	tracking::KalmanType m_kalmanType = tracking::KalmanLinear;
	tracking::FilterGoal m_filterGoal = tracking::FilterCenter;
	tracking::LostTrackType m_lostTrackType = tracking::TrackKCF;
	tracking::MatchType m_matchType = tracking::MatchHungrian;

	std::array<track_t, tracking::DistsCount> m_distType;

	///
	/// \brief m_dt
	/// Time step for Kalman
	///
	track_t m_dt = 1.0f;

	///
	/// \brief m_accelNoiseMag
	/// Noise magnitude for Kalman
	///
	track_t m_accelNoiseMag = 0.1f;

	///
	/// \brief m_distThres
	/// Distance threshold for Assignment problem for tracking::DistCenters or for tracking::DistRects (for tracking::DistJaccard it need from 0 to 1)
	///
	track_t m_distThres = 0.5f;

	///
	/// \brief m_maximumAllowedSkippedFrames
	/// If the object don't assignment more than this frames then it will be removed
	///
	size_t m_maximumAllowedSkippedFrames = 25;

	///
	/// \brief m_maxTraceLength
	/// The maximum trajectory length
	///
	size_t m_maxTraceLength = 50;

	///
	/// \brief m_useAbandonedDetection
	/// Detection abandoned objects
	///
	bool m_useAbandonedDetection = false;
	///
	/// \brief m_minStaticTime
	/// After this time (in seconds) the object is considered abandoned
	///
	int m_minStaticTime = 5;
	///
	/// \brief m_maxStaticTime
	/// After this time (in seconds) the abandoned object will be removed
	///
	int m_maxStaticTime = 25;

	///
	TrackerSettings()
	{
		m_distType[tracking::DistCenters] = 0.0f;
		m_distType[tracking::DistRects] = 0.0f;
		m_distType[tracking::DistJaccard] = 0.5f;
		m_distType[tracking::DistHist] = 0.5f;
		m_distType[tracking::DistHOG] = 0.0f;

	}
};

class CTracker
{
public:
	CTracker();
	CTracker(const CTracker&) = delete;
	CTracker(CTracker&&) = delete;
	CTracker& operator=(const CTracker&) = delete;//Avoiding implicit copy assignment.
	CTracker& operator=(CTracker&&) = delete;

	~CTracker();

	float CalculateIOU(const FaceBox& fb1, const FaceBox& fb2);
	void CalculateIOUs(FaceBox box1, const regions_t& tracks, std::pair<int, float>& ious);
	void Landmark2Box(const std::vector<FaceBox>& boxes1, std::vector<FaceBox>& boxes2);

	void AddNewTracks(FaceTrack& faceTrack,regions_t& tracks);
	void UpdateTracks(FaceTrack& faceTrack, FaceTrack& tracks);
	void DeleteLostTracks(regions_t& tracks);

	void UpdateTrackingState(const cregions_t& regions, int width, int height, int fps);
	void CreateDistaceMatrix(const cregions_t& regions, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost);
	void SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment);
	
private:
	TrackerSettings m_settings;

	size_t m_trackID = 0;
	//regions_t m_tracks;
	CTracks_t c_tracks;

	size_t m_nextTrackID;
};