#pragma once
#include "common.h"

class CTracker
{
public:
	CTracker();
	~CTracker();

	float CalculateIOU(const FaceBox& fb1, const FaceBox& fb2);
	void CalculateIOUs(FaceBox box1, const regions_t& tracks, std::pair<int, float>& ious);

	void AddNewTracks(FaceTrack& faceTrack,regions_t& tracks);
	void UpdateTracks(FaceTrack& faceTrack, FaceTrack& tracks);
	void DeleteLostTracks(regions_t& tracks);

};