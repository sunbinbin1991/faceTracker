#pragma once
#include "common.h"

class CTracker
{
public:
	CTracker();
	~CTracker();

	float CalculateIOU(const FaceBox& fb1, const FaceBox& fb2);
	void CalculateIOUs(FaceBox box1, const regions_t& tracks, std::pair<int, float>& ious);
	void Landmark2Box(const std::vector<FaceBox>& boxes1, std::vector<FaceBox>& boxes2);
	void Landmark2BoxWithFlag(const std::vector<FaceBox>& boxes1, const std::vector<unsigned char>& status, std::vector<FaceBox>& boxes2);

	void AddNewTracks(FaceTrack& faceTrack,regions_t& tracks);
	void UpdateTracks(FaceTrack& faceTrack, FaceTrack& tracks);
	void DeleteLostTracks(regions_t& tracks);

};