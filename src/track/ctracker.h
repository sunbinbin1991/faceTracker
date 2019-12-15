#pragma once
#include "common.h"
#include "HungarianAlg.h"

template<class T> inline
T square(T val)
{
	return val * val;
}

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
	void DeleteLostTracks2(regions_t& tracks);

	void CreateDistaceMatrix(const regions_t& curr_dets, const regions_t& pre_trks, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost);

	void SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment);
	
	//void UpdateTrackingState(const regions_t& regions, float fps) {};
private:

	track_t CalcDistCenter(FaceBox& pre_box, FaceBox& curr_box);
	
	float getIou(const FaceBox& curr_fb, const FaceBox& prev_fb);

	track_t CalcDistRect(FaceBox& pre_box, FaceBox& curr_box);
	

};