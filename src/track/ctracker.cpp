#include "ctracker.h"
#include "iostream"
#include "algorithm"

CTracker::CTracker() {};
CTracker::~CTracker() {};

float CTracker::CalculateIOU(const FaceBox& curr_fb, const FaceBox& prev_fb) {
	float box_area = (curr_fb.x2 - curr_fb.x1 + 1) *(curr_fb.y2 - curr_fb.y1 + 1);
	float area = (prev_fb.x2 - prev_fb.x1 + 1) *(prev_fb.y2 - prev_fb.y1 + 1);
	float xx1 = fmax(prev_fb.x1, curr_fb.x1);
	float xx2 = fmin(prev_fb.x2, curr_fb.x2);
	float yy1 = fmax(prev_fb.y1, curr_fb.y1);
	float yy2 = fmin(prev_fb.y2, curr_fb.y2);

	float w = fmax(0, xx2 - xx1 + 1);
	float h = fmax(0, yy2 - yy1 + 1);
	float inter = w *h;
	return inter / (box_area + area - inter);
};

void CTracker::CalculateIOUs(FaceBox box1, const regions_t& tracks, std::pair<int, float>& ious) {
	float tmpIOU = 0;
	for (int i = 0; i< tracks.size(); i++) {
		tmpIOU = CalculateIOU(box1, tracks[i].bbox_);
		if (ious.second < tmpIOU) {
			ious.second = tmpIOU;
			ious.first = i;
		}
	}
}

void CTracker::Landmark2Box(const std::vector<FaceBox>& boxes1, std::vector<FaceBox> &boxes2) {
	if (boxes1.empty()) {
		return;
	}
	boxes2.resize(boxes1.size());
	for (size_t j = 0; j < boxes1.size(); j++)
	{
		FaceBox temp;
		boxes2[j] = boxes1[j];
		boxes2[j].x1 = 1e5;
		boxes2[j].y1 = 1e5;
		boxes2[j].x2 = -1e5;
		boxes2[j].y2 = -1e5;
		for (size_t i = 0; i <boxes1[j].numpts; i++)
		{
			boxes2[j].x1 = std::min((int)(boxes1[j].ppoint[2 * i]), boxes2[j].x1);
			boxes2[j].y1 = std::min((int)(boxes1[j].ppoint[2 * i + 1]), boxes2[j].y1);
			boxes2[j].x2 = std::max((int)(boxes1[j].ppoint[2 * i]), boxes2[j].x2);
			boxes2[j].y2 = std::max((int)(boxes1[j].ppoint[2 * i + 1]), boxes2[j].y2);
		}
	}
	
}

void CTracker::Landmark2BoxWithFlag(const std::vector<FaceBox>& boxes1, const std::vector<unsigned char>& status, std::vector<FaceBox> &boxes2) {
	if (boxes1.empty()) {
		return;
	}
	boxes2.resize(boxes1.size());
	for (size_t j = 0; j < boxes1.size(); j++)
	{
		FaceBox temp;
		boxes2[j] = boxes1[j];
		boxes2[j].x1 = 1e5;
		boxes2[j].y1 = 1e5;
		boxes2[j].x2 = -1e5;
		boxes2[j].y2 = -1e5;
		int numpts_num = boxes1[j].numpts;
		for (size_t i = 0; i <numpts_num; i++)
		{
			if ((unsigned char)status[i*numpts_num + j]) {
				boxes2[j].x1 = std::min((int)(boxes1[j].ppoint[2 * i]), boxes2[j].x1);
				boxes2[j].y1 = std::min((int)(boxes1[j].ppoint[2 * i + 1]), boxes2[j].y1);
				boxes2[j].x2 = std::max((int)(boxes1[j].ppoint[2 * i]), boxes2[j].x2);
				boxes2[j].y2 = std::max((int)(boxes1[j].ppoint[2 * i + 1]), boxes2[j].y2);
			}			
		}
	}

}


void CTracker::AddNewTracks(FaceTrack& faceTrack, regions_t& tracks) {
	faceTrack.existsTimes_++;
	tracks.push_back(faceTrack);
};

void CTracker::UpdateTracks(FaceTrack& currfaceTrack, FaceTrack& prefaceTrack) {
	FaceBox curr_box = currfaceTrack.bbox_;
	FaceBox pre_box = prefaceTrack.bbox_;
	FaceBox newBox = currfaceTrack.bbox_;

	float curr_width = curr_box.x2 - curr_box.x1 + 1;
	float curr_height = curr_box.y2 - curr_box.y1 + 1;
	float pre_width = pre_box.x2 - pre_box.x1 + 1;
	float pre_height = pre_box.y2 - pre_box.y1 + 1;
	float cent_w = (curr_width + pre_width) / 2;
	float cent_h = (curr_height + pre_height) / 2;

	float curr_center_x0 = curr_box.x1 + curr_width / 2;
	float curr_center_y0 = curr_box.y1 + curr_height / 2;
	float pre_center_x0 =  pre_box.x1 + pre_width / 2;
	float pre_center_y0 =  pre_box.y1 + pre_height / 2;
	float cent_x0 = (pre_center_x0 + curr_center_x0) / 2;
	float cent_y0 = (pre_center_y0 + curr_center_y0) / 2;

	newBox.x1 = cent_x0 - cent_w/2;
	newBox.x2 = cent_x0 + cent_w/2;
	newBox.y1 = cent_y0 - cent_h/2;
	newBox.y2 = cent_y0 + cent_h/2;

	prefaceTrack.bbox_ = newBox;

};

void CTracker::DeleteLostTracks(regions_t& tracks) {
	regions_t removed;
	std::vector<int> delete_idx;
	for (int i = 0; i < tracks.size(); i++) {
		int thr_lost_count = tracks[i].age_ - tracks[i].existsTimes_;
		if (thr_lost_count>5) {
			delete_idx.push_back(i);
		}
	}
	if (delete_idx.empty()) {
		return ;
	}
	std::sort(delete_idx.begin(), delete_idx.end());
	removed.reserve(delete_idx.size());
	for (int i = delete_idx.size() - 1; i >= 0; i--) {
		auto it = std::next(tracks.begin(), delete_idx[i]);
		removed.push_back(std::move(*it));
		tracks.erase(it);
	}
};


void CTracker::DeleteLostTracks2(regions_t& tracks) {
	regions_t removed;
	std::vector<int> delete_idx;
	for (int i = 0; i < tracks.size(); i++) {

		//int thr_lost_count = tracks[i].age_ - tracks[i].existsTimes_;
		int thr_lost_count = tracks[i].lostTimes_;
		if (thr_lost_count>5) {
			delete_idx.push_back(i);
		}
	}
	if (delete_idx.empty()) {
		return;
	}
	std::sort(delete_idx.begin(), delete_idx.end());
	removed.reserve(delete_idx.size());
	for (int i = delete_idx.size() - 1; i >= 0; i--) {
		auto it = std::next(tracks.begin(), delete_idx[i]);
		removed.push_back(std::move(*it));
		tracks.erase(it);
	}
};

void CTracker::CreateDistaceMatrix(const regions_t& curr_dets, const regions_t& pre_trks, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost) {
	const size_t N = pre_trks.size();	// Tracking objects
	maxCost = 0;

	for (size_t i = 0; i < pre_trks.size(); ++i)
	{
		for (size_t j = 0; j < curr_dets.size(); ++j)
		{
			auto dist = maxPossibleCost;
			dist = 0;
			//size_t ind = 0;
			FaceBox pre_box = pre_trks[i].bbox_;
			FaceBox curr_box = curr_dets[j].bbox_;
			dist = CalcDistCenter(pre_box, curr_box)*(1-getIou(pre_box,curr_box));
			printf(" coust \t %f ", dist);
			costMatrix[i + j * N] = dist;
		}
	}
}

void CTracker::SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment)
{
	AssignmentProblemSolver APS;
	APS.Solve(costMatrix, N, M, assignment, AssignmentProblemSolver::optimal);
}

track_t CTracker::CalcDistCenter(FaceBox& pre_box, FaceBox& curr_box)
{
	cv::Point2f cent_pre = pre_box.getCenter();
	cv::Point2f cent_pre2 = curr_box.getCenter();
	cv::Point2f diff = cent_pre2 - cent_pre;
	return sqrtf(square(diff.x) + square(diff.y));
};

float CTracker::getIou(const FaceBox& curr_fb, const FaceBox& prev_fb) {
	float box_area = (curr_fb.x2 - curr_fb.x1 + 1) *(curr_fb.y2 - curr_fb.y1 + 1);
	float area = (prev_fb.x2 - prev_fb.x1 + 1) *(prev_fb.y2 - prev_fb.y1 + 1);
	float xx1 = fmax(prev_fb.x1, curr_fb.x1);
	float xx2 = fmin(prev_fb.x2, curr_fb.x2);
	float yy1 = fmax(prev_fb.y1, curr_fb.y1);
	float yy2 = fmin(prev_fb.y2, curr_fb.y2);

	float w = fmax(0, xx2 - xx1 + 1);
	float h = fmax(0, yy2 - yy1 + 1);
	float inter = w *h;
	return inter / (box_area + area - inter);
};


track_t CTracker::CalcDistRect(FaceBox& pre_box, FaceBox& curr_box) 
{
	float iou = getIou(pre_box, curr_box);
	return iou;
}
