#include "ctracker.h"
#include "iostream"
#include "algorithm"



CTracker::CTracker() {
	printf("CTracker init");
};
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
			boxes2[j].x1 = min((int)(boxes1[j].ppoint[2 * i]), boxes2[j].x1);
			boxes2[j].y1 = min((int)(boxes1[j].ppoint[2 * i + 1]), boxes2[j].y1);
			boxes2[j].x2 = max((int)(boxes1[j].ppoint[2 * i]), boxes2[j].x2);
			boxes2[j].y2 = max((int)(boxes1[j].ppoint[2 * i + 1]), boxes2[j].y2);
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
}

void CTracker::UpdateTrackingState(const cregions_t& regions, int width ,int height,int fps)
	{
		const size_t N = c_tracks.size();	// Tracking objects
		const size_t M = regions.size();	// Detections or regions

		assignments_t assignment(N, -1); // Assignments regions -> tracks
		
		if (!c_tracks.empty())
		{
			// Distance matrix between all tracks to all regions
			distMatrix_t costMatrix(N * M);
			const track_t maxPossibleCost = static_cast<track_t>(width * height);
			track_t maxCost = 0;
			CreateDistaceMatrix(regions, costMatrix, maxPossibleCost, maxCost);

			// Solving assignment problem (tracks and predictions of Kalman filter)
			if (m_settings.m_matchType == tracking::MatchHungrian)
			{
				SolveHungrian(costMatrix, N, M, assignment);
			}

			// clean assignment from pairs with large distance
			for (size_t i = 0; i < assignment.size(); i++)
			{
				if (assignment[i] != -1)
				{
					if (costMatrix[i + assignment[i] * N] > m_settings.m_distThres)
					{
						assignment[i] = -1;
						c_tracks[i]->SkippedFrames()++;
					}
				}
				else
				{
					// If track have no assigned detect, then increment skipped frames counter.
					c_tracks[i]->SkippedFrames()++;
				}
			}

			// If track didn't get detects long time, remove it.
			for (int i = 0; i < static_cast<int>(c_tracks.size()); i++)
			{
				if (c_tracks[i]->SkippedFrames() > m_settings.m_maximumAllowedSkippedFrames ||
					c_tracks[i]->IsStaticTimeout(cvRound(fps * (m_settings.m_maxStaticTime - m_settings.m_minStaticTime))))
				{
					c_tracks.erase(c_tracks.begin() + i);
					assignment.erase(assignment.begin() + i);
					i--;
				}
			}
		}

		// Search for unassigned detects and start new tracks for them.
		for (size_t i = 0; i < regions.size(); ++i)
		{
			if (find(assignment.begin(), assignment.end(), i) == assignment.end())
			{
				c_tracks.push_back(std::make_unique<CTrack>(regions[i],
					m_settings.m_kalmanType,
					m_settings.m_dt,
					m_settings.m_accelNoiseMag,
					m_nextTrackID++,
					m_settings.m_filterGoal == tracking::FilterRect,
					m_settings.m_lostTrackType));
			}
		}

		// Update Kalman Filters state
		const ptrdiff_t stop_i = static_cast<ptrdiff_t>(assignment.size());
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < stop_i; ++i)
		{
			// If track updated less than one time, than filter state is not correct.
			if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
			{
				c_tracks[i]->SkippedFrames() = 0;
				c_tracks[i]->Update(
					regions[assignment[i]], true,
					m_settings.m_maxTraceLength,
					width,
					height
				);
			}
			else // if not continue using predictions
			{
				c_tracks[i]->Update(CRegion(), false, m_settings.m_maxTraceLength, width,height);
			}
		}
	}

///
/// \brief CTracker::CreateDistaceMatrix
/// \param regions
/// \param costMatrix
/// \param maxPossibleCost
/// \param maxCost
///
void CTracker::CreateDistaceMatrix(const cregions_t& regions, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost)
{
	const size_t N = c_tracks.size();	// Tracking objects
	maxCost = 0;

	for (size_t i = 0; i < c_tracks.size(); ++i)
	{
		const auto& track = c_tracks[i];

		for (size_t j = 0; j < regions.size(); ++j)
		{
			auto dist = maxPossibleCost;
				dist = 0;
				size_t ind = 0;
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistCenters)
				{
					dist += m_settings.m_distType[ind] * track->CalcDistCenter(regions[j]);
				}
				++ind;
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistRects)
				{
					dist += m_settings.m_distType[ind] * track->CalcDistRect(regions[j]);
				}
				++ind;
				//if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistJaccard)
				//{
				//	dist += m_settings.m_distType[ind] * track->CalcDistJaccard(regions[j]);
				//}
				//++ind;
				//if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistHOG)
				//{
				//	dist += m_settings.m_distType[ind] * track->CalcDistHOG(regions[j]);
				//}
				//++ind;
				assert(ind == tracking::DistsCount);

			costMatrix[i + j * N] = dist;
			if (dist > maxCost)
			{
				maxCost = dist;
			}
		}
	}
}

///
/// \brief CTracker::SolveHungrian
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
///
void CTracker::SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment)
{
	AssignmentProblemSolver APS;
	APS.Solve(costMatrix, N, M, assignment, AssignmentProblemSolver::optimal);
}

CTracks_t CTracker::GetTracks() const
{
	CTracks_t tracks;
	if (!c_tracks.empty())
	{
		tracks.reserve(c_tracks.size());
		for (const auto& track : c_tracks)
		{
			tracks.push_back(track);
		}
	}
	return tracks;
}

///
/// \brief CTracker::SolveBipartiteGraphs
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///