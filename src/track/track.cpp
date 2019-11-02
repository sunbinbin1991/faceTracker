#include <thread>

#include "track.h"
#include "../example/utils.hpp"

tracker::tracker()
{
	m_detector = std::unique_ptr<MTCNN>(new MTCNN());
	InitDetector();

	//detectionThread = std::thread(&tracker::Detecting, this);

}

tracker::~tracker() = default;

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
				m_tracker->CalculateIOUs(tempTrk.bbox_, m_tracks, ious);
				if (ious.second < 0.2) {					
					//iou<0.5 add new track
					tempTrk.id_ = m_trackID++; 
					m_tracker->AddNewTracks(tempTrk, m_tracks);
				}
				else {	
					//iou>0.5 merge with previous track
					m_tracker->UpdateTracks(tempTrk, m_tracks[ious.first]);
					m_tracks[ious.first].existsTimes_++;
				}
			}
		}
		
		//every trk age++
		for (auto &trk : m_tracks)
		{
			trk.age_++;
		}
		m_tracker->DeleteLostTracks(m_tracks);

		for (auto &trk : m_tracks)
		{
			printf("age = %d time = %d", trk.age_, trk.existsTimes_);
		}

	}
}

void tracker::TrackingSyncProcess(const cv::Mat& frame, regions_t& regs) {
	regions_t curr_detections;
	Detecting(frame, curr_detections);
	
	Tracking(frame, curr_detections);

	regs.assign(std::begin(m_tracks), std::end(m_tracks));
}

void tracker::DrawTracks(cv::Mat& frame) {

	for (const FaceTrack& ftrk : m_tracks) {
		FaceBox box = ftrk.bbox_;
		drawFace(frame, box);
		sprintf(idtext, "ID %d", ftrk.id_);
		//sprintf(angletext, "%d %d %d", int(ftrk.bbox_.angles[0]), int(ftrk.bbox_.angles[1]), int(ftrk.bbox_.angles[2]));
		//sprintf(scoretext, "score %f", ftrk.bbox_.score);
		//cv::putText(frame, scoretext, cv::Point(box.x1, box.y1 - 80), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1, 8);
		cv::putText(frame, idtext, cv::Point(box.x1, box.y1 - 40), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1, 8);
		//cv::putText(frame, angletext, cv::Point(box.x1, box.y1), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1, 8);
	}
}