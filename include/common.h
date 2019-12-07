#pragma once
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

struct FaceBox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	cv::Point2f center_;
	float area;
	float ppoint[256];
	float regreCoord[4];
	float angles[3];
	int numpts = 106;
	void setCenter() {
		center_.x = (x1 + x2)>>1;
		center_.y = (y1 + y2)>>1;
	}
};

//frame Info
class FaceTrack {	

public:
	int id_;
	FaceBox bbox_;
	int existsTimes_ = 0;
	int age_ = 0;
	FaceTrack() {};	
	FaceTrack(int id, const FaceBox& rect) 
	: id_(id)
	, bbox_(rect)
	{}
};

using regions_t = std::vector<FaceTrack>;
using track_t = float;
using Point_t = cv::Point_<track_t>;
#define Mat_t CV_32FC

///
/// \brief The CRegion class
///
class CRegion
{
public:
	CRegion()
		: m_type(""), m_confidence(-1)
	{
	}

	CRegion(const cv::Rect& rect)
		: m_brect(rect)
	{
		B2RRect();
	}

	CRegion(const cv::RotatedRect& rrect)
		: m_rrect(rrect)
	{
		R2BRect();
	}

	CRegion(const cv::Rect& rect, const std::string& type, float confidence)
		: m_brect(rect), m_type(type), m_confidence(confidence)
	{
		B2RRect();
	}

	cv::RotatedRect m_rrect;
	cv::Rect m_brect;

	std::string m_type;
	float m_confidence = -1;

	mutable cv::Mat m_hist;

private:
	///
	/// \brief R2BRect
	/// \return
	///
	cv::Rect R2BRect()
	{
		m_brect = m_rrect.boundingRect();
		return m_brect;
	}
	///
	/// \brief B2RRect
	/// \return
	///
	cv::RotatedRect B2RRect()
	{
		m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
		return m_rrect;
	}
};

typedef std::vector<CRegion> cregions_t;




namespace tracking
{
	///
	/// \brief The Detectors enum
	///
	enum Detectors
	{
		MTCNN,
		Motion_VIBE,
		Motion_MOG,
		Motion_GMG,
		Motion_CNT,
		Motion_SuBSENSE,
		Motion_LOBSTER,
		Pedestrian_HOG,
		Pedestrian_C4,
		SSD_MobileNet,
		Yolo_OCV,
		Yolo_Darknet
	};

	///
	/// \brief The DistType enum
	///
	enum DistType
	{
		DistCenters,   // Euclidean distance between centers, pixels
		DistRects,     // Euclidean distance between bounding rectangles, pixels
		DistJaccard,   // Intersection over Union, IoU, [0, 1]
		DistHist,      // Bhatacharia distance between histograms, [0, 1]
		DistHOG,       // Euclidean distance between HOG descriptors, [0, 1]
		DistsCount
	};

	///
	/// \brief The FilterGoal enum
	///
	enum FilterGoal
	{
		FilterCenter,
		FilterRect
	};

	///
	/// \brief The KalmanType enum
	///
	enum KalmanType
	{
		KalmanLinear,
		KalmanUnscented,
		KalmanAugmentedUnscented
	};

	///
	/// \brief The MatchType enum
	///
	enum MatchType
	{
		MatchHungrian,
		MatchBipart
	};

	///
	/// \brief The LostTrackType enum
	///
	enum LostTrackType
	{
		TrackNone,
		TrackKCF,
		TrackMIL,
		TrackMedianFlow,
		TrackGOTURN,
		TrackMOSSE,
		TrackCSRT,
		TrackDAT,
		TrackSTAPLE,
		TrackLDES
	};
}

