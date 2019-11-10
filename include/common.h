#pragma once
#include <vector>
#include <string>

struct FaceBox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	float ppoint[256];
	float regreCoord[4];
	float angles[3];
	int numpts;
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
	, bbox_(rect){}
};

using regions_t = std::vector<FaceTrack>;


// namespace Shape {

// 	template <typename T> class Rect {
// 	public:
// 		Rect() {}
// 		Rect(T x, T y, T w, T h) {
// 			this->x = x;
// 			this->y = y;
// 			this->width = w;
// 			height = h;

// 		}
// 		T x;
// 		T y;
// 		T width;
// 		T height;

// 		cv::Rect convert_cv_rect(int _height, int _width)
// 		{
// 			cv::Rect Rect_(static_cast<int>(x*_width), static_cast<int>(y*_height),
// 				static_cast<int>(width*_width), static_cast<int>(height*_height));
// 			return Rect_;
// 		}
// 	};
// }


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

