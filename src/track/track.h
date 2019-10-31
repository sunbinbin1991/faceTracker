#include <atomic>
#include <mutex>

#include "opencv2/opencv.hpp"

#include "common.h"

class track
{
public:
    track();
    ~track();

	//detect related init
	void init_detector();

	void tracking(const cv::Mat& frame) {};


private:
	std::atomic<bool> flag = ATOMIC_VAR_INIT(false);

// tracking strategy 0 : KCF


// tracking match strategy


};


