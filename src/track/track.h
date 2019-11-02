#include <atomic>
#include <mutex>

#include "opencv2/opencv.hpp"

#include "common.h"
#include "../detect/mtcnn.h"

class track
{
public:
    track();
    ~track();
	void tracking(const cv::Mat& frame);

private :
	//detect related init	
	void init_detector();

private:
	cv::Mat m_curr_frame;

	std::atomic<bool> m_flag = ATOMIC_VAR_INIT(false);

	std::unique_ptr<MTCNN> m_detector;//

// tracking strategy 0 : KCF


// tracking match strategy


};


