#include "kalman.h"

TkalmanFilter::TkalmanFilter(tracking::KalmanType type, 
	cv::Rect rect,
	float deltaTime, 
	float accelNoiseMag)
	:
	m_initialized(false),
	m_deltaTime(deltaTime),
	m_deltaTimeMin(deltaTime),
	m_deltaTimeMax(2 * deltaTime){

	m_deltaStep = (m_deltaTimeMax - m_deltaTimeMin) / m_deltaStepsCount;

	m_initialRects.push_back(rect);
	m_lastRectResult = rect;
};

// -------------------------------------------------------------------------- -
void TkalmanFilter::CreateLinear(Point_t xy0, Point_t xyv0)
{
	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object.
	// Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.

	// 4 state variables, 2 measurements
	m_linearKalman = std::make_unique<cv::KalmanFilter>(4, 2, 0);
	// Transition cv::Matrix
	m_linearKalman->transitionMatrix = (cv::Mat_<track_t>(4, 4) <<
		1, 0, m_deltaTime, 0,
		0, 1, 0, m_deltaTime,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// init...
	m_lastPointResult = xy0;
	m_linearKalman->statePre.at<track_t>(0) = xy0.x; // x
	m_linearKalman->statePre.at<track_t>(1) = xy0.y; // y

	m_linearKalman->statePre.at<track_t>(2) = xyv0.x;
	m_linearKalman->statePre.at<track_t>(3) = xyv0.y;

	m_linearKalman->statePost.at<track_t>(0) = xy0.x;
	m_linearKalman->statePost.at<track_t>(1) = xy0.y;

	cv::setIdentity(m_linearKalman->measurementMatrix);

	m_linearKalman->processNoiseCov = (cv::Mat_<track_t>(4, 4) <<
		pow(m_deltaTime, 4.0) / 4.0, 0, pow(m_deltaTime, 3.0) / 2.0, 0,
		0, pow(m_deltaTime, 4.0) / 4.0, 0, pow(m_deltaTime, 3.0) / 2.0,
		pow(m_deltaTime, 3.0) / 2.0, 0, pow(m_deltaTime, 2.0), 0,
		0, pow(m_deltaTime, 3.0) / 2.0, 0, pow(m_deltaTime, 2.0));


	m_linearKalman->processNoiseCov *= m_accelNoiseMag;

	cv::setIdentity(m_linearKalman->measurementNoiseCov, cv::Scalar::all(0.1));

	cv::setIdentity(m_linearKalman->errorCovPost, cv::Scalar::all(.1));

	m_initialized = true;
}

//---------------------------------------------------------------------------
void TkalmanFilter::CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0)
{
	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object.
	// Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.

	// 6 state variables (x, y, dx, dy, width, height), 4 measurements (x, y, width, height)
	m_linearKalman = std::make_unique<cv::KalmanFilter>(8, 4, 0);
	// Transition cv::Matrix
	m_linearKalman->transitionMatrix = (cv::Mat_<track_t>(8, 8) <<
		1, 0, 0, 0, m_deltaTime, 0, 0, 0,
		0, 1, 0, 0, 0, m_deltaTime, 0, 0,
		0, 0, 1, 0, 0, 0, m_deltaTime, 0,
		0, 0, 0, 1, 0, 0, 0, m_deltaTime,
		0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 1);

	// init...
	m_linearKalman->statePre.at<track_t>(0) = rect0.x;      // x
	m_linearKalman->statePre.at<track_t>(1) = rect0.y;      // y
	m_linearKalman->statePre.at<track_t>(2) = rect0.width;  // width
	m_linearKalman->statePre.at<track_t>(3) = rect0.height; // height
	m_linearKalman->statePre.at<track_t>(4) = rectv0.x;     // dx
	m_linearKalman->statePre.at<track_t>(5) = rectv0.y;     // dy
	m_linearKalman->statePre.at<track_t>(6) = 0;            // dw
	m_linearKalman->statePre.at<track_t>(7) = 0;            // dh

	m_linearKalman->statePost.at<track_t>(0) = rect0.x;
	m_linearKalman->statePost.at<track_t>(1) = rect0.y;
	m_linearKalman->statePost.at<track_t>(2) = rect0.width;
	m_linearKalman->statePost.at<track_t>(3) = rect0.height;

	cv::setIdentity(m_linearKalman->measurementMatrix);

	track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
	track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
	track_t n3 = pow(m_deltaTime, 2.f);
	m_linearKalman->processNoiseCov = (cv::Mat_<track_t>(8, 8) <<
		n1, 0, 0, 0, n2, 0, 0, 0,
		0, n1, 0, 0, 0, n2, 0, 0,
		0, 0, n1, 0, 0, 0, n2, 0,
		0, 0, 0, n1, 0, 0, 0, n2,
		n2, 0, 0, 0, n3, 0, 0, 0,
		0, n2, 0, 0, 0, n3, 0, 0,
		0, 0, n2, 0, 0, 0, n3, 0,
		0, 0, 0, n2, 0, 0, 0, n3);

	m_linearKalman->processNoiseCov *= m_accelNoiseMag;

	cv::setIdentity(m_linearKalman->measurementNoiseCov, cv::Scalar::all(0.1));

	cv::setIdentity(m_linearKalman->errorCovPost, cv::Scalar::all(.1));

	m_initialized = true;
}


//---------------------------------------------------------------------------
cv::Rect TkalmanFilter::GetRectPrediction()
{
	if (m_initialized)
	{
		cv::Mat prediction;

		switch (m_type)
		{
		case tracking::KalmanLinear:
			prediction = m_linearKalman->predict();
			break;

		case tracking::KalmanUnscented:
		case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
			prediction = m_uncsentedKalman->predict();
#else
			prediction = m_linearKalman->predict();
			std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
			break;
		}

		m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
	}
	else
	{
		printf("kalman filter not initialize proper!");
	}
	return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}


//---------------------------------------------------------------------------
cv::Rect TkalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
	if (!m_initialized)
	{
		if (m_initialRects.size() < MIN_INIT_VALS)
		{
			if (dataCorrect)
			{
				m_initialRects.push_back(rect);
			}
		}
		if (m_initialRects.size() == MIN_INIT_VALS)
		{
			std::vector<Point_t> initialPoints;
			Point_t averageSize(0, 0);
			for (const auto& r : m_initialRects)
			{
				initialPoints.emplace_back(static_cast<track_t>(r.x), static_cast<track_t>(r.y));
				averageSize.x += r.width;
				averageSize.y += r.height;
			}
			averageSize.x /= MIN_INIT_VALS;
			averageSize.y /= MIN_INIT_VALS;

			track_t kx = 0;
			track_t bx = 0;
			track_t ky = 0;
			track_t by = 0;
			get_lin_regress_params(initialPoints, 0, MIN_INIT_VALS, kx, bx, ky, by);
			cv::Rect_<track_t> rect0(kx * (MIN_INIT_VALS - 1) + bx, ky * (MIN_INIT_VALS - 1) + by, averageSize.x, averageSize.y);
			Point_t rectv0(kx, ky);

			switch (m_type)
			{
			case tracking::KalmanLinear:
				CreateLinear(rect0, rectv0);
				break;

			case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
				CreateUnscented(rect0, rectv0);
#else
				CreateLinear(rect0, rectv0);
				std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
				break;

			case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
				CreateAugmentedUnscented(rect0, rectv0);
#else
				CreateLinear(rect0, rectv0);
				std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
				break;
			}
		}
	}

	if (m_initialized)
	{
		cv::Mat measurement(4, 1, Mat_t(1));
		if (!dataCorrect)
		{
			measurement.at<track_t>(0) = m_lastRectResult.x;  // update using prediction
			measurement.at<track_t>(1) = m_lastRectResult.y;
			measurement.at<track_t>(2) = m_lastRectResult.width;
			measurement.at<track_t>(3) = m_lastRectResult.height;
		}
		else
		{
			measurement.at<track_t>(0) = static_cast<track_t>(rect.x);  // update using measurements
			measurement.at<track_t>(1) = static_cast<track_t>(rect.y);
			measurement.at<track_t>(2) = static_cast<track_t>(rect.width);
			measurement.at<track_t>(3) = static_cast<track_t>(rect.height);
		}
		// Correction
		cv::Mat estimated;
		switch (m_type)
		{
		case tracking::KalmanLinear:
		{
			estimated = m_linearKalman->correct(measurement);

			m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRectResult.y = estimated.at<track_t>(1);
			m_lastRectResult.width = estimated.at<track_t>(2);
			m_lastRectResult.height = estimated.at<track_t>(3);

			// Inertia correction
			track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - rect.x) + sqr(estimated.at<track_t>(1) - rect.y) + sqr(estimated.at<track_t>(2) - rect.width) + sqr(estimated.at<track_t>(3) - rect.height));
			if (currDist > m_lastDist)
			{
				m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
			}
			else
			{
				m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);
			}
			m_lastDist = currDist;

			m_linearKalman->transitionMatrix.at<track_t>(0, 4) = m_deltaTime;
			m_linearKalman->transitionMatrix.at<track_t>(1, 5) = m_deltaTime;
			m_linearKalman->transitionMatrix.at<track_t>(2, 6) = m_deltaTime;
			m_linearKalman->transitionMatrix.at<track_t>(3, 7) = m_deltaTime;

			break;
		}

		case tracking::KalmanUnscented:
		case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
			estimated = m_uncsentedKalman->correct(measurement);

			m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRectResult.y = estimated.at<track_t>(1);
			m_lastRectResult.width = estimated.at<track_t>(6);
			m_lastRectResult.height = estimated.at<track_t>(7);
#else
			estimated = m_linearKalman->correct(measurement);

			m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRectResult.y = estimated.at<track_t>(1);
			m_lastRectResult.width = estimated.at<track_t>(2);
			m_lastRectResult.height = estimated.at<track_t>(3);
			std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
			break;
		}
	}
	else
	{
		if (dataCorrect)
		{
			m_lastRectResult.x = static_cast<track_t>(rect.x);
			m_lastRectResult.y = static_cast<track_t>(rect.y);
			m_lastRectResult.width = static_cast<track_t>(rect.width);
			m_lastRectResult.height = static_cast<track_t>(rect.height);
		}
	}
	return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}
