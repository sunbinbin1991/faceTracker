#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "common.h"
using std::string;
#define M_PI 3.1415926

inline void drawFace(cv::Mat& img, const FaceBox& face, const cv::Scalar& color = cv::Scalar(255, 0, 0)) {
	//printf("%f %f %f %f\n", face.xmin, face.ymin, face.xmax, face.ymax);
	cv::Rect rect(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
	//printf("face %f, %f %f %f\n", face.xmin, face.ymin, face.xmax - face.xmin, face.ymax - face.ymin);
	cv::rectangle(img, rect, color, 2);
	for (int i = 0; i < face.numpts; i++) {
		float x = face.ppoint[2 * i];
		float y = face.ppoint[2 * i + 1];
		cv::Point ppt(x, y);
		cv::circle(img, ppt, 2, color, -1);
	}
	char angletext[256] = "IU is Mine";
	cv::putText(img, angletext, cv::Point(face.x1, face.y1-10), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1, 8);
}

inline void drawFaces(cv::Mat& img, const std::vector<FaceBox>& faces, const cv::Scalar& color = cv::Scalar(255, 0, 0)) {
	for (auto& face : faces) {
		drawFace(img, face, color);
	}
}

inline void DrawTracks(cv::Mat& frame,regions_t m_tracks) {
	char idtext[256];
	for (const FaceTrack& ftrk : m_tracks) {
		FaceBox box = ftrk.bbox_;
		drawFace(frame, box);
		sprintf(idtext, "ID %d", ftrk.id_);
		cv::putText(frame, idtext, cv::Point(box.x1, box.y1 - 40), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1, 8);
	}
}

inline cv::Point2f get_point(const float* landmarks, int i) {
	return cv::Point2f(landmarks[2 * i], landmarks[2 * i + 1]);
}

inline cv::Point2f get_projection(const cv::Point2f& p1, const cv::Point2f & p2, const cv::Point2f& p3) {
	cv::Point2f rtp;
	if ((std::fabs(p1.y - p2.y))<1e-5) {
		rtp.y = p1.y;
		rtp.x = p3.x;
	}
	else {
		float k = (p1.x - p2.x) / (p1.y - p2.y);
		rtp.y = (k*k*p1.y - k*p1.x + k*p3.x + p3.y) / (k*k + 1);
		rtp.x = p1.x + k*(rtp.y - p1.y);
	}
	return rtp;
}

inline void angle_estimate_5pts(const float* landmarks, float* angles) {
	const int eyel = 0;
	const int eyer = 1;
	const int nose = 2;
	const int mousel = 3;
	const int mouser = 4;
	cv::Point2f eye_middle, mouse_middle;
	eye_middle.x = (landmarks[eyel * 2] + landmarks[eyer * 2]) / 2.f;
	eye_middle.y = (landmarks[eyel * 2 + 1] + landmarks[eyer * 2 + 1]) / 2.f;
	mouse_middle.x = (landmarks[mousel * 2] + landmarks[mouser * 2]) / 2.f;
	mouse_middle.y = (landmarks[mousel * 2 + 1] + landmarks[mouser * 2 + 1]) / 2.f;
	auto point_nose = get_point(landmarks, nose);
	// left face square box
	auto left_righttop = get_projection(eye_middle, mouse_middle, get_point(landmarks, eyel));
	auto left_lefttop = get_projection(get_point(landmarks, eyel), left_righttop, get_point(landmarks, mousel));
	auto left_leftdown = get_projection(get_point(landmarks, mousel), left_lefttop, mouse_middle);
	auto left_rightdown = mouse_middle;
	// right face square box
	auto right_lefttop = get_projection(eye_middle, mouse_middle, get_point(landmarks, eyer));
	auto right_righttop = get_projection(get_point(landmarks, eyer), right_lefttop, get_point(landmarks, mouser));
	auto right_rightdown = get_projection(get_point(landmarks, mouser), right_righttop, mouse_middle);
	auto right_leftdown = mouse_middle;

	// roll
	float angle_roll = std::atan((mouse_middle.x - eye_middle.x) / (mouse_middle.y - eye_middle.y));
	angles[2] = -180 * angle_roll / M_PI;
	// yaw
	auto noise_to_leftbox_left = get_projection(left_lefttop, left_leftdown, get_point(landmarks, nose));
	float dis_left = std::sqrt(std::pow((point_nose.x - noise_to_leftbox_left.x), 2) +
		std::pow((point_nose.y - noise_to_leftbox_left.y), 2));
	auto noise_to_leftbox_right = get_projection(right_righttop, right_rightdown, get_point(landmarks, nose));
	float dis_right = std::sqrt(std::pow((point_nose.x - noise_to_leftbox_right.x), 2) +
		std::pow((point_nose.y - noise_to_leftbox_right.y), 2));
	float yaw = std::atan(std::fmax(dis_left, dis_right) / std::fmin(dis_left + 0.0001, dis_right + 0.0001)) - 45. / 180 * M_PI;
	if (dis_left >= dis_right)
		angles[1] = -180 * yaw / M_PI;
	else
		angles[1] = 180 * yaw / M_PI;
	// pitch
	float scale = 1.1;
	// computer the dis from noise to the eye middle
	float dis_p_to_top = std::sqrt(std::pow((point_nose.x - eye_middle.x), 2) + std::pow((point_nose.y - eye_middle.y), 2));
	// computer the dis from noise to the mouth middle
	float dis_p_to_down = std::sqrt(std::pow((point_nose.x - mouse_middle.x), 2) + std::pow((point_nose.y - mouse_middle.y), 2));
	//    printf("ratio %f\n", dis_p_to_top/dis_p_to_down);
	float pitch = 1.1*std::atan(std::fmax(dis_p_to_top, scale * dis_p_to_down) / std::fmin(dis_p_to_top + 0.0001, scale * dis_p_to_down + 0.0001)) - 45. / 180 * M_PI;
	if (dis_p_to_top >= dis_p_to_down * scale)
		angles[0] = -180 * pitch / M_PI;
	else
		angles[0] = 180 * pitch / M_PI;
}

inline void printBox(const FaceBox& face) {
	printf("%d %d %d %d \n", face.x1, face.y1, face.x2, face.y2);
	for (size_t i = 0; i < face.numpts; i++)
	{
		//printf("%f %f ", face.ppoint[2 * i], face.ppoint[2 * i + 1]);
	}
	printf("\n");
}

inline void printBoxs(const std::vector<FaceBox>& faces) {
	for (size_t i = 0; i < faces.size(); i++)
	{
		printf("det \t");
		printBox(faces[i]);
	}
}

inline void printTrks(const regions_t& reg_s) {
	for (size_t i = 0; i < reg_s.size(); i++)
	{
		printf("trk \t");
		printBox(reg_s[i].bbox_);
	}
}

////字符串分割函数
//inline std::vector<std::string> split(std::string str,std::string pattern)
//{
//    std::string::size_type pos;
//    std::vector<std::string> result;
//    str+=pattern;//扩展字符串以方便操作
//    int size=str.size();
//
//    for(int i=0; i<size; i++)
//    {
//        pos=str.find(pattern,i);
//        if(pos<size)
//        {
//            std::string s=str.substr(i,pos-i);
//            result.push_back(s);
//            i=pos+pattern.size()-1;
//        }
//    }
//    return result;
//}