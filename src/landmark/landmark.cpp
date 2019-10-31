#include "landmark.h"

landmark::landmark()
{
	std::string model_prefix = "D:/git/track/faceTracker/src/landmark/models/ncnn/";
	std::string param_file = model_prefix + "lnet106_112.param";
	std::string bin_file = model_prefix + "lnet106_112.bin";
	_landmark_net.load_param(param_file.c_str());
	_landmark_net.load_model(bin_file.c_str());
}
void landmark::get_landmark(const cv::Mat& image, std::vector<FaceBox> &faces) {
	_image = image;
	int img_w = image.cols;
	int img_h = image.rows;
	for (int i = 0; i < faces.size(); i++)
	{
		FaceBox bb = faces[i];
		faces[i].numpts = 112;
		cv::Rect face_rect(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1);
		cv::Mat img_face = _image(face_rect).clone();

		ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
			ncnn::Mat::PIXEL_BGR, img_face.cols, img_face.rows, 112, 112);
		in.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor ex = _landmark_net.create_extractor();
		ex.input("data", in);
		ncnn::Mat out;
		ex.extract("bn6_3", out);
		for (int j = 0; j < 106; ++j) {
			float x = abs(out[2 * j] * img_face.cols) + face_rect.x;
			float y = abs(out[2 * j + 1] * img_face.rows) + face_rect.y;
			faces[i].ppoint[2 * j] = x;
			faces[i].ppoint[2 * j+1] = y;
		}
	}
}

landmark::~landmark()
{
}
