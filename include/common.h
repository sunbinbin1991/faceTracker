#pragma once
struct Bbox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	float ppoint[256];
	float regreCoord[4];
	int numpts;
};

namespace Shape {

	template <typename T> class Rect {
	public:
		Rect() {}
		Rect(T x, T y, T w, T h) {
			this->x = x;
			this->y = y;
			this->width = w;
			height = h;

		}
		T x;
		T y;
		T width;
		T height;

		cv::Rect convert_cv_rect(int _height, int _width)
		{
			cv::Rect Rect_(static_cast<int>(x*_width), static_cast<int>(y*_height),
				static_cast<int>(width*_width), static_cast<int>(height*_height));
			return Rect_;
		}
	};
}
