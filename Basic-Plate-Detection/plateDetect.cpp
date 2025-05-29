#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
	cv::Mat original = cv::imread("images/araba3.png");
	cv::Mat output = original.clone();
	cv::Mat kernel = (cv::Mat_<uint8_t>(3, 3) << 0, 1, 0,
												 1, 1, 1,
												 0, 1, 0);

	cv::cvtColor(original, output, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(output, output, cv::Size(5, 5), 1.8);
	cv::Canny(output, output, 100, 300);
	cv::dilate(output, output, kernel, cv::Point(-1, -1), 1);

	std::vector<std::vector<cv::Point>> contours; 
	std::vector<cv::Vec4i> hierarchy;
	
	cv::findContours(output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Rect bestCandidate;

	for (size_t i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		double area = rect.width * rect.height;

		double aspectRatio = (double)rect.width / rect.height;
		std::vector<cv::Point> approx;
		cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);

		if (aspectRatio > 3 && aspectRatio < 5 && approx.size()==4 && area>500) {
			cv::polylines(original, approx, true, cv::Scalar(0, 255, 0), 3);
			bestCandidate = rect;
		}
	}

	if (bestCandidate.area() > 0) {
		cv::Mat plate = original(bestCandidate);
		cv::imshow("Plaka Kesildi", plate);
	}
	else {
		std::cout << "Plaka bulunamadı!" << std::endl;
	}

	cv::namedWindow("Input", cv::WINDOW_NORMAL);
	cv::resizeWindow("Input", 500, 500);
	cv::imshow("Input", original);

	cv::namedWindow("Output", cv::WINDOW_NORMAL);
	cv::resizeWindow("Output", 500, 500);
	cv::imshow("Output", output);

	cv::waitKey(0);

	return 0;
}