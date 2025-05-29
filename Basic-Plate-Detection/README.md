# License Detection with C++
This repo aims to understand how detect algorithms work. The codes written give correct results only for the loaded images. The code needs to be improved for better results.

The following for loop tries to find rectangles and selects the rectangle closest to the plate proportions.

	for (size_t i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		double area = rect.width * rect.height;

		double aspectRatio = (double)rect.width / rect.height;
		std::vector<cv::Point> approx;
		cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);

		// Plakalar genelde en-boy oranı 2-5 arasında olan dikdörtgenlerdir
		if (aspectRatio > 3 && aspectRatio < 5 && approx.size()==4 && area>500) {
			cv::polylines(original, approx, true, cv::Scalar(0, 255, 0), 3);
			bestCandidate = rect;
		}
	}
