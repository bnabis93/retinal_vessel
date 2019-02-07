#include"Unsupervised.h"
//Constructor 
Unsupervise::Unsupervise()
{
	imgcol = 0, imgrow = 0;
	oriimg = cv::imread("test_01.jpg");
	imgrow = oriimg.rows, imgcol = oriimg.cols;
	std::cout << imgrow << " " << imgcol << std::endl;
}
void Unsupervise::Processing() {
	Preprocess();
	VenulesDetecor();
	CapillDetector();
}

void Unsupervise::Preprocess() {
	
	preimg= oriimg.clone();
	//Copy img. 
	cv::imwrite("1_Preprocess1.jpg", preimg);
	
	preimg = BlackRingRemoval();
	preimg = GrayScaleTrans();
	preimg = BlobImage();
	cv::imwrite("1_Preprocess2.jpg", preimg);
	
}
cv::Mat Unsupervise::BlobImage() {
	//Gaussian, Median, Bilateral 
	//Want to remove background edge + remove Noise .
	//I think that median or gaussian filter is adaptive.
	//background edge remove, but vessel edge should preseve.
	cv::Mat gassimg = preimg.clone();
	cv::Mat medimg = preimg.clone();
	cv::Mat bilimg = preimg.clone();
	cv::Mat bresult;
	//Gaussian filter
	//parameter(in order) input img, output img, kernal size, sigmaX, sigmaY
	/*
	cv::GaussianBlur(gassimg, gassimg, cv::Size(3,3),1.0, 1.0);
	drawBGRHist(gassimg);
	cv::imwrite("1_GaussianBlue.jpg", gassimg);
	return gassimg;
	*/
	//Median filter
	//parameter(in order) input, output, kernal size (kernal size is always odd number.)

	
	cv::medianBlur(medimg, medimg, 3); //3*3 kernal
	drawBGRHist(medimg);
	cv::imwrite("1_Median_filter.jpg", medimg);
	return medimg;
	

	//Bilateral
	//parameter(in order) input, output, (int) diameter, sigma color, sigma space
	/*
	cv::bilateralFilter(bilimg,bresult, 5, 30, 30);
	drawBGRHist(bresult);
	cv::imwrite("1_bilateral.jpg", bresult);
	return bresult;
	*/
	//return medimg;
}
cv::Mat Unsupervise::BlackRingRemoval() {
	/*mean Removal
	can select 2 mathods.
	Histogram Equalization. and pixel processing.
	first, get histogram
	gray scale histogram
	histogram is just to see the value without any function.*/

	cv::Mat tempimg = preimg.clone();
	cv::Mat tempgray;
	int r = 0, g = 0, b = 0;
	//
	//Hough circle
	cv::cvtColor(tempimg, tempgray, CV_RGB2GRAY);
	std::vector<cv::Vec3f> circles;
	cv::Vec3i c;

	cv::HoughCircles(tempgray, circles, cv::HOUGH_GRADIENT, 1, tempgray.rows / 2, 50, 50, 100, 380);
	for (size_t i = 0; i < circles.size(); i++) {
		
		c = circles[i]; //x,y, distance
		cv::circle(tempgray, cv::Point(c[0], c[1]), c[2], cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
		//cv::circle(tempgray, cv::Point(c[0], c[1]), 2, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
		std::cout << c[0] << c[1] << std::endl;
	}
	for (int y = 0; y < tempimg.rows; ++y) {
		for (int x = 0; x < tempimg.cols; ++x) {
			double dist = sqrt((x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]));
			
			if (dist > (c[2]-1) ) {
				tempimg.at<cv::Vec3b>(y, x)[2] = 0;
				tempimg.at<cv::Vec3b>(y, x)[1] = 0;
				tempimg.at<cv::Vec3b>(y, x)[0] = 0;
			}
		}
	}


	cv::imwrite("1_CircleDetect.jpg", tempgray);
	cv::imwrite("1_AfterCircleDetect.jpg", tempimg);
	/*for (int y = 0; y < tempimg.rows; ++y) {
		for (int x = 0; x < tempimg.cols; ++x) {
			if ((tempimg.at<cv::Vec3b>(y, x)[0] <= 30) || tempimg.at<cv::Vec3b>(y, x)[1] <= 30 || tempimg.at<cv::Vec3b>(y, x)[2] <= 30)
			{
				tempimg.at<cv::Vec3b>(y, x)[2] = 0;
				tempimg.at<cv::Vec3b>(y, x)[1] = 0;
				tempimg.at<cv::Vec3b>(y, x)[0] = 0;
			}
		}
	}*/

	for (int y = imgrow/2 -50; y < imgrow/2 +50; ++y) {
		for (int x = imgcol/2 -50; x < imgcol/2 +50; ++x) {
			b += tempimg.at<cv::Vec3b>(y, x)[0];
			g += tempimg.at<cv::Vec3b>(y, x)[1];
			r += tempimg.at<cv::Vec3b>(y, x)[2];
		}
	}
	r = r / (10000);
	g = g / (10000);
	b = b / (10000);
	/*
		Average the entire image.
	r = r / (tempimg.rows * tempimg.cols);
	g = g / (tempimg.rows * tempimg.cols);
	b = b / (tempimg.rows * tempimg.cols);
	*/
	for (int y = 0; y < tempimg.rows; ++y) {
		for (int x = 0; x < tempimg.cols; ++x) { 
			if ((tempimg.at<cv::Vec3b>(y, x)[0] <=30) || tempimg.at<cv::Vec3b>(y, x)[1] <= 30|| tempimg.at<cv::Vec3b>(y, x)[2] <= 30)
			{
				tempimg.at<cv::Vec3b>(y, x)[2] = r;
				tempimg.at<cv::Vec3b>(y, x)[1] = g;
				tempimg.at<cv::Vec3b>(y, x)[0] = b;
			}
		}
	}

	cv::imwrite("1_RemovePreprocess.jpg", tempimg);
	return tempimg;
}
cv::Mat Unsupervise::GrayScaleTrans() {
	//extract green ch.
	cv::Mat gchimg = preimg.clone();
	cv::Mat cirimg = preimg.clone();

	/*for (int y = 0; y <gchimg.rows; ++y) {
		for (int x = 0; x < gchimg.cols; ++x) {
			gchimg.at<cv::Vec3b>(y, x)[0] =0;
			gchimg.at<cv::Vec3b>(y, x)[2] =0;
		}
	}*/
	cv::cvtColor(gchimg, gchimg, CV_RGB2GRAY);
	
	cv::imwrite("1_GrayPreprocess1.jpg", gchimg);
	for (int y = 0; y <gchimg.rows; ++y) {
		for (int x = 0; x < gchimg.cols; ++x) {
			gchimg.at<uchar>(y, x) = 255 - gchimg.at<uchar>(y, x);
		}
	}//Black white Invert
	cv::imwrite("1_GrayPreprocess2.jpg", gchimg);
	return gchimg;
}

void Unsupervise::drawHist(cv::Mat inputimg) {
	int histsize[1];
	float hrange[2];
	const float* ranges[1];
	int ch[1];
	double maxval = 0;
	double minval = 0;

	histsize[0] = 256;
	hrange[0] = 0.0;
	hrange[1] = 255.0;
	ranges[0] = hrange;
	ch[0] = 0;
	// Gray ch. If you want another ch, change array index 0 -> another number(ex, ch[0]={0} = blue, ch[0] = {1} = Green)
	cv::MatND hist;
	cv::calcHist(&inputimg, 1, ch, cv::Mat(), hist, 1, histsize,ranges);
	cv::minMaxLoc(hist, &minval, &maxval, 0, 0);
	int hpt = static_cast<int>(0.9*histsize[0]);
	cv::Mat histimg(histsize[0], histsize[0], CV_8U, cv::Scalar(255));
	int tempcnt = 0;

	cv::normalize(hist, hist, 0, histimg.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 0; i < histsize[0]; i++) {
		float binval = hist.at<float>(i);
		if (binval == 0)// want to know number of Background 
			tempcnt++;
		int intens = static_cast<int>(binval*hpt / maxval);
		line(histimg, cv::Point(i, histsize[0]), cv::Point(i, histsize[0] - intens), cv::Scalar::all(0));
		std::cout << binval << " ";
	}
	cv::imwrite("1_histimg.jpg", histimg);
}

void Unsupervise::drawBGRHist(cv::Mat inputimg) {
	cv::Mat bgr_planes[3];
	cv::split(inputimg, bgr_planes);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = true;
	cv::Mat b_hist, g_hist, r_hist;

	cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat Bhistimg(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat Ghistimg(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat Rhistimg(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::normalize(b_hist, b_hist, 0, Bhistimg.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, Ghistimg.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, Rhistimg.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < histSize; i++) {
		cv::line(Bhistimg, cv::Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8, 0);

		cv::line(Ghistimg, cv::Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0), 2, 8, 0);

		cv::line(Rhistimg, cv::Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255), 2, 8, 0);

	}
	//cv::namedWindow("hist", CV_WINDOW_AUTOSIZE);
	//cv::imshow("hist", histimg);
	cv::imwrite("2_B hist.jpg",Bhistimg);
	cv::imwrite("2_G hist.jpg", Ghistimg);
	cv::imwrite("2_R hist.jpg", Rhistimg);
	//cv::waitKey(0);
}

void Unsupervise::VenulesDetecor() {
	//Venule Detector
	venuimg = preimg.clone();
	venuimg = TophatTrans();
}

cv::Mat Unsupervise::TophatTrans() {
	//kernal의 크기에 따라 어떤 변화가 있는지 알아보기
	//tophat filter 관해서 자세히 알아보기
	cv::Mat topimg;
	cv::Mat kernal(17,17, CV_8U, cv::Scalar(255));
	//kernal size for top-hat.
	//kernal size는 object detection에 사용된다. (kernal size보다 작은 걸 가져옴)
	cv::morphologyEx(venuimg, topimg, cv::MORPH_TOPHAT, kernal);
	//Tophat + image binarization.
	//make 9*9 kernal and do tophat morphology.
	//and binarization using otsu's method (threshold is not general value)

	cv::imwrite("1_pre_topimg.jpg", topimg);
	cv::threshold(topimg, topimg, 20, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
	cv::imwrite("1_topimg.jpg", topimg);

	return topimg;
}

void Unsupervise::CapillDetector() {
	capimg = preimg.clone();
	capimg = BlackWhiteInv();
	capimg = Centerline();
}

cv::Mat Unsupervise::BlackWhiteInv() {
	cv::Mat bwiimg = capimg.clone();

	for (int y = 0; y <bwiimg.rows; ++y) {
		for (int x = 0; x < bwiimg.cols; ++x) {
			bwiimg.at<uchar>(y, x) = 255 - bwiimg.at<uchar>(y, x);
		}
	}//Black white Invert

	cv::imwrite("3_Cap_BWIimg.jpg", bwiimg);
	return bwiimg;

}

cv::Mat Unsupervise::Centerline() {
	//first-order derivative filter
	/* 3*5 filter.
	| -1 -2 0 2 1 |
	| -2 -4 0 4 2 |
	| -1 -2 0 2 1 |

	*/
	/*
	cv::Mat kernal = (cv::Mat_<char>(3,5)<< -1, -2, 0, 2, 1,
											-2, -4, 0, 4, 2,
											-1, -2 ,0, 2, 1);
	*/
	cv::Mat kernal = (cv::Mat_<char>(5, 3) << -1, -2, -1,
		-2, -4, -2,
		0, 0, 0,
		2, 4, 2,
		1, 2, 1);
	cv::Mat filteredimg = capimg.clone();
	
	cv::Mat rotated;
	//cv::GaussianBlur(filteredimg, filteredimg, cv::Size(5, 3), 2.0, 2.0);
	//cv::imwrite("1_Cap_gasimg.jpg", gasimg);| cv::THRESH_OTSU
	//return gasimg;
	cv::filter2D(filteredimg, filteredimg, -1, kernal, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	//cv::imwrite("3_derivateFilter.jpg", filteredimg);

	cv::threshold(filteredimg, filteredimg, 25,255, cv::THRESH_BINARY);
	//cv::imwrite("3_binary.jpg", filteredimg);
	cv::Mat rotateArray[19];
	for (int i = 1; i <= 18; i++) {
		Rotate(filteredimg, i * 10.0, rotateArray[i-1]);
		if (i == 1)
			rotated = rotateArray[i - 1];
		else
			cv::add(rotated, rotateArray[i - 1], rotated);
	}
	cv::imwrite("3_filtered.jpg", rotated);
	//cv::imwrite("3_overlaped.jpg", overlapImg);
	cv::Mat connectedImg,stats,centroids;

	int num_of_labels = cv::connectedComponentsWithStats(rotated, connectedImg, stats, centroids, 4, CV_32S);
	for (int y = 0; y<connectedImg.rows; ++y) {

		int *label = connectedImg.ptr<int>(y);
		cv::Vec3b* pixel = connectedImg.ptr<cv::Vec3b>(y);


		for (int x = 0; x < connectedImg.cols; ++x) {


			if (label[x] == 3) {
				pixel[x][2] = 0;
				pixel[x][1] = 255;
				pixel[x][0] = 0;
			}
		}
	}


	cv::imwrite("3_connected.jpg", connectedImg);


	return filteredimg;
}

void Unsupervise::Rotate(cv::Mat& src, double angle, cv::Mat & dst) {
	cv::Point2f ptCp(src.cols*0.5, src.rows*0.5); //will include black padding 
	cv::Mat M = cv::getRotationMatrix2D(ptCp, angle, 1.0);
	cv::warpAffine(src, dst, M, src.size(), cv::INTER_CUBIC);
}