#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<vector>
/*
현재 System 가장 큰 문제점
임계값이 이미지마다 조금씩 다르므로, 최적화를 따로 해주어야 한다.
이는 큰 문제...
이를 해결 해야 한다.

*/
class Unsupervise
{
public:
	Unsupervise();
	//~Unsupervise();
	void Processing();
	///////////////////////////////////////////
	void Preprocess();
	cv::Mat BlobImage();
	cv::Mat BlackRingRemoval();
	cv::Mat GrayScaleTrans();
	///////////////////////////////////////////
	void VenulesDetecor();
	cv::Mat TophatTrans();
	///////////////////////////////////////////
	void CapillDetector();
	cv::Mat BlackWhiteInv();
	void Rotate(cv::Mat& src, double angle, cv::Mat & dst);
	cv::Mat Centerline();
	cv::Mat FloodFill4(cv::Mat src, cv::Mat dst);
	cv::Mat FloodFill8(cv::Mat src, cv::Mat dst);
	///////////////////////////////////////////
	void PostProcess();
	cv::Mat ImageOverlap();
	cv::Mat PostDenoise();
	///////////////////////////////////////////
	void drawHist(cv::Mat inputimg);
	void drawBGRHist(cv::Mat inputimg);
private:
	cv::Mat oriimg; 
	cv::Mat preimg;
	cv::Mat venuimg;
	cv::Mat capimg;
	cv::Mat postimg;
	int imgcol;
	int imgrow;
};
