#ifndef COMPUTER_VISION_HPP
#define COMPUTER_VISION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace jetracer::vision
{
	// Constantes para dimensões da imagem e memória compartilhada
	constexpr int WIDTH = 128;
	constexpr int HEIGHT = 128;
	constexpr int SIZE = WIDTH * HEIGHT;

	float getXAtY(float y, float y0, float x0, float vx, float vy);

	bool extractLanePoints(const cv::Mat &frame,
						   float image_center,
						   float &y_ref,
						   std::vector<cv::Point> &left_point,
						   std::vector<cv::Point> &right_point);

	float calculateTrackCenter(const std::vector<cv::Point> &left,
							   const std::vector<cv::Point> &right,
							   float y_ref,
							   float displacement_cm,
							   float scale,
							   cv::Mat &frame);

	void draw_overlay(cv::Mat &frame,
					  float erro,
					  float pid,
					  const std::string &file_name,
					  const std::string &txt_lane,
					  float image_center,
					  float center_track,
					  float y_ref);
} // namespace jetracer::vision

#endif // COMPUTER_VISION_HPP
