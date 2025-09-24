#ifndef COMPUTER_VISION_HPP
#define COMPUTER_VISION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace jetracer::vision
{
	// Constants for image dimensions and shared memory
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

	// Functions for danger zone detection and display
	bool sampleLaneEdgesByRow(const cv::Mat &mask,
							  int y_start, int y_end, int step,
							  float image_center,
							  std::vector<cv::Point> &left_curve,
							  std::vector<cv::Point> &right_curve,
							  int min_run = 3);

	void drawDangerZoneCurved(cv::Mat &frame,
							  const std::vector<cv::Point> &left_curve,
							  const std::vector<cv::Point> &right_curve,
							  float displacement_cm, float scale);

	float calculateDangerZoneOccupancy(const cv::Mat &mask,
									   const std::vector<cv::Point> &left_curve,
									   const std::vector<cv::Point> &right_curve,
									   float displacement_cm,
									   float scale);

	void createDangerZoneMask(const cv::Mat &original_mask,
							  const std::vector<cv::Point> &left_curve,
							  const std::vector<cv::Point> &right_curve,
							  float displacement_cm, float scale,
							  cv::Mat &danger_zone_mask);

	float calculateDangerZoneOccupancyFromMask(const cv::Mat &original_mask,
											   const cv::Mat &danger_zone_mask);

	// Functions for drivable area danger zone
	void createDrivableDangerZoneMask(const cv::Mat &drivable_mask,
									  float displacement_cm, float scale,
									  cv::Mat &drivable_danger_zone_mask);

	float calculateDrivableDangerZoneOccupancy(const cv::Mat &drivable_mask,
											   const cv::Mat &drivable_danger_zone_mask);

	void drawDrivableDangerZone(cv::Mat &frame,
								const cv::Mat &drivable_danger_zone_mask);
} // namespace jetracer::vision

#endif // COMPUTER_VISION_HPP
