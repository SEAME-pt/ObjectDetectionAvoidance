#include "jetracer/computer_vision.hpp"
#include <limits>

namespace jetracer::vision
{
	constexpr int roi_numbers_in_frame = 7;

	float getXAtY(float y, float y0, float x0, float vx, float vy)
	{
		return x0 + (y - y0) * (vx / vy);
	}

	void draw_overlay(cv::Mat &frame, float erro, float pid, const std::string &file_name, const std::string &txt_lane, float image_center, float center_track, float y_ref)
	{
		cv::line(frame, {int(image_center), int(y_ref)}, {int(image_center), frame.rows}, {0, 150, 0}, 2);
		cv::line(frame, {int(center_track), int(y_ref)}, {int(center_track), frame.rows}, {200, 200, 200}, 2);
		cv::line(frame, {0, int(y_ref)}, {frame.cols, int(y_ref)}, {255, 255, 255}, 1);
		cv::circle(frame, {int(center_track), int(y_ref)}, 5, {255, 0, 0}, -1);

		char buffer[100];
		std::snprintf(buffer, sizeof(buffer), "Lateral error: %.2f deg", erro);
		std::string txt_erro(buffer);
		std::snprintf(buffer, sizeof(buffer), "PID correction: %.2f deg", pid);
		std::string txt_pid(buffer);

		cv::putText(frame, file_name, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);
		cv::putText(frame, txt_erro, {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
		cv::putText(frame, txt_pid, {10, 75}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
		cv::putText(frame, txt_lane, {10, 95}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
	}

	bool extractLanePoints(const cv::Mat &frame, float image_center, float &y_ref, std::vector<cv::Point> &left_point, std::vector<cv::Point> &right_point)
	{
		int height = frame.rows;

		for (int i = 3; i < roi_numbers_in_frame; ++i)
		{
			int roi_y = (height * i) / roi_numbers_in_frame;
			int roi_height = (height * (i + 1)) / roi_numbers_in_frame - roi_y;
			y_ref = roi_y + roi_height / 2;

			cv::Mat roi = frame(cv::Rect(0, roi_y, frame.cols, roi_height)).clone();
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			float best_left = std::numeric_limits<float>::max();
			float best_right = std::numeric_limits<float>::max();
			cv::Point2f center_left, center_right;

			for (const auto &c : contours)
			{
				if (c.size() < 5)
					continue;

				cv::Vec4f line;
				cv::fitLine(c, line, cv::DIST_L2, 0, 0.01, 0.01);
				float vx = line[0], x0 = line[2];
				float slope = line[1] / (vx + 1e-5);
				float dist = std::abs(x0 - image_center);

				if (slope < -0.3f && dist < best_left)
				{
					best_left = dist;
					center_left = {x0, line[3] + roi_y};
				}
				else if (slope > 0.3f && dist < best_right)
				{
					best_right = dist;
					center_right = {x0, line[3] + roi_y};
				}
			}

			if (best_left < std::numeric_limits<float>::max())
				left_point = {{int(center_left.x), int(center_left.y)}, {int(center_left.x), int(center_left.y + 5)}};

			if (best_right < std::numeric_limits<float>::max())
				right_point = {{int(center_right.x), int(center_right.y)}, {int(center_right.x), int(center_right.y + 5)}};

			if (!left_point.empty() || !right_point.empty())
				return true;
		}
		return false;
	}

	float calculateTrackCenter(const std::vector<cv::Point> &left, const std::vector<cv::Point> &right, float y_ref, float displacement_cm, float scale, cv::Mat &frame)
	{
		if (!left.empty() && !right.empty())
		{
			cv::Vec4f l1, l2;
			cv::fitLine(left, l1, cv::DIST_L2, 0, 0.01, 0.01);
			cv::fitLine(right, l2, cv::DIST_L2, 0, 0.01, 0.01);
			float x_left = getXAtY(y_ref, l1[3], l1[2], l1[0], l1[1]);
			float x_right = getXAtY(y_ref, l2[3], l2[2], l2[0], l2[1]);
			cv::circle(frame, {int(x_left), int(y_ref)}, 4, {200, 255, 200}, -1);
			cv::circle(frame, {int(x_right), int(y_ref)}, 4, {200, 100, 255}, -1);
			return (x_left + x_right) / 2.0f;
		}
		if (!right.empty())
		{
			cv::Vec4f l2;
			cv::fitLine(right, l2, cv::DIST_L2, 0, 0.01, 0.01);
			float x_right = getXAtY(y_ref, l2[3], l2[2], l2[0], l2[1]);
			cv::circle(frame, {int(x_right), int(y_ref)}, 4, {200, 100, 255}, -1);
			return x_right - (displacement_cm / scale);
		}
		if (!left.empty())
		{
			cv::Vec4f l1;
			cv::fitLine(left, l1, cv::DIST_L2, 0, 0.01, 0.01);
			float x_left = getXAtY(y_ref, l1[3], l1[2], l1[0], l1[1]);
			cv::circle(frame, {int(x_left), int(y_ref)}, 4, {200, 255, 200}, -1);
			return x_left + (displacement_cm / scale);
		}
		return -1.0f;
	}

	float calculateDangerZoneOccupancy(const cv::Mat &mask,
									   const std::vector<cv::Point> &left_curve,
									   const std::vector<cv::Point> &right_curve,
									   float displacement_cm,
									   float scale)
	{
		// Mark parameters as intentionally unused (for future compatibility)
		(void)displacement_cm;
		(void)scale;

		if (mask.empty() || left_curve.empty() || right_curve.empty())
		{
			return 0.0f;
		}

		// Create danger zone mask
		cv::Mat danger_zone_mask = cv::Mat::zeros(mask.size(), CV_8UC1);

		// Define the danger zone as the area between lanes in the lower half of the image
		int height = mask.rows;
		int start_y = height / 2; // Start from the middle of the image

		// Create points for the danger zone
		std::vector<cv::Point> danger_zone_points;

		// Add points from left lane (from middle down)
		for (const auto &point : left_curve)
		{
			if (point.y >= start_y)
			{
				danger_zone_points.push_back(point);
			}
		}

		// Add points from right lane (from middle down)
		for (const auto &point : right_curve)
		{
			if (point.y >= start_y)
			{
				danger_zone_points.push_back(point);
			}
		}

		// If we don't have enough points, return 0
		if (danger_zone_points.size() < 4)
		{
			return 0.0f;
		}

		// Create danger zone polygon
		cv::fillPoly(danger_zone_mask, std::vector<std::vector<cv::Point>>{danger_zone_points}, cv::Scalar(255));

		// Calculate total area of danger zone
		int total_danger_area = cv::countNonZero(danger_zone_mask);
		if (total_danger_area == 0)
		{
			return 0.0f;
		}

		// Apply danger zone mask to original mask
		cv::Mat masked_obstacles;
		cv::bitwise_and(mask, danger_zone_mask, masked_obstacles);

		// Count white pixels (obstacles) in danger zone
		int obstacle_pixels = cv::countNonZero(masked_obstacles);

		// Calculate occupancy percentage
		float occupancy = static_cast<float>(obstacle_pixels) / static_cast<float>(total_danger_area);

		return occupancy;
	}

	// Scans the lower half of the mask and extracts lane curves.
	// min_run helps ignore noise by requiring at least N contiguous white pixels.
	bool sampleLaneEdgesByRow(const cv::Mat &mask,
							  int y_start, int y_end, int step,
							  float image_center,
							  std::vector<cv::Point> &left_curve,
							  std::vector<cv::Point> &right_curve,
							  int min_run)
	{
		CV_Assert(mask.type() == CV_8UC1);
		y_start = std::max(0, y_start);
		y_end = std::min(mask.rows, y_end);
		left_curve.clear();
		right_curve.clear();

		const int width = mask.cols;
		bool any = false;

		for (int y = y_start; y < y_end; y += step)
		{
			const uchar *row = mask.ptr<uchar>(y);

			// --- search left lane (left half; from center to left) ---
			int run = 0, sumx = 0;
			int x_left = -1;
			for (int x = int(image_center) - 1; x >= 0; --x)
			{
				if (row[x] > 0)
				{
					run++;
					sumx += x;
				}
				else
				{
					if (run >= min_run)
					{
						x_left = sumx / run;
						break;
					}
					run = 0;
					sumx = 0;
				}
			}
			if (x_left == -1 && run >= min_run)
				x_left = sumx / run;

			// --- search right lane (right half; from center to right) ---
			run = 0;
			sumx = 0;
			int x_right = -1;
			for (int x = int(image_center); x < width; ++x)
			{
				if (row[x] > 0)
				{
					run++;
					sumx += x;
				}
				else
				{
					if (run >= min_run)
					{
						x_right = sumx / run;
						break;
					}
					run = 0;
					sumx = 0;
				}
			}
			if (x_right == -1 && run >= min_run)
				x_right = sumx / run;

			if (x_left >= 0)
			{
				left_curve.emplace_back(x_left, y);
				any = true;
			}
			if (x_right >= 0)
			{
				right_curve.emplace_back(x_right, y);
				any = true;
			}
		}
		return any && (!left_curve.empty() || !right_curve.empty());
	}

	// Draws the curved danger zone between the two curves.
	// If only one lane exists, estimates the other by offsetting by lane_width_pixels.
	void drawDangerZoneCurved(cv::Mat &frame,
							  const std::vector<cv::Point> &left_curve,
							  const std::vector<cv::Point> &right_curve,
							  float displacement_cm, float scale)
	{
		if (frame.empty())
			return;

		const int start_y = frame.rows * 2 / 5; // Lower 3/5 of frame
		const int width = frame.cols;
		const int end_y = frame.rows - 1;
		const int min_pts = 8; // avoids degenerate polygons
		const int lane_width_pixels = std::max(1, int(displacement_cm / scale));

		std::vector<cv::Point> poly;

		if (!left_curve.empty() && !right_curve.empty())
		{
			// left: from bottom to top (large y -> small)
			for (auto it = left_curve.rbegin(); it != left_curve.rend(); ++it)
				if (it->y >= start_y && it->y <= end_y)
					poly.push_back(*it);

			// right: from top to bottom (small y -> large)
			for (const auto &p : right_curve)
				if (p.y >= start_y && p.y <= end_y)
					poly.push_back(p);
		}
		else if (!right_curve.empty())
		{
			// only right → estimate left by offset
			for (auto it = right_curve.rbegin(); it != right_curve.rend(); ++it)
			{
				if (it->y >= start_y && it->y <= end_y)
				{
					int xl = std::max(0, it->x - lane_width_pixels);
					poly.emplace_back(xl, it->y);
				}
			}
			for (const auto &p : right_curve)
				if (p.y >= start_y && p.y <= end_y)
					poly.push_back(p);
		}
		else if (!left_curve.empty())
		{
			// only left → estimate right by offset
			for (const auto &p : left_curve)
				if (p.y >= start_y && p.y <= end_y)
					poly.push_back(p);

			for (auto it = left_curve.rbegin(); it != left_curve.rend(); ++it)
			{
				if (it->y >= start_y && it->y <= end_y)
				{
					int xr = std::min(width - 1, it->x + lane_width_pixels);
					poly.emplace_back(xr, it->y);
				}
			}
		}

		if ((int)poly.size() >= min_pts)
		{
			if (frame.channels() == 1)
				cv::fillPoly(frame, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
			else
				cv::fillPoly(frame, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(0, 0, 255));
		}
	}

	// Creates a separate mask only for the danger zone
	void createDangerZoneMask(const cv::Mat &original_mask,
							  const std::vector<cv::Point> &left_curve,
							  const std::vector<cv::Point> &right_curve,
							  float displacement_cm, float scale,
							  cv::Mat &danger_zone_mask)
	{
		// Initialize danger zone mask as zeros
		danger_zone_mask = cv::Mat::zeros(original_mask.size(), CV_8UC1);

		if (original_mask.empty())
			return;

		const int start_y = original_mask.rows * 2 / 5; // Lower 3/5 of frame
		const int width = original_mask.cols;
		const int end_y = original_mask.rows - 1;
		const int min_pts = 8; // avoids degenerate polygons
		const int lane_width_pixels = std::max(1, int(displacement_cm / scale));

		std::vector<cv::Point> poly;

		if (!left_curve.empty() && !right_curve.empty())
		{
			// left: from bottom to top (large y -> small)
			for (auto it = left_curve.rbegin(); it != left_curve.rend(); ++it)
				if (it->y >= start_y && it->y <= end_y)
					poly.push_back(*it);

			// right: from top to bottom (small y -> large)
			for (const auto &p : right_curve)
				if (p.y >= start_y && p.y <= end_y)
					poly.push_back(p);
		}
		else if (!right_curve.empty())
		{
			// only right → estimate left by offset
			for (auto it = right_curve.rbegin(); it != right_curve.rend(); ++it)
			{
				if (it->y >= start_y && it->y <= end_y)
				{
					int xl = std::max(0, it->x - lane_width_pixels);
					poly.emplace_back(xl, it->y);
				}
			}
			for (const auto &p : right_curve)
				if (p.y >= start_y && p.y <= end_y)
					poly.push_back(p);
		}
		else if (!left_curve.empty())
		{
			// only left → estimate right by offset
			for (const auto &p : left_curve)
				if (p.y >= start_y && p.y <= end_y)
					poly.push_back(p);

			for (auto it = left_curve.rbegin(); it != left_curve.rend(); ++it)
			{
				if (it->y >= start_y && it->y <= end_y)
				{
					int xr = std::min(width - 1, it->x + lane_width_pixels);
					poly.emplace_back(xr, it->y);
				}
			}
		}

		// Create filled danger zone polygon
		if ((int)poly.size() >= min_pts)
		{
			cv::fillPoly(danger_zone_mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
		}

		// DO NOT apply bitwise_and - we want to show the filled danger zone, not just the obstacles
		// The mask now shows the complete danger zone area in white
	}

	// Calculates danger zone occupancy based on the danger zone mask
	float calculateDangerZoneOccupancyFromMask(const cv::Mat &original_mask,
											   const cv::Mat &danger_zone_mask)
	{
		if (original_mask.empty() || danger_zone_mask.empty())
		{
			return 0.0f;
		}

		// Calculate total danger zone area (white pixels in danger zone mask)
		int total_danger_area = cv::countNonZero(danger_zone_mask);
		if (total_danger_area == 0)
		{
			return 0.0f;
		}

		// Apply danger zone mask to original mask to get only obstacles in the zone
		cv::Mat obstacles_in_danger_zone;
		cv::bitwise_and(original_mask, danger_zone_mask, obstacles_in_danger_zone);

		// Count white pixels (obstacles) in danger zone
		int obstacle_pixels = cv::countNonZero(obstacles_in_danger_zone);

		// Calculate occupancy percentage
		float occupancy = static_cast<float>(obstacle_pixels) / static_cast<float>(total_danger_area);

		return occupancy;
	}

	// ====== FUNCTIONS FOR DRIVABLE AREA DANGER ZONE ======

	// Creates a danger zone mask based on the drivable area
	void createDrivableDangerZoneMask(const cv::Mat &drivable_mask,
									  float displacement_cm, float scale,
									  cv::Mat &drivable_danger_zone_mask)
	{
		// Initialize danger zone mask as zeros
		drivable_danger_zone_mask = cv::Mat::zeros(drivable_mask.size(), CV_8UC1);

		// Always create danger zone, even if drivable mask is empty
		// This allows detecting obstacles even when no drivable area is detected

		const int start_y = drivable_mask.rows * 2 / 5; // Lower 3/5 of frame
		const int end_y = drivable_mask.rows - 1;
		const int width = drivable_mask.cols;
		const int center_x = width / 2;
		const int danger_width_pixels = std::max(10, int(displacement_cm / scale));

		// Create rectangular danger zone in the lower central part
		cv::Rect danger_zone_rect(
			center_x - danger_width_pixels / 2, // x
			start_y,							// y
			danger_width_pixels,				// width
			end_y - start_y						// height
		);

		// Ensure rectangle is within image bounds
		danger_zone_rect &= cv::Rect(0, 0, width, drivable_mask.rows);

		// Fill danger zone in mask
		if (danger_zone_rect.area() > 0)
		{
			cv::rectangle(drivable_danger_zone_mask, danger_zone_rect, cv::Scalar(255), -1);
		}
	}

	// Calculates drivable danger zone occupancy (detected obstacles)
	float calculateDrivableDangerZoneOccupancy(const cv::Mat &drivable_mask,
											   const cv::Mat &drivable_danger_zone_mask)
	{
		if (drivable_danger_zone_mask.empty())
		{
			return 0.0f;
		}

		// Calculate total danger zone area
		int total_danger_area = cv::countNonZero(drivable_danger_zone_mask);
		if (total_danger_area == 0)
		{
			return 0.0f;
		}

		// If drivable mask is empty, consider entire danger zone as obstacle
		if (drivable_mask.empty() || cv::countNonZero(drivable_mask) == 0)
		{
			// Entire danger zone is considered obstacle (100% occupancy)
			return 1.0f;
		}

		// Detect obstacles: areas where drivable mask is 0 (not drivable) within danger zone
		cv::Mat obstacles_in_danger_zone;
		cv::bitwise_and(~drivable_mask, drivable_danger_zone_mask, obstacles_in_danger_zone);

		// Count obstacle pixels in danger zone
		int obstacle_pixels = cv::countNonZero(obstacles_in_danger_zone);

		// Calculate occupancy percentage
		float occupancy = static_cast<float>(obstacle_pixels) / static_cast<float>(total_danger_area);

		return occupancy;
	}

	// Draws drivable danger zone on frame
	void drawDrivableDangerZone(cv::Mat &frame,
								const cv::Mat &drivable_danger_zone_mask)
	{
		if (frame.empty() || drivable_danger_zone_mask.empty())
			return;

		// Convert to 3 channels if necessary
		cv::Mat frame_3ch;
		if (frame.channels() == 1)
		{
			cv::cvtColor(frame, frame_3ch, cv::COLOR_GRAY2BGR);
		}
		else
		{
			frame_3ch = frame;
		}

		// Create colored mask for drivable danger zone (blue)
		cv::Mat danger_overlay = cv::Mat::zeros(frame_3ch.size(), CV_8UC3);
		danger_overlay.setTo(cv::Scalar(255, 0, 0), drivable_danger_zone_mask); // BGR: blue

		// Apply transparency and overlay
		cv::addWeighted(frame_3ch, 0.7, danger_overlay, 0.3, 0, frame_3ch);

		// Copy back to original frame
		if (frame.channels() == 1)
		{
			cv::cvtColor(frame_3ch, frame, cv::COLOR_BGR2GRAY);
		}
		else
		{
			frame = frame_3ch;
		}
	}
} // namespace jetracer::vision
