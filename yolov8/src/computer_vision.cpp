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
} // namespace jetracer::vision
