#include "jetracer/pid_controller.hpp"
#include "jetracer/computer_vision.hpp"
#include <algorithm>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace jetracer::pid
{
	constexpr float Kp = 1.5f;
	constexpr float Ki = 0.1f;
	constexpr float Kd = 0.2f;
	constexpr float MAX_ANGLE = 140.0f;
	constexpr float displacement_cm = 21.0f;

	float PIDapply(float error, float dt, PIDStatus &status)
	{
		status.integral_error += error * dt;
		float derivative = (error - status.previous_error) / dt;
		status.previous_error = error;

		float output = Kp * error + Ki * status.integral_error + Kd * derivative;
		return std::clamp(output, -MAX_ANGLE, MAX_ANGLE);
	}

	float PIDexecute(const cv::Mat &original_frame, const std::string &base_name)
	{
		if (original_frame.empty() || original_frame.channels() != 1)
		{
			std::cerr << "Invalid input image." << std::endl;
			return 0.0f;
		}

		cv::Mat frame = original_frame.clone();
		const float image_center = frame.cols / 2.0f;
		const float scale = 40.0f / (frame.cols / 2);

		PIDStatus pid_state;
		std::vector<cv::Point> left, right;
		float y_ref = -1.0f;

		bool found = jetracer::vision::extractLanePoints(frame, image_center, y_ref, left, right);
		if (!found)
		{
			std::cerr << "No lane detected." << std::endl;
			return 0.0f;
		}

		std::string status;
		if (left.empty() && right.empty())
			status = "none";
		else
		{
			if (!left.empty())
				status += "left";
			if (!right.empty())
				status += "right";
			status = "Lane detected: " + status;
		}

		float center_track = jetracer::vision::calculateTrackCenter(left, right, y_ref, displacement_cm, scale, frame);
		float lateral_error = image_center - center_track;
		float error = lateral_error * scale;
		float pid_angle = -PIDapply(error, 0.1f, pid_state);

		std::cout << "Lateral error: " << error << " degrees" << std::endl;
		std::cout << "PID correction: " << pid_angle << " degrees" << std::endl;

		jetracer::vision::draw_overlay(frame, error, pid_angle, base_name, status, image_center, center_track, y_ref);

		fs::create_directories("outputs");
		cv::imwrite("outputs/" + base_name, frame);
		cv::imshow("Image with PID", frame);

		return pid_angle;
	}
} // namespace jetracer::pid
