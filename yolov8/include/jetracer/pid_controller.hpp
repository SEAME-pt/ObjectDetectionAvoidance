#ifndef PID_CONTROLLER_HPP
#define PID_CONTROLLER_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace jetracer::pid
{
	struct PIDStatus
	{
		float integral_error = 0.0f;
		float previous_error = 0.0f;
	};
	float PIDapply(float error, float dt, PIDStatus &status);
	float PIDexecute(const cv::Mat &original_frame, const std::string &base_name);
} // namespace jetracer::pid

#endif // PID_CONTROLLER_HPP
