#ifndef MOTOR_CONTROL_HPP
#define MOTOR_CONTROL_HPP

namespace jetracer::motor_control
{
	// Deadzone and threshold configurations
	namespace thresholds
	{
		static constexpr float SPEED_DEADZONE = 0.001f;		   // 0.1% - Minimum deadzone for maximum responsiveness
		static constexpr int MIN_PWM_THRESHOLD = 500;		   // Reduced minimum PWM for slower car (12%)
		static constexpr int TORQUE_BOOST_THRESHOLD = 1200;	   // Reduced threshold for less amplification
		static constexpr float LOW_SPEED_AMPLIFICATION = 1.8f; // Reduced amplification for slower car
	}

	// Power curve configurations
	namespace power
	{
		static constexpr float POWER_CURVE_FACTOR = 0.7f;	   // Exponential power curve factor (reduced for slower car)
		static constexpr float SPEED_SMOOTHING_FACTOR = 0.88f; // Speed smoothing factor (reduced)
	}

	// Acceleration ramp configurations
	namespace ramps
	{
		static constexpr float ACCELERATION_RAMP = 8.0f;		  // Much increased acceleration rate (% per update)
		static constexpr float DECELERATION_RAMP = 15.0f;		  // Much increased deceleration rate (% per update)
		static constexpr float EMERGENCY_BRAKE_THRESHOLD = 50.0f; // Threshold for emergency braking
	}

	// Safety configurations
	namespace safety
	{
		static constexpr float DIRECTION_CHANGE_THRESHOLD = 10.0f; // Threshold to detect direction change
		static constexpr float MAX_SPEED_PERCENT = 100.0f;		   // Maximum allowed speed (without artificial limitation)

		// Speed limitations for curve safety
		static constexpr float BASE_MAX_SPEED = 40.0f;
		static constexpr float CURVE_SPEED_REDUCTION = 1.1f;	 // Speed boost in curves (110% of base speed) - reduced
		static constexpr float STRAIGHT_SPEED_BOOST = 1.05f;	 // No boost in straights (100% of base speed)
		static constexpr float STEERING_ANGLE_THRESHOLD = 25.0f; // Steering angle to consider as "curve" - increased to be less sensitive
	}
}

#endif // MOTOR_CONTROL_HPP
