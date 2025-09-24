#ifndef PWM_CONFIG_HPP
#define PWM_CONFIG_HPP

namespace jetracer::pwm
{
	// PWM frequency settings
	namespace frequency
	{
		static constexpr int MOTOR_LOW_FREQ = 500;
		static constexpr int MOTOR_MED_FREQ = 1000;
		static constexpr int MOTOR_HIGH_FREQ = 2000;
		static constexpr int SERVO_FREQ = 50;
	}

	// Smoothing settings
	namespace smoothing
	{
		// Smoothing windows for different types of movement
		static constexpr int AGGRESSIVE = 3;
		static constexpr int BALANCED = 5;
		static constexpr int SMOOTH = 7;
		static constexpr int ULTRA_SMOOTH = 10;

		// Speed change limits per update
		static constexpr float FAST_CHANGE = 3.0f;
		static constexpr float MEDIUM_CHANGE = 1.5f;
		static constexpr float SLOW_CHANGE = 0.8f;
		static constexpr float ULTRA_SLOW_CHANGE = 0.3f;
	}

	// Timing settings
	namespace timing
	{
		// Control loop update frequencies
		static constexpr int JOYSTICK_UPDATE_MS = 25;
		static constexpr int PID_UPDATE_MS = 30;
		static constexpr int MOTOR_UPDATE_MS = 10;
	}

	// Safety settings
	namespace safety
	{
		static constexpr float MAX_SPEED_PERCENT = 46.0f;
		static constexpr float EMERGENCY_STOP_DELAY_MS = 150.0f;
	}
}

#endif // PWM_CONFIG_HPP
