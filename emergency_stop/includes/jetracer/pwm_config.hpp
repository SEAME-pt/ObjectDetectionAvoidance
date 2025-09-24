#ifndef PWM_CONFIG_HPP
#define PWM_CONFIG_HPP

namespace jetracer::pwm
{
	// PWM frequency configurations
	namespace frequency
	{
		// Recommended frequencies for different applications
		static constexpr int MOTOR_LOW_FREQ = 500;	 // 500 Hz - For low power motors
		static constexpr int MOTOR_MED_FREQ = 1000;	 // 1000 Hz - Standard for DC motors
		static constexpr int MOTOR_HIGH_FREQ = 2000; // 2000 Hz - For high precision motors
		static constexpr int SERVO_FREQ = 50;		 // 50 Hz - Standard for servos
	}

	// Smoothing configurations
	namespace smoothing
	{
		// Smoothing windows for different types of movement
		static constexpr int AGGRESSIVE = 3;	// Fast response, less smooth
		static constexpr int BALANCED = 5;		// Balance between smoothness and response
		static constexpr int SMOOTH = 7;		// Very smooth, slower response
		static constexpr int ULTRA_SMOOTH = 10; // Extremely smooth

		// Speed change limits per update
		static constexpr float FAST_CHANGE = 3.0f;		 // Fast change (reduced)
		static constexpr float MEDIUM_CHANGE = 1.5f;	 // Medium change (default, reduced)
		static constexpr float SLOW_CHANGE = 0.8f;		 // Slow change (reduced)
		static constexpr float ULTRA_SLOW_CHANGE = 0.3f; // Very slow change (reduced)
	}

	// Timing configurations
	namespace timing
	{
		// Control loop update frequencies
		static constexpr int JOYSTICK_UPDATE_MS = 25; // 40 Hz - Joystick (more time to process frames)
		static constexpr int PID_UPDATE_MS = 30;	  // 33 Hz - PID (more time to process)
		static constexpr int MOTOR_UPDATE_MS = 10;	  // 100 Hz - Motors (reduced to give more time)
	}

	// Safety configurations
	namespace safety
	{
		static constexpr float MAX_SPEED_PERCENT = 46.0f;
		static constexpr float EMERGENCY_STOP_DELAY_MS = 150.0f; // Delay for emergency stop (increased)
	}
}

#endif // PWM_CONFIG_HPP
