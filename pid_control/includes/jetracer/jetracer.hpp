#ifndef JETRACER_HPP
#define JETRACER_HPP

#include <cstdint>
#include <string>
#include <atomic>
#include <iostream>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include "jetracer/i2c_device.hpp"

namespace jetracer::control
{
	class JetRacer
	{
	public:
		JetRacer(int servo_addr, int motor_addr);
		~JetRacer();

		void start();
		void stop();
		bool is_running() const;
		void set_speed(float speed);
		void set_steering(int angle);
		void smooth_steering(int target_angle, int increment);
		int servo_delay_ms_ = 30;

	private:
		void init_servo();
		void init_motors();
		void set_servo_pwm(int channel, int on_value, int off_value);
		void set_motor_pwm(int channel, int value);
		void process_joystick();

		static constexpr int MAX_ANGLE_ = 140;
		static constexpr int SERVO_LEFT_PWM_ = 140;
		static constexpr int SERVO_CENTER_PWM_ = 280;
		static constexpr int SERVO_RIGHT_PWM_ = 420;

		int servo_addr_;
		int motor_addr_;
		std::atomic<bool> running_;
		hardware::I2CDevice servo_device_;
		hardware::I2CDevice motor_device_;
		int current_angle_ = 0;
		float current_speed_ = 0.0f;
	};
} // namespace jetracer::control

#endif // JETRACER_HPP
