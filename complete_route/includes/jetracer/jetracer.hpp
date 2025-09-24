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
#include <deque>
#include <array>
#include <chrono>
#include <thread>
#include "jetracer/i2c_device.hpp"
#include "jetracer/pwm_config.hpp"
#include "jetracer/motor_control.hpp"

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

		void set_constant_speed_mode(bool enabled);
		void set_test_speed(float speed_percent);
		void set_test_duration(int seconds);
		void start_speed_test();
		void stop_speed_test();
		bool is_test_mode() const { return test_mode_; }

		int servo_delay_ms_ = 30;

		static constexpr int PWM_FREQUENCY_HZ = pwm::frequency::MOTOR_MED_FREQ;
		static constexpr int SPEED_SMOOTHING_WINDOW = pwm::smoothing::BALANCED;
		static constexpr float MAX_SPEED_CHANGE_PER_UPDATE = pwm::smoothing::MEDIUM_CHANGE;

	private:
		void init_servo();
		void init_motors();
		void set_servo_pwm(int channel, int on_value, int off_value);
		void set_motor_pwm(int channel, int value);
		void set_motor_pwm_smooth(int channel, int value);
		void process_joystick();
		void process_test_mode();
		float smooth_speed(float target_speed);
		float calculate_safe_speed(float target_speed);

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

		std::deque<float> speed_history_;
		float smoothed_speed_ = 0.0f;

		float filtered_speed_ = 0.0f;
		float target_speed_ = 0.0f;
		float last_speed_command_ = 0.0f;
		unsigned long last_movement_time_ = 0;

		std::atomic<bool> test_mode_{false};
		std::atomic<float> test_speed_{0.0f};
		std::atomic<int> test_duration_{0};
		std::atomic<bool> test_running_{false};
		std::chrono::steady_clock::time_point test_start_time_;
		std::thread test_thread_;
	};
}

#endif
