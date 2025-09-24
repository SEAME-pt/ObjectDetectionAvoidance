#include "jetracer/jetracer.hpp"
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <cstring>

namespace jetracer::control
{
	JetRacer::JetRacer(int servo_addr, int motor_addr)
		: servo_addr_(servo_addr),
		  motor_addr_(motor_addr),
		  running_(false),
		  servo_device_("/dev/i2c-1", servo_addr),
		  motor_device_("/dev/i2c-1", motor_addr)
	{
		init_servo();
		init_motors();
	}

	JetRacer::~JetRacer()
	{
		stop();
	}

	void JetRacer::init_servo()
	{
		try
		{
			servo_device_.write_byte(0x00, 0x06);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			servo_device_.write_byte(0x00, 0x10);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			servo_device_.write_byte(0xFE, 0x79);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			servo_device_.write_byte(0x01, 0x04);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			servo_device_.write_byte(0x00, 0x20);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		catch (const std::exception &e)
		{
			std::cerr << "Servo initialization failed: " << e.what() << std::endl;
			stop();
		}
	}

	void JetRacer::init_motors()
	{
		try
		{
			motor_device_.write_byte(0x00, 0x20);

			int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / PWM_FREQUENCY_HZ - 1));
			int oldmode = motor_device_.read_byte(0x00);
			int newmode = (oldmode & 0x7F) | 0x10;

			motor_device_.write_byte(0x00, newmode);
			motor_device_.write_byte(0xFE, prescale);
			motor_device_.write_byte(0x00, oldmode);
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			motor_device_.write_byte(0x00, oldmode | 0xA1);

			std::cout << "[INFO] Motors initialized with PWM frequency: " << PWM_FREQUENCY_HZ << " Hz" << std::endl;
			std::cout << "[INFO] Calculated prescale: " << prescale << std::endl;
		}
		catch (const std::exception &e)
		{
			std::cerr << "Motor initialization failed: " << e.what() << std::endl;
			stop();
		}
	}

	void JetRacer::set_steering(int angle)
	{
		angle = std::clamp(angle, -MAX_ANGLE_, MAX_ANGLE_);

		int pwm = 0;
		if (angle < 0)
		{
			std::cout << "Setting steering to left: " << angle << std::endl;
			pwm = SERVO_CENTER_PWM_ + (angle / static_cast<float>(MAX_ANGLE_)) * (SERVO_CENTER_PWM_ - SERVO_LEFT_PWM_);
		}
		else if (angle > 0)
		{
			pwm = SERVO_CENTER_PWM_ + (angle / static_cast<float>(MAX_ANGLE_)) * (SERVO_RIGHT_PWM_ - SERVO_CENTER_PWM_);
			std::cout << "Setting steering to right: " << angle << std::endl;
		}
		else
		{
			pwm = SERVO_CENTER_PWM_;
			std::cout << "Setting steering to center: " << angle << std::endl;
		}

		set_servo_pwm(0, 0, pwm);
		current_angle_ = angle;

		std::this_thread::sleep_for(std::chrono::milliseconds(servo_delay_ms_));
	}

	void JetRacer::smooth_steering(int target_angle, int increment)
	{
		target_angle = std::clamp(target_angle, -MAX_ANGLE_, MAX_ANGLE_);
		int step = (target_angle > current_angle_) ? increment : -increment;

		while ((step > 0 && current_angle_ < target_angle) || (step < 0 && current_angle_ > target_angle))
		{
			current_angle_ += step;
			if ((step > 0 && current_angle_ > target_angle) || (step < 0 && current_angle_ < target_angle))
			{
				current_angle_ = target_angle;
			}
			set_steering(current_angle_);
		}
	}

	void JetRacer::set_servo_pwm(int channel, int on_value, int off_value)
	{
		int base_reg = 0x06 + (channel * 4);
		servo_device_.write_byte(base_reg, on_value & 0xFF);
		servo_device_.write_byte(base_reg + 1, on_value >> 8);
		servo_device_.write_byte(base_reg + 2, off_value & 0xFF);
		servo_device_.write_byte(base_reg + 3, off_value >> 8);
	}

	void JetRacer::set_motor_pwm(int channel, int value)
	{
		value = std::clamp(value, 0, 4095);
		int base_reg = 0x06 + (channel * 4);
		motor_device_.write_byte(base_reg, 0);
		motor_device_.write_byte(base_reg + 1, 0);
		motor_device_.write_byte(base_reg + 2, value & 0xFF);
		motor_device_.write_byte(base_reg + 3, value >> 8);
	}

	float JetRacer::smooth_speed(float target_speed)
	{

		speed_history_.push_back(target_speed);

		if (speed_history_.size() > SPEED_SMOOTHING_WINDOW)
		{
			speed_history_.pop_front();
		}

		float sum = 0.0f;
		for (float speed : speed_history_)
		{
			sum += speed;
		}
		float average_speed = sum / speed_history_.size();

		float max_change = MAX_SPEED_CHANGE_PER_UPDATE;
		float speed_diff = average_speed - smoothed_speed_;

		if (std::abs(speed_diff) > max_change)
		{
			if (speed_diff > 0)
			{
				smoothed_speed_ += max_change;
			}
			else
			{
				smoothed_speed_ -= max_change;
			}
		}
		else
		{
			smoothed_speed_ = average_speed;
		}

		return smoothed_speed_;
	}

	float JetRacer::calculate_safe_speed(float target_speed)
	{

		float max_safe_speed = motor_control::safety::BASE_MAX_SPEED;

		bool is_turning = std::abs(current_angle_) > motor_control::safety::STEERING_ANGLE_THRESHOLD;

		bool is_car_stopped = (std::abs(current_speed_) < 1.0f);

		if (is_turning && !is_car_stopped)
		{

			max_safe_speed *= motor_control::safety::CURVE_SPEED_REDUCTION;

			static int curve_debug_counter = 0;
			if ((++curve_debug_counter % 50) == 0)
			{
				std::cout << "[CURVE] Angle: " << current_angle_
						  << "°, Max speed: " << max_safe_speed << "% (boost applied to overcome resistance)" << std::endl;
			}
		}
		else if (!is_turning)
		{

			max_safe_speed *= motor_control::safety::STRAIGHT_SPEED_BOOST;

			static int straight_debug_counter = 0;
			if ((++straight_debug_counter % 100) == 0)
			{
				std::cout << "[STRAIGHT] Angle: " << current_angle_
						  << "°, Max speed: " << max_safe_speed << "% (normal speed)" << std::endl;
			}
		}
		else
		{

			if (is_car_stopped && is_turning)
			{
				static int startup_curve_debug_counter = 0;
				if ((++startup_curve_debug_counter % 30) == 0)
				{
					std::cout << "[STARTUP_CURVE] Car stopped in turn - allowing maximum speed to overcome inertia of turned wheels" << std::endl;
				}
			}
		}

		if (std::abs(target_speed) > max_safe_speed)
		{

			float sign = (target_speed > 0) ? 1.0f : -1.0f;
			target_speed = sign * max_safe_speed;

			static int limit_debug_counter = 0;
			if ((++limit_debug_counter % 30) == 0)
			{
				std::cout << "[SAFETY] Speed limited to " << target_speed
						  << "% (max safe: " << max_safe_speed << "%)" << std::endl;
			}
		}

		return target_speed;
	}

	void JetRacer::set_motor_pwm_smooth(int channel, int value)
	{

		value = std::clamp(value, 0, 4095);
		int base_reg = 0x06 + (channel * 4);

		bool is_car_stopped = (std::abs(current_speed_) < 1.0f);

		if (is_car_stopped)
		{

			motor_device_.write_byte(base_reg, 0);
			motor_device_.write_byte(base_reg + 1, 0);
			motor_device_.write_byte(base_reg + 2, value & 0xFF);
			motor_device_.write_byte(base_reg + 3, value >> 8);
			return;
		}

		static std::array<int, 9> last_pwm_values = {0};
		static std::array<int, 9> target_pwm_values = {0};

		target_pwm_values[channel] = value;

		int current_pwm = last_pwm_values[channel];
		int pwm_diff = target_pwm_values[channel] - current_pwm;

		int max_pwm_change = 200;
		if (std::abs(pwm_diff) > max_pwm_change)
		{
			if (pwm_diff > 0)
			{
				current_pwm += max_pwm_change;
			}
			else
			{
				current_pwm -= max_pwm_change;
			}
		}
		else
		{
			current_pwm = target_pwm_values[channel];
		}

		last_pwm_values[channel] = current_pwm;

		motor_device_.write_byte(base_reg, 0);
		motor_device_.write_byte(base_reg + 1, 0);
		motor_device_.write_byte(base_reg + 2, current_pwm & 0xFF);
		motor_device_.write_byte(base_reg + 3, current_pwm >> 8);
	}

	void JetRacer::set_speed(float speed)
	{

		static int set_speed_debug_counter = 0;

		if ((++set_speed_debug_counter % 20) == 0)
		{
			std::cout << "[DEBUG] set_speed() - Requested speed: " << speed << "%" << std::endl;
		}

		float original_speed = speed;
		speed = calculate_safe_speed(speed);

		unsigned long current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
										 std::chrono::steady_clock::now().time_since_epoch())
										 .count();

		if (std::abs(original_speed) != std::abs(speed) && set_speed_debug_counter % 20 == 0)
		{
			std::cout << "[SAFETY] Speed adjusted from " << original_speed
					  << "% to " << speed << "% for safety" << std::endl;
		}

		bool emergency_brake = false;
		float speed_reduction = std::abs(speed) - std::abs(last_speed_command_);

		if (last_speed_command_ != 0.0f && speed_reduction < -motor_control::ramps::EMERGENCY_BRAKE_THRESHOLD)
		{
			emergency_brake = true;
		}

		if ((last_speed_command_ > motor_control::safety::DIRECTION_CHANGE_THRESHOLD && speed < -motor_control::safety::DIRECTION_CHANGE_THRESHOLD) ||
			(last_speed_command_ < -motor_control::safety::DIRECTION_CHANGE_THRESHOLD && speed > motor_control::safety::DIRECTION_CHANGE_THRESHOLD))
		{
			emergency_brake = true;
		}

		if (std::abs(speed) < motor_control::thresholds::SPEED_DEADZONE * 100.0f)
		{
			speed = 0.0f;
		}

		bool is_car_stopped = (std::abs(current_speed_) < 1.0f);

		if (is_car_stopped && std::abs(speed) > 5.0f && set_speed_debug_counter % 10 == 0)
		{
			std::cout << "[STARTUP] Car stopped - applying direct acceleration: " << speed << "%" << std::endl;
		}

		float power_curve = 1.0f;
		if (std::abs(speed) > 0.0f)
		{

			power_curve = 1.0f + (std::abs(speed) / 100.0f) * motor_control::power::POWER_CURVE_FACTOR;
			speed *= power_curve;
		}

		speed = std::max(-100.0f, std::min(speed, 100.0f));

		if (emergency_brake)
		{

			filtered_speed_ = speed;
			target_speed_ = speed;
		}
		else if (is_car_stopped)
		{

			filtered_speed_ = speed;
			target_speed_ = speed;
			if (set_speed_debug_counter % 10 == 0)
			{
				std::cout << "[STARTUP] Car really stopped - applying direct acceleration: " << speed << "%" << std::endl;
			}
		}
		else if (std::abs(speed - current_speed_) > 5.0f)
		{

			filtered_speed_ = speed;
			target_speed_ = speed;
			if (set_speed_debug_counter % 10 == 0)
			{
				std::cout << "[RESPONSIVE] Sudden change detected - applying direct command: " << speed << "%" << std::endl;
			}
		}
		else
		{

			filtered_speed_ = motor_control::power::SPEED_SMOOTHING_FACTOR * speed +
							  (1.f - motor_control::power::SPEED_SMOOTHING_FACTOR) * filtered_speed_;
		}

		if (!emergency_brake && !is_car_stopped && std::abs(speed - current_speed_) <= 5.0f)
		{

			float speed_diff = filtered_speed_ - target_speed_;
			float ramp_rate;

			bool is_braking = (filtered_speed_ < target_speed_ && target_speed_ > 0) ||
							  (filtered_speed_ > target_speed_ && target_speed_ < 0) ||
							  (std::abs(filtered_speed_) < std::abs(target_speed_));

			if (is_braking)
			{

				ramp_rate = motor_control::ramps::DECELERATION_RAMP;
			}
			else
			{

				ramp_rate = motor_control::ramps::ACCELERATION_RAMP;
			}

			if (std::abs(speed_diff) > ramp_rate)
			{
				target_speed_ += (speed_diff > 0 ? ramp_rate : -ramp_rate);
			}
			else
			{
				target_speed_ = filtered_speed_;
			}
		}
		else
		{

			target_speed_ = filtered_speed_;
			if (set_speed_debug_counter % 10 == 0)
			{
				std::cout << "[DIRECT] Applying direct command (no ramp): " << filtered_speed_ << "%" << std::endl;
			}
		}

		int pwm_value = static_cast<int>(std::abs(target_speed_) / 100.0f * 4095);

		if (set_speed_debug_counter % 20 == 0)
		{
			std::cout << "[DEBUG] set_speed() - Calculated PWM: " << pwm_value
					  << " for speed: " << target_speed_ << "%" << std::endl;
		}

		if (pwm_value > 0 && pwm_value < motor_control::thresholds::MIN_PWM_THRESHOLD)
		{
			pwm_value = motor_control::thresholds::MIN_PWM_THRESHOLD;
		}

		if (pwm_value > 0 && pwm_value < 1000)
		{
			pwm_value = static_cast<int>(pwm_value * 2.0f);
		}

		if (pwm_value > 0 && pwm_value < motor_control::thresholds::TORQUE_BOOST_THRESHOLD)
		{
			pwm_value = static_cast<int>(pwm_value * motor_control::thresholds::LOW_SPEED_AMPLIFICATION);
		}

		if (is_car_stopped && pwm_value > 0)
		{

			float boost_multiplier = 2.0f;

			if (std::abs(current_angle_) > motor_control::safety::STEERING_ANGLE_THRESHOLD)
			{
				boost_multiplier = 3.5f;
				static int curve_startup_debug_counter = 0;
				if ((++curve_startup_debug_counter % 10) == 0)
				{
					std::cout << "[STARTUP_CURVE] Applying extra boost for car stopped in turn: PWM " << pwm_value << std::endl;
				}
			}

			pwm_value = static_cast<int>(pwm_value * boost_multiplier);

			static int startup_debug_counter = 0;
			if ((++startup_debug_counter % 10) == 0)
			{
				std::cout << "[STARTUP] Applying startup boost for car really stopped: PWM " << pwm_value << std::endl;
			}
		}

		pwm_value = std::min(pwm_value, 4095);

		if (set_speed_debug_counter % 20 == 0)
		{
			std::cout << "[DEBUG] set_speed() - Target: " << target_speed_
					  << "%, PWM: " << pwm_value
					  << ", Filtered: " << filtered_speed_ << "%" << std::endl;
		}

		if (target_speed_ > 0)
		{
			set_motor_pwm_smooth(0, pwm_value);
			set_motor_pwm_smooth(1, 0);
			set_motor_pwm_smooth(2, pwm_value);
			set_motor_pwm_smooth(5, pwm_value);
			set_motor_pwm_smooth(6, 0);
			set_motor_pwm_smooth(7, pwm_value);
		}
		else if (target_speed_ < 0)
		{
			set_motor_pwm_smooth(0, pwm_value);
			set_motor_pwm_smooth(1, pwm_value);
			set_motor_pwm_smooth(2, 0);
			set_motor_pwm_smooth(6, pwm_value);
			set_motor_pwm_smooth(7, pwm_value);
			set_motor_pwm_smooth(8, 0);
		}
		else
		{

			for (int channel = 0; channel < 9; ++channel)
			{
				set_motor_pwm_smooth(channel, 0);
			}
		}

		current_speed_ = target_speed_;
		last_speed_command_ = speed;

		if (std::abs(speed) > 1.0f)
		{
			last_movement_time_ = current_time;
		}
	}

	void JetRacer::set_constant_speed_mode(bool enabled)
	{
		test_mode_ = enabled;
		if (enabled)
		{
			std::cout << "[TEST MODE] Constant speed test mode ACTIVATED" << std::endl;
			std::cout << "[TEST MODE] Use set_test_speed() and start_speed_test() to test" << std::endl;
		}
		else
		{
			std::cout << "[TEST MODE] Test mode DEACTIVATED - returning to joystick control" << std::endl;
			stop_speed_test();
		}
	}

	void JetRacer::set_test_speed(float speed_percent)
	{

		speed_percent = std::max(-50.0f, std::min(50.0f, speed_percent));
		test_speed_ = speed_percent;
		std::cout << "[TEST MODE] Test speed set to: " << speed_percent << "%" << std::endl;
	}

	void JetRacer::set_test_duration(int seconds)
	{
		test_duration_ = std::max(1, std::min(300, seconds));
		std::cout << "[TEST MODE] Test duration set to: " << test_duration_ << " seconds" << std::endl;
	}

	void JetRacer::start_speed_test()
	{
		if (!test_mode_)
		{
			std::cout << "[ERROR] Test mode is not activated. Use set_constant_speed_mode(true) first." << std::endl;
			return;
		}

		if (test_running_)
		{
			std::cout << "[WARNING] Test already running. Stopping previous test..." << std::endl;
			stop_speed_test();
		}

		test_running_ = true;
		test_start_time_ = std::chrono::steady_clock::now();

		test_thread_ = std::thread(&JetRacer::process_test_mode, this);
		test_thread_.detach();

		std::cout << "[TEST MODE] Test started with speed: " << test_speed_ << "% for " << test_duration_ << " seconds" << std::endl;
		std::cout << "[TEST MODE] Use stop_speed_test() to stop the test" << std::endl;
	}

	void JetRacer::stop_speed_test()
	{
		if (test_running_)
		{
			test_running_ = false;
			set_speed(0);
			std::cout << "[TEST MODE] Test stopped. Car stopped." << std::endl;
		}
	}

	void JetRacer::process_test_mode()
	{
		std::cout << "[TEST MODE] Applying constant speed: " << test_speed_ << "%" << std::endl;

		set_speed(test_speed_);

		auto start_time = std::chrono::steady_clock::now();
		while (test_running_)
		{
			auto current_time = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

			if (elapsed >= test_duration_)
			{
				std::cout << "[TEST MODE] Test duration reached (" << test_duration_ << "s). Stopping..." << std::endl;
				break;
			}

			set_speed(test_speed_);

			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		if (test_running_)
		{
			set_speed(0);
			test_running_ = false;
			std::cout << "[TEST MODE] Test finished. Car stopped." << std::endl;
		}
	}

	void JetRacer::process_joystick()
	{
		if (SDL_Init(SDL_INIT_JOYSTICK) < 0)
		{
			std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
			return;
		}

		SDL_Joystick *joystick = SDL_JoystickOpen(0);
		if (!joystick)
		{
			std::cerr << "Failed to open joystick: " << SDL_GetError() << std::endl;
			SDL_Quit();
			return;
		}

		while (running_)
		{

			if (test_mode_ && test_running_)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}

			static int debug_counter = 0;
			if ((++debug_counter % 100) == 0)
			{
				std::cout << "[JOYSTICK] Processing joystick input..." << std::endl;
			}

			SDL_JoystickUpdate();

			int left_joystick_y = SDL_JoystickGetAxis(joystick, 1);

			if (debug_counter % 100 == 0)
			{
				float speed_percent = -left_joystick_y / 32767.0f * 100;
				std::cout << "[JOYSTICK] Y: " << left_joystick_y << " -> Speed: " << speed_percent << "%" << std::endl;
			}

			set_speed(-left_joystick_y / 32767.0f * 100);

			std::this_thread::sleep_for(std::chrono::milliseconds(pwm::timing::JOYSTICK_UPDATE_MS));
		}

		SDL_JoystickClose(joystick);
		SDL_Quit();
	}

	void JetRacer::start()
	{
		running_ = true;
		std::thread joystick_thread(&JetRacer::process_joystick, this);
		joystick_thread.detach();
	}

	void JetRacer::stop()
	{
		running_ = false;
		stop_speed_test();
		set_speed(0);
		set_steering(0);
	}

	bool JetRacer::is_running() const
	{
		return running_.load();
	}

}
