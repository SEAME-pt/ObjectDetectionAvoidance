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

			int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 100 - 1));
			int oldmode = motor_device_.read_byte(0x00);
			int newmode = (oldmode & 0x7F) | 0x10;

			motor_device_.write_byte(0x00, newmode);
			motor_device_.write_byte(0xFE, prescale);
			motor_device_.write_byte(0x00, oldmode);
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			motor_device_.write_byte(0x00, oldmode | 0xA1);
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

	void JetRacer::set_speed(float speed, int angle)
	{
		//    std::cout << "[DEBUG] set_speed = " << speed << std::endl;
//		if ((angle <= 130 || angle >= -130) && speed > 30)
//			speed = 30;

                (void)angle;
                if (speed > 30)
                      speed = 30;


		speed = std::clamp(speed, -100.0f, 100.0f);
		int pwm_value = static_cast<int>(std::abs(speed) / 100.0f * 4095);

		if (speed > 0)
		{
			set_motor_pwm(0, pwm_value);
			set_motor_pwm(1, 0);
			set_motor_pwm(2, pwm_value);
			set_motor_pwm(5, pwm_value);
			set_motor_pwm(6, 0);
			set_motor_pwm(7, pwm_value);
		}
		else if (speed < 0)
		{
			set_motor_pwm(0, pwm_value);
			set_motor_pwm(1, pwm_value);
			set_motor_pwm(2, 0);
			set_motor_pwm(6, pwm_value);
			set_motor_pwm(7, pwm_value);
			set_motor_pwm(8, 0);
		}
		else
		{
			for (int channel = 0; channel < 9; ++channel)
			{
				set_motor_pwm(channel, 0);
			}
		}

		current_speed_ = speed;
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

		bool r2_previous_state = false;
		while (running_)
		{
			SDL_JoystickUpdate();
			// Speed control
			int left_joystick_y = SDL_JoystickGetAxis(joystick, 1);
			// Directional control
			// int right_joystick_x = SDL_JoystickGetAxis(joystick, 2);
			// Autonomous mode toggle
			int r2_button = SDL_JoystickGetButton(joystick, 9);
			// Se o botão R2 for pressionado e o modo autônomo estiver desativado
			if (r2_button == 1 && r2_previous_state == false)
			{
				// Toggle autonomous mode
				autonomous_mode_ = !autonomous_mode_;
				std::cout << "Autonomous mode: ON" << std::endl;
				if (current_speed_ > 0)
					set_speed(current_speed_, current_angle_);
			}
			r2_previous_state = r2_button;

			if (!autonomous_mode_)
			{
				//autonomous_mode_ = false;
				// std::cout << "Autonomous mode: OFF" << std::endl;
				set_speed(-left_joystick_y / 32767.0f * 100, current_angle_);
			}
			//r2_previous_state = r2_button;
			// set_speed(-left_joystick_y / 32767.0f * 100, current_angle_);
			// smooth_steering(right_joystick_x / 32767.0f * MAX_ANGLE_, 10);

			std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
		set_speed(0, 0);
		set_steering(0);
	}

bool JetRacer::is_running() const
	{
		return running_.load();
	}

} // namespace jetracer::control
