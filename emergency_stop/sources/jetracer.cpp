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
		// Add new target speed to history
		speed_history_.push_back(target_speed);

		// Keep only the last SPEED_SMOOTHING_WINDOW speeds
		if (speed_history_.size() > SPEED_SMOOTHING_WINDOW)
		{
			speed_history_.pop_front();
		}

		// Calculate average of speeds in history
		float sum = 0.0f;
		for (float speed : speed_history_)
		{
			sum += speed;
		}
		float average_speed = sum / speed_history_.size();

		// Apply maximum change limitation per update
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
		// Apply base speed limitation (as before)
		float max_safe_speed = motor_control::safety::BASE_MAX_SPEED;

		// Check if turning based on current steering angle
		bool is_turning = std::abs(current_angle_) > motor_control::safety::STEERING_ANGLE_THRESHOLD;

		// CHECK IF CAR IS STOPPED - if so, allows maximum speed to overcome inertia
		bool is_car_stopped = (std::abs(current_speed_) < 1.0f);

		if (is_turning && !is_car_stopped)
		{
			// In curve AND car moving: applies boost to overcome resistance of turned wheels
			max_safe_speed *= motor_control::safety::CURVE_SPEED_REDUCTION;

			// Debug log for curves
			static int curve_debug_counter = 0;
			if ((++curve_debug_counter % 50) == 0)
			{ // Log every ~2.5 seconds
				std::cout << "[CURVE] Angle: " << current_angle_
						  << "°, Max speed: " << max_safe_speed << "% (boost applied to overcome resistance)" << std::endl;
			}
		}
		else if (!is_turning)
		{
			// Straight: normal speed without boost
			max_safe_speed *= motor_control::safety::STRAIGHT_SPEED_BOOST;

			// Debug log for straight lines
			static int straight_debug_counter = 0;
			if ((++straight_debug_counter % 100) == 0)
			{ // Log every ~5 seconds
				std::cout << "[STRAIGHT] Angle: " << current_angle_
						  << "°, Max speed: " << max_safe_speed << "% (normal speed)" << std::endl;
			}
		}
		else
		{
			// Car stopped in curve: allows maximum speed to overcome inertia
			if (is_car_stopped && is_turning)
			{
				static int startup_curve_debug_counter = 0;
				if ((++startup_curve_debug_counter % 30) == 0)
				{ // Log a cada ~1.5 segundos
					std::cout << "[STARTUP_CURVE] Carro parado em curva - permitindo velocidade máxima para vencer inércia das rodas viradas" << std::endl;
				}
			}
		}

		// Apply safe speed limitation
		if (std::abs(target_speed) > max_safe_speed)
		{
			// Keep the sign (forward/backward) but limit the magnitude
			float sign = (target_speed > 0) ? 1.0f : -1.0f;
			target_speed = sign * max_safe_speed;

			// Log applied limitation
			static int limit_debug_counter = 0;
			if ((++limit_debug_counter % 30) == 0)
			{ // Log a cada ~1.5 segundos
				std::cout << "[SAFETY] Velocidade limitada para " << target_speed
						  << "% (máximo seguro: " << max_safe_speed << "%)" << std::endl;
			}
		}

		return target_speed;
	}

	void JetRacer::set_motor_pwm_smooth(int channel, int value)
	{
		// PWM implementation with smoothing for smoother transitions
		value = std::clamp(value, 0, 4095);
		int base_reg = 0x06 + (channel * 4);

		// CHECK IF CAR IS STOPPED - if so, apply PWM directly without smoothing
		bool is_car_stopped = (std::abs(current_speed_) < 1.0f);

		if (is_car_stopped)
		{
			// CAR REALLY STOPPED: Apply PWM directly for maximum responsiveness
			motor_device_.write_byte(base_reg, 0);
			motor_device_.write_byte(base_reg + 1, 0);
			motor_device_.write_byte(base_reg + 2, value & 0xFF);
			motor_device_.write_byte(base_reg + 3, value >> 8);
			return;
		}

		// CAR IN MOTION: Apply normal smoothing
		static std::array<int, 9> last_pwm_values = {0};
		static std::array<int, 9> target_pwm_values = {0};

		// Define o valor alvo
		target_pwm_values[channel] = value;

		// Calculate difference for smoothing
		int current_pwm = last_pwm_values[channel];
		int pwm_diff = target_pwm_values[channel] - current_pwm;

		// Apply smoothing with maximum change rate
		int max_pwm_change = 200; // Maximum PWM change per update
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

		// Atualiza o valor atual
		last_pwm_values[channel] = current_pwm;

		// Aplica o PWM suavizado
		motor_device_.write_byte(base_reg, 0);
		motor_device_.write_byte(base_reg + 1, 0);
		motor_device_.write_byte(base_reg + 2, current_pwm & 0xFF);
		motor_device_.write_byte(base_reg + 3, current_pwm >> 8);
	}

	void JetRacer::set_speed(float speed)
	{
		// CHECK IF MOTOR IS LOCKED - if so, ignore speed commands
		if (motor_locked_)
		{
			// Log occasionally to avoid spam
			static int motor_lock_debug_counter = 0;
			if ((++motor_lock_debug_counter % 100) == 0)
			{ // Log a cada ~5 segundos
				std::cout << "[MOTOR LOCK] Motor travado - ignorando comando de velocidade: " << speed << "%" << std::endl;
			}
			return; // Exit function without processing command
		}

		// CHECK IF STILL IN EMERGENCY BRAKING PERIOD
		if (is_emergency_braking_active())
		{
			// Apply continuous braking for 2 seconds
			for (int channel = 0; channel < 9; ++channel)
			{
				set_motor_pwm(channel, 4095); // Maximum PWM to maintain braking
			}

			// Log occasionally to avoid spam
			static int emergency_brake_debug_counter = 0;
			if ((++emergency_brake_debug_counter % 50) == 0)
			{ // Log every ~2.5 seconds
				std::cout << "[EMERGENCY BRAKE] Frenagem ativa - aplicando PWM máximo para travar motor" << std::endl;
			}
			return; // Exit function without processing speed command
		}

		// DEBUG: Function entry log
		static int set_speed_debug_counter = 0;

		// Speed log
		if ((++set_speed_debug_counter % 20) == 0)
		{ // Log every ~1 second
			std::cout << "[DEBUG] set_speed() - Velocidade solicitada: " << speed << "%" << std::endl;
		}

		// SAFETY LIMITATION APPLICATION (as before, but more intelligent)
		float original_speed = speed;
		speed = calculate_safe_speed(speed);

		// Timestamp for stopped state detection
		unsigned long current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
										 std::chrono::steady_clock::now().time_since_epoch())
										 .count();

		// Log applied limitation (if there was change)
		if (std::abs(original_speed) != std::abs(speed) && set_speed_debug_counter % 20 == 0)
		{
			std::cout << "[SAFETY] Velocidade ajustada de " << original_speed
					  << "% para " << speed << "% por segurança" << std::endl;
		}

		// EMERGENCY BRAKING DETECTION
		bool emergency_brake = false;
		float speed_reduction = std::abs(speed) - std::abs(last_speed_command_);

		// Detect if there is a sharp speed reduction (>50% reduction)
		if (last_speed_command_ != 0.0f && speed_reduction < -motor_control::ramps::EMERGENCY_BRAKE_THRESHOLD)
		{
			emergency_brake = true;
			// std::cout << "[EMERGENCY BRAKE] Frenagem de emergência detectada! Redução: " << speed_reduction << "%" << std::endl;
		}

		// Detect rapid direction change (forward to backward or vice-versa)
		if ((last_speed_command_ > motor_control::safety::DIRECTION_CHANGE_THRESHOLD && speed < -motor_control::safety::DIRECTION_CHANGE_THRESHOLD) ||
			(last_speed_command_ < -motor_control::safety::DIRECTION_CHANGE_THRESHOLD && speed > motor_control::safety::DIRECTION_CHANGE_THRESHOLD))
		{
			emergency_brake = true;
			// std::cout << "[EMERGENCY BRAKE] Mudança de direção detectada!" << std::endl;
		}

		// Apply deadzone to avoid oscillations at low speed
		if (std::abs(speed) < motor_control::thresholds::SPEED_DEADZONE * 100.0f)
		{
			speed = 0.0f;
		}

		// Simple logic to detect stopped state
		bool is_car_stopped = (std::abs(current_speed_) < 1.0f);

		// SPECIAL LOG FOR STARTUP - show when car is really stopped
		if (is_car_stopped && std::abs(speed) > 5.0f && set_speed_debug_counter % 10 == 0)
		{
			std::cout << "[STARTUP] Carro parado - aplicando aceleração direta: " << speed << "%" << std::endl;
		}

		// Amplify motor force - intelligent limitation already applied above
		// Speed has already been limited by the calculate_safe_speed() function

		// Intelligent power curve reactivated
		float power_curve = 1.0f;
		if (std::abs(speed) > 0.0f)
		{
			// Exponential curve for better response at low speeds
			power_curve = 1.0f + (std::abs(speed) / 100.0f) * motor_control::power::POWER_CURVE_FACTOR;
			speed *= power_curve;
		}

		speed = std::max(-100.0f, std::min(speed, 100.0f));

		// SPECIAL PROCESSING FOR EMERGENCY BRAKING AND STARTUP
		if (emergency_brake)
		{
			// Skip smoothing and apply immediate braking
			filtered_speed_ = speed;
			target_speed_ = speed;
			// std::cout << "[EMERGENCY BRAKE] Aplicando frenagem imediata!" << std::endl;
		}
		else if (is_car_stopped)
		{
			// CAR REALLY STOPPED: APPLY COMMANDS DIRECTLY WITHOUT SMOOTHING
			// Check if last command was also close to zero (car really stopped)
			filtered_speed_ = speed;
			target_speed_ = speed;
			if (set_speed_debug_counter % 10 == 0)
			{
				std::cout << "[STARTUP] Carro realmente parado - aplicando aceleração direta: " << speed << "%" << std::endl;
			}
		}
		else if (std::abs(speed - current_speed_) > 5.0f)
		{
			// SHARP SPEED CHANGE: apply command directly for maximum responsiveness
			filtered_speed_ = speed;
			target_speed_ = speed;
			if (set_speed_debug_counter % 10 == 0)
			{
				std::cout << "[RESPONSIVE] Mudança brusca detectada - aplicando comando direto: " << speed << "%" << std::endl;
			}
		}
		else
		{
			// NORMAL INTELLIGENT SMOOTHING (only for small changes)
			filtered_speed_ = motor_control::power::SPEED_SMOOTHING_FACTOR * speed +
							  (1.f - motor_control::power::SPEED_SMOOTHING_FACTOR) * filtered_speed_;
		}

		// INTELLIGENT speed ramp with aggressive braking and responsive startup
		if (!emergency_brake && !is_car_stopped && std::abs(speed - current_speed_) <= 5.0f)
		{
			// ONLY apply ramp for small and gradual changes
			float speed_diff = filtered_speed_ - target_speed_;
			float ramp_rate;

			// Determine type of change: acceleration or deceleration
			bool is_braking = (filtered_speed_ < target_speed_ && target_speed_ > 0) ||
							  (filtered_speed_ > target_speed_ && target_speed_ < 0) ||
							  (std::abs(filtered_speed_) < std::abs(target_speed_));

			if (is_braking)
			{
				// BRAKING: use very fast deceleration rate
				ramp_rate = motor_control::ramps::DECELERATION_RAMP;
			}
			else
			{
				// ACCELERATION: use normal rate
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
			// For sharp changes or startup: apply directly
			target_speed_ = filtered_speed_;
			if (set_speed_debug_counter % 10 == 0)
			{
				std::cout << "[DIRECT] Aplicando comando direto (sem rampa): " << filtered_speed_ << "%" << std::endl;
			}
		}

		// Converter para PWM com melhor aproveitamento da faixa e threshold otimizado
		int pwm_value = static_cast<int>(std::abs(target_speed_) / 100.0f * 4095);

		// DEBUG: Log do PWM calculado
		if (set_speed_debug_counter % 20 == 0)
		{
			std::cout << "[DEBUG] set_speed() - PWM calculado: " << pwm_value
					  << " para velocidade: " << target_speed_ << "%" << std::endl;
		}

		// Apply higher minimum threshold to ensure initial force
		if (pwm_value > 0 && pwm_value < motor_control::thresholds::MIN_PWM_THRESHOLD)
		{
			pwm_value = motor_control::thresholds::MIN_PWM_THRESHOLD;
		}

		// FORCED AMPLIFICATION for very small commands (ensure movement)
		if (pwm_value > 0 && pwm_value < 1000)
		{
			pwm_value = static_cast<int>(pwm_value * 2.0f); // 100% additional amplification
		}

		// Intelligent torque amplification reactivated
		if (pwm_value > 0 && pwm_value < motor_control::thresholds::TORQUE_BOOST_THRESHOLD)
		{
			pwm_value = static_cast<int>(pwm_value * motor_control::thresholds::LOW_SPEED_AMPLIFICATION);
		}

		// SPECIAL BOOST FOR STARTUP (when car is really stopped)
		if (is_car_stopped && pwm_value > 0)
		{
			// Apply additional boost to overcome initial inertia (reduced for slower car)
			float boost_multiplier = 2.0f; // Base boost reduced to 100%

			// EXTRA BOOST for cars stopped in curves (harder to get out of place)
			if (std::abs(current_angle_) > motor_control::safety::STEERING_ANGLE_THRESHOLD)
			{
				boost_multiplier = 3.5f; // 250% boost for curves (increased to overcome resistance of turned wheels)
				static int curve_startup_debug_counter = 0;
				if ((++curve_startup_debug_counter % 10) == 0)
				{ // Log every ~0.5 seconds
					std::cout << "[STARTUP_CURVE] Aplicando boost extra para carro parado em curva: PWM " << pwm_value << std::endl;
				}
			}

			pwm_value = static_cast<int>(pwm_value * boost_multiplier);

			// Log de debug para partida
			static int startup_debug_counter = 0;
			if ((++startup_debug_counter % 10) == 0)
			{ // Log a cada ~0.5 segundos
				std::cout << "[STARTUP] Aplicando boost de partida para carro realmente parado: PWM " << pwm_value << std::endl;
			}
		}

		// Ensure it doesn't exceed maximum
		pwm_value = std::min(pwm_value, 4095);

		// DEBUG: Log of calculated PWM values
		if (set_speed_debug_counter % 20 == 0)
		{
			std::cout << "[DEBUG] set_speed() - Target: " << target_speed_
					  << "%, PWM: " << pwm_value
					  << ", Filtered: " << filtered_speed_ << "%" << std::endl;
		}

		// Controle suave mas responsivo reativado
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
			// Parada suave com rampa
			for (int channel = 0; channel < 9; ++channel)
			{
				set_motor_pwm_smooth(channel, 0);
			}
		}

		current_speed_ = target_speed_;
		last_speed_command_ = speed;

		// Atualizar timestamp de movimento (simplificado)
		if (std::abs(speed) > 1.0f)
		{
			last_movement_time_ = current_time;
		}
	}

	// ===== METHODS FOR CONSTANT SPEED TESTING =====

	void JetRacer::set_constant_speed_mode(bool enabled)
	{
		test_mode_ = enabled;
		if (enabled)
		{
			std::cout << "[TEST MODE] Modo de teste de velocidade constante ATIVADO" << std::endl;
			std::cout << "[TEST MODE] Use set_test_speed() e start_speed_test() para testar" << std::endl;
		}
		else
		{
			std::cout << "[TEST MODE] Modo de teste DESATIVADO - voltando ao controle por joystick" << std::endl;
			stop_speed_test();
		}
	}

	void JetRacer::set_test_speed(float speed_percent)
	{
		// Limit test speed for safety
		speed_percent = std::max(-50.0f, std::min(50.0f, speed_percent));
		test_speed_ = speed_percent;
		std::cout << "[TEST MODE] Velocidade de teste definida para: " << speed_percent << "%" << std::endl;
	}

	void JetRacer::set_test_duration(int seconds)
	{
		test_duration_ = std::max(1, std::min(300, seconds)); // 1 segundo a 5 minutos
		std::cout << "[TEST MODE] Duração do teste definida para: " << test_duration_ << " segundos" << std::endl;
	}

	void JetRacer::start_speed_test()
	{
		if (!test_mode_)
		{
			std::cout << "[ERROR] Modo de teste não está ativado. Use set_constant_speed_mode(true) primeiro." << std::endl;
			return;
		}

		if (test_running_)
		{
			std::cout << "[WARNING] Teste já está em execução. Parando teste anterior..." << std::endl;
			stop_speed_test();
		}

		test_running_ = true;
		test_start_time_ = std::chrono::steady_clock::now();

		// Iniciar thread de teste
		test_thread_ = std::thread(&JetRacer::process_test_mode, this);
		test_thread_.detach();

		std::cout << "[TEST MODE] Teste iniciado com velocidade: " << test_speed_ << "% por " << test_duration_ << " segundos" << std::endl;
		std::cout << "[TEST MODE] Use stop_speed_test() para parar o teste" << std::endl;
	}

	void JetRacer::stop_speed_test()
	{
		if (test_running_)
		{
			test_running_ = false;
			set_speed(0); // Parar o carro
			std::cout << "[TEST MODE] Teste parado. Carro parado." << std::endl;
		}
	}

	void JetRacer::process_test_mode()
	{
		std::cout << "[TEST MODE] Aplicando velocidade constante: " << test_speed_ << "%" << std::endl;

		// Aplicar velocidade de teste
		set_speed(test_speed_);

		// Wait for test duration
		auto start_time = std::chrono::steady_clock::now();
		while (test_running_)
		{
			auto current_time = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

			if (elapsed >= test_duration_)
			{
				std::cout << "[TEST MODE] Duração do teste atingida (" << test_duration_ << "s). Parando..." << std::endl;
				break;
			}

			// Manter velocidade constante
			set_speed(test_speed_);

			// Wait before next update
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		// Parar o carro ao final do teste
		if (test_running_)
		{
			set_speed(0);
			test_running_ = false;
			std::cout << "[TEST MODE] Teste finalizado. Carro parado." << std::endl;
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
			// Check if in test mode - if so, don't process joystick
			if (test_mode_ && test_running_)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}

			// DEBUG: Log to verify if joystick is being processed
			static int debug_counter = 0;
			if ((++debug_counter % 100) == 0)
			{ // Log every ~2.5 seconds
				std::cout << "[JOYSTICK] Processando entrada do joystick..." << std::endl;
			}

			SDL_JoystickUpdate();

			int left_joystick_y = SDL_JoystickGetAxis(joystick, 1); // Speed control
			// int right_joystick_x = SDL_JoystickGetAxis(joystick, 2); // Directional control

			// Detect R2 button (button 7 on standard controller)
			bool r2_current = SDL_JoystickGetButton(joystick, 7);

			// Detect R2 click (transition from not pressed to pressed)
			if (r2_current && !r2_button_was_pressed_)
			{
				// Toggle do cruise control
				if (cruise_control_active_)
				{
					// Desativar cruise control
					cruise_control_active_ = false;
					std::cout << "[CRUISE CONTROL] Desativado - Retornando ao controle manual" << std::endl;
				}
				else
				{
					// Ativar cruise control com a velocidade atual
					float current_speed = -left_joystick_y / 32767.0f * 100;
					if (std::abs(current_speed) > 5.0f)
					{ // Only activate if there is significant speed
						cruise_control_active_ = true;
						cruise_control_speed_ = current_speed;
						std::cout << "[CRUISE CONTROL] Ativado - Velocidade: " << current_speed << "%" << std::endl;
					}
					else
					{
						std::cout << "[CRUISE CONTROL] Não ativado - Velocidade muito baixa: " << current_speed << "%" << std::endl;
					}
				}
			}
			r2_button_was_pressed_ = r2_current;

			// DEBUG: Log dos valores do joystick
			if (debug_counter % 100 == 0)
			{
				float speed_percent = -left_joystick_y / 32767.0f * 100;
				std::cout << "[JOYSTICK] Y: " << left_joystick_y << " -> Velocidade: " << speed_percent << "%";
				if (cruise_control_active_)
				{
					std::cout << " [CRUISE: " << cruise_control_speed_ << "%]";
				}
				std::cout << std::endl;
			}

			// Aplicar velocidade baseada no modo atual
			if (cruise_control_active_)
			{
				// Modo cruise control - usar velocidade salva
				set_speed(cruise_control_speed_);
			}
			else
			{
				// Modo manual - usar joystick
				set_speed(-left_joystick_y / 32767.0f * 100);
			}
			// smooth_steering(right_joystick_x / 32767.0f * MAX_ANGLE_, 10);

			// Configurable update frequency for smoother response
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
		stop_speed_test(); // Parar teste se estiver rodando
		set_speed(0);
		set_steering(0);
	}

	bool JetRacer::is_running() const
	{
		return running_.load();
	}

	void JetRacer::emergency_stop()
	{
		std::cout << "[EMERGENCY STOP] Parada de emergência ativada!" << std::endl;

		// TRAVAR O MOTOR para evitar movimento em "ponto morto"
		motor_locked_ = true;
		std::cout << "[EMERGENCY STOP] Motor travado para evitar movimento em ponto morto" << std::endl;

		// ACTIVATE EMERGENCY BRAKING FOR 3 SECONDS
		emergency_braking_active_ = true;
		emergency_brake_start_time_ = std::chrono::steady_clock::now();
		std::cout << "[EMERGENCY STOP] Frenagem de emergência ativada por 3 segundos" << std::endl;

		// Apply immediate braking with maximum PWM to lock motor
		// Send maximum PWM to both directions to create resistance
		for (int channel = 0; channel < 9; ++channel)
		{
			set_motor_pwm(channel, 4095); // Maximum PWM to lock
			set_motor_pwm(channel, 4095);
		}
		std::cout << "[EMERGENCY STOP] PWM máximo aplicado para travar o motor" << std::endl;

		// Wait a moment to ensure locking is effective
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		// Now completely stop the motor
		for (int channel = 0; channel < 9; ++channel)
		{
			set_motor_pwm(channel, 0);
		}

		// Center the steering
		set_steering(0);

		// Stop speed tests if they are active
		if (test_mode_)
		{
			stop_speed_test();
			std::cout << "[EMERGENCY STOP] Teste de velocidade interrompido" << std::endl;
		}

		// Desativar cruise control se estiver ativo
		if (cruise_control_active_)
		{
			cruise_control_active_ = false;
			std::cout << "[EMERGENCY STOP] Cruise control desativado" << std::endl;
		}

		// Reset state variables
		current_speed_ = 0.0f;
		target_speed_ = 0.0f;
		filtered_speed_ = 0.0f;
		last_speed_command_ = 0.0f;

		// Log detalhado
		std::cout << "[EMERGENCY STOP] Sistema parado - velocidade: 0%, direção: centralizada" << std::endl;
		std::cout << "[EMERGENCY STOP] Motor travado - aguardando zona de perigo ficar livre..." << std::endl;
		std::cout << "[EMERGENCY STOP] Frenagem ativa por 3 segundos - como se pisasse no freio" << std::endl;
	}

	// ====== FUNCTION TO RELEASE MOTOR LOCK ======

	void JetRacer::release_motor_lock()
	{
		if (motor_locked_)
		{
			motor_locked_ = false;
			std::cout << "[MOTOR LOCK] Travamento do motor liberado - sistema pronto para movimento" << std::endl;
		}
		else
		{
			std::cout << "[MOTOR LOCK] Motor já estava liberado" << std::endl;
		}
	}

	// ====== FUNCTION TO CHECK IF STILL BRAKING ======

	bool JetRacer::is_emergency_braking_active()
	{
		if (!emergency_braking_active_)
		{
			return false;
		}

		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - emergency_brake_start_time_).count();

		// Se passou dos 3 segundos, desativar a frenagem
		if (elapsed >= EMERGENCY_BRAKE_DURATION_MS)
		{
			emergency_braking_active_ = false;
			return false;
		}

		return true;
	}

	// ====== CRUISE CONTROL FUNCTIONS ======

	void JetRacer::set_cruise_control_mode(bool enabled)
	{
		if (enabled && !cruise_control_active_)
		{
			// Ativar cruise control
			cruise_control_active_ = true;
			std::cout << "[CRUISE CONTROL] Modo ativado programaticamente" << std::endl;
		}
		else if (!enabled && cruise_control_active_)
		{
			// Desativar cruise control
			cruise_control_active_ = false;
			std::cout << "[CRUISE CONTROL] Modo desativado programaticamente" << std::endl;
		}
	}

	void JetRacer::set_cruise_control_speed(float speed)
	{
		cruise_control_speed_ = speed;
		std::cout << "[CRUISE CONTROL] Velocidade definida programaticamente: " << speed << "%" << std::endl;
	}

} // namespace jetracer::control
