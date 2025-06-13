// File: sources/main.cpp
#include "jetracer/pid_controller.hpp"
#include "jetracer/jetracer.hpp"
#include "jetracer/computer_vision.hpp"
#include "jetracer/pid_controller.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <atomic>
#include <thread>

jetracer::control::JetRacer *jetracer_ptr = nullptr;

void signal_handler(int)
{
	std::cout << std::endl << "[!] Ctrl+C detected. Stopping the JetRacer..." << std::endl;
	if (jetracer_ptr)
		jetracer_ptr->stop();
	std::_Exit(0);
}

int main()
{
	std::cout << "=== PID with joystick (manual speed) + shared memory ===" << std::endl;

	try
	{
		jetracer::control::JetRacer jetracer(0x40, 0x60);
		jetracer_ptr = &jetracer;

		signal(SIGINT, signal_handler);
		jetracer.start();

		int shm_fd = shm_open("mask_shared", O_RDWR, 0666);
		if (shm_fd == -1)
		{
			std::cerr << "Error oppening shared memory." << std::endl;
			return 1;
		}

		uint8_t *shm_ptr = (uint8_t *)mmap(nullptr, jetracer::vision::SIZE + 1, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
		if (shm_ptr == MAP_FAILED)
		{
			std::cerr << "Error mapping memory." << std::endl;
			return 1;
		}

		uint8_t *flag_ptr = shm_ptr;
		uint8_t *img_ptr = shm_ptr + 1;

		cv::Mat mask(jetracer::vision::HEIGHT, jetracer::vision::WIDTH, CV_8UC1, img_ptr);
		int frame_id = 0;

		while (true)
		{
			if (flag_ptr[0] != 1)
			{
				usleep(3000);
				continue;
			}

			std::ostringstream name_stream;
			name_stream << "frame_" << std::setfill('0') << std::setw(4) << frame_id++ << ".jpg";

			float pid_angle = jetracer::pid::PIDexecute(mask.clone(), name_stream.str());
			jetracer.smooth_steering(static_cast<int>(pid_angle), 5);

			flag_ptr[0] = 0;

			if (cv::waitKey(1) == 27)
				break;
		}

		jetracer.stop();
		std::cout << "Finishing." << std::endl;
		return 0;
	}
	catch (const std::exception &e)
	{
		std::cerr << "[ERROR] " << e.what() << std::endl;
		if (jetracer_ptr)
			jetracer_ptr->stop();
		return 1;
	}
}
