// Path: yolov8/main.cpp
#include "jetracer/pid_controller.hpp"
#include "jetracer/jetracer.hpp"
#include "jetracer/computer_vision.hpp"
#include "jetracer/yolo_infer.hpp"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

using namespace nvinfer1;
Logger gLogger;

const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

jetracer::control::JetRacer *jetracer_ptr = nullptr;

void signal_handler(int)
{
	std::cout << "\n[!] Ctrl+C detected. Stopping the JetRacer..." << std::endl;
	if (jetracer_ptr)
		jetracer_ptr->stop();
	std::_Exit(0);
}

int main()
{
	std::cout << "=== PID + YOLOv8 segmentation ===" << std::endl;

	try
	{
		// Leitura da calibração da câmera
		cv::Mat cameraMatrix, distCoeffs;
		cv::FileStorage fs("/home/jetson/calibration_camera/calibration.yml", cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cerr << "[ERROR] Not able to open calibration file." << std::endl;
			return -1;
		}
		fs["CameraMatrix"] >> cameraMatrix;
		fs["DistCoeffs"] >> distCoeffs;
		fs.release();

		jetracer::control::JetRacer jetracer(0x40, 0x60);
		jetracer_ptr = &jetracer;
		signal(SIGINT, signal_handler);
		jetracer.start();

		// YOLO + TensorRT
		std::string engine_name = "/home/jetson/Documents/e-codes/cpp/cpp_yolo/yolov8/best.engine";
		std::string cuda_post_process = "c";
		std::string labels_filename = "/home/jetson/Documents/e-codes/cpp/cpp_yolo/yolov8/my_classes.txt";
		cudaSetDevice(kGpuId);

		IRuntime *runtime = nullptr;
		ICudaEngine *engine = nullptr;
		IExecutionContext *context = nullptr;
		deserialize_engine(engine_name, &runtime, &engine, &context);

		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		cuda_preprocess_init(kMaxInputImageSize);

		float *device_buffers[3];
		float *output_buffer_host = nullptr;
		float *output_seg_buffer_host = nullptr;
		float *decode_ptr_host = nullptr;
		float *decode_ptr_device = nullptr;

		std::unordered_map<int, std::string> labels_map;
		read_labels(labels_filename, labels_map);
		assert(kNumClass == labels_map.size());

		prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &device_buffers[2],
					   &output_buffer_host, &output_seg_buffer_host,
					   &decode_ptr_host, &decode_ptr_device, cuda_post_process);

		// Pipeline GStreamer da câmera
		std::string pipeline =
			"nvarguscamerasrc sensor-id=0 ! "
			"video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
			"nvvidconv ! video/x-raw, format=BGRx ! "
			"videoconvert ! video/x-raw, format=BGR ! "
			"appsink";

		cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
		if (!cap.isOpened())
		{
			std::cerr << "[ERROR] Failed to open camera." << std::endl;
			return 1;
		}

		cv::Mat frame, undistorted;
		int frame_id = 0;
		auto start = std::chrono::steady_clock::now();
		int fps_counter = 0;
		float fps = 0.0f;

		while (true)
		{
			cap >> frame;
			if (frame.empty())
				break;

			cv::undistort(frame, undistorted, cameraMatrix, distCoeffs);

			std::vector<cv::Mat> img_batch = {undistorted};
			cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
			infer(*context, stream, (void **)device_buffers, output_buffer_host, output_seg_buffer_host, 1,
				  decode_ptr_host, decode_ptr_device, engine->getBindingDimensions(1).d[0], cuda_post_process);

			std::vector<std::vector<Detection>> res_batch;
			batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);
			auto &res = res_batch[0];
			auto masks = process_mask(output_seg_buffer_host, kOutputSegSize, res);

			cv::imshow("Camera fixed", undistorted);

			if (!masks.empty())
			{
				cv::Mat mask_bin;
				cv::threshold(masks[0], mask_bin, 0.5, 255, cv::THRESH_BINARY);
				mask_bin.convertTo(mask_bin, CV_8UC1);

				cv::imshow("Segmented Mascara", mask_bin);

				std::ostringstream name_stream;
				name_stream << "frame_" << std::setfill('0') << std::setw(4) << frame_id++ << ".jpg";

				float pid_angle = jetracer::pid::PIDexecute(mask_bin.clone(), name_stream.str());
				jetracer.smooth_steering(static_cast<int>(pid_angle), 5);
			}

			fps_counter++;
			auto now = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
			if (elapsed >= 1)
			{
				fps = fps_counter / (float)elapsed;
				std::cout << "FPS: " << fps << std::endl;
				start = now;
				fps_counter = 0;
			}

			if (cv::waitKey(1) == 27) break;
		}

		// Cleanup
		jetracer.stop();
		cap.release();
		cv::destroyAllWindows();
		cudaStreamDestroy(stream);
		CUDA_CHECK(cudaFree(device_buffers[0]));
		CUDA_CHECK(cudaFree(device_buffers[1]));
		CUDA_CHECK(cudaFree(device_buffers[2]));
		CUDA_CHECK(cudaFree(decode_ptr_device));
		delete[] decode_ptr_host;
		delete[] output_buffer_host;
		delete[] output_seg_buffer_host;
		delete context;
		delete engine;
		delete runtime;
		cuda_preprocess_destroy();

		std::cout << "Finished" << std::endl;
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
