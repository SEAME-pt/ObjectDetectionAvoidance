#include "jetracer/yolo_infer.hpp"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include <fstream>
#include <cmath>
#include <cassert>

using namespace nvinfer1;
extern Logger gLogger;

const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

static cv::Rect get_downscale_rect(float bbox[4], float scale)
{
	float left = bbox[0];
	float top = bbox[1];
	float right = bbox[0] + bbox[2];
	float bottom = bbox[1] + bbox[3];

	left = std::max(0.0f, left);
	top = std::max(0.0f, top);
	right = std::min((float)kInputW, right);
	bottom = std::min((float)kInputH, bottom);

	left /= scale;
	top /= scale;
	right /= scale;
	bottom /= scale;

	return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

std::vector<cv::Mat> process_mask(const float *proto, int proto_size, std::vector<Detection> &dets)
{
	std::vector<cv::Mat> masks;
	for (size_t i = 0; i < dets.size(); i++)
	{
		cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
		auto r = get_downscale_rect(dets[i].bbox, 4);

		for (int x = r.x; x < r.x + r.width; x++)
		{
			for (int y = r.y; y < r.y + r.height; y++)
			{
				float e = 0.0f;
				for (int j = 0; j < 32; j++)
				{
					e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
				}
				e = 1.0f / (1.0f + expf(-e));
				mask_mat.at<float>(y, x) = e;
			}
		}
		cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
		masks.push_back(mask_mat);
	}
	return masks;
}

void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context)
{
	std::ifstream file(engine_name, std::ios::binary);
	assert(file.good());

	file.seekg(0, file.end);
	size_t size = file.tellg();
	file.seekg(0, file.beg);
	char *serialized_engine = new char[size];
	file.read(serialized_engine, size);
	file.close();

	*runtime = createInferRuntime(gLogger);
	assert(*runtime);
	*engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
	assert(*engine);
	*context = (*engine)->createExecutionContext();
	assert(*context);
	delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_seg_buffer_device, float **output_buffer_host, float **output_seg_buffer_host,
                    float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process)
{
	assert(engine->getNbBindings() == 3);
	const int inputIndex = engine->getBindingIndex(kInputTensorName);
	const int outputIndex = engine->getBindingIndex(kOutputTensorName);
	const int outputIndex_seg = engine->getBindingIndex("proto");

	assert(inputIndex == 0);
	assert(outputIndex == 1);
	assert(outputIndex_seg == 2);

	CUDA_CHECK(cudaMalloc((void **)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)output_seg_buffer_device, kBatchSize * kOutputSegSize * sizeof(float)));

	if (cuda_post_process == "c")
	{
		*output_buffer_host = new float[kBatchSize * kOutputSize];
		*output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];
	}
	else if (cuda_post_process == "g")
	{
		if (kBatchSize > 1)
		{
			std::cerr << "GPU post-processing ainda não implementado para batch > 1" << std::endl;
			exit(1);
		}
		*decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
		CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
	}
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, float *output_seg,
           int batchsize, float *decode_ptr_host, float *decode_ptr_device, int model_bboxes,
           std::string cuda_post_process)
{
	context.enqueue(batchsize, buffers, stream, nullptr);

	if (cuda_post_process == "c")
	{
		CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(output_seg, buffers[2], batchsize * kOutputSegSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
	}
	else if (cuda_post_process == "g")
	{
		CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
		cuda_decode((float *)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
		cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
		CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
			sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));
}
