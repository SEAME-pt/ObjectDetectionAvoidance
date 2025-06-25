#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>
#include <vector>
#include "types.h"
#include "cuda_utils.h"

void deserialize_engine(std::string &engine_name, nvinfer1::IRuntime **runtime, nvinfer1::ICudaEngine **engine,
                        nvinfer1::IExecutionContext **context);

void prepare_buffer(nvinfer1::ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_seg_buffer_device, float **output_buffer_host, float **output_seg_buffer_host,
                    float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process);

void infer(nvinfer1::IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, float *output_seg,
           int batchsize, float *decode_ptr_host, float *decode_ptr_device, int model_bboxes,
           std::string cuda_post_process);

std::vector<cv::Mat> process_mask(const float *proto, int proto_size, std::vector<Detection> &dets);
