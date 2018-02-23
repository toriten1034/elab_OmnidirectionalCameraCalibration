#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
void cuda_stitching(cv::cuda::GpuMat &src ,cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::Rect roi);
