#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
namespace OmnidirectionalCamera{
	namespace cuda{	
		void DivAndClip(cv::cuda::GpuMat &src ,cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::Rect roi);
		void Join(cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::cuda::GpuMat &dst , int vdiff, int blendWidth);
	}
}
