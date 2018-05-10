#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
namespace OmnidirectionalCamera{
  namespace cuda{	
    void DivAndClip(cv::cuda::GpuMat &src ,cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::Rect roi);
    void RingStitch(cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::cuda::GpuMat &dst , int vdiff, int blendWidth);
    void SideBySideStitch(cv::cuda::GpuMat &left, cv::cuda::GpuMat &right, cv::cuda::GpuMat &dst ,int v_diff,int blend_width);
  }
}
