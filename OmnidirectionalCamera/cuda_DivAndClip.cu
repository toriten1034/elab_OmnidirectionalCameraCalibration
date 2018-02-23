#include "OmnidirectionalCamera.cuh"

#include <opencv2/cudev/ptr2d/glob.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*******************************
* cuda_DivAndClip_kernel
*	arguments
* 	src   : input  data pointer (GlobPtrSz)
*	right : output data pointer (GlobPtrSz)
*	left  : output data pointer (GlobPtrSz)
*******************************/
__global__ void cuda_DivAndClip_kernel(const cv::cudev::GlobPtrSz<uchar> src,  cv::cudev::GlobPtrSz<uchar> right ,cv::cudev::GlobPtrSz<uchar> left ,int x_offset ,int y_offset){
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int src_color_tid =  (y+y_offset) * src.step + (3 * (x + x_offset));
    const int right_color_tid = y * right.step + (3 * x);

    int offset = src.cols/2;
    if((x < right.cols) && (y < right.rows)){
	right.data[right_color_tid + 0]  =  src.data[src_color_tid + 0];
	right.data[right_color_tid + 1]  =  src.data[src_color_tid + 1];
	right.data[right_color_tid + 2]  =  src.data[src_color_tid + 2];

	left.data[right_color_tid + 0]  =   src.data[src_color_tid + 0 + offset];
	left.data[right_color_tid + 1]  =   src.data[src_color_tid + 1 + offset];
	left.data[right_color_tid + 2]  =   src.data[src_color_tid + 2 + offset];
   }
}

/*******************************
* cuda_DivAndClip
*	arguments
* 	src   : input  data pointer (GpuMat)
*	right : output data pointer (GpuMat)
*	left  : output data pointer (GpuMat)
*******************************/
void OmnidirectionalCamera::cuda::DivAndClip(cv::cuda::GpuMat &src ,cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::Rect roi){

   int x_offset = roi.x;
   int y_offset = roi.y;
	
	//create image pointer
    cv::cudev::GlobPtrSz<uchar> p_Right = cv::cudev::globPtr(right.ptr<uchar>(), right.step, right.rows, right.cols * right.channels());
    cv::cudev::GlobPtrSz<uchar> p_Left  = cv::cudev::globPtr(left.ptr<uchar>() , left.step , left.rows , left.cols  *  left.channels());
    cv::cudev::GlobPtrSz<uchar> p_Src   = cv::cudev::globPtr(src.ptr<uchar>()  , src.step  , src.rows  , src.cols   * src.channels());

    const dim3 block(32, 24);
    const dim3 grid(cv::cudev::divUp(right.cols, block.x), cv::cudev::divUp(right.rows, block.y));


    cuda_DivAndClip_kernel<<<grid, block>>>( p_Src ,p_Right,p_Left ,x_offset, y_offset);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

}
