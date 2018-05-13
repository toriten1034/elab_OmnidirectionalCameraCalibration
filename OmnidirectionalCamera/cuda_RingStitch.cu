#include "OmnidirectionalCamera.cuh"

#include <opencv2/cudev/ptr2d/glob.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*******************************
* cuda_RingStitch_kernel
*	arguments
* 	right : input  data pointer (GlobPtrSz)
*	left  : input  data pointer (GlobPtrSz)
*	dst   : output data pointer (GlobPtrSz)
*	vdiff : vertical diffarence offset (int)
*	blendWidth : alpha blend area width to stiting (int)
*******************************/
__global__ void cuda_RingStitch_kernel( const cv::cudev::GlobPtrSz<uchar> right ,const cv::cudev::GlobPtrSz<uchar> left , cv::cudev::GlobPtrSz<uchar> dst, int vdiff, int blendWidth){
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
   
    const int dst_color_tid =  y * dst.step + (3 * x);
   
    const int right_color_tid = y * right.step + (3 *  (x+ (blendWidth/2)) );
    const int left_color_tid  = ( (y+vdiff) * left.step )  + (3 * (x+ (blendWidth/2) - (dst.cols/6)));

    if((x < dst.cols/3) && (y < dst.rows)){
 	if(x < ((dst.cols/6)-blendWidth/2 ) ){
		dst.data[dst_color_tid + 0] = right.data[right_color_tid + 0];
		dst.data[dst_color_tid + 1] = right.data[right_color_tid + 1];
		dst.data[dst_color_tid + 2] = right.data[right_color_tid + 2];
	}else if(x < (dst.cols/6)){ //blending area
		float alpha = (float)(x - ( (dst.cols/6) - (blendWidth/2) ) )/(float)blendWidth*2;
		dst.data[dst_color_tid + 0] = right.data[right_color_tid + 0]*(1-alpha) + left.data[left_color_tid + 0]*alpha;
		dst.data[dst_color_tid + 1] = right.data[right_color_tid + 1]*(1-alpha) + left.data[left_color_tid + 1]*alpha;
		dst.data[dst_color_tid + 2] = right.data[right_color_tid + 2]*(1-alpha) + left.data[left_color_tid + 2]*alpha;
	}else if(x < ((dst.cols/3)-blendWidth/2 ) ){
		dst.data[dst_color_tid + 0] =  left.data[left_color_tid + 0];
		dst.data[dst_color_tid + 1] =  left.data[left_color_tid + 1];
		dst.data[dst_color_tid + 2] =  left.data[left_color_tid + 2];
	}else if(x < (dst.cols/3)){ //blending area
		float alpha = (float)(x - ( (dst.cols/3) - (blendWidth/2) ) )/(float)blendWidth*2;
		dst.data[dst_color_tid + 0] =  left.data[left_color_tid + 0]*(1-alpha) + right.data[(right_color_tid - (dst.cols))+ 0]*alpha;
		dst.data[dst_color_tid + 1] =  left.data[left_color_tid + 1]*(1-alpha) + right.data[(right_color_tid - (dst.cols))+ 1]*alpha;
		dst.data[dst_color_tid + 2] =  left.data[left_color_tid + 2]*(1-alpha) + right.data[(right_color_tid - (dst.cols))+ 2]*alpha;
	}
   }
}

/*******************************
* cuda_RingStitch
*	arguments
* 	right : input  data pointer (GpuMat)
*	left  : input  data pointer (GpuMat)
*	dst   : output data pointer (GpuMat)
*	vdiff : vertical diffarence offset (int)
*	blendWidth : alpha blend area width to stiting (int)
*******************************/
void  OmnidirectionalCamera::cuda::RingStitch(cv::cuda::GpuMat &right, cv::cuda::GpuMat &left, cv::cuda::GpuMat &dst , int vdiff, int blendWidth){

    //create image pointer
    cv::cudev::GlobPtrSz<uchar> p_Right = cv::cudev::globPtr(right.ptr<uchar>(), right.step, right.rows, right.cols * right.channels());
    cv::cudev::GlobPtrSz<uchar> p_Left  = cv::cudev::globPtr(left.ptr<uchar>() , left.step , left.rows , left.cols  * left.channels());
    cv::cudev::GlobPtrSz<uchar> p_Dst   = cv::cudev::globPtr(dst.ptr<uchar>()  , dst.step  , dst.rows  , dst.cols   * dst.channels());

    const dim3 block(32, 32);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows , block.y));

    cuda_RingStitch_kernel<<<grid, block>>>( p_Right, p_Left, p_Dst , vdiff, blendWidth);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

}

