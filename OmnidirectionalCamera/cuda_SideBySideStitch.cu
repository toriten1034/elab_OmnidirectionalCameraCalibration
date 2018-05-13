#include "OmnidirectionalCamera.cuh"

#include <opencv2/cudev/ptr2d/glob.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*******************************
* cuda_SideBySideStitch_kernel
*	arguments
* 	right : input  data pointer (GlobPtrSz)
*	left  : input  data pointer (GlobPtrSz)
*	dst   : output data pointer (GlobPtrSz)
*	v_diff : vertical diffarence offset (int)
*	blend_width : alpha blend area width to stiting (int)
*******************************/
__global__ void cuda_SideBySideStitch_kernel( const cv::cudev::GlobPtrSz<uchar> left ,const cv::cudev::GlobPtrSz<uchar> right , cv::cudev::GlobPtrSz<uchar> dst, int v_diff, int blend_width){
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    const int dst_color_tid =  (y * dst.step) + (3 * x);

    const int left_color_tid = ((x-(dst.cols/6) + blend_width ) * right.step)  + (3 * (y +  v_diff) );
    const int right_color_tid  = (((dst.cols/6) -  x   + blend_width ) * left.step )  + (3 * (dst.rows - y +  v_diff ) );

    const double alpha = (double)( x - ((dst.cols/6) - (blend_width/2)) ) / (double)blend_width;
    
    if(x < (dst.cols/6) - (blend_width/2) ){
      dst.data[dst_color_tid + 0] = right.data[right_color_tid + 0];
      dst.data[dst_color_tid + 1] = right.data[right_color_tid + 1];
      dst.data[dst_color_tid + 2] = right.data[right_color_tid + 2];
    }else if((dst.cols/6) - (blend_width/2) <= x && x <= (dst.cols/6) + (blend_width/2)){
      dst.data[dst_color_tid + 0] = (int)((double)right.data[right_color_tid + 0]*(1-alpha) + (double)left.data[left_color_tid + 0]*alpha );
      dst.data[dst_color_tid + 1] = (int)((double)right.data[right_color_tid + 1]*(1-alpha) + (double)left.data[left_color_tid + 1]*alpha );
      dst.data[dst_color_tid + 2] = (int)((double)right.data[right_color_tid + 2]*(1-alpha) + (double)left.data[left_color_tid + 2]*alpha );
    }else if((dst.cols/6) + (blend_width/2) < x){
      dst.data[dst_color_tid + 0] = left.data[left_color_tid + 0];
      dst.data[dst_color_tid + 1] = left.data[left_color_tid + 1];
      dst.data[dst_color_tid + 2] = left.data[left_color_tid + 2];
    }
}

/*******************************
* cuda_SideBySideStitch
*	arguments
* 	right : input  data pointer (GpuMat)
*	left  : input  data pointer (GpuMat)
*	dst   : output data pointer (GpuMat)
*	v_diff : vertical diffarence offset (int)
*	blend_width : alpha blend area width to stiting (int)
*******************************/
void  OmnidirectionalCamera::cuda::SideBySideStitch(cv::cuda::GpuMat &left, cv::cuda::GpuMat &right, cv::cuda::GpuMat &dst ,int v_diff,int blend_width){

    //create image pointer
    cv::cudev::GlobPtrSz<uchar> p_Right = cv::cudev::globPtr(right.ptr<uchar>(), right.step, right.rows, right.cols * right.channels());
    cv::cudev::GlobPtrSz<uchar> p_Left  = cv::cudev::globPtr(left.ptr<uchar>() , left.step , left.rows , left.cols  * left.channels());
    cv::cudev::GlobPtrSz<uchar> p_Dst   = cv::cudev::globPtr(dst.ptr<uchar>()  , dst.step  , dst.rows  , dst.cols   * dst.channels());

    const dim3 block(32, 32);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows , block.y));

    cuda_SideBySideStitch_kernel<<<grid, block>>>( p_Right, p_Left, p_Dst , v_diff, blend_width);
    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}
