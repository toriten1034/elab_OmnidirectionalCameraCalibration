#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include<omp.h>
#include "OmnidirectionalCamera.hpp"

/**********************************************
 *This Function Join 2 remmaped omnidirectional 
 * Image smooth
 *left : input :input remmaped a side Image
 *right : input :input remmaped other sideI mage 
 *dst : output: Output remapped Image;
 *x_map: input : xmap
 *y_map: input : ym
 *********************************************/
  
  
void OmnidirectionalCamera::RingStitch(cv::Mat left, cv::Mat right, cv::Mat dst ,int v_diff,int blend_width){
    
  int dst_width = dst.cols;
  int height = left.rows;
  for(int i = 0; i < height-v_diff; i++){
    cv::Vec3b *p_dst   = dst.ptr<cv::Vec3b>(i);
    cv::Vec3b *p_left = left.ptr<cv::Vec3b>(i);
    cv::Vec3b *p_right = right.ptr<cv::Vec3b>(i + v_diff);
    for(int j = 0 ; j  < ( (dst_width/2)- blend_width); j++ ){
      p_dst[j] = p_left[j + blend_width];
      p_dst[j + (dst_width/2)] = p_right[j + blend_width];
    }
    for(int j = ( (dst_width/2)- blend_width); j < (dst_width); j++){
      int Bindex = j - ( (dst_width/2)- blend_width);
      double alpha = ( j - ( (dst_width/2)- blend_width) ) / (double)blend_width;
	
      p_dst[j][0] = (int)( (double)p_left[j + blend_width][0] * alpha  + (double)p_right[Bindex][0] * (1 - alpha) );
      p_dst[j][1] = (int)( (double)p_left[j + blend_width][1] * alpha  + (double)p_right[Bindex][1] * (1 - alpha) );
      p_dst[j][2] = (int)( (double)p_left[j + blend_width][2] * alpha  + (double)p_right[Bindex][2] * (1 - alpha) );

      p_dst[j + (dst_width/2)][0] = (int)( (double)p_right[j + blend_width][0]*alpha  + (double)p_left[j - ( (dst_width/2)- blend_width)][0] * (1-alpha) );
      p_dst[j + (dst_width/2)][1] = (int)( (double)p_right[j + blend_width][1]*alpha  + (double)p_left[j - ( (dst_width/2)- blend_width)][1] * (1-alpha) );
      p_dst[j + (dst_width/2)][2] = (int)( (double)p_right[j + blend_width][2]*alpha  + (double)p_left[j - ( (dst_width/2)- blend_width)][2] * (1-alpha) );

    }
  }
}


