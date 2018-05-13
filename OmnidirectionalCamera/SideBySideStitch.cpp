#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
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
void OmnidirectionalCamera::SideBySideStitch(cv::Mat left, cv::Mat right, cv::Mat dst ,int v_diff,int blend_width){
    
  int input_width = left.cols;
  int input_height = left.rows;
  int dst_width = dst.cols;
  int dst_height = dst.rows;

    
  cv::Vec3b *p_dst   = dst.ptr<cv::Vec3b>(0);
  cv::Vec3b *p_left = left.ptr<cv::Vec3b>(0);
  cv::Vec3b *p_right = right.ptr<cv::Vec3b>(0);

    
  for(int i = 0; i < dst_height; i++){ //y

    for(int j = 0 ; j  <   (dst_width/2)-(blend_width) ; j++ ){
      int dst_Index = dst_width * i + j;
      int src_IndexLeft = (input_width) * (j + (blend_width*2)  )  + (i+v_diff);
      int src_IndexRight = (input_width) * (input_height - j  ) + (input_width-i) ;

	
      p_dst[dst_Index + (dst_width/2) + (blend_width)][0] = (int)p_left[src_IndexLeft][0]; 
      p_dst[dst_Index + (dst_width/2) + (blend_width)][1] = (int)p_left[src_IndexLeft][1]; 
      p_dst[dst_Index + (dst_width/2) + (blend_width)][2] = (int)p_left[src_IndexLeft][2]; 

      p_dst[dst_Index][0] = (int)p_right[src_IndexRight ][0];
      p_dst[dst_Index][1] = (int)p_right[src_IndexRight ][1];
      p_dst[dst_Index][2] = (int)p_right[src_IndexRight ][2];

    }
      
    for(int j =  (dst_width/2)-(blend_width) ; j < input_height;  j++ ){
      int dst_Index = dst_width * i + j;

      int src_IndexLeft = (input_width) * (j - (input_height - blend_width*2 ))  + (i+v_diff);
      int src_IndexRight = (input_width) * (input_height-j  ) + (input_width-i) ;

      double alpha_Index = (double)(input_height - j)/((double)blend_width*2);

      //	p_dst[dst_Index][0] = p_left[src_IndexLeft][0];
      //	p_dst[dst_Index][1] = p_left[src_IndexLeft][1];
      //	p_dst[dst_Index][2] = p_left[src_IndexLeft][2];
      /*if(alpha_Index < 0 || 1.0 < alpha_Index){
	std::cout << "Error" << std::endl;
	}*/
	
      p_dst[dst_Index][0] = (int) ( (double) p_right[src_IndexRight ][0] *alpha_Index + (double)p_left[src_IndexLeft][0] * (1 - alpha_Index ) );
      p_dst[dst_Index][1] = (int) ( (double) p_right[src_IndexRight ][1] *alpha_Index + (double)p_left[src_IndexLeft][1] * (1 - alpha_Index ) );
      p_dst[dst_Index][2] = (int) ( (double) p_right[src_IndexRight ][2] *alpha_Index + (double)p_left[src_IndexLeft][2] * (1 - alpha_Index ) );


      /*((double)dst_width/2 - j)/(blend_width) + (double)p_right[src_IndexLeft][2]*(1-((double)dst_width - j)/(blend_width/2)); */
	
    }
      
  }
}

