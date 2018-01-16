#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

namespace OmnidirectionalCamera{
  #define ORTHOGRAPHIC  0
  #define STEREOGRAPHIC 1
  #define EQUISOLID     2
  #define EQUIDISTANT   3
  /****************************************************
   *This function generate fish eye image remapper,You 
    can select fish eye lens type by mode argment
   *argments
   *   src   :input  : image size
   *   x_map :output : x remapper
   *   y_map :output : y remapper
   *   mode  :input  : select lens type
   ****************************************************/
  void OmnidirectionalCameraRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode){
    int width = src.width;
    int dist_width  = x_map.cols;
    int height = src.height;
    for(int y = 0; y < height; y++){
      for(int x = 0; x < dist_width; x++){
	double py = y - (height / 2);
	double px = x - (dist_width / 2);
	
	//spherical coordinate
	double theta = (px/((double)dist_width/2)) *(M_PI/2);
	double phi =  (py/((double)height/2)) * (M_PI / 2);
	
	//point on orthogonal projection
	double d_x = sin(theta)*cos(phi)*(dist_width/2);
	double d_y = sin(phi)*(height/2);
	
	double r = sqrt(pow(d_x,2)+pow(d_y,2));
	double r_angle = asin(r/(height/2));
	//correct by focal length
	// double f = (height/height/2)/sin(r_angle)/2;
	double fr;
	switch(mode){
	case ORTHOGRAPHIC:
	  fr = sin(r_angle/2);
	  break;
	case STEREOGRAPHIC:
	  fr = tan(r_angle/2);
	}
	double rate = fr/(r/(height/2));
	
	double c_x = d_x*rate;
	double c_y = d_y*rate;
	
	unsigned short real_x = (int) ((dist_width / 2) +  c_x );
	unsigned short real_y = (int) ((height / 2) +  c_y ) ;

	if(real_y < 0|| height < real_y  ){
	  std::cout << "range error" << std::endl;
	}	
	x_map.ptr< unsigned short >(y)[x] = real_x ;
	y_map.ptr< unsigned short >(y)[x] = real_y ;
      }
    }
  }
  /**********************************************
   *This Function FishEyeImage to panoramaImage
   *src : input :input FishEye Image
   *dst : output: Output remapped Image;
   *xmap: input : xmap
   *ymap: input : ymap
   *********************************************/
  void OmnidirectionalImageRemap(cv::Mat src, cv::Mat dst, cv::Mat xmap, cv::Mat ymap){
    int src_width = src.cols;
    int dst_width = dst.cols;
    if(xmap.cols != dst.cols || xmap.rows != dst.rows){
      std::cout << "error" << std::endl;
    }
    
    int height = xmap.rows;
    cv::Vec3b *p_src = src.ptr<cv::Vec3b>(0);      
    
    for(int y = 0 ; y < height; y++){
      cv::Vec3b *p_dst = dst.ptr<cv::Vec3b>(y);
      unsigned short *p_xmap  = xmap.ptr<unsigned short>(y);
      unsigned short *p_ymap  = ymap.ptr<unsigned short>(y);
      for(int x = 0 ; x < dst_width; x++){
	int index = p_ymap[x] * src_width + p_xmap[x];
	p_dst[x] = p_src[index];
      }
    }
  }
  
  void OmnidirectionalImgJoin(cv::Mat right, cv::Mat left, cv::Mat src ,int diff){
    int width = src.cols/2;
    int height = src.rows;
    for(int i = 0; i < height; i++){
      cv::Vec3b *p_src   = src.ptr<cv::Vec3b>(i);
      cv::Vec3b *p_right = right.ptr<cv::Vec3b>(i);
      cv::Vec3b *p_left  = left.ptr<cv::Vec3b>(i);
      for(int j = 0; j < width; j++){
      p_src[j] = p_left[j];
      p_src[j + width] = p_right[j] ;
      }
    }
  }
}
