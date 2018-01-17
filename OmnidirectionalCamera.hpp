#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

namespace OmnidirectionalCamera{
  const int  ORTHOGRAPHIC  = 0; //orthographic 
  const int  STEREOGRAPHIC = 1; //stereographic 
  const int  EQUISOLID     = 2; //equisolid  e.g insta360 Air
  const int  EQUIDISTANT   = 3; //equidistant fisheye
  
  /****************************************************
   *This function generate fish eye image remapper,You 
    can select fish eye lens type by mode argment
   *argments
   *   src   :input  : image size
   *   x_map :output : x remapper
   *   y_map :output : y remapper
   *   mode  :input  : select lens type
   *   angle :input  : angle of view 0 to 360
   ****************************************************/
  void OmnidirectionalCameraRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,int view_angle){
    if(x_map.cols != y_map.cols || x_map.cols != y_map.cols){
      CV_Error(cv::Error::StsBadSize, "map size is not same");
    }
    if(!(0 <= view_angle and view_angle < 360)){
      CV_Error(cv::Error::StsOutOfRange, "angle is out of range, it must set between 0~360 ");      
    }
    if(!(0 <= view_angle and view_angle < 1800) && mode == ORTHOGRAPHIC){
     CV_Error(cv::Error::StsOutOfRange, "angle is out of range, when use ORTHOGRAPHIC, it must set between 0~180 ");     
    }

    double radian_angle = M_PI*((double)view_angle / 180.0);
      
    int width = src.width;
    int dist_width  = x_map.cols;
    int height = src.height;
    for(int y = 0; y < height; y++){
      for(int x = 0; x < dist_width; x++){
        //Point on dst image
	double py = y - (height / 2);
	double px = x - (dist_width / 2);
	
	double theta = (px/((double)dist_width/2)) *(radian_angle/2);
	double phi =  (py/((double)height/2)) * (radian_angle / 2);
	
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
	  break;
	case EQUISOLID:
	  fr = sqrt(2)*sin(r_angle/2);
	  break;
	case EQUIDISTANT:
	  fr = r_angle/radian_angle;
	  break;
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
  
  //Almost Omnidirectional Camera View of angle is 180
  void OmnidirectionalCameraRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode){
    OmnidirectionalCameraRemapperGen(src, x_map, y_map ,mode,180);
  }
  
  /**********************************************
   *This Function FishEyeImage to panoramaImage
   *src : input :input FishEye Image
   *dst : output: Output remapped Image;
   *x_map: input : xmap
   *y_map: input : ymap
   *********************************************/
  void OmnidirectionalImageRemap(cv::Mat src, cv::Mat dst, cv::Mat x_map, cv::Mat y_map){
    int src_width = src.cols;
    int dst_width = dst.cols;
    
    if(x_map.cols != dst.cols || x_map.rows != dst.rows||y_map.cols != dst.cols || y_map.rows != dst.rows){
      CV_Error(cv::Error::StsBadSize, "map size is not same");
    }
    
    int height = x_map.rows;
    cv::Vec3b *p_src = src.ptr<cv::Vec3b>(0);      
    
    for(int y = 0 ; y < height; y++){
      cv::Vec3b *p_dst = dst.ptr<cv::Vec3b>(y);
      unsigned short *p_x_map  = x_map.ptr<unsigned short>(y);
      unsigned short *p_y_map  = y_map.ptr<unsigned short>(y);
      for(int x = 0 ; x < dst_width; x++){
	int index = p_y_map[x] * src_width + p_x_map[x];
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
