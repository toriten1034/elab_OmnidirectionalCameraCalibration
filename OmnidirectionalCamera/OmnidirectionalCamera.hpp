
#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include<omp.h>

namespace OmnidirectionalCamera{
  const int  ORTHOGRAPHIC  = 0; //orthographic 
  const int  STEREOGRAPHIC = 1; //stereographic 
  const int  EQUISOLID     = 2; //equisolid  e.g insta360 Air
  const int  EQUIDISTANT   = 3; //equidistant fisheye

  /*matrix functions for OmnidirectionalCameraRemapperGen*/
  namespace matrix{
    void rot_x(double src[3],double dst[3], double angle){
      dst[0] = src[0] +       0           +  0;
      dst[1] =    0   + src[1]*cos(angle) -  src[2]*sin(angle);
      dst[2] =    0   + src[1]*sin(angle) +  src[2]*cos(angle);
    }
    
    void rot_y(double src[3],double dst[3], double angle){
      dst[0] = src[0]*cos(angle)  +   0    +  src[2]*sin(angle);
      dst[1] =         0          + src[1] +        0;
      dst[2] = -src[0]*sin(angle) +   0    +  src[2]*cos(angle);
    }
    
    double inner(double a[3], double b[3]){
      return  acos((a[0]*b[0] +  a[1]*b[1] +  a[2]*b[2]) / (sqrt( pow(a[0],2) + pow(a[1],2) + pow(a[2],2)) * sqrt( pow(a[0],2) + pow(a[1],2) + pow(a[2],2) ) ));
    }
  }
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
    if(!(0 <= view_angle and view_angle < 180) && mode == ORTHOGRAPHIC){
     CV_Error(cv::Error::StsOutOfRange, "angle is out of range, when use ORTHOGRAPHIC, it must set between 0~180 ");     
    }

    double angle_rate = (double)view_angle / 180.0;  
    double radian_angle = M_PI*angle_rate;
    
    int width = src.width;
    int dist_width  = x_map.cols;
    int height = src.height;
    for(int y = 0; y < height; y++){
      for(int x = 0; x < dist_width; x++){
        //Point on dst image
	double py = y - (height / 2);
	double px = x - (dist_width / 2);
	//angles
	double theta = (px/((double)dist_width/2)) *(radian_angle/2);
	double phi =  (py/((double)height/2)) * (radian_angle / 2);

	double tmp = M_PI;
	if( theta/angle_rate < -M_PI/2  ||   M_PI/2 < theta/angle_rate){
	  std::cout << "theta range error" << std::endl;	  
	}

	if(phi/angle_rate < -M_PI/2 || M_PI/2 <  phi/angle_rate ){
	  std::cout << "phi range error" << std::endl;	  
	}
	
	//point on orthogonal projection
	double d_x = sin(theta/angle_rate)*cos(phi/angle_rate)*(dist_width/2);
	double d_y = sin(phi/angle_rate)*(height/2);
	    
	double r = sqrt(pow(d_x,2)+pow(d_y,2));
	

	double Vec0[3] = {0,0,1.0};
	double VecX[3];
	matrix::rot_x(Vec0,VecX,theta);
	double VecY[3];
	matrix::rot_y(VecX,VecY,phi);
	//double r_angle = asin(r/(height/2));
	double r_angle = matrix::inner(Vec0,VecY); 
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
	//fit to image width
	fr = fr / angle_rate;
	
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
  void OmnidirectionalCameraGpuRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,int view_angle){
    if(x_map.type() != CV_32FC1 or y_map.type() != CV_32FC1){
      CV_Error(cv::Error::StsBadArg, "invalid type of maps, only CV_32FC1");      
    }

    if(!(0 <= view_angle and view_angle < 360)){
      CV_Error(cv::Error::StsOutOfRange, "angle is out of range, it must set between 0~360 ");      
    }
    if(!(0 <= view_angle and view_angle < 180) && mode == ORTHOGRAPHIC){
     CV_Error(cv::Error::StsOutOfRange, "angle is out of range, when use ORTHOGRAPHIC, it must set between 0~180 ");     
    }

    double angle_rate = (double)view_angle / 180.0;  
    double radian_angle = M_PI*angle_rate;
    
    int width = src.width;
    int dist_width  = x_map.cols;
    int height = src.height;
    for(int y = 0; y < height; y++){
      for(int x = 0; x < dist_width; x++){
        //Point on dst image
	double py = y - (height / 2);
	double px = x - (dist_width / 2);
	//angles
	double theta = (px/((double)dist_width/2)) *(radian_angle/2);
	double phi =  (py/((double)height/2)) * (radian_angle / 2);

	double tmp = M_PI;
	if( theta/angle_rate < -M_PI/2  ||   M_PI/2 < theta/angle_rate){
	  std::cout << "theta range error" << std::endl;	  
	}

	if(phi/angle_rate < -M_PI/2 || M_PI/2 <  phi/angle_rate ){
	  std::cout << "phi range error" << std::endl;	  
	}
	
	//point on orthogonal projection
	double d_x = sin(theta/angle_rate)*cos(phi/angle_rate)*(dist_width/2);
	double d_y = sin(phi/angle_rate)*(height/2);
	    
	double r = sqrt(pow(d_x,2)+pow(d_y,2));
	

	double Vec0[3] = {0,0,1.0};
	double VecX[3];
	matrix::rot_x(Vec0,VecX,theta);
	double VecY[3];
	matrix::rot_y(VecX,VecY,phi);
	//double r_angle = asin(r/(height/2));
	double r_angle = matrix::inner(Vec0,VecY); 
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
	//fit to image width
	fr = fr / angle_rate;
	
	double rate = fr/(r/(height/2));
	
	double c_x = d_x*rate;
	double c_y = d_y*rate;
	
        float real_x = (float) ((dist_width / 2) +  c_x );
	float real_y = (float) ((height / 2) +  c_y ) ;

	if(real_y < 0|| height < real_y  ){
	  std::cout << "range error" << std::endl;
	}	
	x_map.ptr<float>(y)[x] = real_x ;
	y_map.ptr<float>(y)[x] = real_y ;
      }
    }
  }

  //Almost Omnidirectional Camera View of angle is 180
  void OmnidirectionalCameraRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode){
    OmnidirectionalCameraRemapperGen(src, x_map, y_map ,mode,180);
  }
  
  /**********************************************
   *This Function OmnidirectionalImage to panoramaImage
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
    cv::Vec3b *p_dst = dst.ptr<cv::Vec3b>(0);
    unsigned short *p_x_map  = x_map.ptr<unsigned short>(0);
    unsigned short *p_y_map  = y_map.ptr<unsigned short>(0);

    //#pragma omp parallel for
    for(int y = 0 ; y < height; y++){
      //#pragma omp parallel for
      for(int x = 0 ; x < dst_width; x++){
	int index = p_y_map[y*dst_width + x] * src_width + p_x_map[y*dst_width + x];
	p_dst[y*dst_width + x] = p_src[index];
      }
    }
  }

  /**********************************************
   *This Function Join 2 remmaped omnidirectional 
   * Image smooth
   *left : input :input remmaped a side Image
   *right : input :input remmaped other sideI mage 
   *dst : output: Output remapped Image;
   *x_map: input : xmap
   *y_map: input : ym
   *********************************************/
  
  
  void OmnidirectionalImgJoin(cv::Mat left, cv::Mat right, cv::Mat dst ,int h_diff,int blend_width, std::vector< double >& blend_map){
    
    int input_width = left.cols;
    int dst_width = dst.cols;
    int height = left.rows;
    #pragma omp parallel for
    for(int i = 0; i < height-h_diff; i++){
      cv::Vec3b *p_dst   = dst.ptr<cv::Vec3b>(i);
      cv::Vec3b *p_left = left.ptr<cv::Vec3b>(i);
      cv::Vec3b *p_right = right.ptr<cv::Vec3b>(i + h_diff);
      #pragma omp parallel for
      for(int j = 0 ; j  < ( (dst_width/2)- blend_width); j++ ){
	p_dst[j] = p_left[j + blend_width];
	p_dst[j + (dst_width/2)] = p_right[j + blend_width];
      }
      #pragma omp parallel for
      for(int j = ( (dst_width/2)- blend_width); j < (dst_width/2); j++){
	int Aindex = j - ( (dst_width/2)- blend_width);
	int Bindex = j - ( (dst_width/2)- blend_width);

	
	p_dst[j][0] = (int)( (double)p_left[j + blend_width][0] * blend_map[Aindex]  + (double)p_right[Bindex][0] * (1-blend_map[j - ( (dst_width/2)- blend_width)]) );
	p_dst[j][1] = (int)( (double)p_left[j + blend_width][1] * blend_map[Aindex]  + (double)p_right[Bindex][1] * (1-blend_map[j - ( (dst_width/2)- blend_width)]));
	p_dst[j][2] = (int)( (double)p_left[j + blend_width][2] * blend_map[Aindex]  + (double)p_right[Bindex][2] * (1-blend_map[j - ( (dst_width/2)- blend_width)]));

	p_dst[j + (dst_width/2)][0] = (int)( (double)p_right[j + blend_width][0] * blend_map[j - ( (dst_width/2)- blend_width)]  + (double)p_left[j - ( (dst_width/2)- blend_width)][0] * (1-blend_map[j - ( (dst_width/2)- blend_width)]) );
	p_dst[j + (dst_width/2)][1] = (int)( (double)p_right[j + blend_width][1] * blend_map[j - ( (dst_width/2)- blend_width)]  + (double)p_left[j - ( (dst_width/2)- blend_width)][1] * (1-blend_map[j - ( (dst_width/2)- blend_width)]));
	p_dst[j + (dst_width/2)][2] = (int)( (double)p_right[j + blend_width][2] * blend_map[j - ( (dst_width/2)- blend_width)]  + (double)p_left[j - ( (dst_width/2)- blend_width)][2] * (1-blend_map[j - ( (dst_width/2)- blend_width)]));

      }
    }
  }
}
