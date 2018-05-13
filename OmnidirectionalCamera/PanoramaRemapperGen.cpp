#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include "OmnidirectionalCamera.hpp"



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
void OmnidirectionalCamera::PanoramaRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,int view_angle){
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

//Almost Omnidirectional Camera View of angle is 180
void OmnidirectionalCamera::PanoramaRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode){
  PanoramaRemapperGen(src, x_map, y_map ,mode,180);
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

void OmnidirectionalCamera::PanoramaGpuRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,int view_angle){
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

