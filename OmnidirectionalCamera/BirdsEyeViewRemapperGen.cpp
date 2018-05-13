#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <omp.h>
#include "OmnidirectionalCamera.hpp"

/****************************************************
*This function generate bird's eye view rimapper
*argments
*   src   :input  : image size
*   x_map :output : x remapper
*   y_map :output : y remapper
*   mode  :input  : select lens type
*   angle :input  : angle of view 0 to 360
****************************************************/  
void OmnidirectionalCamera::BirdsEyeViewRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,double view_angle){
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
  int height = src.height;

  int dist_width  = x_map.cols;
  int dist_height  = x_map.rows;

  for(int y = 0; y < dist_height; y++){
    for(int x = 0; x < dist_width; x++){
      //Point on dst image
      double py = y;
      double px = x - (dist_width / 2);

      //angles
      double l = sqrt(pow(px,2) + pow(py,2));
      double h = x_map.rows;
      double theta = atan((double)px/(double)( py));
      double phi = (M_PI/2) - atan(l/h);


      if( theta < -M_PI/2  ||   M_PI/2 < theta){
	std::cout << "theta range error" << std::endl;	  
      }

      if(phi < -M_PI/2 || M_PI/2 <=0 ){
	std::cout << "phi range error" << std::endl; 
      }

      //point on orthogonal projection
      double d_x = sin(theta)*cos(phi)*(width/2);
      double d_y = sin(phi)*(height/2);

      double r = sqrt(pow(d_x,2)+pow(d_y,2));


      double Vec0[3] = {0,0,1.0};
      double VecX[3];
      matrix::rot_x(Vec0,VecX,theta);
      double VecY[3];
      matrix::rot_y(VecX,VecY,phi);
      double r_angle = matrix::inner(Vec0,VecY); 

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

      float real_x = (float) ((width / 2) +  c_x );
      float real_y = (float) ((height / 2) +  c_y ) ;

      if(real_y < 0|| height < real_y  ){
	std::cout << "range error" << std::endl;
      }	
      x_map.ptr<float>(y)[x] = real_x;
      y_map.ptr<float>(y)[x] = real_y ;
    }
  }    
}
