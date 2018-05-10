#include <cmath>
#include "OmnidirectionalCamera.hpp"
namespace OmnidirectionalCamera{
  /*matrix functions for PanoramaRemapperGen*/
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
}
