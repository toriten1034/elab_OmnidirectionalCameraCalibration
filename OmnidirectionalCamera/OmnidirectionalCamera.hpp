#ifndef INCLUDED_OMNI
#define INCLUDED_OMNI

#include <iostream>
#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <omp.h>
namespace OmnidirectionalCamera{

  const int  ORTHOGRAPHIC  = 0; //orthographic 
  const int  STEREOGRAPHIC = 1; //stereographic 
  const int  EQUISOLID     = 2; //equisolid  e.g insta360 Air
  const int  EQUIDISTANT   = 3; //equidistant fisheye

  namespace matrix{
    void rot_x(double src[3],double dst[3], double angle);
    void rot_y(double src[3],double dst[3], double angle);
    double inner(double a[3], double b[3]);
  }

  void PanoramaRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,int view_angle);
  void PanoramaGpuRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,int view_angle);
  //Almost Omnidirectional Camera View of angle is 180
  void PanoramaRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode);
  
  void BirdsEyeViewRemapperGen(const cv::Rect src,cv::Mat x_map,cv::Mat y_map ,int mode,double view_angle);
  void RingStitch(cv::Mat left, cv::Mat right, cv::Mat dst ,int v_diff,int blend_width);
  void SideBySideStitch(cv::Mat left, cv::Mat right, cv::Mat dst ,int v_diff,int blend_width);
}

#endif
