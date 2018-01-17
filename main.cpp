#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/stitching.hpp>
#include "OmnidirectionalCamera.hpp"

int HorizonalCrossCount = 9;
int VerticalCrossCount = 6;

//dvide image
void ImgDiv(cv::Mat src, cv::Mat right, cv::Mat left){
  int width = src.cols/2;
  int height = src.rows;
  for(int i = 0; i < height; i++){
    cv::Vec3b *p_src   = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *p_right = right.ptr<cv::Vec3b>(i);
    cv::Vec3b *p_left  = left.ptr<cv::Vec3b>(i);
    for(int j = 0; j < width; j++){
      p_left[j] = p_src[j];
      p_right[j] = p_src[j + width];
    }
  }
}



void ImgClip(cv::Mat src, cv::Mat dst,cv::Rect range){
  int width = range.width;
  int height = range.height;
  int offset_x = range.x;
  int offset_y = range.y;

  for(int i = 0; i < height; i++){
    cv::Vec3b *p_src   = src.ptr<cv::Vec3b>(i+offset_y);
    cv::Vec3b *p_dst   = dst.ptr<cv::Vec3b>(i);
    for(int j = 0; j < width; j++){
      p_dst[j] = p_src[j+ offset_x];
    }
  }
}



int main(int argc , char* argv[]){
  if(argc != 2){
    std::cout << "too few argments" << std::endl;
    return -1;
  }

  int cam_id = std::atoi(argv[1]); 
  int clip = 47;
  int hdiff = 5;
  int focus = 1;
  int mode =  OmnidirectionalCamera::EQUISOLID;
  int wclip = 0;
  int gammma = 1;/*0-20 to 0.0-2.0*/

  std::cout << "s to capture" << std::endl;
  std::cout << "ESC s to Exit Program" << std::endl;

  //set camera configration
  cv::VideoCapture insta360(cam_id);
  insta360.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
  insta360.set( CV_CAP_PROP_FRAME_HEIGHT, 1080);  
  
  cv::Mat src;
  insta360 >> src;
  cv::Rect roi = cv::Rect(clip,clip,src.cols/2 - (2*clip) ,src.rows - (2*clip) );

  const std::string WinName = "Joind";
  int key = 0;

  int clip_t = 0;;
  int hdiff_t = 0;
  while(1){

    if(key == 49){
      if(0 < clip ){
	clip--;
      }  
    }else if (key == 50){
      if(0 < 1000 ){
	clip++;
      }
    }else if(key == 51){
      mode = mode + 1;
      mode = mode % 4;
      std::cout << "mode is "<< mode << std::endl;
    }else if(key == 27){
      break;
    }
    std::cout << "clip range i " << clip << std::endl;
    std::cout << "key is "<< key << std::endl;
    roi = cv::Rect(clip,clip,src.cols/2 - (2*clip) ,src.rows - (2*clip) );
        
    cv::Mat y_mat(roi.width, roi.height,CV_16UC1);
    cv::Mat x_mat(roi.width, roi.height,CV_16UC1);
    OmnidirectionalCamera::OmnidirectionalCameraRemapperGen(roi, x_mat, y_mat , mode , 185);

    while(1){  
            
      //fetch image
      insta360 >> src;
      cv::Mat right(src.cols/2, src.rows,right.type());
      cv::Mat left(src.cols/2, src.rows,right.type()); 
      ImgDiv(src,right,left);
      cv::Mat cliped_r(roi.width, roi.height,src.type());
      cv::Mat cliped_l(roi.width, roi.height,src.type());
      
      ImgClip(right,cliped_r,roi);
      ImgClip(left,cliped_l,roi);
      cv::Mat result_right(x_mat.rows, x_mat.cols, src.type());
      cv::Mat result_left(x_mat.rows, x_mat.cols, src.type());
      
      // cv::imshow("debug2",cliped_r);
      
      OmnidirectionalCamera::OmnidirectionalImageRemap(cliped_r, result_right,x_mat,y_mat);
      OmnidirectionalCamera::OmnidirectionalImageRemap(cliped_l, result_left,x_mat,y_mat);
      
      cv::Mat Join( result_right.rows -hdiff,result_right.cols*2 - (2*wclip),result_right.type());
      OmnidirectionalCamera::OmnidirectionalImgJoin(result_right,result_left,Join ,hdiff,5);
      cv::imshow(WinName,Join);
      cv::createTrackbar("Clip",WinName,&clip,50);
      cv::createTrackbar("Height diff",WinName,&hdiff,50);
      cv::createTrackbar("wclip",WinName,&wclip,50);
 
      key = cv::waitKey(1);
      
      // if input any key 
      if(key != 255 || clip_t != clip){

	clip_t = clip;
	hdiff_t = hdiff;
	break;
      }
    }
  }
}

