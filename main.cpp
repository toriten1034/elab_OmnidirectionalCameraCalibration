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
#include<omp.h>
#include <time.h>

int HorizonalCrossCount = 9;
int VerticalCrossCount = 6;

//dvide image
void ImgDiv(cv::Mat src, cv::Mat right, cv::Mat left ){
  int width = right.cols;
  int height = right.rows;
  
  cv::Vec3b *p_right = right.ptr<cv::Vec3b>(0);
  cv::Vec3b *p_left  = left.ptr<cv::Vec3b>(0);
#pragma omp parallel for
  for(int y = 0 ; y < height; y++){
    cv::Vec3b *p_src   = src.ptr<cv::Vec3b>(y);
    #pragma omp parallel for
    for(int x = 0 ; x < width; x++){
      p_left[(width*y) +  x] = p_src[ x];
      p_right[(width*y) +  x] = p_src[x+ width];
    }
  }
}



void ImgClip(cv::Mat src, cv::Mat dst,cv::Rect range){
  int width = range.width;
  int height = range.height;
  int offset_x = range.x;
  int offset_y = range.y;
  cv::Vec3b *p_dst = dst.ptr<cv::Vec3b>(0);
#pragma omp parallel for
  for(int i = 0; i < height; i++){
    cv::Vec3b *p_src   = src.ptr<cv::Vec3b>(i+offset_y);
    #pragma omp parallel for
    for(int j = 0; j < width; j++){
      p_dst[i*width + j] = p_src[j+ offset_x];
    }
  }
}

void ImgDivAndClip(cv::Mat src, cv::Mat right, cv::Mat left ,cv::Rect range ){
  int src_width  = src.cols/2;
  int src_height = src.rows;
  int width    = range.width;
  int height   = range.height;
  int offset_x = range.x;
  int offset_y = range.y;

  cv::Vec3b *p_right = right.ptr<cv::Vec3b>(0);
  cv::Vec3b *p_left  = left.ptr<cv::Vec3b>(0);
  //#pragma omp parallel 
  for(int y = 0 ; y < height; y++){
    cv::Vec3b *p_src   = src.ptr<cv::Vec3b>(y + offset_y);
    //#pragma omp parallel 
    for(int x = 0 ; x < width; x++){
      p_left[(width*y) +  x] = p_src[ x + offset_x];
      p_right[(width*y) +  x] = p_src[x+ src_width + offset_x];
    }
  }

}

// cv::parallel_for_利用コード
class TestParallelLoopBody : public cv::ParallelLoopBody
{
private:
  cv::Mat &_src;
  cv::Mat &_dst;
  cv::Mat &_x_map;
  cv::Mat &_y_map;

public:
  TestParallelLoopBody
  (
   cv::Mat &src, cv::Mat &dst, cv::Mat &x_map, cv::Mat &y_map
   )
    : _src(src), _dst(dst), _x_map(x_map), _y_map(y_map) { }
  
  void operator() (const cv::Range& range) const
  {
    int row0 = range.start;
    int row1 = range.end;
    cv::Mat dstStripe = _dst.rowRange(row0, row1);
    cv::Mat x_mapStripe = _x_map.rowRange(row0, row1);
    cv::Mat y_mapStripe = _y_map.rowRange(row0, row1);
    
    //my_threshold(srcStripe, dstStripe, _thresh, _max_value);
    OmnidirectionalCamera::OmnidirectionalImageRemap( _src, dstStripe,  x_mapStripe, y_mapStripe);
  }
};






int main(int argc , char* argv[]){
  double f = 1000.0f / cv::getTickFrequency();
  
  if(argc != 2){
    std::cout << "too few argments" << std::endl;
    return -1;
  }

  #ifdef HAVE_TBB
  printf("YES tbb");
  #endif
  
  int cam_id = std::atoi(argv[1]); 
  int clip = 25;
  int hdiff = 0;
  int mode =  OmnidirectionalCamera::EQUISOLID;
  int blend_width = 13;
  int gammma = 1;/*0-20 to 0.0-2.0*/

  std::cout << "s to capture" << std::endl;
  std::cout << "ESC s to Exit Program" << std::endl;

  //set camera configration
  cv::VideoCapture insta360(cam_id);
  cv::VideoWriter streaming;
  
  //streaming.open("appsrc ! videoconvert ! x264enc tune=zerolatency byte-stream=True bitrate=32768 key-int-max=1  threads = 1   ! h264parse ! rtph264pay config-interval=5 pt=96 ! udpsink host=192.168.100.119 port=5678",  0, 30, cv::Size(3008, 1504 ), true);
  streaming.open("appsrc ! videoconvert n-threads = 4 ! x264enc tune=zerolatency byte-stream=True bitrate=15000 key-int-max=1  threads = 1  sliced-threads=true ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.100.119 port=5678",  0, 30, cv::Size(2048, 1024 ), true);

  if (!streaming.isOpened()) {
    printf("=ERR= can't create capture\n");
    return -1;
  }

  
  // insta360.set(CV_CAP_PROP_FRAME_WIDTH, 2048);
  // insta360.set( CV_CAP_PROP_FRAME_HEIGHT, 1024);
  insta360.set(CV_CAP_PROP_FPS, 60); 

  cv::Mat tmp;
  insta360 >> tmp;
  cv::Mat src(tmp.cols,tmp.rows,tmp.type());
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

    OmnidirectionalCamera::OmnidirectionalCameraRemapperGen(roi, x_mat, y_mat , mode , 183);

    cv::Mat y_mat_s(1024, 1024,CV_16UC1);
    cv::Mat x_mat_s(1024, 1024,CV_16UC1);

    cv::resize(y_mat,y_mat_s,cv::Size(1024,1024), 0);
    cv::resize(x_mat,x_mat_s,cv::Size(1024,1024), 0);

    //    cv::Mat send(3008,1504,CV_8UC3);
    cv::Mat send(2048,1024,CV_8UC3);

    insta360 >> src;

    std::vector< double > blend_map;
    for(int i = blend_width - 1 ; i >= 0; i--){
      blend_map.push_back(((double)i/blend_width));
    }


    int64 cnt = 0;
    int64 sum = 0;
    while(1){  
      int64 start = cv::getTickCount();
      //fetch image
      insta360 >> src;

      cv::Mat cliped_r(roi.width, roi.height,src.type());
      cv::Mat cliped_l(roi.width, roi.height,src.type());
      

      ImgDivAndClip(src,cliped_r,cliped_l,roi);
      
      cv::Mat result_right(x_mat_s.rows, x_mat_s.cols, src.type());
      cv::Mat result_left(x_mat_s.rows, x_mat_s.cols, src.type());
      
      
      OmnidirectionalCamera::OmnidirectionalImageRemap(cliped_r, result_right,x_mat_s,y_mat_s);
      OmnidirectionalCamera::OmnidirectionalImageRemap(cliped_l, result_left,x_mat_s,y_mat_s);
      
      cv::Mat Join( result_right.rows -hdiff,(result_right.cols - blend_width)*2,result_right.type());
      OmnidirectionalCamera::OmnidirectionalImgJoin(result_right,result_left,Join ,hdiff,blend_width,blend_map);
      //*****streaming***********
      cv::resize(Join,send,cv::Size(2048,1024), 0);
     
      streaming.write(send);
      cv::imshow(WinName,Join);
   
      //****display***********
      cv::imshow(WinName,src);
      cv::createTrackbar("Clip",WinName,&clip,70);
      cv::createTrackbar("Height diff",WinName,&hdiff,70);
      cv::createTrackbar("blend width",WinName,&blend_width,70);
      
      int64 end = cv::getTickCount();
      sum += (end - start);
      cnt += 1;
      std::cout << "my_threshold(parallel_for_): "  << sum / cnt * f << "[ms]" << std::endl;
  
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

