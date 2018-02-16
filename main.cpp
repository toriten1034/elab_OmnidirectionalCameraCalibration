#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <thread>
#include <boost/lockfree/queue.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/stitching.hpp>
#include "OmnidirectionalCamera.hpp"
#include <omp.h>
#include <time.h>


int HorizonalCrossCount = 9;
int VerticalCrossCount = 6;


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


int MainLockfree(cv::VideoCapture& Camera, cv::VideoWriter& stream);

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
   streaming.open("appsrc ! videoconvert n-threads = 4 ! vaapih264enc tune=zerolatency byte-stream=True bitrate=15000 key-int-max=1  threads = 1  sliced-threads=true ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.100.119 port=5678",  0, 30, cv::Size(2048, 1024 ), true);

  //  streaming.open("appsrc !  videoconvert   ! autovideosink",  0, 30, cv::Size(2048, 1024 ), true);

  
  if (!streaming.isOpened()) {
    printf("=ERR= can't create capture\n");
    return -1;
  }

  //  MainLockfree(insta360 , streaming);

      
  
  // insta360.set(CV_CAP_PROP_FRAME_WIDTH, 2048);
  // insta360.set( CV_CAP_PROP_FRAME_HEIGHT, 1024);
  //insta360.set(CV_CAP_PROP_FPS, 60); 

  cv::Mat tmp;
  insta360.read(tmp);
  cv::Mat src(tmp.cols,tmp.rows,tmp.type());
  insta360.read(src);
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

    insta360.read(src);

    std::vector< double > blend_map;
    for(int i = blend_width - 1 ; i >= 0; i--){
      blend_map.push_back(((double)i/blend_width));
    }


    int64 cnt = 0;
    int64 sum = 0;
    while(1){  
      int64 start = cv::getTickCount();
      //fetch image
      insta360.read(src);
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

/*
int MainLockfree(cv::VideoCapture& Camera, cv::VideoWriter& stream){
  using std::thread;
  using boost::lockfree::queue;
  queue<cv::Mat*> src_que(2);
  queue<cv::Mat*> dst_que(2);

  double f = 1000.0f / cv::getTickFrequency();
  
  auto source_th = thread([&]{
      int64 start = 0;
      int64 end = 0;

      while(true){
	cv::Mat* src = new cv::Mat{};
	if(Camera.read(*src)){
	  while(!src_que.push(src)){ "retry"; };
	  end = cv::getTickCount();
	  std::cout << "my_threshold(parallel_for_): "  << (end - start) * f << "[ms]" << std::endl;
	  start = end;

	}else{
	  delete src;
	  while(!src_que.push(nullptr)){ "retry"; }
        break;
	}
      }
    });
  
  auto serial_th = thread{[&](){
      while(true){

	cv::Mat* src = nullptr;
	while(!src_que.pop(src)){"retry";}
	if(src != nullptr){
	  cv::Mat *dst = new cv::Mat{};
	  cv::Mat tmp;
	  cv::cvtColor(*src, tmp, cv::COLOR_RGB2GRAY);
	  cv::cvtColor(tmp, *dst, cv::COLOR_GRAY2RGB);
	  cv::resize(*dst,*dst,cv::Size(2048,1024), 0);
     	  delete src;
	  while(!dst_que.push(dst)){ "retry"; }
	}else{
	  while(!dst_que.push(nullptr)){"retry";}
	  break;
	}	
      }
    }};

  auto sink_th = thread{[&](){
    while(true){
      cv::Mat *dst = nullptr;
      while(!dst_que.pop(dst)){"retry";}
      if(dst != nullptr){
	stream.write(*dst);
	delete dst;
      }else{
	break;
      }
    }
    }};
  source_th.join();
  serial_th.join();
  sink_th.join();
  return EXIT_SUCCESS;

}

*/
