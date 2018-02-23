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
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaimgproc.hpp>
//shared memory
#include <sys/ipc.h>
#include <sys/shm.h>

#include "OmnidirectionalCamera.cuh"


int MainLockfree(cv::VideoCapture& Camera, cv::VideoWriter& stream);

int main(int argc , char* argv[]){
  double f = 1000.0f / cv::getTickFrequency();


  if(argc <= 2 || argc % 2 == 0){
    std::cout << "too few argments" << std::endl;
    return -1;
  }

  
  int cam_id = 0;
  int sharedMemoryKey = -1; // -1 is shared memory disable
  std::string ipAddress("192.168.100.119");    
  int clip = 28; //clip range 
  int vdiff = 6; //vertical diffarence offset
  int blendWidth = 51; //alpha blend width to stitting


  // initialize by option
  for(int i = 1; i < argc ; i+=2){
    std::string op(argv[i]);
    if(op == std::string("-id")){
      cam_id =  std::atoi(argv[i+1]);
    }else if(op == std::string("-share")){
      sharedMemoryKey =   std::atoi(argv[i+1]);
    }else if(op == std::string("-ip")){
      ipAddress = std::string(argv[i+1]);
    }else if(op == std::string("-clip")){
      clip = std::atoi(argv[i+1]);
    }else if(op == std::string("-diff")){
      vdiff = std::atoi(argv[i+1]);
    }else if(op == std::string("-bw")){
      blendWidth = std::atoi(argv[i+1]);
    }
  }

  
  std::cout << "*******program status************************"  << std::endl;
  std::cout << "(-id)    camera id is          :" << cam_id << std::endl;
  std::cout << "(-share) shared memory key     :" << ((sharedMemoryKey == -1)? "disable" : std::to_string(sharedMemoryKey)) << std::endl;
  std::cout << "(-ip)    ip address is         :" << ipAddress  << std::endl;
  std::cout << "(-clip)  clip range is         :" << clip  << std::endl;
  std::cout << "(-diff)  ertical diffarence is :" << vdiff  << std::endl;
  std::cout << "(-bw)    alpha blend width is  :" << blendWidth  << std::endl;

  int *sharedMemory;
  int sharedMemoryID;
  if(sharedMemoryKey != -1){
    sharedMemoryID = shmget(sharedMemoryKey,4,0);
    if(sharedMemoryID == -1){
      std::cout << "bad key" << std::endl;
      return 1;
    }
    sharedMemory = (int*)shmat(sharedMemoryID, NULL, SHM_RDONLY);
  }
  

  
  int mode =  OmnidirectionalCamera::EQUISOLID; //set fisheye lens type
  std::cout << "ESC s to Exit Program" << std::endl;

  //set camera configration
  cv::VideoCapture insta360(cam_id);
  cv::VideoWriter streaming;

  char gstreamerCommand[200];
  sprintf(gstreamerCommand,"appsrc ! videoconvert ! x264enc tune=zerolatency byte-stream=True bitrate=15000 key-int-max=1 threads = 1 ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=%s port=5678",ipAddress.c_str());
  streaming.open(gstreamerCommand, 0, 30, cv::Size(2048, 1024 ), true);

  if (!streaming.isOpened()) {
    printf("=ERR= can't create capture\n");
    return -1;
  }
  
  insta360.set(CV_CAP_PROP_FRAME_WIDTH, 2048);
  insta360.set( CV_CAP_PROP_FRAME_HEIGHT, 1024);
  insta360.set(CV_CAP_PROP_FPS, 60); 

  cv::Mat tmp;
  insta360.read(tmp);
  cv::Mat src(tmp.cols,tmp.rows,tmp.type());
  insta360.read(src);
  cv::Rect roi = cv::Rect(clip,clip,src.cols/2 - (2*clip) ,src.rows - (2*clip) );

  const std::string WinName = "Joind";
  int key = 0;
  int clip_t = 0;;
  int vdiff_t = 0;

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


    if(sharedMemoryKey != -1){
      if(!*sharedMemory){
	break;
      }
    }

    std::cout << "clip range i " << clip << std::endl;
    std::cout << "key is "<< key << std::endl;
    roi = cv::Rect(clip,clip,src.cols/2 - (2*clip) ,src.rows - (2*clip) );
        
    cv::Mat yMap(roi.width, roi.height,CV_32FC1);
    cv::Mat xMap(roi.width, roi.height,CV_32FC1);

    OmnidirectionalCamera::OmnidirectionalCameraGpuRemapperGen(roi, xMap, yMap,  mode , 183);
    
    cv::cuda::GpuMat d_xMap(xMap.size(), xMap.type());
    cv::cuda::GpuMat d_yMap(xMap.size(), yMap.type());

    d_xMap.upload( xMap );
    d_yMap.upload( yMap );

    //    cv::Mat send(3008,1504,CV_8UC3);
    cv::Mat send(2048,1024,CV_8UC3);

    insta360.read(src);

    int64 cnt = 0;
    int64 sum = 0;

    int64 a,b;
    while(1){  
      int64 start = cv::getTickCount();
      //fetch image
      insta360.read(src);

      //********** computing image in gpu *************
      //Mat named d_xxx is GpuMat
      cv::cuda::GpuMat d_src(src);
      cv::cuda::GpuMat d_clipedRight(roi.width, roi.height,src.type());
      cv::cuda::GpuMat d_clipedLeft(roi.width, roi.height,src.type());

      OmnidirectionalCamera::cuda::DivAndClip(d_src ,d_clipedRight, d_clipedLeft,roi);
      

      cv::cuda::GpuMat d_resultRight(xMap.rows, xMap.cols, src.type());
      cv::cuda::GpuMat d_resultLeft(xMap.rows, xMap.cols, src.type());

      cv::cuda::remap(d_clipedRight , d_resultRight, d_xMap, d_yMap ,cv::INTER_LINEAR );
      cv::cuda::remap(d_clipedLeft  , d_resultLeft, d_xMap, d_yMap ,cv::INTER_LINEAR );

      cv::cuda::GpuMat d_Join(d_resultRight.rows -vdiff, (d_resultRight.cols - blendWidth)*2, d_resultRight.type());
      OmnidirectionalCamera::cuda::Join(d_resultRight,d_resultLeft,d_Join ,vdiff ,blendWidth );

      cv::cuda::GpuMat d_send(send);
     
      //*****streaming***********
      cv::cuda::resize(d_Join,d_send,cv::Size(2048,1024), 0);
      d_send.download(send);
     
      streaming.write(send);
   
      //****display***********
      cv::imshow(WinName,send);
      cv::createTrackbar("Clip",WinName,&clip,70);
      cv::createTrackbar("Height diff",WinName,&vdiff,70);
      cv::createTrackbar("blend width",WinName,&blendWidth,70);
      
      int64 end = cv::getTickCount();
      sum += (end - start);
      cnt += 1;
      std::cout << "frame interval:"  << sum / cnt * f << "[ms]" << std::endl;
  
      key = cv::waitKey(1);
      
      // if input any key 
      if(key != 255 || clip_t != clip){
	clip_t = clip;
	vdiff_t = vdiff;
	break;
      }

      if(sharedMemoryKey != -1){
	if(!*sharedMemory){
	  break;
	}
      }
      
    }
  }
}

