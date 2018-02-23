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



int main(int argc , char* argv[]){


  //set camera configration
  cv::VideoCapture sink("udpsrc port=5000 -v host=localhost ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=30/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink",CAP_GSTREAMER);
  cv::Mat receive(2048,1024,CV_8UC3);
  
  while(1){  
    
    //fetch image
    insta360 >> src;
    cv::imshow(WinName,send);   
    key = cv::waitKey(1);
    
      // if input any key 
    if(key != 255 ){
      break;
    }
  }
}

