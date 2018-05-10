VERSION=1.0
#for g++
CFLAGS=-g -Wall -fPIC -c  -std=c++11 -O0

#for nvcc
NVCC=nvcc
NVCCFLAGS = -c -std=c++11 -arch=sm_30 -Xcompiler "-fPIC"

INSTALL_PATH =  /usr/local/lib

main: libOmnidirectionalCamera libCudaOmnidirectionalCamera
	mkdir build
	mv  libOmnidirectionalCamera.so  build/libOmnidirectionalCamera.so
	mv  libCudaOmnidirectionalCamera.so  build/libCudaOmnidirectionalCamera.so
	echo "success"
#
# cpu
#
libOmnidirectionalCamera: PanoramaRemapperGen.lo BirdsEyeViewRemapperGen.lo RingStitch.lo SideBySideStitch.lo matrix.lo
	$(CXX) -shared -Wl,-soname=libOmnidirectionalCamera.so,-rpath,/usr/local/lib/ -o libOmnidirectionalCamera.so  OmnidirectionalCamera.hpp  PanoramaRemapperGen.lo BirdsEyeViewRemapperGen.lo SideBySideStitch.lo RingStitch.lo matrix.lo

matrix.lo:	matrix.cpp OmnidirectionalCamera.hpp
	$(CXX) $(CFLAGS) `pkg-config --cflags opencv` matrix.cpp `pkg-config --libs opencv` -o matrix.lo

PanoramaRemapperGen.lo:	PanoramaRemapperGen.cpp OmnidirectionalCamera.hpp
	$(CXX) $(CFLAGS) `pkg-config --cflags opencv` PanoramaRemapperGen.cpp `pkg-config --libs opencv` -o PanoramaRemapperGen.lo

BirdsEyeViewRemapperGen.lo: BirdsEyeViewRemapperGen.cpp OmnidirectionalCamera.hpp
	$(CXX) $(CFLAGS) `pkg-config --cflags opencv` BirdsEyeViewRemapperGen.cpp `pkg-config --libs opencv` -o BirdsEyeViewRemapperGen.lo

RingStitch.lo:  RingStitch.cpp OmnidirectionalCamera.hpp
	$(CXX) $(CFLAGS) `pkg-config --cflags opencv` RingStitch.cpp `pkg-config --libs opencv` -o RingStitch.lo

SideBySideStitch.lo:   SideBySideStitch.cpp OmnidirectionalCamera.hpp
	$(CXX) $(CFLAGS) `pkg-config --cflags opencv` SideBySideStitch.cpp `pkg-config --libs opencv` -o SideBySideStitch.lo	

#
# gpu 
#
libCudaOmnidirectionalCamera: cuda_DivAndClip.o cuda_SideBySideStitch.o cuda_RingStitch.o 
	$(NVCC) -std=c++11 -arch=sm_30 -Xcompiler "-fPIC" -Xlinker "-rpath,/usr/local/lib" -shared -o libCudaOmnidirectionalCamera.so  cuda_DivAndClip.o cuda_SideBySideStitch.o cuda_RingStitch.o 

cuda_DivAndClip.o: cuda_DivAndClip.cu OmnidirectionalCamera.cuh
	$(NVCC) $(NVCCFLAGS) `pkg-config --cflags opencv` cuda_DivAndClip.cu `pkg-config --libs opencv`  -o cuda_DivAndClip.o

cuda_SideBySideStitch.o: cuda_SideBySideStitch.cu OmnidirectionalCamera.cuh
	$(NVCC) $(NVCCFLAGS) `pkg-config --cflags opencv` cuda_SideBySideStitch.cu `pkg-config --libs opencv` -o cuda_SideBySideStitch.o

cuda_RingStitch.o: cuda_RingStitch.cu OmnidirectionalCamera.cuh
	$(NVCC) $(NVCCFLAGS) `pkg-config --cflags opencv` cuda_RingStitch.cu `pkg-config --libs opencv` -o cuda_RingStitch.o 


#clean
clean: 
	-rm libOmnidirectionalCamera.so matrix.lo PanoramaRemapperGen.lo BirdsEyeViewRemapperGen.lo RingStitch.lo SideBySideStitch.lo
	-rm cuda_DivAndClip.o  cuda_RingStitch.o cuda_SideBySideStitch.o
	-rm -rf build
install:
	-cp  build/libOmnidirectionalCamera.so      /usr/local/lib/libOmnidirectionalCamera.so
	-cp  build/libCudaOmnidirectionalCamera.so  /usr/local/lib/libCudaOmnidirectionalCamera.so
uninstall:
	-rm /usr/local/lib/libOmnidirectionalCamera.so
	-rm /usr/local/lib/libCudaOmnidirectionalCamera.so
