out.o: main.o cuda_DivAndClip.o  cuda_RingStitch.o cuda_SideBySideStitch.o
	nvcc   -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv`  -O0  -o VrStreamer  main.o cuda_RingStitch.o  cuda_DivAndClip.o cuda_SideBySideStitch.o

main.o: main.cpp OmnidirectionalCamera.hpp
	g++  -g  -c  -std=c++11 -O0   `pkg-config --cflags opencv`  main.cpp `pkg-config --libs opencv`

cuda_DivAndClip.o: cuda_DivAndClip.cu
	nvcc  -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv` -c cuda_DivAndClip.cu  
cuda_RingStitch.o: cuda_RingStitch.cu 
	nvcc  -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv` -c cuda_RingStitch.cu
cuda_SideBySideStitch.o: cuda_SideBySideStitch.cu 
	nvcc  -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv` -c cuda_SideBySideStitch.cu

clean: 
	rm main.o cuda_DivAndClip.o  cuda_RingStitch.o SideBySideStitch.o  VrStreamer
