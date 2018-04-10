out.o: main.o cuda_DivAndClip.o  cuda_Join.o
	nvcc   -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv`  -O0  -o VrStreamer  main.o cuda_DivAndClip.o  cuda_Join.o

main.o: main.cpp OmnidirectionalCamera.hpp
	g++  -g  -c  -std=c++11 -O0   `pkg-config --cflags opencv`  main.cpp `pkg-config --libs opencv`

cuda_DivAndClip.o: cuda_DivAndClip.cu
	nvcc  -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv` -c cuda_DivAndClip.cu  
cuda_Join.o: cuda_Join.cu 
	nvcc  -std=c++11  `pkg-config --cflags opencv` `pkg-config --libs opencv` -c cuda_Join.cu  
clean: 
	rm main.o cuda_DivAndClip.o  cuda_Join.o  VrStreamer
