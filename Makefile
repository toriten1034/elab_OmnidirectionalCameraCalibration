main: main.cpp fish_eye.hpp
	g++ -g  -O0 -o fish_eye `pkg-config --cflags opencv`  main.cpp `pkg-config --libs opencv` 
