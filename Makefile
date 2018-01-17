main: main.cpp 
	g++ -g  -O0 -o Omnio `pkg-config --cflags opencv`  main.cpp `pkg-config --libs opencv` 
