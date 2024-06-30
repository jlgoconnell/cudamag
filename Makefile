cudamag:
	clear
	nvcc -Xcompiler -fPIC -shared -o cudainterface.so cudainterface.cpp cudamag.cu