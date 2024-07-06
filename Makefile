cudamag:
	clear
	nvcc -Xcompiler -fPIC -shared -o cudamag_cu.so cudamag.cu