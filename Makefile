cudamag:
	clear
	nvcc main.cu cudamag.cu magnet.cu -o mag.out
	./mag.out