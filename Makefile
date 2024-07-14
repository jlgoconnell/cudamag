cudamag:
	clear
	nvcc -Xcompiler -fPIC -shared -o cudamag_cu.so cudamag.cu -l cublas
	python3 ./test_script.py