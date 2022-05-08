CC = gcc # Flag for implicit rules
CFLAGS = -g -O1 -pg # Flag gives to compiler, for implicit rules(allow variable initial within loop); Turn on debug info; if using OpenMP
CPPFLAGS = -lrt -lm # Extra flag gives to C preprocessor, for dynamic linking with pthread library; and for math library
objects = test_1d_conv

all: $(objects)

# gcc -O1 test_mmm_block.c -lrt -o test_mmm_block

# %.o : %.c
# 	$(CC) -c $(CFLAGS) $< -lrt -o $@
test_1d_conv: ./test_1d_conv.cu
	nvcc -G $(CFLAGS) $< -o $@
	./$@ >> ./test_1d_conv.txt
	gprof ./$@ gmon.out > analysis.txt
	nvprof ./$@ >> analysis.txt

clear: #Remove all object code
	rm -rf *.o 

zip: Lab8_Report.pdf makefile 
	zip Zhengqi_Dong_lab8.zip test_1d_conv.cu

# lab3: lab3.o flight_profiles.o helper_function.o plane_allocation_and_recycling.o
# 	gcc -g $^ -L. -llinkedlist -latc -lncurses -lm -o $@