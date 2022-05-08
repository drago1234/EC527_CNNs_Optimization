# Defines
CC         = gcc
OBJ_DIR    = ./obj
SRC_DIR    = ./src
INCL_DIR   = ./include
OUTPUT_DIR = ./output
OBJECTS    = $(addprefix $(OBJ_DIR)/, read_csv.o write_csv.o forward_propagation.o back_propagation.o mlp_trainer.o mlp_classifier.o)
INCLUDES   = $(addprefix $(INCL_DIR)/, read_csv.h write_csv.h forward_propagation.h back_propagation.h mlp_trainer.h mlp_classifier.h parameters.h)
CFLAGS     = -g -Wall -O1 -pg		# -pg: with gprof
EXECUTABLE = MLP

CPPFLAGS =  -O1 -lrt -lm # Extra flag gives to C preprocessor, for dynamic linking with pthread library; and for math library

# Generate the executable file
$(EXECUTABLE): $(SRC_DIR)/main.c $(OBJECTS)
	$(CC) $(CFLAGS) $< $(OBJECTS) -o $(EXECUTABLE) -I $(INCL_DIR) -lm
	./MLP 3 4,5,5 softmax,relu,tanh 1 sigmoid 0.01 1 data/data_train.csv 1096 5 data/data_test.csv 275 5 

# Compile and Assemble C source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INCLUDES)
	$(CC) $(CFLAGS) -I $(INCL_DIR) -c $< -o $@

# Clean the generated executable file and object files
clean:
	rm -f $(OBJECTS)
	rm -rf $(EXECUTABLE)*

############## ==> Customized GPU Test obj
cuda_add: $(SRC_DIR)/cuda_add.cu
	nvcc -g -G $(CPPFLAGS) $< -o $@
	./$@ >> $(OUTPUT_DIR)/$@.txt

cuda_sor:  $(SRC_DIR)/cuda_sor.cu
	nvcc -g -G $(CPPFLAGS) $< -o $@
	./$@ >> $(OUTPUT_DIR)/$@.txt

test_1d_conv: $(SRC_DIR)/test_1d_conv.cu
	nvcc -g -G $(CPPFLAGS) $< -o $@
	./$@ >> $(OUTPUT_DIR)/$@.txt

test_1d_conv_no_padding: $(SRC_DIR)/test_1d_conv_no_padding.cu
	nvcc -g -G $(CPPFLAGS) $< -o $@
	./$@ >> $(OUTPUT_DIR)/test_1d_conv_no_padding_task5.txt

test_mat_mul: $(SRC_DIR)/test_mat_mul.cu
	nvcc -g -G $(CPPFLAGS) $< -o $@
	./$@
