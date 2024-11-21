CC=g++
#CFLAGS=-Ofast -w -std=c++14 -fPIC
CFALGS=-Ofast -static

OBJ_DIR=obj
OBJ=$(OBJ_DIR)/model_basic.o\
		$(OBJ_DIR)/model.o\

TARGET=libann

.PHONY : all clean test remake write

all : mkobj $(TARGET)

mkobj :
	-mkdir $(OBJ_DIR)
	-mkdir lib

test : all
	$(CC) -c -DTEST main_example.cpp -o $(OBJ_DIR)/main_example.o -I./inc
	$(CC) -o test $(OBJ_DIR)/main_example.o -L./lib -lann -static

write : all
	$(CC) -c -DWRITE main_example.cpp -o $(OBJ_DIR)/main_example.o -I./inc
	$(CC) -o test $(OBJ_DIR)/main_example.o -L./lib -lann -static

$(TARGET) : $(OBJ)
	ar rsc lib/$(TARGET).a $(OBJ)

$(OBJ_DIR)/%.o : src/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS) -I./src -I./inc

clean :
	-rm $(OBJ_DIR)/*.o
	-rmdir $(OBJ_DIR)
	-rm lib/$(TARGET).a
	-rmdir lib
	-rm test

remake : clean all
