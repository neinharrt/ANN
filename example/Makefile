CC=g++
#CFLAGS=-Ofast -w -std=c++14 -fPIC
CFALGS=-O2 -static -std=c++20

OBJ_DIR=obj

.PHONY : all clean main

all : mkobj main

mkobj :
	-mkdir $(OBJ_DIR)

main: 
	$(CC) -c main.cpp -o $(OBJ_DIR)/main_example.o -I./inc
	$(CC) -o test $(OBJ_DIR)/main_example.o -L./lib -lann -static

clean :
	-rm $(OBJ_DIR)/*.o
	-rmdir $(OBJ_DIR)
	-rm test

remake : clean all
