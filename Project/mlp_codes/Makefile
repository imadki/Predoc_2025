CC = gcc
CFLAGS = -O0 -Wall -g
OBJ = main.o model.o utils.o
TARGET = mlp

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -lm -o $(TARGET)

main.o: main.c model.h utils.h
	$(CC) $(CFLAGS) -c main.c

model.o: model.c model.h utils.h
	$(CC) $(CFLAGS) -c model.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o $(TARGET)
