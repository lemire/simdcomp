# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h

CFLAGS = -fPIC -std=c99 -O3 -Wall -Wextra -Wno-unused-parameter -pedantic
LDFLAGS = -shared
all:  unit libsimdcomp.so
test: 
	./unit
install: $(OBJECTS)
	cp libsimdcomp.so /usr/local/lib
	cp $(HEADERS) /usr/local/include 



HEADERS=./include/simdbitpacking.h ./include/simdcomputil.h ./include/simdintegratedbitpacking.h ./include/simdcomp.h 

uninstall:
	for h in $(HEADERS) ; do rm  /usr/local/$$h; done
	rm  /usr/local/lib/libsimdcomp.so

OBJECTS= simdbitpacking.o simdintegratedbitpacking.o simdcomputil.o

libsimdcomp.so: $(OBJECTS)
	$(CC) $(CFLAGS) -o libsimdcomp.so $(OBJECTS)  $(LDFLAGS) 



simdcomputil.o: ./src/simdcomputil.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdcomputil.c -Iinclude  

simdbitpacking.o: ./src/simdbitpacking.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdbitpacking.c -Iinclude  

simdintegratedbitpacking.o: ./src/simdintegratedbitpacking.c  $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdintegratedbitpacking.c -Iinclude  

unit: ./src/unit.c    $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o unit ./src/unit.c -Iinclude  $(OBJECTS)

clean: 
	rm -f unit *.o libsimdcomp.so
