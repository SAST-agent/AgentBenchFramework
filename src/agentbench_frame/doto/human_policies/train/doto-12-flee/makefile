all: main.out

main.out: json.o logic.o geometry.o main.o Attack.o Common.o Map.o Math.o playerAI.o SafeOp.o TaskScheduler.o
ifeq ($(OS),Windows_NT)
	g++ json.o logic.o geometry.o main.o Attack.o Common.o Map.o Math.o playerAI.o SafeOp.o TaskScheduler.o -o main.out -static-libstdc++ -lwsock32 -std=c++11 -Wall -Wextra -O2
else
	g++ json.o logic.o geometry.o main.o Attack.o Common.o Map.o Math.o playerAI.o SafeOp.o TaskScheduler.o -o main.out -static-libstdc++ -std=c++11 -pthread -Wall -Wextra -O2
endif

json.o:clean
	g++ -c jsoncpp/jsoncpp.cpp -o json.o -static-libstdc++ -std=c++11 -O2

logic.o:clean
	g++ -c logic.cpp -static-libstdc++ -std=c++11 -O2

geometry.o:clean
	g++ -c geometry.cpp -static-libstdc++ -std=c++11 -O2

main.o:clean
	g++ -c main.cpp -static-libstdc++ -std=c++11 -O1

Attack.o:clean
	g++ -c Attack.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

Common.o:clean
	g++ -c Common.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

Map.o:clean
	g++ -c Map.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

Math.o:clean
	g++ -c Math.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

playerAI.o:clean
	g++ -c playerAI.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

SafeOp.o:clean
	g++ -c SafeOp.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

TaskScheduler.o:clean
	g++ -c TaskScheduler.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2

.PHONY: clean
clean:makefile
ifeq ($(OS),Windows_NT)
	-del -r *.out *.o
else
	-rm -r *.out *.o
endif
