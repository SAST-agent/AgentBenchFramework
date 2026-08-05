all: main.out

main.out: json.o logic.o geometry.o main.o Attack.o Common.o Map.o Math.o playerAI.o SafeOp.o TaskScheduler.o Analyzer.o MoveOperator.o AttackOperator.o Sampler.o
ifeq ($(OS),Windows_NT)
	g++ json.o logic.o geometry.o main.o Attack.o Common.o Map.o Math.o playerAI.o SafeOp.o TaskScheduler.o Analyzer.o MoveOperator.o AttackOperator.o Sampler.o -o main.out -static-libstdc++ -lwsock32 -std=c++11 -Wall -Wextra -O2
else
	g++ json.o logic.o geometry.o main.o Attack.o Common.o Map.o Math.o playerAI.o SafeOp.o TaskScheduler.o Analyzer.o MoveOperator.o AttackOperator.o Sampler.o -o main.out -static-libstdc++ -std=c++11 -pthread -Wall -Wextra -O2
endif

json.o:clean
	g++ -c SDK/jsoncpp/jsoncpp.cpp -o json.o -static-libstdc++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0

logic.o:clean
	g++ -c SDK/logic.cpp -static-libstdc++ -std=c++11 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

geometry.o:clean
	g++ -c SDK/geometry.cpp -static-libstdc++ -std=c++11 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

main.o:clean
	g++ -c SDK/main.cpp -static-libstdc++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0

Attack.o:clean
	g++ -c Attack.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

Common.o:clean
	g++ -c Common.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

Map.o:clean
	g++ -c Map.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

Math.o:clean
	g++ -c Math.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

playerAI.o:clean
	g++ -c playerAI.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

SafeOp.o:clean
	g++ -c SafeOp.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TaskScheduler.o:clean
	g++ -c TaskScheduler.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

Analyzer.o:clean
	g++ -c Analyzer.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

MoveOperator.o:clean
	g++ -c MoveOperator.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

AttackOperator.o:clean
	g++ -c AttackOperator.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

Sampler.o:clean
	g++ -c Sampler.cpp -static-libstdc++ -std=c++11 -Wall -Wextra -O2 -D_GLIBCXX_USE_CXX11_ABI=0

.PHONY: clean
clean:makefile
ifeq ($(OS),Windows_NT)
	-del -r *.out *.o
else
	-rm -r *.out *.o
endif
