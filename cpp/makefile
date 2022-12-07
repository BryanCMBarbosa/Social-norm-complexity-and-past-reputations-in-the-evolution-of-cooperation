build:
	g++ src/Individual.cpp src/Simulation.cpp main.cpp -o main -I /opt/homebrew/opt/boost/include -std=c++17

clean:
	@echo "Cleaning..."
	rm -f main
	@echo "Done."

clean_csv:
	rm -f *.csv

exec:
	./main