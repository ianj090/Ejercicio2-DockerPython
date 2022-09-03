# Ejercicio 2 Dockerfile con Python

Ian Jenatz 20190014

## Steps

1. Download all files
2. Run the following command in terminal: `docker image build -t flask_docker .`
3. Run the following command in terminal: `docker run -d -p 5000:5000 flask_docker`
4. Navigate to localhost:5000/inputs or 127.0.0.1/5000 to access the app
5. An example input is: 

X Coefficient: 10
Y Coefficient: 15
Amount of Constraints: 3

10, 15
282, 400, <=, 2000
4, 40, <=, 140
1, 0, <=, 5

Final output should be: Optimal Solution: Max Value is 70, Point of Interest is (4, 2)