/*
Student Name: Aysu Sayın
Student Number: 2016400051
Compile Status: Compiling
Program Status: Working
Notes:
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <bits/stdc++.h>
#include <time.h>

using namespace std;

// Copies the contents of original two dimensional array to the copy array.
void copy(int **original, int **copy, int row_size, int col_size) {
    for (int i = 0; i < row_size; i++) {
        copy[i] = new int[col_size];
        for (int j = 0; j < col_size; j++) {
            copy[i][j] = original[i][j];
        }
    }
}

// Calculate the sum of the elements in the given 2d array within the given bounds except for the element with the row
// number row_i and column number col_j.
int calculateSum(int **arr, int row_i, int col_j, int row_lower_bound, int row_upper_bound, int col_lower_bound,
                 int col_upper_bound) {
    int sum = 0;
    for (int a = row_lower_bound; a <= row_upper_bound; a++) {
        for (int b = col_lower_bound; b <= col_upper_bound; b++) {
            sum += arr[a][b];
        }
    }
    sum = sum - arr[row_i][col_j];
    return sum;
}

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // End the program with a single warning message if the required arguments are not given.
    if (world_rank == 0 && argc < 5) {
        cout << "Run this code with 4 arguments." << endl;
        cout << "e.g. mpiexec -n [processor_no] ./mpi_project [input_file] [output_file] [beta] [pi]" << endl;
        MPI_Finalize();
        return 0;
    } else if (argc < 5) {
        MPI_Finalize();
        return 0;
    }
    double beta = atof(argv[3]);  // Input image is generated from Ising Model parametrized by beta.
    double pi = atof(argv[4]);  // Probability of a random flip
    int size = 200;  // Size of the image is size*size
    int rowSize = size / (world_size - 1); // Row size of the subarrays

    if (world_rank == 0) {  // Master Process
        ofstream outputTextFile;  // Output text file
        ifstream imageTextFile;  // Input text file
        imageTextFile.open(argv[1]);
        int **imageArr = new int *[size];  // Text array of the original image
        // Read from input file
        for (int i = 0; i < size; i++) {
            imageArr[i] = new int[size];
            for (int j = 0; j < size; j++) {
                imageTextFile >> imageArr[i][j];
            }
        }
        // Send a subarray to each slave processor
        for (int i = 1; i < world_size; i++) {
            for (int j = 0; j < rowSize; j++) {
                MPI_Send(imageArr[rowSize * (i - 1) + j], size, MPI_INT, i, j, MPI_COMM_WORLD);
            }
        }
        // Recieve subarrays from slave processors
        for (int i = 1; i < world_size; i++) {
            for (int j = 0; j < rowSize; j++) {
                MPI_Recv(imageArr[rowSize * (i - 1) + j], size, MPI_INT, i, j, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
        outputTextFile.open(argv[2]);
        // Write to output file
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                outputTextFile << imageArr[i][j] << " ";
            }
            outputTextFile << "\n";
        }
        cout << "Output generated: " << argv[2] << endl;
    } else {
        int **imageSubArr = new int *[rowSize];  // Subarray of the original image
        int **flippedArr = new int *[rowSize];  // Subarray who's elements are going to be flipped with a probability
        int *nextRow = new int[size];  // The first row of flipped subarray of the process with rank+1
        int *preRow = new int[size];  // The last row of flipped subarray of the process with rank-1
        double gamma = 0.5 * log((1 - pi) / pi);
        // Recieve the subarray
        for (int r = 0; r < rowSize; r++) {
            imageSubArr[r] = new int[size];
            MPI_Recv(imageSubArr[r], size, MPI_INT, 0, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Copy original subarray to the array to be flipped.
        copy(imageSubArr, flippedArr, rowSize, size);
        int step =
                500000 / (world_size - 1);  // Count of the steps (How many times we are going to propose a bit flip.)
        int i, j; // Row no and column no
        // Initialize random number generator
        random_device r;
        seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        mt19937 gen(seed);
        // Initialize distributions
        uniform_int_distribution<> dis0(0, rowSize - 1);
        uniform_int_distribution<> dis1(0, size - 1);
        uniform_real_distribution<double> dis2(0, 1);
        for (int s = 0; s < step; s++) {
            // Choose a random bit.
            i = dis0(gen);
            j = dis1(gen);
            // Send and receive the last row.
            // To prevent deadlock let the processes with even rank send the message first.
            if (world_rank != world_size - 1 && world_rank % 2 == 0) {
                MPI_Send(imageSubArr[rowSize - 1], size, MPI_INT, world_rank + 1, rowSize - 1, MPI_COMM_WORLD);
            }
            // To prevent deadlock let the processes with odd rank receive the message first.
            if (world_rank != 1 && world_rank % 2 == 1) {
                MPI_Recv(preRow, size, MPI_INT, world_rank - 1, rowSize - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Then the processes with odd rank send the message.
            if (world_rank != world_size - 1 && world_rank % 2 == 1) {
                MPI_Send(imageSubArr[rowSize - 1], size, MPI_INT, world_rank + 1, rowSize - 1, MPI_COMM_WORLD);
            }
            // Then the processes with even rank receive the message.
            if (world_rank != 1 && world_rank % 2 == 0) {
                MPI_Recv(preRow, size, MPI_INT, world_rank - 1, rowSize - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Send and receive first row.
            // To prevent deadlock let the processes with even rank send the message first.
            if (world_rank != 1 && world_rank % 2 == 0) {
                MPI_Send(imageSubArr[0], size, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
            }
            // To prevent deadlock let the processes with odd rank receive the message first.
            if (world_rank != world_size - 1 && world_rank % 2 == 1) {
                MPI_Recv(nextRow, size, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Then the processes with odd rank send the message.
            if (world_rank != 1 && world_rank % 2 == 1) {
                MPI_Send(imageSubArr[0], size, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
            }
            // Then the processes with even rank receive the message.
            if (world_rank != world_size - 1 && world_rank % 2 == 0) {
                MPI_Recv(nextRow, size, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Determine the lower bounds and upper bounds for the calcSum function and if the process needs to
            // communicate with another process, calculate the sum of the elements in the other process then add
            // the result of the calcSum function.
            int rowLowerBound, rowUpperBound, colLowerBound, colUpperBound, sum;
            if (i == 0) {
                rowLowerBound = 0;
                rowUpperBound = i + 1;
                if (j == 0) {
                    // Upper left corner
                    colLowerBound = 0;
                    colUpperBound = 1;
                    sum = preRow[0] + preRow[1];
                } else if (j == size - 1) {
                    // Upper bits
                    colLowerBound = j - 1;
                    colUpperBound = j;
                    sum = preRow[colLowerBound] + preRow[colUpperBound];
                } else {
                    // Upper right corner
                    colLowerBound = j - 1;
                    colUpperBound = j + 1;
                    sum = preRow[colLowerBound] + preRow[j] + preRow[colUpperBound];
                }
            } else if (i == rowSize - 1) {
                rowLowerBound = i - 1;
                rowUpperBound = i;
                if (j == 0) {
                    // Lower left corner
                    colLowerBound = 0;
                    colUpperBound = 1;
                    sum = nextRow[0] + nextRow[1];
                } else if (j == size - 1) {
                    // Lower bits
                    colLowerBound = j - 1;
                    colUpperBound = j;
                    sum = nextRow[colLowerBound] + nextRow[colUpperBound];
                } else {
                    // Lower right corner
                    colLowerBound = j - 1;
                    colUpperBound = j + 1;
                    sum = nextRow[colLowerBound] + nextRow[j] + nextRow[colUpperBound];
                }
            } else {
                rowLowerBound = i - 1;
                rowUpperBound = i + 1;
                colLowerBound = max(j - 1, 0);
                colUpperBound = min(j + 1, size - 1);
                sum = 0;
            }
            sum = sum + calculateSum(flippedArr, i, j, rowLowerBound, rowUpperBound, colLowerBound, colUpperBound);
            // Calculate acceptanceP
            double acceptanceP =
                    exp(((-2.0) * gamma * imageSubArr[i][j] * flippedArr[i][j]) - (2 * beta * flippedArr[i][j] * sum));
            // Flip the bit with acceptanceP probability
            if (dis2(gen) < min(1.0, acceptanceP)) {
                flippedArr[i][j] = -1 * flippedArr[i][j];
            }
        }
        // Send flipped array to the master process
        for (int r = 0; r < rowSize; r++) {
            MPI_Send(flippedArr[r], size, MPI_INT, 0, r, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}