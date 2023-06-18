#include "cudaHelpers.h"
#include <iostream>

using namespace std;

cudaError_t cudaHandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        cerr << "CUDA ERROR: " << cudaGetErrorString(error) << " in " << file << " at line " << line << endl;
        exit(EXIT_FAILURE);
    }
    return error;
}
