#pragma once

#include "cudaHelpers.h"
#include "glmHelpers.h"

/**
 * Prefix-sum implementation in CUDA
 *
 * @param cuData cuda array of numbers
 * @param size number count
 * @param sum output sum of all numbers (can be null)
 * @param threadCount
 */
void prefixSum(int* cuData, int size, int* sum = nullptr, int threadCount = 64);
