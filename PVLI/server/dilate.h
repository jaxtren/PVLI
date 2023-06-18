#pragma once

#include "cudaHelpers.h"
#include "glmHelpers.h"

/**
 * Dilate color
 *
 * replace empty (black) pixels with averaged 4-neighbors non-empty pixels within same block
 *
 * @param cuColor cuda color buffer
 * @param width
 * @param height
 * @param blockSize size of dilated and cuda block
 * @param thickness dilate distance / number of dilate iterations
 */
void dilate(glm::u8vec3* cuColor, int width, int height, int blockSize, int thickness = 0);
