// CS 61C Fall 2014 Project 3

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include "calcDepthNaive.h"
#include "utils.h"
// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>
#endif

#define ABS(x) (((x) < 0) ? (-(x)) : (x))

// Implements the displacement function

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement, size_t* floatOps)
{
if (maximumDisplacement == 0) {
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			depth[y * imageWidth + x] = 0;
		}
	}
}
else {
__m128 zeroVector = _mm_setzero_ps();
float A[4];
for (int x = 0; x < imageWidth; x++)
	{
			if ((x < featureWidth) || (x >= imageWidth - featureWidth))
			{
				for (int y = 0; y < imageHeight; y++) {
				depth[y * imageWidth + x] = 0;
				}
				continue;
			}
	for (int y = 0; y < imageHeight; y++)
	{
			if ((y < featureHeight) || (y >= imageHeight - featureHeight)) {
				for (int i = 0; i < imageWidth / 8 * 8; i = i + 8) {
					_mm_storeu_ps(&depth[y * imageWidth + i], zeroVector);
					_mm_storeu_ps(&depth[y * imageWidth + i + 4], zeroVector);
				}
				for (int i = imageWidth / 8 * 8; i < imageWidth; i++) {
					depth[y * imageWidth + i] = 0;
				}
				continue;
			}


			float minimumSquaredDifference = FLT_MAX;
			int minimumDy = 0;
			int minimumDx = 0;
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					if (x + dx - featureWidth < 0) {
					
						continue;
					}
					if (x + dx + featureWidth >= imageWidth) {
						break;
					}
			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				if (y + dy - featureHeight < 0) {
					continue;
				}
				if (y + dy + featureHeight >= imageHeight) {
					break;
				}


					float squaredDifference = 0;
					__m128 squaredDiffVector = _mm_setzero_ps();
					#pragma omp parallel
					for (int boxX = -featureWidth; boxX <= featureWidth - 7; boxX = boxX + 8) {
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY = boxY + 1) {

						 
						
							__m128 ci = _mm_loadu_ps(&left[(y + boxY) * imageWidth + (x + boxX)]);
							__m128 ci1 = _mm_loadu_ps(&left[(y + boxY) * imageWidth + (x + boxX) + 4]);
						    __m128 di = _mm_loadu_ps(&right[(y + dy + boxY) * imageWidth + (x + dx + boxX)]);
						    __m128 di1 = _mm_loadu_ps(&right[(y + dy + boxY) * imageWidth + (x + dx + boxX) + 4]);
							__m128 diff1 = _mm_sub_ps(ci1,di1);
							__m128 diff2 = _mm_sub_ps(ci,di);
							__m128 squaredDiff1 = _mm_mul_ps(diff2,diff2);
							__m128 squaredDiff2 = _mm_mul_ps(diff1,diff1);
							squaredDiffVector = _mm_add_ps(squaredDiffVector,squaredDiff1);
							squaredDiffVector = _mm_add_ps(squaredDiffVector,squaredDiff2);
						 }
						}
						

				//	_mm_store_ps(A, squaredDiffVector);
				//	float squaredDifference1 = A[0] + A[1] + A[2] + A[3];
				//		 if (squaredDifference1 > minimumSquaredDifference) {
					//	 	continue;
				//		 }

						for (int boxX = featureWidth - ((2*featureWidth % 8)); boxX <= featureWidth - 3; boxX = boxX + 4) {
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY = boxY + 1) {

						 
						
							__m128 ci = _mm_loadu_ps(&left[(y + boxY) * imageWidth + (x + boxX)]);
							//__m128 ci1 = _mm_loadu_ps(&left[(y + boxY) * imageWidth + (x + boxX) + 4]);
						    __m128 di = _mm_loadu_ps(&right[(y + dy + boxY) * imageWidth + (x + dx + boxX)]);
						   // __m128 di1 = _mm_loadu_ps(&right[(y + dy + boxY) * imageWidth + (x + dx + boxX) + 4]);
							//__m128 diff1 = _mm_sub_ps(ci1,di1);
							__m128 diff2 = _mm_sub_ps(ci,di);
							__m128 squaredDiff1 = _mm_mul_ps(diff2,diff2);
							//__m128 squaredDiff2 = _mm_mul_ps(diff1,diff1);
							squaredDiffVector = _mm_add_ps(squaredDiffVector,squaredDiff1);
							//squaredDiffVector = _mm_add_ps(squaredDiffVector,squaredDiff2);
						 }
						}
						for (int boxX = featureWidth - ((2*featureWidth % 4)); boxX <= featureWidth; boxX = boxX + 1) {
								for (int boxY = -featureHeight; boxY <= featureHeight; boxY = boxY + 1) {
							float difference = left[(y + boxY) * imageWidth + (x + boxX)] - right[(y + dy + boxY)* imageWidth + (x + dx + boxX)];
							squaredDifference += difference * difference;									
					}
				}

					_mm_store_ps(A, squaredDiffVector);
					squaredDifference = squaredDifference + A[0] + A[1] + A[2] + A[3];


					if (((minimumSquaredDifference > squaredDifference) || (minimumSquaredDifference == squaredDifference) && (sqrt(dx*dx + dy*dy) < sqrt(minimumDx*minimumDx + minimumDy*minimumDy))))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}								

				}
				
			}
					depth[y * imageWidth + x] = sqrt(minimumDx*minimumDx + minimumDy*minimumDy);
		}
	}
}
}
	