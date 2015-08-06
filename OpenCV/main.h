#ifndef __OpenCV__main__
#define __OpenCV__main__

#include "Pixar.h"

double computeDeltaC(Pixel Pi, Region Rj);

// implementation of texture uniformity criterion among a pixel and a region
double computeDeltaT(Pixel Pi, Region Rj);

// implementation of the region uniformity criterion (it's a combination among the color and texture feature)
double computeDeltaR(Pixel pi, Region rj);

// implementation of the symmetry uniformity criterion
double computeDeltaS(Pixel Pi, Region rj, Mat C);

// extraction of the 8-neighbours without edge
vector<Pixel*> get8_neighborPixel(vector< vector<Pixel*> > &label,int i,int j);

// pixels extraction from a 7x7 window
vector<Pixel*> get49_neighborPixel(vector< vector<Pixel*> > &label,int i,int j);

// Computing the Gradient Vector Flow
vector<Mat> GVF(Mat f, double mu, int iter);

// Computing the Curve Gradient Vector Flow (CGVF)
Mat Curv(vector<Mat> V);

// Computing affinity symmetry of a pixel
double computeSymmetryAffinity(Mat curv,vector<Pixel*> neighbors,int i,int j);

// computing the STD's color feature gradient of the region
double computeGraR_color(Region r,Region r_old);

// computing the STD's texture feature gradient of the region
double computeGraR_texture(Region r,Region r_old);

// implementation of the criterion for merging regions
void regionMergingCriterion(vector<Region> VR);

// Routine for segmentation
Mat segmentation(vector< vector<Pixel *> > &label,int n_row,int n_col);

// coloring segments of the output image
Mat drawLine(Mat seg, Mat rgb);

#endif
