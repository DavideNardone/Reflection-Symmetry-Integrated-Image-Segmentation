#ifndef OpenCV_Pixar_h
#define OpenCV_Pixar_h

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <vector>
#include <iostream>
#include <string>
#include <unistd.h>
#include <cmath>

using namespace cv;
using namespace std;

// Image representation
class Pixel {
    
private:
    int etichetta;
    bool bordo;
    int x0;
    int y0;
    Vec<unsigned, 3> HSVvec; // 3D vector representing HSV component for each pixel
    
protected:
    vector<double> *Ftexture; // texture feature descriptor for each pixel
    Vec<double, 3> Fcolor;    // color feature descriptor for each pixel
    
public:
    Pixel();
    // Constructor
    Pixel(Vec<unsigned, 3> HSV,bool val,int i,int j,int et);
    ~Pixel();
    
    int getEtichetta() const;
    void setEtichetta(int E);
    
    bool getBordo();
    
    Vec3b getHSVvec();
    
    int getX0();
    int getY0();
    
    virtual Vec3d getFcolorVec();
    virtual vector<double> getFtextureVec();
    
    // color feature implementation for each pixel
    virtual void computeFcolor();
    
    // texture feature implementation for each pixel
    // 7x7 neighbours window for each pixel (monodimensional manner)
    void computeFtexture(vector<Pixel*> &localWindow);
};



// Representation of a region
class Region : public Pixel
{
    
private:
    double wtexture; //uniform weight for the texture criterion
    double wcolor; //uniform weight for the color criterion
    double Sym;
    vector<Pixel *> region_stack_P; // stack of region
    Mat region_stack; //column vector for next tasks
    
public:
    Region();
    ~Region();
    vector<Pixel *> &getStack();
    Mat &getReg();
    double getWcolor();
    double getWtexture();
    
    void setWcolor(double w_col);
    void setWtexture(double w_text);
    
    Vec3d getFcolorVec();
    vector<double> getFtextureVec();
    
    double getFsym(); // Region symmetry descriptor
    
    void computeFcolor(); // implementation of the texture feature for a region
    void computeFtexture(); // implementation of texture feature for each pixel
    void computeFtexture(vector<Pixel*> &localWindow);
    void computeFsym(Mat C); // implementation of the symmetry feature of a region
};



#endif
