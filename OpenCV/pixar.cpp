#include "Pixar.h"

Pixel::Pixel() {
        Ftexture = new vector<double>(8,0);
        etichetta = 0;
        bordo = true;
    }
    
    // Constructor
Pixel::Pixel(Vec<unsigned, 3> HSV,bool val,int i,int j,int et) {
        Ftexture = new vector<double>(8,0);
        etichetta = et;
        bordo = val;
        HSVvec = HSV;
        x0 = i;
        y0 = j;
    }

Pixel::~Pixel(){};

    
int Pixel::getEtichetta() const { return etichetta; }
void Pixel::setEtichetta(int E) { etichetta = E; }
    
bool Pixel::getBordo()	{ return bordo; }
    
Vec3b Pixel::getHSVvec() { return HSVvec; }
    
int Pixel::getX0() { return x0; }
int Pixel::getY0() { return y0; }
    
 Vec3d Pixel::getFcolorVec() { return Fcolor; }
 vector<double> Pixel::getFtextureVec() { return *Ftexture; }
    
    // color feature implementation for each pixel
 void Pixel::computeFcolor()
    {
        Fcolor[0] = HSVvec[2]*HSVvec[1]*cos(2*M_PI*HSVvec[0]);
        Fcolor[1] = HSVvec[2]*HSVvec[1]*sin(2*M_PI*HSVvec[0]);
        Fcolor[2] = HSVvec[2];
    }
    
    // texture feature implementation for each pixel
    // 7x7 neighbours window for each pixel (monodimensional manner)
void Pixel::computeFtexture(vector<Pixel*> &localWindow) {
    
    Mat temp(7,7,CV_8UC3); // 7x7 neighbours window for each pixel
    
    for(int i=0;i<7;i++)
    {
        for (int j=0; j<7; j++)
        {
            temp.at<Vec3b>(i, j) = localWindow.back()->getHSVvec(); // component V (HSV)
            localWindow.pop_back();
            }
            
        }
        
        // Gaber filter parameters
        int lambd = (temp.rows*temp.cols)/5; // band-width
        int sigma = 8.0; // Gauassian variance
        int gamma = 15.0; // spatial relationship
        int psi = 0; // phase
        
        
        vector<Mat> vv; // vector of matrices
        split(temp, vv); // splitting temp (7x7 neighbourhood) into HSV component
        
        // 7x7 kernel declaration
        Mat gg0(7, 7, CV_64F);
        Mat gg45(7, 7, CV_64F);
        Mat gg90(7, 7, CV_64F);
        Mat gg135(7, 7, CV_64F);
        
        // Initialization of a filter bank consisting of Gabor filters to 0,45,90,135 gradi
        gg0 = getGaborKernel(gg0.size(), sigma, 0, lambd, gamma, psi, CV_64F);
        gg45 = getGaborKernel(gg45.size(), sigma, M_PI/4, lambd, gamma, psi, CV_64F);
        gg90 = getGaborKernel(gg90.size(), sigma, M_PI/2, lambd, gamma, psi, CV_64F);
        gg135 = getGaborKernel(gg135.size(), sigma, 3*M_PI/4, lambd, gamma, psi, CV_64F);
        
        // kernel flipping
        // rows and columns kernel rotation to carry out the convolution
        flip(gg0,gg0,-1);
        flip(gg45,gg45,-1);
        flip(gg90,gg90,-1);
        flip(gg135,gg135,-1);
        
        // Convolution result matrices
        Mat m0,m45,m90,m135;
        
        // Convolution Value matrix with Gabor kernel
        Point anchor = Point(1,1); // kernel positioning
        
        // Application of a bank of Gabor filter
        filter2D(vv[2], m0, -1, gg0,Point(gg0.cols - anchor.x - 1, gg0.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m45, -1, gg45,Point(gg45.cols - anchor.x - 1, gg45.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m90, -1, gg90,Point(gg90.cols - anchor.x - 1, gg90.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m135, -1, gg135,Point(gg135.cols - anchor.x - 1, gg135.rows - anchor.y - 1),0,BORDER_REFLECT);
        
        // temporal variable for the mean and std computation
        Vec<double,1> mean0,mean45,mean90,mean135;
        Vec<double,1> std0,std45,std90,std135;
        
        // std computing
        meanStdDev(m0, mean0, std0);
        meanStdDev(m45, mean45, std45);
        meanStdDev(m90, mean90, std90);
        meanStdDev(m135, mean135, std135);
        
        // memorization of the texture feature for each pixel
        Ftexture->at(0) = mean0[0];
        Ftexture->at(1) = mean45[0];
        Ftexture->at(2) = mean90[0];
        Ftexture->at(3) = mean135[0];
        Ftexture->at(4) = std0[0];
        Ftexture->at(5) = std45[0];
        Ftexture->at(6) = std90[0];
        Ftexture->at(7) = std135[0];
    }


Region::Region()
{
    Pixel();
    wtexture=0.5;
    wcolor=0.5;
}

Region::~Region(){};
    
    vector<Pixel *> &Region::getStack() { return region_stack_P; }
    
    Mat &Region::getReg() { return region_stack; }
    
    double Region::getWcolor() { return wcolor; }
    
    double Region::getWtexture() { return wtexture; }
    
    void Region::setWcolor(double w_col) { wcolor = w_col; }
    
    void Region::setWtexture(double w_text) { wtexture = w_text; }
    
    Vec3d Region::getFcolorVec() { return Fcolor; }
    
    vector<double> Region::getFtextureVec() { return *Ftexture; }
    
    double Region::getFsym() { return Sym; } // Region symmetry descriptor

    void Region::computeFcolor()
    {
        Scalar HSVmean = mean(region_stack);
        
        Fcolor[0] = HSVmean[2]*HSVmean[1]*cos(2*M_PI*HSVmean[0]);
        Fcolor[1] = HSVmean[2]*HSVmean[1]*sin(2*M_PI*HSVmean[0]);
        Fcolor[2] = HSVmean[2];
    }
    
    // implementation of the texture feature for a region
void Region::computeFtexture()
    {
        int lambd = (region_stack.rows)/5; // band width
        int sigma = 8.0; // Gaussian variance
        int gamma = 15.0; // spatial relationship
        int psi = 0; // phase
        
        vector<Mat> vv;
        split(region_stack.t(),vv); // splitting of a region into HSV component
        
        // Kernel declaration 7x7
        Mat gg0(7,7,CV_64F);
        Mat gg45(7,7,CV_64F);
        Mat gg90(7,7,CV_64F);
        Mat gg135(7,7,CV_64F);
        
        // Initialization of a filter bank consisting of Gabor filters at 0,45,90,135 degrees
        gg0 = getGaborKernel(gg0.size(), sigma, 0, lambd, gamma, psi, CV_64F);
        gg45 = getGaborKernel(gg45.size(), sigma, M_PI/4, lambd, gamma, psi, CV_64F);
        gg90 = getGaborKernel(gg90.size(), sigma, M_PI/2, lambd, gamma, psi, CV_64F);
        gg135 = getGaborKernel(gg135.size(), sigma, 3*M_PI/4, lambd, gamma, psi, CV_64F);
        
        // Kernels flipping
        // rows and columns kernel rotation to carrying out the convolution
        flip(gg0,gg0,-1);
        flip(gg45,gg45,-1);
        flip(gg90,gg90,-1);
        flip(gg135,gg135,-1);
        
        // Resultant matrices by convolution
        Mat m0,m45,m90,m135;
        
        // Convolution operation among V[Value] component matrix and Gabor kernel
        Point anchor = Point(1,1);
        // Carrying out a bank of Gabor filter
        filter2D(vv[2], m0, -1, gg0,Point(gg0.cols - anchor.x - 1, gg0.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m45, -1, gg45,Point(gg45.cols - anchor.x - 1, gg45.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m90, -1, gg90,Point(gg90.cols - anchor.x - 1, gg90.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m135, -1, gg135,Point(gg135.cols - anchor.x - 1, gg135.rows - anchor.y - 1),0,BORDER_REFLECT);
        
        // temporal variables for the mean and STD computing
        Vec<double,1> mean0,mean45,mean90,mean135;
        Vec<double,1> std0,std45,std90,std135;
        
        // STD computing
        meanStdDev(m0, mean0, std0);
        meanStdDev(m45, mean45, std45);
        meanStdDev(m90, mean90, std90);
        meanStdDev(m135, mean135, std135);
        
        // storing of texture feature for each pixel
        Ftexture->at(0) = mean0[0];
        Ftexture->at(1) = mean45[0];
        Ftexture->at(2) = mean90[0];
        Ftexture->at(3) = mean135[0];
        Ftexture->at(4) = std0[0];
        Ftexture->at(5) = std45[0];
        Ftexture->at(6) = std90[0];
        Ftexture->at(7) = std135[0];
    }
    
    
    // implementation of texture feature for each pixel
void Region::computeFtexture(vector<Pixel*> &localWindow)
    {
        Mat temp(7,7,CV_8UC3); // l'intorno 7x7 del pixel
        
        for(int i=0;i<7;i++)
        {
            for (int j=0; j<7; j++)
            {
                temp.at<Vec3b>(i, j) = localWindow.back()->getHSVvec(); // V component of HSV
                localWindow.pop_back();
            }
        }
        
        // Gaber filter parameters
        int lambd = (temp.rows*temp.cols)/5; // width band
        int sigma = 8.0; // Gaussian variance
        int gamma = 15.0; //spatial relatioship
        int psi = 0; // phase
        
        //splitting into H-S-V component
        vector<Mat> vv; // vector of matrices
        split(temp, vv); // splitting of temp into H-S-V component
        
        // 7x7 kernel declaration
        Mat gg0(7, 7, CV_64F);
        Mat gg45(7, 7, CV_64F);
        Mat gg90(7, 7, CV_64F);
        Mat gg135(7, 7, CV_64F);
        
        // Bank filter consisting of Gaber filter at 0,45,90,135 degrees
        gg0 = getGaborKernel(gg0.size(), sigma, 0, lambd, gamma, psi, CV_64F);
        gg45 = getGaborKernel(gg45.size(), sigma, M_PI/4, lambd, gamma, psi, CV_64F);
        gg90 = getGaborKernel(gg90.size(), sigma, M_PI/2, lambd, gamma, psi, CV_64F);
        gg135 = getGaborKernel(gg135.size(), sigma, 3*M_PI/4, lambd, gamma, psi, CV_64F);
        
        // rows and columns kernel rotation to carrying out the convolution
        flip(gg0,gg0,-1);
        flip(gg45,gg45,-1);
        flip(gg90,gg90,-1);
        flip(gg135,gg135,-1);
        
        // Resultant matrices by the convolution
        Mat m0,m45,m90,m135;
        
        // Convolution Value matrix with Gabor kernel
        // Applying the bank filter
        Point anchor = Point(1,1);
        filter2D(vv[2], m0, -1, gg0,Point(gg0.cols - anchor.x - 1, gg0.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m45, -1, gg45,Point(gg45.cols - anchor.x - 1, gg45.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m90, -1, gg90,Point(gg90.cols - anchor.x - 1, gg90.rows - anchor.y - 1),0,BORDER_REFLECT);
        filter2D(vv[2], m135, -1, gg135,Point(gg135.cols - anchor.x - 1, gg135.rows - anchor.y - 1),0,BORDER_REFLECT);
        
        // temporal variables for the mean and STD computing
        Vec<double,1> mean0,mean45,mean90,mean135;
        Vec<double,1> std0,std45,std90,std135;
        
        // STD computing
        meanStdDev(m0, mean0, std0);
        meanStdDev(m45, mean45, std45);
        meanStdDev(m90, mean90, std90);
        meanStdDev(m135, mean135, std135);
        
        // storing of the texture feature for each pixel
        Ftexture->at(0) = mean0[0];
        Ftexture->at(1) = mean45[0];
        Ftexture->at(2) = mean90[0];
        Ftexture->at(3) = mean135[0];
        Ftexture->at(4) = std0[0];
        Ftexture->at(5) = std45[0];
        Ftexture->at(6) = std90[0];
        Ftexture->at(7) = std135[0];
    }
    
    // implementation of the symmetry feature of a region
void Region::computeFsym(Mat C)
    {
        vector<double>temp;
        
        for(int i=0;i<region_stack_P.size();i++)
            temp.push_back( C.at<double>(region_stack_P[i]->getX0(), region_stack_P[i]->getY0()));
        
        Scalar vv = mean(temp);
        Sym = vv[0];
    }








