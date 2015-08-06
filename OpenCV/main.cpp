//  main.cpp
//  OpenCV

#include "Pixar.h"

// implementation of the color uniformity criterion between a pixel and a region
double computeDeltaC(Pixel Pi, Region Rj)
{
    Vec<double, 3> app;
    Vec<double, 3> app2;
    
    // color feature normalization in [0,1].
    normalize(Pi.getFcolorVec(),app,1,0);
    normalize(Rj.getFcolorVec(),app2,1,0);
    
    // Euclidean distance between Pi and Rj (with abs)
    double norma2 = norm(app, app2, NORM_L2);
    
    return norma2;
}


// implementation of texture uniformity criterion among a pixel and a region
double computeDeltaT(Pixel Pi, Region Rj)
{
    Vec<double, 8> app;
    Vec<double, 8> app2;
    Vec<double, 8> app3;
    
    vector<double> Ftp = Pi.getFtextureVec();
    vector<double> Ftr = Rj.getFtextureVec();
    Mat pi,rj;
    
    
    for (int i =0; i<8; i++) {
        pi.push_back(Ftp.at(i));
        rj.push_back(Ftr.at(i));
    }
    
    // texture feature normalization in [0,1].
    normalize(pi,app,1,0);
    normalize(rj,app2,1,0);
    
    // Euclidean distance between Pi and Rj (with abs)
    double norma2 = norm(app, app2, NORM_L2);
    
    return norma2;
    
}


// implementation of the region uniformity criterion (it's a combination among the color and texture feature)
double computeDeltaR(Pixel pi, Region rj)
{
    double t1 = rj.getWcolor()*computeDeltaC(pi, rj);
    double t2 = rj.getWtexture()*computeDeltaT(pi, rj);
    return t1+t2;
}


// implementation of the symmetry uniformity criterion
double computeDeltaS(Pixel Pi, Region rj, Mat C)
{
    Mat tmp;
    double deltaS;
    
    
    if( rj.getReg().rows == 1)
    {
        
        deltaS = ( ((M_PI/2) +
                    atan( sqrt( (1 + C.at<double>(Pi.getX0(),Pi.getY0() )) * (1 + C.at<double>(rj.getStack().front()->getX0(), rj.getStack().front()->getY0() ))))  )/M_PI) +
        ((1 + abs( sqrt( C.at<double>(Pi.getX0(),Pi.getY0() )) - sqrt( C.at<double>(rj.getStack().front()->getX0(), rj.getStack().front()->getY0() ))))/2);
    }
    else
    {
        // The symmetry affinity value for a region is given by the mean value of the feature pixel
        for(int i=0;i<rj.getStack().size();i++)
            tmp.push_back( C.at<double>(rj.getStack().at(i)->getX0(),rj.getStack().at(i)->getY0() ));
        
        Scalar media = mean(tmp);
        
        deltaS = ( ( (M_PI/2) +
                    atan( sqrt( (1 + C.at<double>(Pi.getX0(),Pi.getY0())) * (1 + media[0])) ))/M_PI)
        + ((1 + abs( sqrt( (C.at<double>(Pi.getX0(),Pi.getY0()))) - sqrt( media[0]) ))/2);
    }
    
    return deltaS;
}


// extraction of the 8-neighbours without edge
vector<Pixel*> get8_neighborPixel(vector< vector<Pixel*> > &label,int i,int j)
{
    vector<Pixel*> out;
    
    if(label[i-1][j]->getBordo()==false) { out.push_back(label[i-1][j]); } //N
    if(label[i-1][j+1]->getBordo()==false) { out.push_back(label[i-1][j+1]); } //NE
    if(label[i][j+1]->getBordo()==false) { out.push_back(label[i][j+1]); } //E
    if(label[i+1][j+1]->getBordo()==false) { out.push_back(label[i+1][j+1]); } //SE
    if(label[i+1][j]->getBordo()==false) { out.push_back(label[i+1][j]); } //S
    if(label[i+1][j-1]->getBordo()==false) { out.push_back(label[i+1][j-1]); } //SW
    if(label[i][j-1]->getBordo()==false) { out.push_back(label[i][j-1]); } //W
    if(label[i-1][j-1]->getBordo()==false) { out.push_back(label[i-1][j-1]); } //NW
    
    return out;
}


// pixels extraction from a 7x7 window
vector<Pixel*> get49_neighborPixel(vector< vector<Pixel*> > &label,int i,int j)
{
    vector<Pixel*> out;
    
    out.push_back(label[i+3][j+3]); //seventh row
    out.push_back(label[i+3][j+2]);
    out.push_back(label[i+3][j+1]);
    out.push_back(label[i+3][j]);
    out.push_back(label[i+3][j-1]);
    out.push_back(label[i+3][j-2]);
    out.push_back(label[i+3][j-3]);
    
    out.push_back(label[i+2][j+3]);  //sixth row
    out.push_back(label[i+2][j+2]);
    out.push_back(label[i+2][j+1]);
    out.push_back(label[i+2][j]);
    out.push_back(label[i+2][j-1]);
    out.push_back(label[i+2][j-2]);
    out.push_back(label[i+2][j-3]);
    
    out.push_back(label[i+1][j+3]);  //fifth row
    out.push_back(label[i+1][j+2]);
    out.push_back(label[i+1][j+1]);
    out.push_back(label[i+1][j]);
    out.push_back(label[i+1][j-1]);
    out.push_back(label[i+1][j-2]);
    out.push_back(label[i+1][j-3]);
    
    out.push_back(label[i][j+3]);  //fourth row
    out.push_back(label[i][j+2]);
    out.push_back(label[i][j+1]);
    out.push_back(label[i][j]);
    out.push_back(label[i][j-1]);
    out.push_back(label[i][j-2]);
    out.push_back(label[i][j-3]);
    
    out.push_back(label[i-1][j+3]);  //third row
    out.push_back(label[i-1][j+2]);
    out.push_back(label[i-1][j+1]);
    out.push_back(label[i-1][j]);
    out.push_back(label[i-1][j-1]);
    out.push_back(label[i-1][j-2]);
    out.push_back(label[i-1][j-3]);
    
    out.push_back(label[i-2][j+3]);  //second row
    out.push_back(label[i-2][j+2]);
    out.push_back(label[i-2][j+1]);
    out.push_back(label[i-2][j]);
    out.push_back(label[i-2][j-1]);
    out.push_back(label[i-2][j-2]);
    out.push_back(label[i-2][j-3]);
    
    out.push_back(label[i-3][j+3]);  //first row
    out.push_back(label[i-3][j+2]);
    out.push_back(label[i-3][j+1]);
    out.push_back(label[i-3][j]);
    out.push_back(label[i-3][j-1]);
    out.push_back(label[i-3][j-2]);
    out.push_back(label[i-3][j-3]);
    
    return out;
}


// Computing the Gradient Vector Flow
vector<Mat> GVF(Mat f, double mu, int iter)
{
    vector<Mat> V;
    Mat fx,fy;
    Mat sqrMagF;
    Mat u,v;
    Mat u_laplace,v_laplace;
    
    // EDGE map computing
    cvtColor(f,f,CV_RGB2GRAY);
    
    // EDGE map normalization in [0,1]
    f=f/255;
    
    // computing first derivates of the normalized EDGE MAP
    Sobel(f, fx, CV_64F, 1, 0, 3, 1, 0, BORDER_REFLECT);
    Sobel(f, fy, CV_64F, 0, 1, 3, 1, 0, BORDER_REFLECT);
    
    // initialization of the GVF
    u = fx;
    v = fy;
    
    // Computing magnitude
    magnitude(fx, fy, sqrMagF);
    
    //Computing GVF
    for(int i=0;i<iter;i++)
    {
        Laplacian(u, u_laplace, CV_64F, 1, 1, 0, BORDER_REFLECT);
        Laplacian(v, v_laplace, CV_64F, 1, 1, 0, BORDER_REFLECT);
        u = (u+mu*u_laplace) - (sqrMagF.mul(u-fx));
        v = (v+mu*v_laplace) - (sqrMagF.mul(v-fy));
    }
    
    // output matrices
    V.push_back(u);
    V.push_back(v);
    
    return V;
}


// Computing the Curve Gradient Vector Flow (CGVF)
Mat Curv(vector<Mat> V)
{
    Mat u = V[0];
    Mat v = V[1];
    
    Mat ux,uy,vx,vy;
    
    double V_norm = norm(u,v,NORM_L2);
    
    // computing first derivates of the GVF
    Sobel(u, ux, CV_64F, 1, 0, 3, 1, 0, BORDER_REFLECT);
    Sobel(u, uy, CV_64F, 0, 1, 3, 1, 0, BORDER_REFLECT);
    Sobel(v, vx, CV_64F, 1, 0, 3, 1, 0, BORDER_REFLECT);
    Sobel(v, vy, CV_64F, 0, 1, 3, 1, 0, BORDER_REFLECT);
    
    Mat C = (1/pow(V_norm, 3)) * (vx+uy).mul(u.mul(v)) - ux.mul(v.mul(v)) - vy.mul(u.mul(u));
    
    return C;
}


// Computing affinity symmetry of a pixel
double computeSymmetryAffinity(Mat curv,vector<Pixel*> neighbors,int i,int j)
{
    double Cij = curv.at<double>(i, j);
    double min = DBL_MAX;
    
    // computing of the minimum difference affinity among a pixel and its neighbors
    for (int k=0; k<neighbors.size(); k++)
    {
        if(neighbors[k]->getX0() < 0 || neighbors[k]->getY0() < 0 || (neighbors[k]->getX0()==i && neighbors[k]->getY0()==j))
            continue;
        
        double tmp = abs(Cij - curv.at<double>(neighbors[k]->getX0(), neighbors[k]->getY0()));
        
        if ( tmp < min )
            min = tmp;
    }
    
    return min;
}

// computing the STD's color feature gradient of the region
double computeGraR_color(Region r,Region r_old)
{
    Mat media_R,media_R_old,std_R,std_R_old;
    
    meanStdDev(r.getFcolorVec(), media_R, std_R);
    meanStdDev(r_old.getFcolorVec(), media_R_old, std_R_old);
    
    return std_R.at<double>(0)/std_R_old.at<double>(0);
}


// computing the STD's texture feature gradient of the region
double computeGraR_texture(Region r,Region r_old)
{
    Mat media_R,media_R_old,std_R,std_R_old;
    
    meanStdDev(r.getFtextureVec(), media_R, std_R);
    meanStdDev(r_old.getFtextureVec(), media_R_old, std_R_old);
    
    return std_R.at<double>(0)/std_R_old.at<double>(0);
}


// implementation of the criterion for merging regions
void regionMergingCriterion(vector<Region> VR)
{
    bool merge;
    
    for (int k=0; k<VR.size(); k++)
    {
        merge = false;
        for(int i=k+1;i<VR.size();i++)
        {
            // region mergion criteria (sum of euclidean distances)
            double n1 = norm(VR[k].getFcolorVec(),VR[i].getFcolorVec(),NORM_L2);
            double n2 = sqrt( pow(VR[k].getFsym() - VR[i].getFsym(),2) );
            
            // threshold value of the region merging
            if(n1+n2 < 0.05) //[0.02, 0.05]
            {
                
                for(int l=0;l<VR[i].getStack().size();l++)
                {
                    VR[i].getStack()[l]->setEtichetta( VR[k].getStack().front()->getEtichetta() ); // merging step
                }
                // removing of the region just merged
                VR.erase(VR.begin()+i);
                merge = true;
            }
        }
        // removing of the k-th merged region (if merge=true the k-th region has been merged with another region at least)
        if(merge == true)
            VR.erase(VR.begin()+k);
    }
    
}



Mat segmentation(vector< vector<Pixel *> > &label,int n_row,int n_col)
{
    map<int,int> ist;
    
    // computing regions frequencies
    for (int i=0; i<n_row+6; i++)
    {
        for (int j=0; j<n_col+6; j++)
        {
            if(label[i][j]->getBordo()==true)
                continue;
            
            // computing histogram of the image
            ist[label[i][j]->getEtichetta()]++;
        }
    }
    
    map<int,int>::iterator it;
    
    
    vector< Vec<unsigned,3> > colore;
    colore.push_back(Vec3b(0,0,0));
    
    // output segmented image
    Mat segmented(n_row,n_col,CV_8UC3,Scalar(255,255,255));
    
    // threshold for the coloration of the regions [***]
    int deltaC = 7;
    
    // labels scanning for the all pixels
    for (int i=0; i<n_row+6; i++)
    {
        for (int j=0; j<n_col+6; j++)
        {
            if(label[i][j]->getBordo()==true)
                continue;
            
            // counting frequencies for each label
            it=ist.find(label[i][j]->getEtichetta());
            
            // thresholding frequencies
            if( it->second < deltaC)
                segmented.at<Vec3b>( label[i][j]->getX0(),label[i][j]->getY0())=colore[0];
        }
    }
    
    
    imshow( "SEGMENTED IMAGE ", segmented);
    //cvWaitKey(0);
    
    // removing noise from the segmented image
    medianBlur(segmented, segmented, 7); // [***]
    
    imshow( "DENOISED IMAGE ", segmented);
    //cvWaitKey(0);
    
    return segmented;
    
}


// coloring segments of the output image
Mat drawLine(Mat seg, Mat rgb)
{
    int n=0, m=0;
    
    n=seg.rows;
    m=seg.cols;
    
    
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            if( (seg.at<Vec3b>(i,j)[0]==0 && seg.at<Vec3b>(i,j+1)[0]==255) || (seg.at<Vec3b>(i,j)[0]==255 && seg.at<Vec3b>(i,j+1)[0]==0) ||
               (seg.at<Vec3b>(i,j)[0]==255 && seg.at<Vec3b>(i-1,j)[0]==0) || (seg.at<Vec3b>(i,j)[0]==255 && seg.at<Vec3b>(i+1,j)[0]==0) )
            {
                rgb.at<Vec3b>(i,j)[0]=0;
                rgb.at<Vec3b>(i,j)[1]=0;
                rgb.at<Vec3b>(i,j)[2]=255;
            }
        }
    }
    
    return rgb;
}






int main(int argc, char **argv )
{
    Mat matHSV;
    int pixel_label=1;
    double deltaG = 0.015; // threshold for the pixel aggregation criterion toward regions [0.0035,0.015] [***]
    
    
    char str[] = "dataset/woman.jpg";
    
    
    Mat matRGB = imread(str,CV_LOAD_IMAGE_COLOR);
    
    imshow("Display window", matRGB);
    
    if(&matRGB==NULL)
        abort();
    
    // Converting color space from RGB to HSV
    cvtColor(matRGB, matHSV, CV_RGB2HSV);
    
    // Matrix of Pixel (see class Pixel)
    vector<vector<Pixel*> > label ( matHSV.rows+6, vector<Pixel*> ( matHSV.cols+6 ) );
    
    // initialization of the Pixel matrix
    for(int i=0;i<matHSV.rows+6;i++)
    {
        for(int j=0;j<matHSV.cols+6;j++)
        {
            // needed action to move in the actual position
            // edges conditions
            if(i==0 || i==1 || i==2 || i==matHSV.rows+6-1 || i==matHSV.rows+6-2 || i==matHSV.rows+6-3 ||
               j==0 || j==1 || j==2 || j==matHSV.cols+6-1 || j==matHSV.cols+6-2 || j==matHSV.cols+6-3)
            {
                // setting edge to zero
                Vec<unsigned,3> HSV_Color_bordi(0,0,0);
                label[i][j] = new Pixel(HSV_Color_bordi,true,-1,-1,-1); // random coordinates (-1,-1) because they are unnecessary
            }
            else
            {
                Vec<unsigned, 3> HSV_Color (matHSV.at<Vec3b>(i-3,j-3)[0], matHSV.at<Vec3b>(i-3,j-3)[1], matHSV.at<Vec3b>(i-3,j-3)[2]);
                label[i][j] = new Pixel(HSV_Color,false,i-3,j-3,0);
            }
        }
    }
    
    cout<<"Computing GVF..."<<endl;
    vector<Mat> gvf = GVF(matRGB, 0.2, 100); // parameters GVG: matrix,mu,iter
    
    cout<<"Computing CGVF..."<<endl;
    Mat curv = Curv(gvf);
    
    Mat SymAff(curv.rows,curv.cols,CV_64F);
    
    cout<<"Computing matrix of symmetry..."<<endl;
    
    vector<Pixel *> local;
    
    for(int i=0;i<curv.rows+6;i++)
    {
        for(int j=0;j<curv.cols+6;j++)
        {
            if(label[i][j]->getBordo() == true)
                continue;
            
            local = get49_neighborPixel(label,i,j);
            double min = computeSymmetryAffinity(curv,local,i-3,j-3); // shifting the coordinates to center the symmetry matrix's origin (since it's not padded)
            SymAff.at<double>(i-3, j-3) = min;
        }
    }
    
    vector<Region> VR;
    bool cond = true;
    Region *r,*r_old;
    
    
    cout<<"RegionGrowing..."<<endl;
    
    // Row-wise scanning
    for(int i=0;i<matHSV.rows+6;i++)
    {
        for(int j=0;j<matHSV.cols+6;j++)
        {
            if(label[i][j]->getBordo()==true) // edge pixel (omitted)
                continue;
            
            if(label[i][j]->getEtichetta() == 0) // pixel to be processed
            {
                // making of a new region
                r = new Region();
                
                // region growing
                r->getStack().push_back(label[i][j]);
                r->getReg().push_back(label[i][j]->getHSVvec());
                
                // updating the color feature for the region
                r->computeFcolor();
                
                // region size = 1 pixel
                if(r->getReg().rows == 1)
                {
                    vector<Pixel *> local49;
                    local49 = get49_neighborPixel(label,i,j); // getting 7x7 neighbourhood window
                    
                    // updating the texture feature for the region
                    r->computeFtexture(local49);
                }
                else // region size > 1
                    
                    // updating the texture feature for the region
                    r->computeFtexture();
                
                // growing of the r region
                label[i][j]->setEtichetta(pixel_label);
            }
            else
                continue; // pixel already processed
            
            
            vector<Pixel *> neighbors;
            int l=i,k=j;
            int pk;
            
            while(1)
            {
                // searching of the 8 neighbours for the (i,j)-th pixel
                neighbors = get8_neighborPixel(label,l,k);
                
                //processing of the eight pixels
                for(pk=0;pk<neighbors.size();pk++)
                {
                    if(neighbors[pk]->getEtichetta() == 0) // pixel not to process
                        
                    {
                        vector<Pixel*> local2_49;
                        local2_49 = get49_neighborPixel(label,neighbors[pk]->getX0()+3,neighbors[pk]->getY0()+3); // shifting into the original position of label
                        
                        // computing of the color and texture features
                        neighbors[pk]->computeFcolor();
                        neighbors[pk]->computeFtexture(local2_49);
                        
                        // computing of the uniformity criterion of the region and its symmetry
                        double deltaR = computeDeltaR(*neighbors[pk], *r);
                        double deltaS = computeDeltaS(*neighbors[pk], *r,SymAff);
                        
                        // computation of the aggregation criterion[delta] for the pixels
                        if( (deltaR*deltaS) < deltaG) // thresholding of the aggregation criterion [0.0035,0.015]
                            
                        {
                            // setting of the label
                            neighbors[pk]->setEtichetta(pixel_label);
                            
                            // saving the current region r
                            r_old = new Region(*r);
                            
                            // growing of the region r
                            r->getStack().push_back(neighbors[pk]);
                            r->getReg().push_back(neighbors[pk]->getHSVvec());
                            
                            // updating the color and texture features for all the region having a size > 1
                            r->computeFcolor();
                            r->computeFtexture();
                            
                            // weights allocation for the region
                            r_old->setWcolor(r->getWcolor());
                            r_old->setWtexture(r->getWtexture());
                            
                            // computing the STD's color and texture feature gradient of the region
                            double graRC = computeGraR_color(*r, *r_old);
                            double graRT = computeGraR_texture(*r, *r_old);
                            
                            // updating the weights
                            if (graRC < graRT)
                            {
                                r->setWtexture(r_old->getWtexture() * (graRC/graRT));
                                r->setWcolor(1-r->getWtexture());
                            }
                            else if(graRC > graRT)
                            {
                                r->setWcolor(r_old->getWcolor() * (graRT/graRC));
                                r->setWtexture(1-r->getWcolor());
                            }
                            
                            // set pk as new pixel P to be processed
                            l = 3 + neighbors[pk]->getX0();
                            k = 3 + neighbors[pk]->getY0();
                            cond = true;
                            break;
                        }
                        else // unlabeled pixel not satifying the criterion
                            cond = false;
                        
                    }
                    else // labeled pixel already belonging to a region
                        cond = false;
                    
                }
                
                // finalizing the i-th region
                if(pk == neighbors.size() && cond == false) // all 8-neighbours have been found and processed, so it needs to:
                {
                    r->computeFsym(SymAff);
                    VR.push_back(*r);
                    delete r;                               // 2) freeing memory for the region r and r_old
                    pixel_label++;                          // 3) change the label
                    break;
                }
            }
        }
    }
    
    
    cout<<"RegionMergingCriterion..."<<endl;
    
    regionMergingCriterion(VR);
    
    cout<<"Segmentation..."<<endl;
    
    Mat seg_out = segmentation(label,matHSV.rows,matHSV.cols);
    
    seg_out = drawLine(seg_out, matRGB);
    
    imshow("RED LINE", seg_out);
    //waitKey(0);
    
    // storing the ouput image
    imwrite("/Users/davidenardone/Desktop/out.jpg", seg_out);
    
    // freeing the memoru for label
    cout<<"De-allocazione memoria"<<endl;
    for(int i=0;i<matHSV.rows+6;i++)
        for(int j=0;j<matHSV.cols+6;j++)
            delete label[i][j];
    
    waitKey(0);
    cout<<"End"<<endl;
    
    
}