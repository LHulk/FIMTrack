#ifndef ADAPTIVEGAUSSIANMIXTUREMODEL_H
#define ADAPTIVEGAUSSIANMIXTUREMODEL_H


//qt
#include <QDebug>

//std
#include <vector>

//opencv
#include <opencv2/opencv.hpp>

//user




class AdaptiveGaussianMixtureModel
{
private:
    struct NormalDist
    {
        float sigma;
        float muG;
        float weight;
        NormalDist() { sigma = 0.0f; muG = 0.0f;}
    };


public:
    AdaptiveGaussianMixtureModel();
    ~AdaptiveGaussianMixtureModel();

    void init(int height, int width);
    cv::Mat update(const cv::Mat& img, const cv::Mat& mask);
    cv::Mat update(const cv::Mat& img);


    float alpha() const;
    void setAlpha(float alpha = 0.001f);

    float foregroundThresh() const;
    void setForegroundThresh(float foregroundThresh = 16.0f);

    float close_to_component_thresh() const;
    void setClose_to_component_thresh(float close_to_component_thresh  = 9.0f);

    float backgroundThresh() const;
    void setBackgroundThresh(float backgroundThresh = 0.9f);

    float initSigma() const;
    void setInitSigma(float initSigma  = 11.0f);

    float complexityPrior() const;
    void setComplexityPrior(float complexityPrior  = 0.05f);

    cv::Mat backgroundImage();

private:
    uchar updatePixel(long posPixel, uchar pixelVal, std::vector<uchar>::iterator usedModes, bool isInMask);
    //speed of update - if the time interval you want to average over is T
    //set _alpha=1/T. It is also usefull at start to make T slowly increase
    //from 1 until the desired T
    float _alpha;

    //threshold on the squared Mahalan. dist. to decide if it is well described
    //by the background model or not. Related to Cthr from the paper.
    //This does not influence the update of the background. A typical value could be 4 sigma
    //and that is _foregroundThresh=4*4=16;
    float _foregroundThresh;

    //threshold on the squared Mahalan. dist. to decide
    //when a sample is close to the existing components. If it is not close
    //to any a new component will be generated. I use 3 sigma => _close_to_component_thresh=3*3=9.
    //Smaller _close_to_component_thresh leads to more generated components and higher _close_to_component_thresh might make
    //lead to small number of components but they can grow too large
    float _close_to_component_thresh;

    //1-cf from the paper
    //hreshold when the component becomes significant enough to be included into
    //the background model. It is the _backgroundThresh=1-cf from the paper. So I use cf=0.1 => _backgroundThresh=0.
    //For alpha=0.001 it means that the mode should exist for approximately 105 frames before
    float _backgroundThresh;

    //initial standard deviation  for the newly generated components.
    //It will will influence the speed of adaptation. A good guess should be made.
    //A simple way is to estimate the typical standard deviation from the images.
    //I used here 10 as a reasonable value
    float _initSigma;

    //complexity reduction prior
    //this is related to the number of samples needed to accept that a component
    //actually exists. We use _complexityPrior=0.05 of all the samples. By setting _complexityPrior=0 you get
    //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)
    float _complexityPrior;

    float _prune;

    //max number of modes - const - 4 is usually enough
    int _maxModes;

    //data
    int _numBands; //only grey needed

    std::vector<NormalDist> _gmm; //mixture of Gaussians
    std::vector<uchar> _modesPerPixel; //number of Gaussian components per pixel


    int _width, _height;
    cv::Mat _backgroundImage;

};

#endif // ADAPTIVEGAUSSIANMIXTUREMODEL_H
