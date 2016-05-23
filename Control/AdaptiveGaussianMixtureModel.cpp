#include "AdaptiveGaussianMixtureModel.h"



AdaptiveGaussianMixtureModel::AdaptiveGaussianMixtureModel()
{

}


AdaptiveGaussianMixtureModel::~AdaptiveGaussianMixtureModel()
{
}

void AdaptiveGaussianMixtureModel::init(int height, int width)
{
    //_alpha = 0.001f;
    _alpha = 0.002f; //opencv
    _foregroundThresh = 4.0f * 4.0f;
    _close_to_component_thresh = 3.0f * 3.0f;
    _backgroundThresh = 0.9f;
    //_initSigma = 11.0f;
    _initSigma = 15.0f;
    _complexityPrior = 0.05f;
    _prune = (-1) *_alpha * _complexityPrior;
    _maxModes = 4;
    _numBands = 1;
    _height = height;
    _width = width;


    _gmm = std::vector<NormalDist>(_height * _width * _maxModes, NormalDist());
    _modesPerPixel = std::vector<uchar>(_height * _width, 0);

    _backgroundImage = cv::Mat(_height, _width, CV_8UC1);

}


cv::Mat AdaptiveGaussianMixtureModel::update(const cv::Mat& img, const cv::Mat& mask)
{
    cv::Mat fgMask = cv::Mat(img.size(), CV_8UC1);
    uchar fgMaskVal;

    long posPixel;
    std::vector<uchar>::iterator usedModes = _modesPerPixel.begin();

    for(int r = 0; r < _height; r++)
    {
        for(int c = 0; c < _width; c++)
        {
            posPixel = ((r * _width) + c) * _maxModes;


            fgMask.at<uchar>(r,c) = updatePixel(posPixel, img.at<uchar>(r, c), usedModes,(mask.at<uchar>(r,c) == 255));
            _backgroundImage.at<uchar>(r,c) = _gmm[posPixel].muG;
            usedModes++;
        }
    }

    return fgMask;
}

cv::Mat AdaptiveGaussianMixtureModel::update(const cv::Mat& img)
{
    cv::Mat fgMask = cv::imread("../fg_mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    fgMask = fgMask(cv::Range(0, 426), cv::Range::all());

    return update(img, fgMask);;
}


uchar AdaptiveGaussianMixtureModel::updatePixel(long posPixel, uchar pixelVal, std::vector<uchar>::iterator usedModes, bool isInMask)
{
    long pos;
    bool bFitsPDF = false;
    bool bBackground = false;

    float fOneMinAlpha = 1 - _alpha;
    int nModes = *usedModes;
    float totalWeight = 0.0f;

    //MODES NEED TO BE IN DESCENDING ORDER BY WEIGHT

    //go through all modes
    for (int iModes = 0; iModes < nModes; iModes++)
    {
        pos = posPixel + iModes;
        float weight = _gmm[pos].weight;

        //fit not found yet
        //look only for first fit
        if (!bFitsPDF)
        {
            //check if it belongs to some of the modes
            //calculate distance
            float var = _gmm[pos].sigma;
            float muG = _gmm[pos].muG;

            float dG = muG - pixelVal;

            //check if it fits the current mode (Factor * sigma)
            float dist = (dG*dG);

            //background? - m_fTb
            //doesnt modify background model, only foreground mask
            //first part --> eq. 8
            if ( (totalWeight < _backgroundThresh) && (dist < _foregroundThresh * var) && !isInMask)
                    bBackground = true;

            //check fit if current Pixel doesnt belong to specified mask
            if (dist < _close_to_component_thresh * var && !isInMask) //equals mahalan distance to current component
            {
                //belongs to the mode
                bFitsPDF = true;

                //update distribution
                float k = _alpha / weight;
                weight = fOneMinAlpha * weight + _prune;
                weight += _alpha; //ownership is one, prune equals alpha times complexity prior (see eq 14)
                _gmm[pos].muG = muG - k * (dG);

                float sigmanew = var + k * (dist - var); //see eq. 6. dist is NOT equal to mahalanobis dist because 1/sigma is missing

                //limit the variance to [4, 5*_initSigma]
                if(sigmanew < 4)
                    _gmm[pos].sigma = 4;
                else if(sigmanew > 5 * _initSigma)
                    _gmm[pos].sigma = 5 * _initSigma;
                else
                    _gmm[pos].sigma = sigmanew;

                //if currentPixel belongs to foreground mask, set weight to lowest possible, what if only one mode therefore weight = 1?
                /*if(isMask && nModes > 1)
                {
                    weight = _gmm[posPixel + nModes - 1].weight;
                }*/
                //sort
                //all other weights are at the same place and
                //only the matched (iModes) is higher -> just find the new place for it
                //insertion sort manner --> get back to top and replace
                //std::sort(_gmm.begin() + posPixel, _gmm.begin() + posPixel + nModes, [](const NormalDist& lhs, const NormalDist& rhs) { return lhs.weight > rhs.weight; } );
                //std::sort(_gmm.begin() + posPixel, _gmm.begin() + posPixel + iModes + 1, [](const NormalDist& lhs, const NormalDist& rhs) { return lhs.weight > rhs.weight; } );
                for (int iLocal = iModes; iLocal > 0; iLocal--)
                {
                    long posLocal = posPixel + iLocal;
                    if (weight < (_gmm[posLocal - 1].weight))
                        break;
                    else
                        std::swap(_gmm[posLocal], _gmm[posLocal - 1]);
                }
            }
            //is not close to current component
            //reduce weight (less significance)
            else
            {
                weight = fOneMinAlpha * weight + _prune; //not close --> onwerhsip is zero --> no need to add alpha
                if (weight < -_prune) //discard modes (see under eq 14)
                {
                    weight = 0.0f;
                    nModes--;
                }
            }
        }
        //fit already found, reduce weight of remaining gaussians, order stays the same
        else
        {
                weight = fOneMinAlpha * weight + _prune;
                if (weight < -_prune) //discard modes (see under eq 14)
                {
                    weight = 0.0;
                    nModes--;
                }
        }

        totalWeight += weight;
        _gmm[pos].weight = weight;
    }

    //renormalize weights
    for (int iLocal = 0; iLocal < nModes; iLocal++)
    {
        _gmm[posPixel+ iLocal].weight = _gmm[posPixel + iLocal].weight/totalWeight;
    }


    //executed iff no gaussian describes current pixel
    //make new if possible or rewrite least significant
    if (!bFitsPDF)
    {
        if (nModes != _maxModes) //if maxed do nothing, overwrite the one at _maxModes
            nModes++;

        pos = posPixel + nModes - 1; //pos points to current mode now

        if (nModes==1)
            _gmm[pos].weight = 1; //only one mode --> set weight to one. no other modes available
        else
            _gmm[pos].weight = _alpha; //normal update

        //renormalize weights
        int iLocal;
        for (iLocal = 0; iLocal < nModes - 1; iLocal++)
        {
            _gmm[posPixel + iLocal].weight *= fOneMinAlpha;
        }

        _gmm[pos].muG = pixelVal;
        _gmm[pos].sigma = _initSigma;

        //sort
        //find the new place for it
        for (iLocal = nModes - 1; iLocal > 0; iLocal--)
        {
            long posLocal = posPixel + iLocal;
            if (_alpha < (_gmm[posLocal - 1].weight))
                break;
            else
                std::swap(_gmm[posLocal - 1], _gmm[posLocal]);
        }
    }

    *usedModes=nModes;

    if(bBackground)
        return 0;
    else
        return 255;
}


//getter and setter
float AdaptiveGaussianMixtureModel::alpha() const
{
    return _alpha;
}

void AdaptiveGaussianMixtureModel::setAlpha(float alpha)
{
    _alpha = alpha;
}

float AdaptiveGaussianMixtureModel::foregroundThresh() const
{
    return _foregroundThresh;
}

void AdaptiveGaussianMixtureModel::setForegroundThresh(float foregroundThresh)
{
    _foregroundThresh = foregroundThresh;
}

float AdaptiveGaussianMixtureModel::close_to_component_thresh() const
{
    return _close_to_component_thresh;
}

void AdaptiveGaussianMixtureModel::setClose_to_component_thresh(float close_to_component_thresh)
{
    _close_to_component_thresh = close_to_component_thresh;
}

float AdaptiveGaussianMixtureModel::backgroundThresh() const
{
    return _backgroundThresh;
}

void AdaptiveGaussianMixtureModel::setBackgroundThresh(float backgroundThresh)
{
    _backgroundThresh = backgroundThresh;
}

float AdaptiveGaussianMixtureModel::initSigma() const
{
    return _initSigma;
}

void AdaptiveGaussianMixtureModel::setInitSigma(float initSigma)
{
    _initSigma = initSigma;
}

float AdaptiveGaussianMixtureModel::complexityPrior() const
{
    return _complexityPrior;
}

void AdaptiveGaussianMixtureModel::setComplexityPrior(float complexityPrior)
{
    _complexityPrior = complexityPrior;
}


cv::Mat AdaptiveGaussianMixtureModel::backgroundImage()
{
    // !!!zivkovich uses only used normal dist with most evidence!!!! - performance?
    /*cv::Mat backgroundImage;
    backgroundImage = (_numBands == 1) ? cv::Mat(_height, _width, CV_8UC1) : cv::Mat(_height, _width, CV_8UC3);
    float meanB, meanG, meanR;

    std::vector<uchar>::iterator usedModes = _modesPerPixel.begin();

    for(int r = 0; r < _height; r++)
    {
        for(int c = 0; c < _width; c++)
        {
            long posPixel = (r * _width + c) * _maxModes;
            float totalWeight = 0.0f;

            for(int i = 0; i < *usedModes; i++)
            {

                if(_numBands == 1)
                {
                    meanB += _gmm[posPixel + i].weight * _gmm[posPixel + i].muB;
                }
                else
                {
                    meanB += _gmm[posPixel + i].weight * _gmm[posPixel + i].muB;
                    meanG += _gmm[posPixel + i].weight * _gmm[posPixel + i].muG;
                    meanR += _gmm[posPixel + i].weight * _gmm[posPixel + i].muR;
                }
                totalWeight += _gmm[posPixel + i].weight;
                if(totalWeight > _backgroundThresh)
                    break;
            }

            float invWeight = 1.0f/totalWeight;


            if(_numBands == 1)
            {
                backgroundImage.at<uchar>(r,c) = invWeight * meanB;
            }
            else
            {
                backgroundImage.at<cv::Vec3b>(r,c) = cv::Vec3b(invWeight * meanB, invWeight * meanG, invWeight * meanR);
            }

            meanB = meanG = meanR = 0;

            usedModes++;

        }
    }*/

    return _backgroundImage;
}















