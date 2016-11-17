#ifndef LIBFACE_KCF_H_
#define LIBFACE_KCF_H_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/tracking.hpp"

using namespace cv;

int ResetKCF(const Mat& frame, const Rect& face_in, double scale_factor);

int DoKCFWithHaar(const Mat& frame, const Mat& cb, const Mat& cr, Rect& face);

int DoKCF(const Mat& frame, Rect& face);

int SetKCFParameters();

// avoid negative output, out of image border for face_out
// recover original coordinate
int RefineKCFOutput(const Rect2d& roi_in, const Size& image_size,
                    double scale_factor, Rect2i& roi_out);

int ExtendKCFRegion(const Rect& roi_in, const Size& image_size,
                    double edge_extend_factor, Rect& roi_out);

int CheckSkinColor(const Mat& cb, const Mat& cr, const Rect& roi, double threshold);

//int DumpYUV(const Mat& frame, const Mat& cb, const Mat& cr, const Rect& roi);

extern double g_current_kcf_scale_factor;
extern int g_haar_failure_times;

#endif // LIBFACE_KCF_H_
