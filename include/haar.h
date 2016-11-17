#ifndef LIBFACE_HAAR_H_
#define LIBFACE_HAAR_H_

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

enum Orientation {
  FRONT = 0, LEFT, RIGHT, ORIENTATION_END,
};

int DoHaar(const Mat& image, Rect& face);
int DoHaarForAllOrientation(const Mat& image, double scale_factor,
                            const Rect *region, int flags, Rect& face);
int DoHaar(const Mat& image, Orientation current_orientation, int flags,
           Size max_face_size, Rect& face);

int LoadHaarClassifier();
int ImagePreProcessing(const Mat& image, double scale_factor,
                       const Rect *region, bool horizontal_flip,
                       bool equalize_histogram, Mat& image_out);
int RestoreCoordinate(const Rect &face_in, const Size& image_size,
                      double scale_factor, const Rect *region,
                      bool horizontal_flip, Rect &face_out);

extern CascadeClassifier *g_front_cascade, *g_profile_cascade;
extern Orientation g_current_orientation_state;

#endif // LIBFACE_HAAR_H_
