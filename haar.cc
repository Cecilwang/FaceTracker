#include "haar.h"
#include <vector>
#include "face.h"
#include "opencv2/imgcodecs.hpp"

using std::vector;

CascadeClassifier *g_front_cascade = NULL, *g_profile_cascade = NULL;
Orientation g_current_orientation_state = FRONT;

int LoadHaarClassifier() {
  g_front_cascade = new CascadeClassifier;
  g_profile_cascade = new CascadeClassifier;
  g_current_orientation_state = FRONT;

  bool ret1 = g_front_cascade->load(g_config->front_xml_path_);
  bool ret2 = g_profile_cascade->load(g_config->profile_xml_path_);

  if (ret1 && ret2)
    return 1;
  else
    return 0;
}

int DoHaar(const Mat& image, Rect& face) {
  Rect face_tmp;
  int max_face_height = image.rows / 2;
  int status;

  status = DoHaar(image, FRONT, CASCADE_SCALE_IMAGE,
                  Size(max_face_height, max_face_height), face_tmp);
  face = face_tmp;
  return status;
}

int DoHaarForAllOrientation(const Mat& image, double scale_factor,
                            const Rect *region, int flags, Rect& face) {
  // cvTimer?
  Mat image_tmp;
  Rect face_tmp, face_result;
  Size max_face_size = Size();
  if (scale_factor != 0.0) {
    max_face_size = Size(cvRound(g_config->max_face_width_ * scale_factor),
                         cvRound(g_config->max_face_height_ * scale_factor));
  }

  int status;
  Orientation orientation_tmp;

  // distinguish orientation, flip image, resize image, calculate coordinates
  // Order: FRONT -> LEFT -> RIGHT or LEFT -> FRONT -> RIGHT
  if (g_current_orientation_state == FRONT
      || g_current_orientation_state == LEFT) {
    ImagePreProcessing(image, scale_factor, region, false, true, image_tmp);
    status = DoHaar(image_tmp, g_current_orientation_state, flags,
                    max_face_size, face_tmp);
    if (status != 0) {
      RestoreCoordinate(face_tmp, image_tmp.size(), scale_factor, region, false,
                        face_result);
      face = face_result;
      DumpImage(to_string(g_local_frame_count) + "-haar-fl", image_tmp, &face_tmp);
      return 1;
    } else {
      orientation_tmp =
          static_cast<Orientation>(1 - g_current_orientation_state);
      status = DoHaar(image_tmp, orientation_tmp, flags, max_face_size,
                      face_tmp);
      if (status != 0) {
        RestoreCoordinate(face_tmp, image_tmp.size(), scale_factor, region,
                          false, face_result);
        face = face_result;
        g_current_orientation_state = orientation_tmp;
        DumpImage(to_string(g_local_frame_count) + "-haar-fl", image_tmp,
                  &face_tmp);
        return 1;
      } else {  // RIGHT
        flip(image_tmp, image_tmp, 1);
        status = DoHaar(image_tmp, RIGHT, flags, max_face_size, face_tmp);
        if (status != 0) {
          RestoreCoordinate(face_tmp, image_tmp.size(), scale_factor, region,
                            true, face_result);
          face = face_result;
          g_current_orientation_state = RIGHT;
          DumpImage(to_string(g_local_frame_count) + "-haar-r", image_tmp,
                    &face_tmp);
          return 1;
        } else {
          face = Rect(-1, -1, 0, 0);
          DumpImage(to_string(g_local_frame_count) + "-haar-n", image_tmp, NULL);
          return 0;
        }
      }
    }
    // Order: RIGHT -> FRONT -> LEFT
  } else {  // g_current_orientation_state == RIGHT
    ImagePreProcessing(image, scale_factor, region, true, true, image_tmp);
    status = DoHaar(image_tmp, RIGHT, flags, max_face_size, face_tmp);
    if (status != 0) {
      RestoreCoordinate(face_tmp, image_tmp.size(), scale_factor, region, true,
                        face_result);
      face = face_result;
      DumpImage(to_string(g_local_frame_count) + "-haar-r", image_tmp, &face_tmp);
      return 1;
    } else { // FRONT or LEFT
      flip(image_tmp, image_tmp, 1);
      status = DoHaar(image_tmp, FRONT, flags, max_face_size, face_tmp);
      if (status != 0) {
        RestoreCoordinate(face_tmp, image_tmp.size(), scale_factor, region,
                          false, face_result);
        face = face_result;
        g_current_orientation_state = FRONT;
        DumpImage(to_string(g_local_frame_count) + "-haar-f", image_tmp,
                  &face_tmp);
        return 1;
      } else {
        status = DoHaar(image_tmp, LEFT, flags, max_face_size, face_tmp);
        if (status != 0) {
          RestoreCoordinate(face_tmp, image_tmp.size(), scale_factor, region,
                            false, face_result);
          face = face_result;
          g_current_orientation_state = LEFT;
          DumpImage(to_string(g_local_frame_count) + "-haar-l", image_tmp,
                    &face_tmp);
          return 1;
        } else {
          face = Rect(-1, -1, 0, 0);
          DumpImage(to_string(g_local_frame_count) + "-haar-n", image_tmp, NULL);
          return 0;
        }
      }
    }
  }
  return status;  // Should never go here.
}

int DoHaar(const Mat& image, Orientation current_orientation, int flags,
           Size max_face_size, Rect& face) {
  vector<Rect> faces;
  vector<int> num_detections;

  Size min_face_size(g_config->min_face_width_, g_config->min_face_height_);

  int max_neighbor = 0, max_neighbor_pos = 0;

  // Detect faces
  if (current_orientation == FRONT) {
    g_front_cascade->detectMultiScale(image, faces, num_detections,
                                      g_config->scale_factor_,
                                      g_config->min_neighbors_front_,
                                      CASCADE_SCALE_IMAGE, min_face_size,
                                      max_face_size);
  } else {
    g_profile_cascade->detectMultiScale(image, faces, num_detections,
                                        g_config->scale_factor_,
                                        g_config->min_neighbors_profile_,
                                        CASCADE_SCALE_IMAGE, min_face_size,
                                        max_face_size);
  }

  // find the largest neighbors in the results
  if (!faces.empty()) { // face found
    for (size_t i = 0; i < num_detections.size(); ++i) {
      if (num_detections[i] > max_neighbor) {
        max_neighbor = num_detections[i];
        max_neighbor_pos = i;
      }
    }
    face = faces[max_neighbor_pos];
    return 1;
  } else { // face not found
    face = Rect(-1, -1, 0, 0);
    return 0;
  }
  return 0; // Never go here.
}

int ImagePreProcessing(const Mat& image, double scale_factor,
                       const Rect *region, bool horizontal_flip,
                       bool equalize_histogram, Mat& image_out) {
  Mat image_tmp, image_tmp1;
  // Image pre-processing---
  // crop
  if (region != NULL) {
    image_tmp = image(*region);
    //DumpImage("crop-"+to_string(g_local_frame_count), image_tmp);
  } else
    image_tmp = image;

  // resize
  if (scale_factor != 0.0) // is this safe?
    resize(image_tmp, image_tmp1, Size(), scale_factor, scale_factor);
  else
    image_tmp1 = image_tmp.clone();

  // Horizontal flipping for profile face
  if (horizontal_flip)
    flip(image_tmp1, image_tmp1, 1);

  // Equalizes the histogram of a grayscale image.
  // TODO: histogram equalization useful or not?
  if (equalize_histogram)
    equalizeHist(image_tmp1, image_tmp1);

  image_out = image_tmp1;

  return 1;
}

int RestoreCoordinate(const Rect &face_in, const Size& image_size,
                      double scale_factor, const Rect *region,
                      bool horizontal_flip, Rect &face_out) {
  // Check (-1, -1, 0, 0)
  if (face_in.x == -1) {
    face_out = face_in;
  } else {
    face_out = face_in;
    if (horizontal_flip) {
      face_out.x = image_size.width - (face_in.x + face_in.width) - 1;
      face_out.y = face_in.y;
    }

    //frlog("%d %d %d %d", face_out.x, face_out.y,face_out.width,face_out.height);
    if (scale_factor != 0.0) {
      face_out.x = cvRound(face_out.x / scale_factor);
      face_out.y = cvRound(face_out.y / scale_factor);
      face_out.width = cvRound(face_out.width / scale_factor);
      face_out.height = cvRound(face_out.height / scale_factor);
    }
    //frlog("%f", scale_factor);
    //frlog("%d %d %d %d", face_out.x, face_out.y,face_out.width,face_out.height);

    if (region != NULL) {
      face_out.x = face_out.x + region->x;
      face_out.y = face_out.y + region->y;
    }
    // do not across border
    Size original_image_size = Size(g_config->kImageWidth_,
                                    g_config->kImageHeight_);
    RectifyCoordinate(face_out, original_image_size, face_out);
  }
  return 1;
}
