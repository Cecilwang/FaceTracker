#include "kcf.h"
#include <string>
#include "haar.h"
#include "face.h"

Ptr<Tracker> g_kcf_tracker;
TrackerKCF::Params* g_kcf_param = NULL;
Rect2d* g_kcf_roi = NULL;

double g_current_kcf_scale_factor = 0.0;
int g_haar_failure_times = 0;

int ResetKCF(const Mat& frame, const Rect& face_in, double scale_factor) {
  Mat image_after_resize;
  bool kcf_result = false;

  if (scale_factor != 0.0) {   // is this safe?
    resize(frame, image_after_resize, Size(), scale_factor, scale_factor);
    g_kcf_roi->x = cvRound(face_in.x * scale_factor);
    g_kcf_roi->y = cvRound(face_in.y * scale_factor);
    g_kcf_roi->width = cvRound(face_in.width * scale_factor);
    g_kcf_roi->height = cvRound(face_in.height * scale_factor);
  } else {
    image_after_resize = frame;
    *g_kcf_roi = face_in;
  }

  g_kcf_tracker = TrackerKCF::createTracker(*g_kcf_param);
  kcf_result = g_kcf_tracker->init(image_after_resize, *g_kcf_roi);
  mylog("INFO: Frame %d, after kcf init, g_kcf_roi: %f %f %f %f", g_local_frame_count,
        g_kcf_roi->x, g_kcf_roi->y, g_kcf_roi->width, g_kcf_roi->height);
  if (!kcf_result) {
    print_and_log("ERROR: Frame %d, KCF tracker initialization failed",
                  g_local_frame_count);
    return 0;
  }

  kcf_result = g_kcf_tracker->update(image_after_resize, *g_kcf_roi);
  mylog("INFO: Frame %d, after kcf update, g_kcf_roi: %f %f %f %f",
        g_local_frame_count, g_kcf_roi->x, g_kcf_roi->y, g_kcf_roi->width,
        g_kcf_roi->height);
  DumpImage2d(to_string(g_local_frame_count) + "-resetkcf", image_after_resize,
              g_kcf_roi);

  if (!kcf_result) {
    print_and_log(
        "ERROR: Frame %d, KCF tracker can't update target after initialization",
        g_local_frame_count);
    return 0;
  }

  return 1;
}

int RefineKCFOutput(const Rect2d& roi_in, const Size& image_size,
                    double scale_factor, Rect2i& roi_out) {
  Rect roi_1(cvRound(roi_in.x), cvRound(roi_in.y), cvRound(roi_in.width),
             cvRound(roi_in.height));
  Rect roi_2, roi_3;

  RectifyCoordinate(roi_1, image_size, roi_2);

  double x = roi_2.x;
  double y = roi_2.y;
  double width = roi_2.width;
  double height = roi_2.height;

  if (scale_factor != 0.0) {
    x = x / scale_factor;
    y = y / scale_factor;
    width = width / scale_factor;
    height = height / scale_factor;
  }

  // TODO: Should we add another RectifyCoordinate here?
  roi_3 = Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height));
  Size original_image_size = Size(g_config->kImageWidth_,
                                  g_config->kImageHeight_);
  RectifyCoordinate(roi_3, original_image_size, roi_out);
  return 1;
}

// Extend the region without changing the center of rectangle
int ExtendKCFRegion(const Rect2d& roi_in, const Size& image_size,
                    double edge_extend_factor, Rect& roi_out) {

  int new_x1, new_y1, new_x2, new_y2;

  new_x1 = cvRound(roi_in.x - (edge_extend_factor / 2.0 - 0.5) * roi_in.width);
  new_y1 = cvRound(roi_in.y - (edge_extend_factor / 2.0 - 0.5) * roi_in.height);
  new_x2 = cvRound(roi_in.x + (edge_extend_factor / 2.0 + 0.5) * roi_in.width);
  new_y2 = cvRound(roi_in.y + (edge_extend_factor / 2.0 + 0.5) * roi_in.height);

  Rect roi_1 = Rect(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1);

  RectifyCoordinate(roi_1, image_size, roi_out);

  return 1;
}

int DoKCF(const Mat& frame, Rect& face) {
  Mat image_after_resize /*, frame_backup*/;
  Rect face_1;
  int kcf_result = 0;

  assert(!g_kcf_tracker.empty());

  if (g_current_kcf_scale_factor != 0.0) {   // is this safe?
    resize(frame, image_after_resize, Size(), g_current_kcf_scale_factor,
           g_current_kcf_scale_factor);
  } else {
    image_after_resize = frame;
  }

  kcf_result = g_kcf_tracker->update(image_after_resize, *g_kcf_roi);
  mylog("INFO: Frame %d, kcf update g_kcf_roi: %f %f %f %f", g_local_frame_count,
        g_kcf_roi->x, g_kcf_roi->y, g_kcf_roi->width, g_kcf_roi->height);
  DumpImage2d(to_string(g_local_frame_count) + "-Dokcf", image_after_resize,
              g_kcf_roi);

  if (kcf_result == 0) {
    print_and_log("ERROR: Frame %d, KCF tracker can't update target",
                  g_local_frame_count);
    return 0;
  }
  RefineKCFOutput(*g_kcf_roi, image_after_resize.size(),
                  g_current_kcf_scale_factor, face_1);
  face = face_1;
  return 1;
}

int DoKCFWithHaar(const Mat& frame, const Mat& cb, const Mat& cr, Rect& face) {
  Mat image_after_resize;
  Rect face_1, face_2, face_3, face_4, face_5, face_6;
  int kcf_result = 0, skin_color_check_result = 0, haar_result = 0;

  assert(!g_kcf_tracker.empty());

  if (g_current_kcf_scale_factor != 0.0) {   // is this safe?
    resize(frame, image_after_resize, Size(), g_current_kcf_scale_factor,
           g_current_kcf_scale_factor);
  } else {
    image_after_resize = frame;
  }

  kcf_result = g_kcf_tracker->update(image_after_resize, *g_kcf_roi);
  if (kcf_result == 0) {
    print_and_log("ERROR: Frame %d: KCF tracker update failed!",
                  g_local_frame_count);
    return 0;
  }

  RefineKCFOutput(*g_kcf_roi, image_after_resize.size(),
                  g_current_kcf_scale_factor, face_1);
  // I think the frame is not changed at this moment;
  skin_color_check_result = CheckSkinColor(cb, cr, face_1,
                                           g_config->skin_color_low_threshold_);
  //DumpYUV(frame, uv_frame, face_1);

  if (skin_color_check_result == 0 || g_kcf_roi->width < 5
      || g_kcf_roi->height < 5
      || g_haar_failure_times >= g_config->upper_limit_of_failure) {
    if (skin_color_check_result == 0)
      mylog("INFO: Frame %d, skin check failed!", g_local_frame_count);
    // TODO: change 5 to a parameter
    if (g_kcf_roi->width < 5 || g_kcf_roi->height < 5)
      mylog("INFO: Frame %d, KCF ROI window is too small!", g_local_frame_count);
    if (g_haar_failure_times >= g_config->upper_limit_of_failure)
      mylog("INFO: Frame %d, Haar failed too many, %d times!",
            g_local_frame_count, g_haar_failure_times);

    haar_result = DoHaarForAllOrientation(
        image_after_resize, 0.0,
        NULL,
        CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH,
        face_2);

    if (haar_result != 0) {
      //double previous_kcf_scale_factor = g_current_kcf_scale_factor;
      //face_3 = face_2;
      RestoreCoordinate(face_2, image_after_resize.size(),
                        g_current_kcf_scale_factor, NULL, false, face_3);
      g_current_kcf_scale_factor = g_config->std_face_height_
          / static_cast<double>(face_3.height);
      int reset_kcf_result = ResetKCF(
          frame, face_3,
          g_current_kcf_scale_factor);
      if (reset_kcf_result == 0) g_before_kcf = true;
      face = face_3;
      g_haar_failure_times = 0;
      mylog("INFO: Frame %d: skin check fail, "
            "but Haar found face in the whole image",
            g_local_frame_count);
      return 1;
    } else {
      face = Rect(-1, -1, 0, 0);
      mylog("INFO: Frame %d: skin check fail. "
            "Haar found no faces in the whole image, neither.",
            g_local_frame_count);
      return 0;
    }
  } else { // We don't need to search the whole image
    ExtendKCFRegion(*g_kcf_roi, image_after_resize.size(),
                    g_config->edge_extend_factor_, face_4);

    // TODO: scale_factor = ??
    haar_result = DoHaarForAllOrientation(image_after_resize, 0.0, &face_4,
                                          CASCADE_SCALE_IMAGE, face_5);
    if (haar_result != 0) {
      //face_6 = face_5;
      RestoreCoordinate(face_5, image_after_resize.size(),
                        g_current_kcf_scale_factor, NULL, false, face_6);
      g_current_kcf_scale_factor = g_config->std_face_height_
          / static_cast<double>(face_6.height);
      ResetKCF(frame, face_6, g_current_kcf_scale_factor);
      face = face_6;
      g_haar_failure_times = 0;
      mylog("INFO: skin check pass, and Haar found face in the extended region");
      return 1;
    } else {  // Haar not found, return KCF result

      g_haar_failure_times++;
      face = face_1;
      // maybe here we should return 0 because haar dose NOT found faces
      // but return 0 will impact fps
      mylog(
          "INFO: Frame %d: skin check pass, but Haar found no face (%d times) "
          "in the extended region. Return KCF result",
          g_local_frame_count, g_haar_failure_times);
      return 1;
    }
  }
  // Never go here.
  return 0;

}

/*
int DumpYUV(const Mat& frame, const uint8_t *uv, const Rect& roi) {
  // open file
  if (g_config->dump_yuv_ == 0) return 0;
  string yuv_file_name = g_config->dump_yuv_path_ + "yuv-"
      + to_string(g_local_frame_count) + ".log";
  FILE *yuv_fp = fopen(yuv_file_name.data(), "w");
  if (!yuv_fp) {
    print_and_log("ERROR: Cannot open dump yuv log file : %s", yuv_file_name.data());
    exit(EXIT_FAILURE);
  }

  // write the head
  fprintf(yuv_fp, "low threshold %f\n", g_config->skin_color_low_threshold_);
  fprintf(yuv_fp, "mid threshold %f\n", g_config->skin_color_middle_threshold_);
  fprintf(yuv_fp, "high threshold %f\n", g_config->skin_color_high_threshold_);
  fprintf(yuv_fp, "cb low %d cb high %d\n", g_config->skin_cb_low_,
          g_config->skin_cb_high_);
  fprintf(yuv_fp, "cr low %d cr high %d\n", g_config->skin_cr_low_,
          g_config->skin_cr_high_);

  //change the region
  int x1 = (roi.x + 1) / 2;
  int y1 = (roi.y + 1) / 2;
  int x2 = x1 + roi.width / 2;
  int y2 = y1 + roi.height / 2;
  const uint8_t *row_ptr, *col_ptr;
  int cb_sum[300];
  int cr_sum[300];
  memset(cb_sum, 0, sizeof(int) * 300);
  memset(cr_sum, 0, sizeof(int) * 300);

  //dump cb
  fprintf(yuv_fp, "CB\n");
  for (int i = y1; i <= y2; ++i) {
    row_ptr = uv + i * g_config->kImageWidth_; // stride = kImageWidth/2*2
    col_ptr = row_ptr + x1 * 2;
    for (int j = x1; j <= x2; ++j) {
      int Cb = static_cast<int>(*col_ptr);
      ++col_ptr;
      //int Cr = static_cast<int>(*col_ptr);
      ++col_ptr;
      fprintf(yuv_fp, "%3d ", Cb);
      cb_sum[Cb]++;
    }
    fprintf(yuv_fp, "\n");
  }

  //dump cr
  fprintf(yuv_fp, "CR\n");
  for (int i = y1; i <= y2; ++i) {
    row_ptr = uv + i * g_config->kImageWidth_; // stride = kImageWidth/2*2
    col_ptr = row_ptr + x1 * 2;
    for (int j = x1; j <= x2; ++j) {
      //int Cb = static_cast<int>(*col_ptr);
      ++col_ptr;
      int Cr = static_cast<int>(*col_ptr);
      ++col_ptr;
      fprintf(yuv_fp, "%3d ", Cr);
      cr_sum[Cr]++;
    }
    fprintf(yuv_fp, "\n");
  }

  //dump y
  fprintf(yuv_fp, "Y\n");
  for (int i = roi.y; i < roi.y + roi.height; ++i) {
    for (int j = roi.x; j < roi.x + roi.width; ++j) {
      fprintf(yuv_fp, "%3d ", (int)(frame.at<uchar>(i, j)));
    }
    fprintf(yuv_fp, "\n");
  }

  //print conforming cb & cr pixel
  int tmp_cb_low = g_config->skin_cb_high_;
  int tmp_cb_high = g_config->skin_cb_low_;
  int tmp_cr_low = g_config->skin_cr_high_;
  int tmp_cr_high = g_config->skin_cr_low_;

  fprintf(yuv_fp, "Conforming pixel cr & cb == 0 else == 1\n");
  for (int i = y1; i <= y2; ++i) {
    row_ptr = uv + i * g_config->kImageWidth_; // stride = kImageWidth/2*2
    col_ptr = row_ptr + x1 * 2;
    for (int j = x1; j <= x2; ++j) {
      int Cb = static_cast<int>(*col_ptr);
      ++col_ptr;
      int Cr = static_cast<int>(*col_ptr);
      ++col_ptr;
      if (g_config->skin_cb_low_ <= Cb && Cb <= g_config->skin_cb_high_
          && g_config->skin_cr_low_ <= Cr && Cr <= g_config->skin_cr_high_) {
        fprintf(yuv_fp, "0");
        if (Cb < tmp_cb_low) tmp_cb_low = Cb;
        if (Cb > tmp_cb_high) tmp_cb_high = Cb;
        if (Cr < tmp_cr_low) tmp_cr_low = Cr;
        if (Cr > tmp_cr_high) tmp_cr_high = Cr;
      } else {
        fprintf(yuv_fp, "1");
      }

    }
    fprintf(yuv_fp, "\n");
  }

  fprintf(yuv_fp, "conforming pixel cb low %d\n", tmp_cb_low);
  fprintf(yuv_fp, "conforming pixel cb high %d\n", tmp_cb_high);
  fprintf(yuv_fp, "conforming pixel cr low %d\n", tmp_cr_low);
  fprintf(yuv_fp, "conforming pixel cr high %d\n", tmp_cr_high);

  //print conforming cb & cr &y pixel
  fprintf(yuv_fp, "Conforming pixel cr & cb & y == 0 else == 1\n");
  for (int i = roi.y; i < roi.y + roi.height; ++i) {
    for (int j = roi.x; j < roi.x + roi.width; ++j) {
      int y = (int)(frame.at<uchar>(i, j));
      int uv_x = j / 2;
      int uv_y = i / 2;
      row_ptr = uv + uv_y * g_config->kImageWidth_; // stride = kImageWidth/2*2
      col_ptr = row_ptr + uv_x * 2;
      int Cb = static_cast<int>(*col_ptr);
      ++col_ptr;
      int Cr = static_cast<int>(*col_ptr);
      if (g_config->skin_cb_low_ <= Cb && Cb <= g_config->skin_cb_high_
          && g_config->skin_cr_low_ <= Cr && Cr <= g_config->skin_cr_high_
          && y > 80) {
        fprintf(yuv_fp, "0");
      } else {
        fprintf(yuv_fp, "1");
      }
    }
    fprintf(yuv_fp, "\n");
  }

  //print cb cr
  fprintf(yuv_fp, "value   CB    CR\n");
  for (int i = 0; i < 256; ++i) {
    fprintf(yuv_fp, "%3d %3d %3d\n", i, cb_sum[i], cr_sum[i]);
  }
  fprintf(yuv_fp, "sum\n");
  for (int i = 0; i < 256; ++i) {
    fprintf(yuv_fp, "%3d %3d %3d\n", i, cb_sum[i], cr_sum[i]);
    cb_sum[i + 1] += cb_sum[i];
    cr_sum[i + 1] += cr_sum[i];
  }

  fclose(yuv_fp);
  return 1;
}
*/

// Check skin color, < 25%, restart haar search in full image.
int CheckSkinColor(const Mat& cb, const Mat& cr, const Rect& roi, double threshold) {
  int x1 = roi.x;
  int y1 = roi.y;
  int x2 = x1 + roi.width;
  int y2 = y1 + roi.height;

  int skin_point_num = 0;

  for (int i = y1; i < y2; ++i) {
    for (int j = x1; j < x2; ++j) {
      int Cb = cb.at<unsigned char>(i,j);
      int Cr = cr.at<unsigned char>(i,j);
      if (g_config->skin_cb_low_ <= Cb && Cb <= g_config->skin_cb_high_
          && g_config->skin_cr_low_ <= Cr && Cr <= g_config->skin_cr_high_) {
        ++skin_point_num;
      }
//      if(skin_point_num >= skin_point_lower_bound)
//        return 1;
    }
  }
  double skin_percent = skin_point_num / (double)(roi.width * roi.height);
  mylog("INFO: Frame %d, Skin percent: %f", g_local_frame_count, skin_percent);
  if (skin_percent > g_config->skin_color_high_threshold_) return 3;
  if (skin_percent > g_config->skin_color_middle_threshold_) return 2;
  if (skin_percent > g_config->skin_color_low_threshold_) return 1;
  return 0;
}

int SetKCFParameters() {
  g_kcf_param = new TrackerKCF::Params;
  // Is this reasonable? use Y\U\V as three customized named color?
  g_kcf_param->desc_pca = TrackerKCF::GRAY;
  g_kcf_param->desc_npca = 0;
  g_kcf_param->compress_feature = true;
  g_kcf_param->compressed_size = 1;

  g_kcf_roi = new Rect2d;

  return 1;
}
