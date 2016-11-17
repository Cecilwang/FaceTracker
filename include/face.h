#ifndef LIBFACE_FACE_H_
#define LIBFACE_FACE_H_

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include "opencv2/core.hpp"
#include <string>

using namespace std;
using namespace cv;

// log utilities
#define frlog(fmt, args...) \
  do { \
    struct timeval now; \
    gettimeofday(&now, NULL); \
    fprintf(stderr, "[%5lu.%06lu] %s() %d: " fmt "\n", \
        now.tv_sec, now.tv_usec, __FUNCTION__, __LINE__, ##args); \
  } while (0)

#define mylog(fmt, args...) \
  do { \
    struct timeval now; \
    gettimeofday(&now, NULL); \
    fprintf(g_log_fp, "[%5lu.%06lu] %s() %d: " fmt "\n", \
        now.tv_sec, now.tv_usec, __FUNCTION__, __LINE__, ##args); \
  } while (0)

#define print_and_log(fmt, args...) \
  do { \
    frlog(fmt, ##args); \
    mylog(fmt, ##args); \
  } while (0)

// timestamp utilities
__attribute_used__ static inline uint64_t gettimestamp_ns() {
  uint64_t val;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  val = ts.tv_sec;
  val *= 1000000000;
  val += ts.tv_nsec;
  return val;
}

extern FILE *g_log_fp;

class Config {
 public:
  Config()
    : front_xml_path_("./data/xml/front.xml"),
      profile_xml_path_("./data/xml/profile.xml"),
      dump_image_path_("./data/img/"),
      dump_yuv_path_("./data/yuv/"),
      scale_factor_(1.05),
      min_neighbors_front_(3),
      min_neighbors_profile_(3),
      min_face_width_(20),
      min_face_height_(20),
      max_face_width_(250),
      max_face_height_(250),
      std_face_width_(30),
      std_face_height_(30),
      haar_interval_(4),
      init_scale_factor_(0.4),
      edge_extend_factor_(3.0),
      skin_color_low_threshold_(0.15),
      skin_color_middle_threshold_(0.2),
      skin_color_high_threshold_(0.9),
      skin_cb_low_(77),
      skin_cb_high_(127),
      skin_cr_low_(133),
      skin_cr_high_(173),
      kImageWidth_(1920),
      kImageHeight_(1080),
      smoothe_(1),
      replay_coor_(0),
      creat_coor_(0),
      dump_image_(0),
      dump_yuv_(0),
      upper_limit_of_failure(30)
  { };
  String front_xml_path_;
  String profile_xml_path_;
  String dump_image_path_;
  String dump_yuv_path_;
  double scale_factor_;
  int min_neighbors_front_;
  int min_neighbors_profile_;
  int min_face_width_;
  int min_face_height_;
  int max_face_width_;
  int max_face_height_;
  int std_face_width_;
  int std_face_height_;
  int haar_interval_;
  double init_scale_factor_;
  double edge_extend_factor_;
  double skin_color_low_threshold_;
  double skin_color_middle_threshold_;
  double skin_color_high_threshold_;
  int skin_cb_low_;
  int skin_cb_high_;
  int skin_cr_low_;
  int skin_cr_high_;
  int kImageWidth_;
  int kImageHeight_;
  int smoothe_;
  int replay_coor_;
  int creat_coor_;
  int dump_image_;
  int dump_yuv_;
  int upper_limit_of_failure;
};

extern Config* g_config;
extern int g_local_frame_count;
extern bool g_before_kcf;

const char* const kConfigPath = "./data/cfg/libface.cfg";

int DoImageProcessing(void *frame_buffer, Rect &face);

// DumpImage for Debug
int DumpImage2d(const String& filename, const Mat& image, Rect2d *region);
int DumpImage(const String& filename, const Mat& image, Rect *region);

// avoid negative output, out of image border, negative width for roi_out
int RectifyCoordinate(const Rect& roi_in, const Size& image_size,
                      Rect& roi_out);

#endif // LIBFACE_FACE_H_
