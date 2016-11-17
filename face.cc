#include "face.h"
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include "opencv2/imgcodecs.hpp"
#include "haar.h"
#include "face.h"
#include "smooth.h"
#include "kcf.h"

FILE *g_log_fp  = NULL;
FILE *g_coor_fp = NULL;

bool g_before_kcf             = true;
int  g_local_frame_count      = 0;
Face_region_state *face_state = NULL;

Config *g_config = NULL;

static int SetAffinity();
static int Initialize();
static int ReadConfig();

static int Initialize() {
  g_before_kcf = true;

  // Read something from configuration file
  if (ReadConfig() == 0) {
    print_and_log("ERROR: Cannot read configuration file!");
    exit(EXIT_FAILURE);
  }

  if (LoadHaarClassifier() == 0) {
    print_and_log("ERROR: Cannot load Haar classifier file!");
    exit(EXIT_FAILURE);
  }

  // SetHaarParameters
  // SetKCFParameters
  SetKCFParameters();

  // init smoothe
  face_state = (struct Face_region_state *)malloc(
    sizeof(struct Face_region_state));
  face_region_init(face_state);

  return 1;
}

static int ReadConfig() {
  g_config = new Config();

  // read configuration file
  FILE *cfg_fp = fopen(kConfigPath, "r");

  if (!cfg_fp) {
    print_and_log("ERROR: Cannot open %s", kConfigPath);
    exit(EXIT_FAILURE);
  }
  fscanf(cfg_fp, "%*s %lf", &(g_config->scale_factor_));
  mylog("g_config->scale_factor_ : %lf", g_config->scale_factor_);
  fscanf(cfg_fp, "%*s %d", &(g_config->min_neighbors_front_));
  mylog("g_config->min_neighbors_front_ : %d", g_config->min_neighbors_front_);
  fscanf(cfg_fp, "%*s %d", &(g_config->min_neighbors_profile_));
  mylog("g_config->min_neighbors_profile_ : %d",
        g_config->min_neighbors_profile_);
  fscanf(cfg_fp, "%*s %d", &(g_config->min_face_width_));
  mylog("g_config->min_face_width_ : %d", g_config->min_face_width_);
  fscanf(cfg_fp, "%*s %d", &(g_config->min_face_height_));
  mylog("g_config->min_face_height_ : %d", g_config->min_face_height_);
  fscanf(cfg_fp, "%*s %d", &(g_config->max_face_width_));
  mylog("g_config->max_face_width_ : %d", g_config->max_face_width_);
  fscanf(cfg_fp, "%*s %d", &(g_config->max_face_height_));
  mylog("g_config->max_face_height_ : %d", g_config->max_face_height_);
  fscanf(cfg_fp, "%*s %d", &(g_config->std_face_width_));
  mylog("g_config->std_face_width_ : %d", g_config->std_face_width_);
  fscanf(cfg_fp, "%*s %d", &(g_config->std_face_height_));
  mylog("g_config->std_face_height_ : %d", g_config->std_face_height_);
  fscanf(cfg_fp, "%*s %d", &(g_config->haar_interval_));
  mylog("g_config->haar_interval_ : %d", g_config->haar_interval_);
  fscanf(cfg_fp, "%*s %lf", &(g_config->init_scale_factor_));
  mylog("g_config->init_scale_factor_ : %lf", g_config->init_scale_factor_);
  fscanf(cfg_fp, "%*s %lf", &(g_config->edge_extend_factor_));
  mylog("g_config->edge_extend_factor_ : %lf", g_config->edge_extend_factor_);
  fscanf(cfg_fp, "%*s %lf", &(g_config->skin_color_low_threshold_));
  mylog("g_config->skin_color_low_threshold_ : %lf",
        g_config->skin_color_low_threshold_);
  fscanf(cfg_fp, "%*s %lf", &(g_config->skin_color_middle_threshold_));
  mylog("g_config->skin_color_middle_threshold_ : %lf",
        g_config->skin_color_middle_threshold_);
  fscanf(cfg_fp, "%*s %lf", &(g_config->skin_color_high_threshold_));
  mylog("g_config->skin_color_high_threshold_ : %lf",
        g_config->skin_color_high_threshold_);
  fscanf(cfg_fp, "%*s %d", &(g_config->skin_cb_low_));
  mylog("g_config->skin_cb_low_ : %d", g_config->skin_cb_low_);
  fscanf(cfg_fp, "%*s %d", &(g_config->skin_cb_high_));
  mylog("g_config->skin_cb_high_ : %d", g_config->skin_cb_high_);
  fscanf(cfg_fp, "%*s %d", &(g_config->skin_cr_low_));
  mylog("g_config->skin_cr_low_ : %d", g_config->skin_cr_low_);
  fscanf(cfg_fp, "%*s %d", &(g_config->skin_cr_high_));
  mylog("g_config->skin_cr_high_ : %d", g_config->skin_cr_high_);
  fscanf(cfg_fp, "%*s %d", &(g_config->kImageWidth_));
  mylog("g_config->kImageWidth_ : %d", g_config->kImageWidth_);
  fscanf(cfg_fp, "%*s %d", &(g_config->kImageHeight_));
  mylog("g_config->kImageHeight_ : %d", g_config->kImageHeight_);
  fscanf(cfg_fp, "%*s %d", &(g_config->smoothe_));
  mylog("g_config->smoothe_ : %d", g_config->smoothe_);
  fscanf(cfg_fp, "%*s %d", &(g_config->replay_coor_));
  mylog("g_config->replay_coor_ : %d", g_config->replay_coor_);
  fscanf(cfg_fp, "%*s %d", &(g_config->creat_coor_));
  mylog("g_config->creat_coor_ : %d", g_config->creat_coor_);
  fscanf(cfg_fp, "%*s %d", &(g_config->dump_image_));
  mylog("g_config->dump_image_ : %d", g_config->dump_image_);
  fscanf(cfg_fp, "%*s %d", &(g_config->dump_yuv_));
  mylog("g_config->dump_yuv_ : %d", g_config->dump_yuv_);
  fscanf(cfg_fp, "%*s %d", &(g_config->upper_limit_of_failure));
  mylog("g_config->upper_limit_of_failure: %d",
        g_config->upper_limit_of_failure);
  fclose(cfg_fp);
  return 1;
}

static int SetAffinity() {
  // Get numbers of CPU cores
  int num_cpus = sysconf(_SC_NPROCESSORS_CONF);
  cpu_set_t my_set;  // Define your cpu_set bit mask.

  CPU_ZERO(&my_set); // Initialize it all to 0, i.e. no CPUs selected.

  // Set the bit that represents all cores.
  for (int cpu = 0; cpu < num_cpus; cpu++) CPU_SET(cpu, &my_set);

  // Set affinity of this calling thread (and its children) to the defined mask.
  if (sched_setaffinity(0, sizeof(my_set), &my_set) != 0) {
    print_and_log("ERROR: Call sched_setaffinity failed!");
    return 0;
  } else {
    print_and_log("INFO: Call sched_setaffinity succeed. Core # = %d",
                  num_cpus);
    return 1;
  }
}

// do some initialization, start tracking check thread
void Create() {
  // create log file for debugging
  struct timeval now;

  gettimeofday(&now, NULL);
  char log_file_name[100];

  sprintf(log_file_name, "%s%ld%s", "./data/log/face-",
          now.tv_sec, ".log");
  g_log_fp = fopen(log_file_name, "w");

  if (!g_log_fp) {
    frlog("ERROR: Cannot open %s!", log_file_name);
    exit(EXIT_FAILURE);
  }

  SetAffinity();
  Initialize();

  print_and_log("replay_coor_ : %d", g_config->replay_coor_);

  if (g_config->replay_coor_ == 1) {
    sprintf(log_file_name, "./data/coor/coor.log");
    g_coor_fp = fopen(log_file_name, "r");

    if (!g_coor_fp) {
      print_and_log("ERROR: Cannot open %s!", log_file_name);
      exit(EXIT_FAILURE);
    }
  } else {
    sprintf(log_file_name, "%s%ld%s", "./data/coor/coor-",
            now.tv_sec, ".log");
    g_coor_fp = fopen(log_file_name, "w");

    if (!g_coor_fp) {
      print_and_log("ERROR: Cannot open %s!", log_file_name);
      exit(EXIT_FAILURE);
    }
  }
}

void Destroy() {
  face_region_clean(face_state);
  fclose(g_log_fp);
  fclose(g_coor_fp);
}

int creat_fake_coordinate(Rect& face) {
  static int x = 0;
  static int y = 0;
  static int w = 10;
  static int h = 10;

  x = x + 1;

  if (x >= g_config->kImageWidth_ - 10) {
    x = 0;
    y = y + 1;

    if (y >= g_config->kImageHeight_ - 10) {
      x = y = 0;
    }
  }
  face = CvRect(x, y, w, h);
  return 1;
}

int DoImageProcessing(Mat frame, Rect& face) {
  int status = 0;

  Mat yuv;
  cvtColor(frame, yuv,            CV_BGR2YCrCb);
  Mat chan[3];
  split(yuv, chan);
  Mat oringnal_frame = chan[0];
  Mat cb = chan[2];
  Mat cr = chan[1];

  if (g_before_kcf) {
    status = DoHaarForAllOrientation(
      oringnal_frame, g_config->init_scale_factor_,
      NULL,
      CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH, face);

    if (status == 1) { // Found
      g_current_kcf_scale_factor = g_config->std_face_height_
                                   / static_cast<double>(face.height);

      if (!ResetKCF(oringnal_frame, face, g_current_kcf_scale_factor)) {
        g_before_kcf               = true;
        g_current_kcf_scale_factor = 0.0;
        mylog("INFO: Frame %d, before KCF, Haar found face [%d %d %d %d], "
              "but ResetKCF fail!",
              g_local_frame_count, face.x, face.y, face.width, face.height);
      } else {
        g_before_kcf         = false;
        g_haar_failure_times = 0;
        mylog("INFO: Frame %d, before KCF, Haar found face [%d %d %d %d] ",
              g_local_frame_count, face.x, face.y, face.width, face.height);
      }
    } else {
      g_before_kcf               = true;
      g_current_kcf_scale_factor = 0.0;
      mylog("INFO: Frame %d, before KCF, Haar found no face!",
            g_local_frame_count);
    }
  } else if (g_local_frame_count % g_config->haar_interval_ != 0) {
    //frlog("DoKCF");
    status = DoKCF(oringnal_frame, face);

    if (status == 0) { // Face not found
      g_before_kcf = true;
    }
  } else {
    //frlog("DoKCFWithHaar");
    status = DoKCFWithHaar(oringnal_frame, cb, cr, face);

    if (status == 0) { // Face not found
      g_before_kcf = true;
    }
  }

  return status;
}

void Run(int argc, char **argv) {
  VideoCapture  cap;
  VideoWriter   output;
  Mat frame;

  if (argc < 3) {
    frlog("open default camera");
    cap.open(0);
    if (!cap.isOpened()) {
      frlog("Cannot open camera or video");
      return;
    }
  }
  if (argc == 3) {
		frlog("oepn video [%s]", argv[1]);
		cap.open(argv[1]);
    if (!cap.isOpened()) {
      frlog("Cannot open camera or video");
      return;
    }
  }

  cap>>frame;
  g_config->kImageWidth_ = frame.cols;
  g_config->kImageHeight_ = frame.rows;
  imshow("tra cker", frame);

  if (argc == 1) {
    frlog("output to [video/output.avi]");
    output.open("./video/output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(g_config->kImageWidth_,
               g_config->kImageHeight_),true);
  }

  if (argc == 2) {
    frlog("output to [%s]", argv[1]);
    output.open(argv[1], CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(g_config->kImageWidth_,
               g_config->kImageHeight_),
                true);
  }

  if (argc == 3) {
    frlog("output to [%s]",  argv[2]);
    cap.open(argv[1]);
    output.open(argv[2], CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(g_config->kImageWidth_,
               g_config->kImageHeight_),
                true);
  }

  while (true) {
    cap >> frame;

    uint64_t time_start        = 0, time_end = 0, time_used = 0;
    static uint64_t total_time = 0;
    static int total_count     = 0;
    struct Face_region face_region, face_region_after_smoothe;

    time_start = gettimestamp_ns();
    int  status = 1;
    Rect face;

    if ((g_config->replay_coor_ == 0) &&
        (g_config->creat_coor_ == 0)) status = DoImageProcessing(frame, face);
    else if (g_config->replay_coor_ == 1) {
      if (fscanf(g_coor_fp, "%d %d %d %d", &face.x, &face.y, &face.width,
                 &face.height) != 4) {
        print_and_log("ERROR: coor.log is EOF or error !");
        exit(EXIT_FAILURE);
      }
    } else if (g_config->creat_coor_ == 1) {
      creat_fake_coordinate(face);
    }

    if (status != 0) {
      // smoothe the face
      if (g_config->smoothe_) {
        face_region = face_region_get(face.x,
                                      face.y,
                                      face.width,
                                      face.height);
        face_region_after_smoothe = face_region_smoothe(face_state,
                                                        face_region);
        face = CvRect(face_region_after_smoothe.x, face_region_after_smoothe.y,
                      face_region_after_smoothe.w, face_region_after_smoothe.h);
      }


      // record the face coordinate
      if (g_config->replay_coor_ == 0) {
        fprintf(g_coor_fp, "%d %d %d %d\n", face.x, face.y, face.width,
                face.height);
      }

      rectangle(frame, face, Scalar(0, 255, 0), 3, 1);
    } else {
      // when lose the target, should reset the smooth function
      face_region_clean(face_state);
      face_state = (struct Face_region_state *)malloc(
        sizeof(struct Face_region_state));
      face_region_init(face_state);
    }

    time_end  = gettimestamp_ns();
    time_used = (time_end - time_start) / 1000000; // time in ms

    output << frame;
    imshow("tra cker", frame);

    mylog("INFO: Frame: %d, Face: [%d %d %d %d], time: %llu ms",
          g_local_frame_count, face.x, face.y, face.width, face.height,
          time_used);

    total_time += time_used;
    ++g_local_frame_count;
    ++total_count;

    if (total_count >= 100) {
      print_and_log("INFO: Face tracking fps: %f",
                    total_count / (total_time / 1000.0f));
      total_count = 0;
      total_time  = 0;
    }

    char c = waitKey(30);
    if(c == 'q') break;
  }
}

int DumpImage(const String& filename, const Mat& image, Rect *region) {
  if (g_config->dump_image_) {
    Mat image_temp;
    image_temp = image.clone();

    if (region != NULL) {
      int x_upper_bound = image.cols - 1;
      int y_upper_bound = image.rows - 1;
      int x1, x2, y1, y2;

      x1 = region->x;

      if (x1 < 0) {
        x1 = 0;
      } else if (x1 >= x_upper_bound) {
        x1 = x_upper_bound;
      }

      y1 = region->y;

      if (y1 < 0) {
        y1 = 0;
      } else if (y1 >= y_upper_bound) {
        y1 = y_upper_bound;
      }

      if (region->width < 0) {
        x2 = x1;
      } else {
        x2 = region->x + region->width;
      }

      if (x2 < x1) {
        x2 = x1;
      } else if (x2 > x_upper_bound) {
        x2 = x_upper_bound;
      }

      if (region->height < 0) {
        y2 = y1;
      } else {
        y2 = region->y + region->height;
      }

      if (y2 < y1) {
        y2 = y1;
      } else if (y2 > y_upper_bound) {
        y2 = y_upper_bound;
      }

      Rect tmp_region = Rect(cvRound(x1), cvRound(y1), cvRound(x2 - x1),
                             cvRound(y2 - y1));
      rectangle(image_temp, tmp_region, Scalar(0, 0, 0), 3, 1);
    }
    imwrite(g_config->dump_image_path_ + filename + ".jpg", image_temp);
    return 1;
  } else {
    return 0;
  }
}

int DumpImage2d(const String& filename, const Mat& image, Rect2d *region) {
  if (g_config->dump_image_ == 0) return 0;

  if (region == NULL) return DumpImage(filename, image, NULL);

  Rect *region_tmp = new Rect;
  region_tmp->x      = cvRound(region->x);
  region_tmp->y      = cvRound(region->y);
  region_tmp->width  = cvRound(region->width);
  region_tmp->height = cvRound(region->height);
  return DumpImage(filename, image, region_tmp);
}

int RectifyCoordinate(const Rect& roi_in, const Size& image_size,
                      Rect& roi_out) {
  int x1_upper_bound = image_size.width - 2;
  int y1_upper_bound = image_size.height - 2;
  int x2_upper_bound = image_size.width;
  int y2_upper_bound = image_size.height;

  // OpenCV typically assumes that the top and left boundary (x1, y1) of the
  // rectangle are inclusive, while the right and bottom boundaries (x2, y2)
  // are not.
  // For roi_out, we don't want a 0~1 width or 0~1 height rectangle. The minimum
  // width or height should be 2
  int x1, y1, x2, y2;

  x1 = roi_in.x;

  if (x1 < 0) {
    x1 = 0;
  } else if (x1 >= x1_upper_bound) {
    x1 = x1_upper_bound;
  }

  y1 = roi_in.y;

  if (y1 < 0) {
    y1 = 0;
  } else if (y1 >= y1_upper_bound) {
    y1 = y1_upper_bound;
  }

  x2 = roi_in.x + roi_in.width;

  if (x2 < x1 + 2) {
    x2 = x1 + 2;
  } else if (x2 > x2_upper_bound) {
    x2 = x2_upper_bound;
  }

  y2 = roi_in.y + roi_in.height;

  if (y2 < y1 + 2) {
    y2 = y1 + 2;
  } else if (y2 > y2_upper_bound) {
    y2 = y2_upper_bound;
  }

  roi_out = Rect(x1, y1, x2 - x1, y2 - y1);
  return 1;
}

int main(int argc, char **argv) {
  Create();
  namedWindow( "tra cker", 1 );
  Run(argc, argv);
  Destroy();
}
