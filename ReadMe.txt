系统环境：
	OS： lINUX MINT 17.3 64位
	RAM: 8G
	CPU: I5-4690K


依赖库：
	1.opencv3.1
	2.opencv_contrib3.1
	3.opencv所依赖的其他库
	（注：lib文件夹下包含在上述系统环境编译后的opencv和opencv_contrib的运行库）


运行指南：

make clean; make

./face
	会打开默认摄像头并将结果生成video/output.avi视频
./face output.avi
	会打开默认摄像头并将结果视频存入output.avi
./face input.avi output.avi
	会打开input.avi处理，讲结果视频存入output.avi

运行时 按q停止程序 或等待视频自动处理完成

（注1：程序不会处理参数错误请谨慎执行）
（注2：若要使用lib下的库，请手动添加到LD_LIBRARY_PATH环境变量下）
