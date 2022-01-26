//#include "pch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> //dnn模块
#include <time.h>
 
using namespace std;
using namespace cv;
using namespace ::dnn; //调用DNN命名空间
clock_t start, finish;
 
String model_file = "../model/bvlc_reference_caffenet.caffemodel"; //模型结构文件
String model_text = "../model/deploy.prototxt";  //模型数据
 
											  //图像深度学习检测
double detect_NN(Mat detectImg, Net net)
{
	if (net.empty())
	{
		cout << "no model!" << endl;
		return -1;
	}
 
	//initialize images(输入图像初始化)
	Mat src = detectImg.clone();
	if (src.empty())
	{
		return -1;
	}
 
	//图像识别转换
	//第一个参数输入图像，第二个参数图像放缩大小，第三个参数输入图像尺寸,第四个参数模型训练图像三个通道RGB的均值（均值文件）
	start = clock();
 
	Mat inputBlob;
 
	//resize(src, src, Size(227, 227));
	// 参数分别为输入图像，归一化参数，模型大小，BGR均值
	inputBlob = blobFromImage(src, 1.0, Size(227, 227), Scalar(92.71, 106.44, 118.11));
 
	Mat prob; //输出结果
			  //循环
	for (int i = 0; i < 1; i++)
	{
		net.setInput(inputBlob, "data");
		prob = net.forward("prob"); //输出层2
	}
	Mat probMat = prob.reshape(1, 1); //转化为1行2列
	Point classNumber;				  //最大值的位置
	double classProb;
	//最大值多少
	//最大最小值查找，忽略最小值
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("classidx is:%d\n", classidx);
	printf("prob is %f\n", classProb);
	finish = clock();
	double duration = (double)(finish - start);
	printf("run time is %f ms\n", duration);
	return duration;
}
 
int main()
{
	Net net = readNetFromCaffe(model_text, model_file);
	Mat detectImg = imread("../data/cat.jpg");
	double runTime = detect_NN(detectImg, net);
	return 0;
}
