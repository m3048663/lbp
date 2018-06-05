#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include "opencv2/contrib/contrib.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>

using namespace cv;
using namespace std;

Mat getLocalRegionLBPH(const Mat& src,int minValue,int maxValue,bool normed);

//原始LBP特征计算

void getOriginLBPFeature(cv::Mat& src,cv::Mat& dst)
{
    for(int i=1; i < src.rows-1;i++)
    {
        for(int j=1; j < src.cols-1;j++)
        {
            unsigned char center = src.at<uchar>(i,j);
            unsigned char lbpCode = 0;
            lbpCode |= (src.at<uchar>(i-1,j-1) > center) << 7;
            lbpCode |= (src.at<uchar>(i-1,j  ) > center) << 6;
            lbpCode |= (src.at<uchar>(i-1,j+1) > center) << 5;
            lbpCode |= (src.at<uchar>(i  ,j+1) > center) << 4;
            lbpCode |= (src.at<uchar>(i+1,j+1) > center) << 3;
            lbpCode |= (src.at<uchar>(i+1,j  ) > center) << 2;
            lbpCode |= (src.at<uchar>(i+1,j-1) > center) << 1;
            lbpCode |= (src.at<uchar>(i  ,j-1) > center) << 0;
	    //std::cout << "lbpcode = " << bitset<sizeof(int)*8>(lbpCode) << std::endl;
            dst.at<uchar>(i-1,j-1) = lbpCode;
        }
    }
}


void getOriginLBP(cv::Mat &img)
{
	Mat ori_dst = Mat(img.rows - 2, img.cols -2, CV_8UC1);
	getOriginLBPFeature(img, ori_dst);
	//cv::imwrite("false_cr_lbp.jpg",ori_dst);	
	namedWindow("lbp",CV_WINDOW_NORMAL);
	imshow("lbp",ori_dst);
}

//圆形LBP特征计算，效率优化版本，声明时默认neighbors=8
void getCircularLBPFeature(cv::Mat &src,cv::Mat &dst,int radius,int neighbors)
{

    for(int k=0;k<neighbors;k++)
    {
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //为双线性插值做准备
        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1-tx) * (1-ty);
        float w2 =    tx  * (1-ty);
        float w3 = (1-tx) *    ty;
        float w4 =    tx  *    ty;
        //循环处理每个像素
        for(int i=radius;i<src.rows-radius;i++)
        {
            for(int j=radius;j<src.cols-radius;j++)
            {
                //获得中心像素点的灰度值
                float center = src.at<float>(i,j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src.at<float>(i+x1,j+y1) * w1 + src.at<float>(i+x1,j+y2) *w2 \
                    + src.at<float>(i+x2,j+y1) * w3 +src.at<float>(i+x2,j+y2) *w4;
                
                dst.at<uchar>(i-radius,j-radius) |= (neighbor>center) <<(neighbors-k-1);
            }
        }
    }
}


//等价模式LBP特征计算
//计算跳变次数
int getHopTimes(int n)
{
    int count = 0;
    bitset<8> binaryCode = n;
    for(int i=0;i<8;i++)
    {
        if(binaryCode[i] != binaryCode[(i+1)%8])
        {
            count++;
        }
    }
    return count;
}
void getUniformPatternLBPFeature(cv::Mat &src,cv::Mat &dst,int radius,int neighbors)
{
    //LBP特征值对应图像灰度编码表，直接默认采样点为8位
    uchar temp = 1;
    uchar table[256] = {0};
    for(int i=0;i<256;i++)
    {
        if(getHopTimes(i)<3)
        {
            table[i] = temp;
            temp++;
        }
    }
    //是否进行UniformPattern编码的标志
    bool flag = false;
    //计算LBP特征图
    for(int k=0;k<neighbors;k++)
    {
        if(k==neighbors-1)
        {
            flag = true;
            }
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //为双线性插值做准备
        //对采样点偏移量分别进行上下取整
        int fx = static_cast<int>(floor(rx));
        int cx = static_cast<int>(ceil(rx));
        int fy = static_cast<int>(floor(ry));
        int cy = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - fx;
        float ty = ry - fy;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1-tx) * (1-ty);
        float w2 =    tx  * (1-ty);
        float w3 = (1-tx) *    ty;
        //float w4 =    tx  *    ty;
	float w4 = 1 - w1 - w2 - w3;
        //循环处理每个像素
        for(int i=radius;i<src.rows-radius;i++)
        {
            for(int j=radius;j<src.cols-radius;j++)
            {
                //获得中心像素点的灰度值
                float center = src.at<float>(i,j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src.at<float>(i+fy,j+fx) * w1 + src.at<float>(i+fy,j+cx) *w2 \
                    + src.at<float>(i+cy,j+fx) * w3 +src.at<float>(i+cy,j+cx) *w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                    dst.at<uchar>(i-radius,j-radius) |= (neighbor>center) <<(neighbors-k-1);
                //进行LBP特征的UniformPattern编码
                if(flag)
                {
                    dst.at<uchar>(i-radius,j-radius) = table[dst.at<uchar>(i-radius,j-radius)];
                }
            }
        }
    }
}


void getCircleLBP(cv::Mat img, int radius, int neighbors)
{
	Mat dst = Mat(img.rows-2*radius, img.cols-2*radius, CV_8UC1);
	getUniformPatternLBPFeature(img, dst, radius, neighbors);
 	namedWindow("circlelbp");
	imshow("circlelbp",dst);
}


//计算LBP特征图像的直方图LBPH
Mat getLBPH(cv::Mat &src,int numPatterns,int grid_x,int grid_y,bool normed)
{
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    //定义LBPH的行和列，grid_x*grid_y表示将图像分割成这么些块，numPatterns表示LBP值的模式种类
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns,  CV_32FC1);
    if(src.empty())
    {
        return result.reshape(1,1);
    }
    int resultRowIndex = 0;
    //对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
    for(int i=0;i<grid_x;i++)
    {
        for(int j=0;j<grid_y;j++)
        {
            //图像分块
            Mat src_cell = Mat(src,Range(i*height,(i+1)*height),Range(j*width,(j+1)*width));
            //计算直方图
            Mat hist_cell = getLocalRegionLBPH(src_cell,0,(numPatterns-1),true);
            //将直方图放到result中
            Mat rowResult = result.row(resultRowIndex);
            hist_cell.reshape(1,1).convertTo(rowResult,CV_32FC1);
            resultRowIndex++;
        }
    }
    return result.reshape(1,1);
}
//计算一个LBP特征图像块的直方图
Mat getLocalRegionLBPH(const Mat& src,int minValue,int maxValue,bool normed)
{
    //定义存储直方图的矩阵
    Mat result;
    //计算得到直方图bin的数目，直方图数组的大小
    int histSize = maxValue - minValue + 1;
    //定义直方图每一维的bin的变化范围
    float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
    //定义直方图所有bin的变化范围
    const float* ranges = { range };
    //计算直方图，src是要计算直方图的图像，1是要计算直方图的图像数目，0是计算直方图所用的图像的通道序号，从0索引
    //Mat()是要用的掩模，result为输出的直方图，1为输出的直方图的维度，histSize直方图在每一维的变化范围
    //ranges，所有直方图的变化范围（起点和终点）
    calcHist(&src,1,0,Mat(),result,1,&histSize,&ranges,true,false);
    //归一化
    if(normed)
    {
        result /= (int)src.total();
    }
    //结果表示成只有1行的矩阵
    return result.reshape(1,1);
}

Mat getHistogramImage(const Mat &img)
{	
	MatND hist;
	float hranges[2] = {0.0, 64.0};
	const float* ranges[1];
	ranges[0] = hranges;
	int channels[1];
	channels[0] = 0;
	int histSize[1];
	histSize[0] = 65;
	
	calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges,true, false);
	double Max_val = 0.0f;
	double Min_val = 64.0f; 
	minMaxLoc(hist, &Min_val, &Max_val, 0, 0);
	
	Mat histImg(65, 65, CV_8U, Scalar(65));
	double rate = (histSize[0] / Max_val) * 0.9;

	for (int h = 0; h < 65; h++)
	{
		float binVal = hist.at<float>(h);
		cv::line(histImg, cv::Point(h,histSize[0]), Point(h, histSize[0]-binVal*rate), Scalar::all(0));
	}
	return histImg;
}

void BGR2YCbCr(const Mat &img, Mat &ycbcr, Mat &y, Mat &cb, Mat &cr)
{
	vector<Mat> channels;
	cvtColor(img, ycbcr, CV_BGR2YCrCb);
	split(ycbcr, channels);
	y = channels.at(0);
	cb = channels.at(1);
	cr = channels.at(2);
	std::cout << y.channels() << std::endl;

}

void getYCbCr(const Mat &img)
{
	Mat ycbcr = Mat(img.rows, img.cols,CV_8UC3,Scalar(0));
	Mat y = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	Mat cb = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	Mat cr = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	BGR2YCbCr(img, ycbcr, y, cb, cr);
	cv::imwrite("false_y.jpg",y);
	cv::imwrite("false_cb.jpg",cb);
	cv::imwrite("false_cr.jpg",cr);	
	//cv::imshow("y",y);
	//cv::imshow("cr",cr);
	//cv::imshow("cb",cb);
}


int main()
{
	//Mat img = cv::imread("/home/ubuntu/lbp/false.jpeg");
	//namedWindow("origin");
	//imshow("origin",img);
	//cvtColor(img, img, CV_BGR2GRAY);

	//getYCbCr(img);	
	Mat lbp = cv::imread("home/ubuntu/lbp/false_y.jpg");
	std::cout << lbp.channels() << std::endl;
	getOriginLBP(lbp);
	
	//Mat histImg(y.rows, y.cols, CV_8U, Scalar(0));
	//histImg = getHistogramImage(y);
	//namedWindow("hist",CV_WINDOW_NORMAL);
	//cv::imshow("hist", histImg);


	//getCircularLBP(img, dst1, radius, neighbors);


     	//Mat histImg(65, 65, CV_8U, Scalar(65));
	//histImg = getHistogramImage(ori_dst);
	//namedWindow("hist",CV_WINDOW_NORMAL);
	//cv::imshow("hist", histImg);


	cv::waitKey(0);
}
