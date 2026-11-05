#ifndef __VNAVIGATORINTERFACE_H__
#define __VNAVIGATORINTERFACE_H__
#include <vector>
#include <string>
#include <memory>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief: 视频图像自定位算法接口类
 *
 * 该接口类用于隔离具体算法与调用程序，
 * 视频图像自定位算法必须按照此接口类的形式封装成动态链接库。
 * 
 * 接口类为外部调用程序提供了算法初始化，自定位模型解算的接口，参与比测的各个
 * 单位需要按照接口的定义实现实际的功能，否则无法进入比测环节。
 *
 * 发布版本：v1.0
 * 发布时间：2026-8-25
 * 版权：发布方保留所有权力
 * 作者：nobody1k
 */


/*
 * @brief: 定义视频图像自定位算法接口的名字空间
 *
 * 
*/
namespace VNAVIGATORINTERFACE
{
	// 以下接口的对齐方式为1字节对齐
	#pragma pack(push, 1)
	/*
	 * @brief: 定义自定位结果的结构
	 *
	 * */
	struct vNaviResult
	{
	public:
		vNaviResult(): longtitude(0.0),
        	latitude(0.0),
        	altitude(0.0f),
        	selfEval(0.0f)
		{
		}
		double longtitude;		// 表示相机的经度，WGS8椭球坐标系，十进制度表示，精确到小数点后8位
		double latitude;		// 表示相机的纬度，WGS84椭球坐标系，十进制度表示，精确到小数点后8位
        	double altitude;        	// 表示相机的高度，高度为正常高（即海拔高度），单位为米，精确到小数点后2位
        	float selfEval;         	// 自评估结果，0.0表示自评估认为计算得到的相机位置精度不满足指标要求，1.0表示
                                		// 自评估认为计算得到的相机位置满足指标要求
	};
	/*
	 * @brief: 定义与图像对应的参数的结构，每一幅图像有一个结构
	 * 
	 * */
	struct vTelemParams
	{
	public:
		vTelemParams():number(0),
		droneLongtitude(0.0),
		droneLatitude(0.0),
		droneAltitude(0.0f),
		droneYaw(0.0f),
		dronePitch(0.0f),
		droneRoll(0.0f),
		sensorYaw(0.0f),
		sensorPitch(0.0f),
		sensorRoll(0.0f),
		innsYaw(0.0f),
		innsPitch(0.0f),
		innsRoll(0.0f),
		hfov(0.0f),
		vfov(0.0f),
		ccdSize(0.0f),
		isEO(false),
        	droneLongtitude_gt(0.0),
        	droneLatitude_gt(0.0),
        	droneAltitude_gt(0.0f)
		{
		}
		unsigned int number;		// 表示按时间顺序获取的第几组参数
		double droneLongtitude;		// 平台的经度，WGS84坐标系，单位为度，精确到第8位小数
		double droneLatitude;		// 平台的纬度，WGS84坐标系，单位为度，精确到第8位小数
		float droneAltitude;		// 平台的高度，正常高，单位为米，精确到第2位小数
		float droneYaw;			// 平台的航向角，坐标系为北东地(NED)右手坐标系(详见NED坐标系定义)，单位为度，精确到第2位小数
		float dronePitch;		// 平台的俯仰角，NED坐标系，单位为度，精确到第2位小数
		float droneRoll;		// 平台的滚转角，NED坐标系，单位为度，精确到第2位小数
		float sensorYaw;		// 传感器相对于安装面的角度，右手系，X轴同光轴指向，绕Z轴旋转获得的角度，范围为[0,360)
		float sensorPitch;		// 传感器相对于安装面的角度，右手系，X轴同光轴指向，绕Y轴旋转获得的角度，范围为[-180,180]
		float sensorRoll;		// 传感器相对于安装面的角度，右手系，X轴同光轴指向，绕X轴旋转获得的角度，范围为[-180,180]
		float innsYaw;			// 内置惯导的航向角，定义同平台的角度
		float innsPitch;		// 内置惯导的俯仰角，定义同平台的角度
		float innsRoll;			// 内置惯导的滚转角，定义同平台的角度
		float hfov;			// 传感器的水平视场角，单位为度，精确到第2位小数
		float vfov;			// 传感器的垂直视场角，单位为度，精确到第2位小数
		float ccdSize;			// 传感器像元的物理尺寸，单位位微米，精确到第1位小数
		bool isEO;			// 传感器是否为光电转塔，光电转塔为true，其他为false
						// 光电转塔的光轴与内置惯导x轴指向平行，其他传感器的光轴与内置惯导
						// 的x轴指向不平行，需要通过传感器相对于安装面的角度来确定
        	double droneLongtitude_gt;  	// 经度真值
        	double droneLatitude_gt;     	// 纬度真值
        	float droneAltitude_gt;     	// 高度真值 
	};
	#pragma pack(pop)



	/*
	 * @brief: 视频图像自定位算法接口类
	 *
	 * 该类采用虚函数方式定义了具体算法需要实现的若干接口。
	 * 各个定位算法需要实现具体的接口，比测评估程序调用
	 * 这些接口，对算法进行评估。
	 *
	 * */
	class IVideoNavigator
	{
	public:
		/*
		 * @brief: 初始化自定位算法
		 *
		 * 该接口用于初始化自定位算法，初始化不计入总体耗时
		 *
		 * 输入：1. const std::string &initFilePath 输入参数，表示输入到 
		 * 	 算法的初始化参数文件路径，所有的算法初始化均通过该初始化参数文件提供。
		 *  	2. int imgType 输入参数，视频图像类型，红外图像为0，真彩色图像为1；
		 *	3. int dayOrNight 输入参数，昼间或夜间图像，昼间为0，夜间为1；
		 *	4. bool isSlant, 输入参数，是否为大倾斜图像，true表示该图像为大倾斜图像，
		 *	   false表示该图像不是大倾斜图像；
		 *  	5. bool isEO, 输入参数，是否为光电转塔的图像，ture表示是光电转塔图像
		       	   false表示不是光电转塔图像。
		 *
		 * 返回值：int 返回状态，0表示初始化失败，1表示初始化成功，其他
		 *         值由算法自行定义。
		 *
		 * */
		virtual int initVNavigator(const std::string &initFilePath,
								int imgType,
								int dayOrNight,
								bool isSlant,
								bool isEO) = 0;
		/*
		 * @brief: 释放自定位算法资源，关闭自定位算法
		 *
		 * 该接口用于释放自定位算法运行期间分配的资源
		 *
		 * 输入：无
		 *
		 * 返回值：int 返回状态，0表示关闭失败，1表示关闭成功，其他
		 *         值由算法自行定义
		 *
		 * */
		virtual int closeVNavigator() = 0;
		/*
		 * @brief: 根据输入视频图像和参数，解算一次自定位模型，得到自定位结果，解算时间计入到总体耗时
		 * 
		 * 该接口为实时运行阶段自定位模型解算的函数，每输入一幅视频图像和参数
		 * ，该接口将解算一组自定位模型，保存在算法类内部，并输出自定位结果。
		 *
		 * 输入：1. const unsigned char * pImgData 输入参数，视频图像数据，当图像为
		 *	    真彩色三通道时，图像数据组织方式为：RGB RGB RGB......;
		 *	 2. int imgWidth 输入参数，视频图像的宽度，单位为像素；
		 *	 3. imgHeight 输入参数，视频图像的高度，单位为像素；
		 *	 4. int imgChannels 输入参数，视频图像的通道数，灰度图像通道数为1，
		 *	    真彩色图像的通道数为3；
		 *	 5. int imgType 输入参数，视频图像类型，红外图像为0，真彩色图像为1；
		 *	 6. int dayOrNight 输入参数，昼间或夜间图像，昼间为0，夜间为1；
		 *	 7. bool isSlant, 输入参数，是否为大倾斜图像，true表示该图像为大倾斜图像，
		 *	    false表示该图像不是大倾斜图像；
		 *	 8. const vTelemParams &telem 与图像对应的参数结构
         	 *   	 9. vNaviResult &outputResult 输出结果
		 *	 10. void *pParams = nullptr 输入参数，算法需要的其他参数，可通过该指针传入
		 *	    算法需要的其他参数，默认为NULL。
		 *
		 * 返回值：int 返回状态，0表示定位模型解算不成功，1表示成功，其他
         	 *         值由算法自行定义
		 *
		 * */
		virtual int stepVNavigator(const unsigned char * pImgData,
				int imgWidth,
				int imgHeight,
				int imgChannels,
				int imgType,
				int dayOrNight,
				bool isSlant,
				const vTelemParams &telem,
                		vNaviResult &outputResult,
				void *pParams = nullptr) = 0;

	};// 视频图像自定位算法接口类结束

};// 视频图像自定位算法接口类名字空间结束


/*
 * @brief: 创建视频图像自定位算法接口类的函数
 *
 * 通过该函数创建一个指向自定位算法接口类的智能指针对象
 *
 * */

std::shared_ptr<VNAVIGATORINTERFACE::IVideoNavigator> createVideoNavigator();

/*
 * @brief: 释放自定位算法接口资源的函数
 *
 * 通过该函数释放接口智能指针对象（可不调用，由智能指针自行释放资源）
 *
 *
 * */
void freeVideoNavigator(std::shared_ptr<VNAVIGATORINTERFACE::IVideoNavigator> pVideoNavigator);


#ifdef __cplusplus
}
#endif


#endif // __VNAVIGATORINTERFACE_H_
