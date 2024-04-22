#pragma once
#include <cmath>
#include <vector>
#include <string>
#include "Net.h"

using std::vector;
using std::string;

namespace Utils 
{
    static double sigmoid(double x) 
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    //读取文件的数据
    vector<double> getFileData(const string& filename);

    //接收训练数据
    vector<Sample> getTrainData(const string& filename);

    //接收测试数据（已训练）
    vector<Sample> getTestData(const string& filename);

    //输出数据
    void OutputToFile(vector<double>* outData, const string& filename);
}