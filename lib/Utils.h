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

    //��ȡ�ļ�������
    vector<double> getFileData(const string& filename);

    //����ѵ������
    vector<Sample> getTrainData(const string& filename);

    //���ղ������ݣ���ѵ����
    vector<Sample> getTestData(const string& filename);

    //�������
    void OutputToFile(vector<double>* outData, const string& filename);
}