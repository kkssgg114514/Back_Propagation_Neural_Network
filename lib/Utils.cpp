#include <iostream>
#include <fstream>
#include "Utils.h"
#include "Config.h"

#if defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <direct.h>
#else
#include <unistd.h>
#endif

vector<double> Utils::getFileData(const string& filename)
{
    std::vector<double> res;

    std::ifstream in(filename);
    //判断是否打开成功
    if (in.is_open())
    {
        //不断读取文件数据
        while (!in.eof())
        {
            //暂时从文件写入缓冲区
            double buffer;
            in >> buffer;
            res.push_back(buffer);
        }
        //关闭文件
        in.close();
    }
    else
    {
        //打开失败
        std::cout << "打开文件失败:" << filename << std::endl;
    }
    return res;
}

vector<Sample> Utils::getTrainData(const string& filename)
{
    std::vector<Sample> res;

    std::vector<double> buffer = getFileData(filename);

    //循环读入训练数据，输入输出数量不固定，每次跳转数量根据输入输出数量决定
    for (size_t i = 0; i < buffer.size(); i += Config::INNODE + Config::OUTNODE)
    {
        Sample tmp;
        for (size_t t = 0; t < Config::INNODE; t++)
        {
            tmp.in.push_back(buffer.at(i + t));
        }
        for (size_t t = 0; t < Config::OUTNODE; t++)
        {
            tmp.out.push_back(buffer.at(i + Config::INNODE + t));
        }
        res.push_back(tmp);
    }
    return res;
}

vector<Sample> Utils::getTestData(const string& filename)
{
    std::vector<Sample> res;

    std::vector<double> buffer = getFileData(filename);

    //循环读入测试数据，输入数量不固定，每次跳转数量根据输入数量决定
    for (size_t i = 0; i < buffer.size(); i += Config::INNODE)
    {
        Sample tmp;
        for (size_t t = 0; t < Config::INNODE; t++)
        {
            tmp.in.push_back(buffer.at(i + t));
        }
        res.push_back(tmp);
    }
    return res;
}
