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
    //�ж��Ƿ�򿪳ɹ�
    if (in.is_open())
    {
        //���϶�ȡ�ļ�����
        while (!in.eof())
        {
            //��ʱ���ļ�д�뻺����
            double buffer;
            in >> buffer;
            res.push_back(buffer);
        }
        //�ر��ļ�
        in.close();
    }
    else
    {
        //��ʧ��
        std::cout << "���ļ�ʧ��:" << filename << std::endl;
    }
    return res;
}

vector<Sample> Utils::getTrainData(const string& filename)
{
    std::vector<Sample> res;

    std::vector<double> buffer = getFileData(filename);

    //ѭ������ѵ�����ݣ���������������̶���ÿ����ת�����������������������
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

    //ѭ������������ݣ������������̶���ÿ����ת��������������������
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
