#pragma once
#include <vector>
#include "Config.h"

using std::vector;

struct Sample
{
    //������������ǹ̶�����
    std::vector<double> in;
    std::vector<double> out;

    Sample();

    Sample(const vector<double>& feature, const vector<double>& label);

    void display();

};

struct Node
{
    double value{};
    double bias{};  //ƫ��ֵ
    double bias_delta{};    //ƫ��ֵ����

    std::vector<double> weight; //Ȩֵ
    std::vector<double> weight_delta;   //Ȩֵ����ֵ

    explicit Node(size_t nextLayerSize);
};

class Net
{
private:
    //����������
    Node* inputLayer[Config::INNODE]{};
    Node* hiddenLayer[Config::HIDE0NODE]{};
    Node* outputLayer[Config::OUTNODE]{};

    void grad_zero();

    void forward();

    double calculateLoss(const vector<double>& label);

    void backward(const vector<double>& label);

    void revise(size_t batch_size);

public:
    Net();

    bool train(const vector<Sample>& trainDataSet);

    Sample predict(const vector<double>& feature);

    vector<Sample> predict(const vector<Sample>& predictDataSet);
};