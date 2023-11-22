#pragma once
#include <vector>
#include "Config.h"

using std::vector;

struct Sample
{
    //输入输出都不是固定数量
    std::vector<double> in;
    std::vector<double> out;

    Sample();

    Sample(const vector<double>& feature, const vector<double>& label);

    void display();

};

struct Node
{
    double value{};
    double bias{};  //偏置值
    double bias_delta{};    //偏置值修正

    std::vector<double> weight; //权值
    std::vector<double> weight_delta;   //权值修正值

    explicit Node(size_t nextLayerSize);
};

class Net
{
private:
    //创建神经网络
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