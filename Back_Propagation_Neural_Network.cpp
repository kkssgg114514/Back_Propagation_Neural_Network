#include <iostream>
#include "lib/Net.h"
#include "lib/Utils.h"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) 
{

    //创建网络对象
    Net net;

    //读取训练数据
    const vector<Sample> trainDataSet = Utils::getTrainData("data/train.txt");

    //训练神经网络
    net.train(trainDataSet);

    //用神经网络预计样本
    const vector<Sample> testDataSet = Utils::getTestData("data/test.txt");
    vector<Sample> predSet = net.predict(testDataSet);
    for (auto& pred : predSet) 
    {
        pred.display();
    }

    return 0;
}