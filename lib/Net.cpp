#include "Net.h"
#include "Utils.h"
#include <random>
#include <iostream>

Net::Net()
{
    //生成随机数，使用种子
    std::mt19937 rd;
    rd.seed(std::random_device()());
    //创建随机数生成器
    std::uniform_real_distribution<double> distribution(-1, 1);

    //遍历
    //初始化输入层
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        inputLayer[i] = new Node(Config::HIDE0NODE);
        //对于输入层连接的每一个隐藏层节点都要设置权重
        //一开始是随机数，修正值是0.0
        for (size_t j = 0; j < Config::HIDE0NODE; j++)
        {
            inputLayer[i]->weight[j] = distribution(rd);
            inputLayer[i]->weight_delta[j] = 0.0f;
        }
    }

    //初始化隐藏层0
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        hiddenLayer[i] = new Node(Config::OUTNODE);
        //隐藏层的偏置值也要初始化
        hiddenLayer[i]->bias = distribution(rd);
        //对于每一个与隐藏层链接的输出层都要设置权重
        for (size_t j = 0; j < Config::OUTNODE; j++)
        {
            hiddenLayer[i]->weight[j] = distribution(rd);
            hiddenLayer[i]->weight_delta[j] = 0.0f;
        }
    }

    //初始化输出层
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        outputLayer[i] = new Node(0);
        outputLayer[i]->bias = distribution(rd);
        outputLayer[i]->bias_delta = 0.0f;
    }
}

void Net::grad_zero()
{
    //清空输入层偏置变化值
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        inputLayer[i]->weight_delta.assign(inputLayer[i]->weight_delta.size(), 0.0f);
    }

    //清空隐藏层偏置变化值和权重变化值
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        hiddenLayer[i]->bias_delta = 0.0f;
        hiddenLayer[i]->weight_delta.assign(hiddenLayer[i]->weight_delta.size(), 0.0f);
    }

    //清空输出层偏置值
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        outputLayer[i]->bias_delta = 0.0f;
    }
}

void Net::forward()
{
    //正向传播
            //生成隐藏层的值
    for (size_t j = 0; j < Config::HIDE0NODE; j++)
    {
        double sum = 0.0f;
        for (size_t i = 0; i < Config::INNODE; i++)
        {
            sum += inputLayer[i]->value * inputLayer[i]->weight.at(j);
        }
        sum -= hiddenLayer[j]->bias;

        hiddenLayer[j]->value = Utils::sigmoid(sum);
    }

    //生成输出层的值
    for (size_t j = 0; j < Config::OUTNODE; j++)
    {
        double sum = 0.0f;
        for (size_t i = 0; i < Config::HIDE0NODE; i++)
        {
            sum += hiddenLayer[i]->value * hiddenLayer[i]->weight[j];
        }
        sum -= outputLayer[j]->bias;

        outputLayer[j]->value = Utils::sigmoid(sum);
    }
}

double Net::calculateLoss(const vector<double>& label)
{
    //计算误差
    double error = 0.0f;
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        double tmp = std::fabs(outputLayer[i]->value - label.at(i));
        error += tmp * tmp / 2;
    }
    return error;
}

void Net::backward(const vector<double>& label)
{
    //反向传播
    //输出层节点的偏置值修正值
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        double bias_delta = -(label.at(i) - outputLayer[i]->value) *
            outputLayer[i]->value * (1.0f - outputLayer[i]->value);
        outputLayer[i]->bias_delta += bias_delta;
    }

    //隐藏层到输出层的权值修正值
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        for (size_t j = 0; j < Config::OUTNODE; j++)
        {
            double weight_delta = (label.at(j) - outputLayer[j]->value) *
                outputLayer[j]->value * (1.0f - outputLayer[j]->value) *
                hiddenLayer[i]->value;
            hiddenLayer[i]->weight_delta[j] += weight_delta;
        }
    }

    //隐藏层节点偏置值修正值
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        double sum = 0.0f;
        for (size_t j = 0; j < Config::OUTNODE; j++)
        {
            sum += -(label.at(j) - outputLayer[j]->value) *
                outputLayer[j]->value * (1.0f - outputLayer[j]->value) *
                hiddenLayer[i]->weight.at(j);
        }
        hiddenLayer[i]->bias_delta += sum * hiddenLayer[i]->value * (1.0f - hiddenLayer[i]->value);
    }

    //输入层到隐藏层的权值修正值
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        for (size_t j = 0; j < Config::HIDE0NODE; j++)
        {
            double sum = 0.0f;
            for (size_t k = 0; k < Config::OUTNODE; k++)
            {
                sum += (label.at(k) - outputLayer[k]->value) *
                    outputLayer[k]->value * (1.0f - outputLayer[k]->value) *
                    hiddenLayer[j]->weight.at(k);
            }
            inputLayer[i]->weight_delta[j] += sum *
                hiddenLayer[j]->value * (1.0f - hiddenLayer[j]->value) *
                inputLayer[i]->value;
        }
    }
}

void Net::revise(size_t batch_size)
{
    auto batch_size_double = (double)batch_size;

    //调整输入层到隐藏层的所有权值
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        for (size_t j = 0; j < Config::HIDE0NODE; j++)
        {
            inputLayer[i]->weight[j] += Config::rate * inputLayer[i]->weight_delta.at(j) / batch_size_double;
        }
    }

    //调整所有隐藏层的偏置值和隐藏层到输出层的权值
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        hiddenLayer[i]->bias += Config::rate *
            hiddenLayer[i]->bias_delta / batch_size_double;

        for (size_t j = 0; j < Config::OUTNODE; j++)
        {
            hiddenLayer[i]->weight[j] += Config::rate *
                hiddenLayer[i]->weight_delta.at(j) / batch_size_double;
        }
    }

    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        outputLayer[i]->bias += Config::rate *
            outputLayer[i]->bias_delta / batch_size_double;
    }
}

bool Net::train(const vector<Sample>& trainDataSet)
{
    for (size_t times = 0; times < Config::mosttimes; times++)
    {
        grad_zero();

        //误差最大值
        double error_max = 0.0f;

        //更新误差值（####）
        for (const Sample& trainSample : trainDataSet)
        {
            //输入
            for (size_t i = 0; i < Config::INNODE; i++)
            {
                inputLayer[i]->value = trainSample.in.at(i);
            }

            forward();

            double error = calculateLoss(trainSample.out);
            error_max = std::max(error_max, error);

            backward(trainSample.out);
        }

        if (error_max < Config::threshold)
        {
            std::cout << "训练成功！迭代次数：" << times + 1 << std::endl;
            std::cout << "最终误差：" << error_max << std::endl;
            return true;
        }
        else if (times % 5000 == 0) 
        {
            std::cout << "迭代了：" << times << "次\n最大误差值为：" << error_max;
            //printf_s("#epoch %-7lu - max_loss: %lf\n", times, error_max);
        }
        revise(trainDataSet.size());
    }
    //printf("Failed within %lu epoch.", Config::max_epoch);
    std::cout << "最大迭代次数：" << Config::mosttimes << "次" << std::endl;

    return false;
}

Sample Net::predict(const vector<double>& feature)
{
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        inputLayer[i]->value = feature.at(i);
    }

    forward();

    vector<double> label(Config::OUTNODE);

    for (size_t k = 0; k < Config::OUTNODE; ++k)
    {
        label[k] = outputLayer[k]->value;
    }
    Sample pred = Sample(feature, label);
    return pred;
}

vector<Sample> Net::predict(const vector<Sample>& predictDataSet)
{
    vector<Sample> predSet;

    for (auto& sample : predictDataSet) {
        Sample pred = predict(sample.in);
        predSet.push_back(pred);
    }

    return predSet;
}

Node::Node(size_t nextLayerSize) 
{
    weight.resize(nextLayerSize);
    weight_delta.resize(nextLayerSize);
}

Sample::Sample() = default;

Sample::Sample(const vector<double>& feature, const vector<double>& label) 
{
    this->in= feature;
    this->out = label;
}

void Sample::display() 
{
    Utils::OutputToFile(&out, "data\\res.txt");
}