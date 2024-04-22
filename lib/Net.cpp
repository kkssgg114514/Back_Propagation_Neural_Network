#include "Net.h"
#include "Utils.h"
#include <random>
#include <iostream>

Net::Net()
{
    //�����������ʹ������
    std::mt19937 rd;
    rd.seed(std::random_device()());
    //���������������
    std::uniform_real_distribution<double> distribution(-1, 1);

    //����
    //��ʼ�������
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        inputLayer[i] = new Node(Config::HIDE0NODE);
        //������������ӵ�ÿһ�����ز�ڵ㶼Ҫ����Ȩ��
        //һ��ʼ�������������ֵ��0.0
        for (size_t j = 0; j < Config::HIDE0NODE; j++)
        {
            inputLayer[i]->weight[j] = distribution(rd);
            inputLayer[i]->weight_delta[j] = 0.0f;
        }
    }

    //��ʼ�����ز�0
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        hiddenLayer[i] = new Node(Config::OUTNODE);
        //���ز��ƫ��ֵҲҪ��ʼ��
        hiddenLayer[i]->bias = distribution(rd);
        //����ÿһ�������ز����ӵ�����㶼Ҫ����Ȩ��
        for (size_t j = 0; j < Config::OUTNODE; j++)
        {
            hiddenLayer[i]->weight[j] = distribution(rd);
            hiddenLayer[i]->weight_delta[j] = 0.0f;
        }
    }

    //��ʼ�������
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        outputLayer[i] = new Node(0);
        outputLayer[i]->bias = distribution(rd);
        outputLayer[i]->bias_delta = 0.0f;
    }
}

void Net::grad_zero()
{
    //��������ƫ�ñ仯ֵ
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        inputLayer[i]->weight_delta.assign(inputLayer[i]->weight_delta.size(), 0.0f);
    }

    //������ز�ƫ�ñ仯ֵ��Ȩ�ر仯ֵ
    for (size_t i = 0; i < Config::HIDE0NODE; i++)
    {
        hiddenLayer[i]->bias_delta = 0.0f;
        hiddenLayer[i]->weight_delta.assign(hiddenLayer[i]->weight_delta.size(), 0.0f);
    }

    //��������ƫ��ֵ
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        outputLayer[i]->bias_delta = 0.0f;
    }
}

void Net::forward()
{
    //���򴫲�
            //�������ز��ֵ
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

    //����������ֵ
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
    //�������
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
    //���򴫲�
    //�����ڵ��ƫ��ֵ����ֵ
    for (size_t i = 0; i < Config::OUTNODE; i++)
    {
        double bias_delta = -(label.at(i) - outputLayer[i]->value) *
            outputLayer[i]->value * (1.0f - outputLayer[i]->value);
        outputLayer[i]->bias_delta += bias_delta;
    }

    //���ز㵽������Ȩֵ����ֵ
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

    //���ز�ڵ�ƫ��ֵ����ֵ
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

    //����㵽���ز��Ȩֵ����ֵ
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

    //��������㵽���ز������Ȩֵ
    for (size_t i = 0; i < Config::INNODE; i++)
    {
        for (size_t j = 0; j < Config::HIDE0NODE; j++)
        {
            inputLayer[i]->weight[j] += Config::rate * inputLayer[i]->weight_delta.at(j) / batch_size_double;
        }
    }

    //�����������ز��ƫ��ֵ�����ز㵽������Ȩֵ
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

        //������ֵ
        double error_max = 0.0f;

        //�������ֵ��####��
        for (const Sample& trainSample : trainDataSet)
        {
            //����
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
            std::cout << "ѵ���ɹ�������������" << times + 1 << std::endl;
            std::cout << "������" << error_max << std::endl;
            return true;
        }
        else if (times % 5000 == 0) 
        {
            std::cout << "�����ˣ�" << times << "��\n������ֵΪ��" << error_max;
            //printf_s("#epoch %-7lu - max_loss: %lf\n", times, error_max);
        }
        revise(trainDataSet.size());
    }
    //printf("Failed within %lu epoch.", Config::max_epoch);
    std::cout << "������������" << Config::mosttimes << "��" << std::endl;

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