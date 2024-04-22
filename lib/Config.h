#pragma once

namespace Config
{
	const size_t INNODE = 41;    //输入层节点数
	const size_t HIDE0NODE = 50;  //隐藏层节点数，一般大于输入层
	const size_t OUTNODE = 1;   //输出层节点数

	const double rate = 0.8f; //步长
	const double threshold = 1e-4;    //误差允许值（误差进入此范围认为成功）
	const size_t mosttimes = 1e6; //训练次数
}