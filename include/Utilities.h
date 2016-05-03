#pragma once
#include <array>
#include <vector>
#include "DataSource.h"


class Utilities
{
public:
	static std::array<unsigned int, 4> ConfusionMatrix(const DataSource& datasource, const std::vector<float>& result, unsigned int dependent = 0, float threshold = 0.5f);
	static float TSS(std::array<unsigned int, 4> confusion);
	static std::string GetTimeStampedFilename(const std::string& FileName);
	static float AUC(const std::vector<float>& predictions, const std::vector<float>& labels, float threshold = 0.5f);
	static float AUC(const std::vector<float>& predictions, const DataSource& datasource, int dependent=0);


};

