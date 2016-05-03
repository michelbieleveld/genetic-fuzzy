#include "Utilities.h"
#include <chrono>
#include <ctime>
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <iostream>
#include <algorithm>


float Utilities::AUC(const std::vector<float>& predictions, const DataSource& datasource, int dependent)
{
	std::vector<float> labels;
	labels.reserve(datasource.rows.size());
	for (unsigned int i = 0; i < datasource.rows.size(); i++)
		labels.push_back(datasource.rows[i].data[dependent]);
	return AUC(predictions, labels);
}

float Utilities::AUC(const std::vector<float>& predictions, const std::vector<float>& labels, float threshold)
{
	// Machine Learning : ECML 2007 : 18th European Conference on Machine Learning ...
	// An improved model selection heuristic for AUC, Shaomin Wu

	std::vector<std::tuple<bool, float>> z;
	z.reserve(predictions.size());
	unsigned m = 0;
	unsigned n = 0;

	for (unsigned int i = 0; i < predictions.size(); i++)
	{
		bool presence = (labels[i] > threshold);
		if (presence)
			m += 1;
		else
			n += 1;
		z.push_back(std::make_pair(presence, predictions[i]));
	}

	std::sort(z.begin(), z.end(), [](const std::tuple<bool, float> &left, const std::tuple<bool, float> &right) {
		return std::get<1>(left) > std::get<1>(right);
	});
	
	float AUC = 0;
	unsigned int c = 0;

	for (unsigned int i = 0; i < z.size(); i++)
	{
		if (std::get<0>(z[i]))
			c += 1;
		else
			AUC += c;
	}
	AUC /= m * n;
	return AUC;
}

std::array<unsigned int, 4> Utilities::ConfusionMatrix(const DataSource& datasource, const std::vector<float>& result, unsigned int dependent, float threshold)
{
	std::array<unsigned int, 4> confusion{};
	// tp fp fn tn
	for (unsigned int r = 0; r < datasource.rows.size(); r++)
	{
		float observed = datasource.rows[r].data[dependent];
		float predicted = result[r];
		if (observed > threshold)
		{
			if (predicted > threshold)
				confusion[0] += 1;
			else
				confusion[2] += 1;
		}
		else
		{
			if (predicted > threshold)
				confusion[1] += 1;
			else
				confusion[3] += 1;
		}
	}
	return confusion;
}

float Utilities::TSS(std::array<unsigned int, 4> confusion)
{
	if (confusion[0] + confusion[2] == 0) return 0;
	if (confusion[3] + confusion[1] == 0) return 0;
	float sensitivity = (float)confusion[0] / (float)(confusion[0] + confusion[2]);
	float specificity = (float)confusion[3] / (float)(confusion[3] + confusion[1]);
	return sensitivity + specificity - 1.0f;
}

std::string Utilities::GetTimeStampedFilename(const std::string& FileName)
{
	return FileName;
	/*
	std::time_t t = std::time(nullptr);
	struct tm timeinfo;
	localtime_s(&timeinfo, &t);

	std::stringstream sfilename;

	if (FileName.find_last_of(".") != std::string::npos)
	{
		sfilename << FileName.substr(0, FileName.find_last_of("."));
		sfilename << std::put_time(&timeinfo, "%Y%m%d%H%M%S");
		sfilename << FileName.substr(FileName.find_last_of("."));
	}
	else
		sfilename << FileName << std::put_time(&timeinfo, "%Y%m%d%H%M%S");
	return sfilename.str();
	*/
}
