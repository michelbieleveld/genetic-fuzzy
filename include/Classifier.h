#pragma once
#include <vector>
#include "DataSource.h"

class Classifier
{
public:
	virtual std::vector<float> classify(const DataSource& datasource) = 0;
};

