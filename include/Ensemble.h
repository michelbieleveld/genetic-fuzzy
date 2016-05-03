#pragma once
#include "Classifier.h"
#include "DataSource.h"
#include <vector>
#include <memory>


class Ensemble : public Classifier
{
public:
	Ensemble();
	Ensemble(const std::string filepath);
	~Ensemble();

	std::vector<std::unique_ptr<Classifier>> classifiers;
	void AddClassifiers(const std::string filepath);

	std::vector<float> classify(const DataSource& datasource);

	

};

