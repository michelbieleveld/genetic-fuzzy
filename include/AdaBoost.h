#pragma once
#include <vector>
#include "Individual.h"
#include "Classifier.h"
#include "DataSource.h"

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>


struct Fitness {
	float fitness;
	int rank;
};

class AdaBoost : public Classifier
{
public:
	AdaBoost();
	AdaBoost(const std::vector<float>& _alpha, const std::vector<Individual>& _classifiers);
	~AdaBoost();

	std::vector<Individual> classifiers;
	std::vector<float> alpha;

	static AdaBoost run(const DataSource& datasource, const DataSource& datasourceValidate, unsigned int boosts, unsigned int generations, unsigned int populationSize, unsigned int instructions, unsigned int dependent, unsigned int shards);
	static std::vector<std::tuple<unsigned int, std::vector<float>>> evaluate(const std::vector<std::tuple<unsigned int, std::vector<float>>>& results, const DataSource& datasource, int dependent = 0);
	
	std::vector<float> classify(const DataSource& datasource);
	
	void save(std::string filepath);
	static AdaBoost load(std::string filepath);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(alpha), CEREAL_NVP(classifiers));
	}

};

