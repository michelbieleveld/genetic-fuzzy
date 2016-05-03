#pragma once
#include <array>
#include <vector>
#include "Individual.h"
#include "DataSource.h"


class MPU
{
public:
	MPU();
	~MPU();

	static const std::array<char, 7> opcodes;
	static const std::array<char, 1> opcodes_push;
	static const std::array<char, 3> opcodes_pop;
	static const std::array<char, 3> opcodes_other;

	static std::vector<std::vector<float>> run(const std::vector<Individual>& individuals, const DataSource& datasource);
	static std::vector<float> run(const Individual& individual,const DataSource& datasource);
};

