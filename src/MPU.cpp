#include "MPU.h"


const std::array<char, 7> MPU::opcodes = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06 };
const std::array<char, 1> MPU::opcodes_push = { 0x00 };
const std::array<char, 3> MPU::opcodes_pop = { 0x01, 0x02, 0x06 };
const std::array<char, 3> MPU::opcodes_other = { 0x03, 0x04, 0x05 };

MPU::MPU()
{
}


MPU::~MPU()
{
}


std::vector<std::vector<float>> MPU::run(const std::vector<Individual>& individuals,const DataSource& datasource)
{
	std::vector<std::vector<float>> results;
	results.reserve(individuals.size());
	for (unsigned int i = 0; i < individuals.size(); i++)
		results.push_back(run(individuals[i], datasource));
	return results;
}


std::vector<float> MPU::run(const Individual& individual, const DataSource& datasource)
{
	std::vector<float> result;
	result.reserve(datasource.rows.size());
	
	const int ndependent = datasource.ndependents;
	const int nindependent = datasource.nindependents;

	for (unsigned int r = 0; r < datasource.rows.size(); r++)
	{
		std::vector<float> stack;
		float a = 0.0f;
		float t = 0.0f;
		float x, y;
		//std::vector<float> data = datasource.rows[r].data;

		for (unsigned int pc = 0; pc < individual.instructions.size(); pc++)
		{
			switch (individual.instructions[pc].instruction)
			{
				case 0x00: // PUSH push
					stack.push_back(a);
					break;
				case 0x01: // AND pop
					t = stack.back();
					stack.pop_back();
					if (t > a) a = t;
					break;
				case 0x02: // OR pop
					t = stack.back();
					stack.pop_back();
					if (t < a) a = t;
					break;
				case 0x03: // LDA other
					a = individual.ram[pc];
					break;
				case 0x04: // TRAPMF other
					a = datasource.rows[r].data[ndependent + individual.instructions[pc].operand];
					x = individual.instructions[pc].constants[1] - individual.instructions[pc].constants[0];
					y = individual.instructions[pc].constants[3] - individual.instructions[pc].constants[2];
					x = (a - individual.instructions[pc].constants[0]) / x;
					y = (individual.instructions[pc].constants[3] - a) / y;
					a = 1.0f;
					if (x < a) a = x;
					if (y < a) a = y;
					if (a < 0) a = 0.0f;
					break;
				case 0x05: // NOT other
					a = 1 - a;
					break;
				case 0x06: // IF pop
					if (a > individual.ram[pc])
					{
						a = stack.back();
						stack.pop_back();
					}
			}
		}

		result.push_back(a);
	}
	return result;
}

