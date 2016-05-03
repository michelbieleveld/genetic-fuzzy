#pragma once
#include <array>
#include <vector>
#include "DataSource.h"
#include "RandomGenerator.h"
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <mpi.h>


struct Instruction
{
	int instruction;
	int operand;
	float constants[4];

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(instruction), CEREAL_NVP(operand), CEREAL_NVP(constants)); 
	}
};

class Individual
{
public:

	std::vector<Instruction> instructions;
	std::vector<float> ram;

	Individual();
	Individual(std::vector<Instruction> _instructions, std::vector<float> _ram);
	~Individual();

	static std::vector<Individual> initializeGeneration_datasource(int size, int instructions, const DataSource& datasource, float conf_min = 1.281552f, float conf_max = 1.959964f);
	static Individual initialize_datasource(RandomGenerator& rnd, int instructions, const DataSource& datasource, float conf_min = 1.281552f, float conf_max = 1.959964f);

	static std::vector<Individual> initializeGeneration(RandomGenerator& rnd, int size, int instructions, float conf_min = 1.281552f, float conf_max = 1.959964f);
	static Individual initialize(RandomGenerator& rnd, int instructions, float conf_min = 1.281552f, float conf_max = 1.959964f);

	static void validize(RandomGenerator& rnd, std::vector<Instruction>& program);
	static Individual migration(const Individual& individual, MPI_Comm& mpi_comm);
	static std::array<Individual, 2> crossover(RandomGenerator& rnd, const Individual& a, const Individual& b);
	static Individual mutate(RandomGenerator& rnd, const Individual& a, float max_mutations = 0.5f);
	static std::vector<Individual> nextGeneration(const DataSource& datasource, const std::vector<Individual>& gen, std::vector<std::tuple<unsigned int, float>> results, int generation, MPI_Comm& mpi_comm, float fraction_crossover = 0.0f, float fraction_mutate = 1.0f, unsigned int elite = 3, int dependent = 0, float keep = 0.5f, unsigned int tournament_k = 3, float tournament_p = 0.5f);


	static Individual simplify(const Individual& individual);
	static std::string ToPrint(const Individual& individual, int position = -2);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(instructions), CEREAL_NVP(ram)); 
	}
};

template<class T>
unsigned int binary_find_index(const std::vector<T>& v, T val);

namespace std {
	template <>
	struct hash<Instruction>
	{
		size_t operator()(const Instruction& v) const
		{
			size_t h = 0;
			std::hash<int> hasher;
			std::hash<float> hasher2;
			h ^= hasher(v.instruction) + 0x9e3779b9 + (h << 6) + (h >> 2);
			h ^= hasher(v.operand) + 0x9e3779b9 + (h << 6) + (h >> 2);
			h ^= hasher2(v.constants[0]) + 0x9e3779b9 + (h << 6) + (h >> 2);
			h ^= hasher2(v.constants[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
			h ^= hasher2(v.constants[2]) + 0x9e3779b9 + (h << 6) + (h >> 2);
			h ^= hasher2(v.constants[3]) + 0x9e3779b9 + (h << 6) + (h >> 2);
			return h;
		}
	};

	template<>
	struct hash < vector<Instruction> >
	{
		size_t operator()(const vector<Instruction>& v) const
		{
			size_t h = 0;
			std::hash<Instruction> hasher;
			for (const auto& e : v) {
				h ^= hasher(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
			}
			return h;
		}
	};
}

std::ostream & operator<<(std::ostream & str, Individual const & v);