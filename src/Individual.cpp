#include "Individual.h"
#include "MPU.h"
#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <set>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cereal/archives/json.hpp>
#include <string>
#ifdef OPENMP_FOUND
#include <omp.h>
#endif

Individual::Individual(std::vector<Instruction> _instructions, std::vector<float> _ram)
{
	instructions = _instructions;
	ram = _ram;
}

Individual::Individual()
{

}

Individual::~Individual()
{
}

void Individual::validize(RandomGenerator& rnd, std::vector<Instruction>& program)
{
	std::vector<unsigned int> stack;

	for (unsigned int i = 0; i < program.size(); i++)
	{
		if (std::find(MPU::opcodes_push.begin(), MPU::opcodes_push.end(), program[i].instruction) != MPU::opcodes_push.end())
			stack.push_back(i);
		else if (std::find(MPU::opcodes_pop.begin(), MPU::opcodes_pop.end(), program[i].instruction) != MPU::opcodes_pop.end())
		{
			if (stack.size() > 0)
				stack.pop_back();
			else
			{
				program[i].instruction = rnd.getOpcodePush();
				stack.push_back(i);
			}
		}
	}
	for (unsigned int i = 1; i < stack.size(); i += 2)
		program[stack[i]].instruction = rnd.getOpcodePop();
	if (stack.size() % 2 == 1)
		program[stack[stack.size() - 1]].instruction = rnd.getOpcodeOther();
}

std::vector<Individual> Individual::initializeGeneration(RandomGenerator& rnd, int size, int instructions, float conf_min, float conf_max)
{
	std::vector<Individual> gen;
	gen.reserve(size);
	for (int i = 0; i < size; i++)
	{
		Individual indv = initialize(rnd, instructions, conf_min, conf_max);
		gen.push_back(indv);
	}
	return gen;
}


Individual Individual::initialize(RandomGenerator& rnd, int instructions, float conf_min, float conf_max)
{
	/*
	qnorm(0.7) 0.5244005
	qnorm(0.8) 0.8416212
	qnorm(0.9) 1.28155
	qnorm(0.975) 1.959964
	*/


	std::vector<Instruction> program;
	program.reserve(instructions);

	std::vector<float> ram;
	ram.reserve(instructions);
	
	for (int i = 0; i < instructions; i++)
		ram.push_back(rnd.getRam());

	for (int i = 0; i < instructions; i++)
	{
		float c[4];
		for (unsigned int r = 0; r < 4; r++)
			c[r] = rnd.getRam();
		std::sort(std::begin(c), std::end(c));
		int o = rnd.getOpcode();
		int p = rnd.getOperand();
		Instruction inst = { o, p, { c[0], c[1], c[2], c[3] } };
		program.push_back(inst);
	}
	
	validize(rnd, program);
	
	Individual indv(program, ram);
	return indv;
}


std::vector<Individual> Individual::initializeGeneration_datasource(int size, int instructions, const DataSource& datasource, float conf_min, float conf_max)
{
	std::vector<Individual> gen;
	

	#pragma omp parallel
	{
		std::vector<Individual> gen_private;
		RandomGenerator rnd(datasource.nindependents);

		#pragma omp for nowait
		for (unsigned int i = 0; i < size; i++)
		{
			Individual indv = Individual::initialize_datasource(rnd, instructions, datasource, conf_min, conf_max);
			gen_private.push_back(indv);

		}

		#pragma omp critical
		gen.insert(gen.end(), gen_private.begin(), gen_private.end());
	}
	return gen;
}

Individual Individual::initialize_datasource(RandomGenerator& rnd, int instructions, const DataSource& datasource, float conf_min, float conf_max)
{
	/*
	qnorm(0.7) 0.5244005
	qnorm(0.8) 0.8416212
	qnorm(0.9) 1.28155
	qnorm(0.975) 1.959964
	*/

	std::vector<Instruction> program;
	program.reserve(instructions);

	std::vector<float> ram;
	ram.reserve(instructions);

	for (int i = 0; i < instructions; i++)
		ram.push_back(rnd.getRam());

	int dependents = datasource.ndependents;
	int independents = datasource.nindependents;

	std::vector<std::tuple<float, float>> stats = datasource.stats;

	for (int i = 0; i < instructions; i++)
	{
		int o = rnd.getOpcode();
		int p = rnd.getOpcode();
		
		std::array<float, 2> confLevels;
		confLevels[0] = rnd.getRam() * (conf_max - conf_min) + conf_min;
		confLevels[1] = rnd.getRam() * (conf_max - conf_min) + conf_min;
		std::sort(std::begin(confLevels), std::end(confLevels));

		float m = std::get<0>(stats[dependents + p]);
		float sd = std::get<1>(stats[dependents + p]);


		// adding constants so i can remove  the ifs not not make values [1]-[0]=0
		std::array<float,4> c = {std::max<float>(m - sd*confLevels[1], 0),
			std::max<float>(m - sd*confLevels[0], 0) + 0.0000001f,
			std::min<float>(m + sd*confLevels[0], 1) + 0.0000001f,
			std::min<float>(m + sd*confLevels[1], 1) + 0.0000002f };

		Instruction instr{ o, p, { c[0], c[1], c[2], c[3] } };

		program.push_back(instr);
	}

	validize(rnd, program);
	Individual indv(program, ram);
	return indv;
}

void us_isect(std::vector<unsigned int> &outvector,
	const std::unordered_set<unsigned int> &in1,
	const std::unordered_set<unsigned int> &in2)
{
	if (in2.size() < in1.size()) {
		us_isect(outvector, in2, in1);
		return;
	}

	std::unordered_set<unsigned int> out;
	outvector.clear();

	for (std::unordered_set<unsigned int>::const_iterator it = in1.begin(); it != in1.end(); it++)
	{
		if (in2.find(*it) != in2.end())
			out.insert(*it);
	}
	std::copy(out.begin(), out.end(), std::back_inserter(outvector));
}

std::array<Individual,2> Individual::crossover(RandomGenerator& rnd, const Individual& a, const Individual& b)
{
	std::vector<unsigned int> popa;
	std::vector<unsigned int> pusha;
	std::vector<unsigned int> popb;
	std::vector<unsigned int> pushb;

	popa.reserve(a.instructions.size());
	pusha.reserve(a.instructions.size());
	popb.reserve(b.instructions.size());
	pushb.reserve(b.instructions.size());

	for (unsigned int i = 0; i < a.instructions.size(); i++)
	{
		int instr = a.instructions[i].instruction;
		if (std::find(MPU::opcodes_push.begin(), MPU::opcodes_push.end(), instr) != MPU::opcodes_push.end())
			pusha.push_back(i);
		else if (std::find(MPU::opcodes_pop.begin(), MPU::opcodes_pop.end(), instr) != MPU::opcodes_pop.end())
			popa.push_back(i);
	}

	for (unsigned int i = 0; i < b.instructions.size(); i++)
	{
		int instr = b.instructions[i].instruction;
		if (std::find(MPU::opcodes_push.begin(), MPU::opcodes_push.end(), instr) != MPU::opcodes_push.end())
			pushb.push_back(i);
		else if (std::find(MPU::opcodes_pop.begin(), MPU::opcodes_pop.end(), instr) != MPU::opcodes_pop.end())
			popb.push_back(i);
	}

	std::vector<unsigned int> cpopa(pusha.size());
	std::vector<unsigned int> cpopb(pushb.size());

	for (unsigned int i = 0; i < pusha.size(); i++)
	{
		for (unsigned int j = 0; j < popa.size(); j++)
		{
			if (popa[j] > pusha[i])
			{
				cpopa[i] = popa[j];
				break;
			}
		}
	}

	for (unsigned int i = 0; i < pushb.size(); i++)
	{
		for (unsigned int j = 0; j < popb.size(); j++)
		{
			if (popb[j] > pushb[i])
			{
				cpopb[i] = popb[j];
				break;
			}
		}
	}

	
	std::vector<unsigned int> counta(pusha.size());
	for (unsigned int i = 0; i < pusha.size(); i++)
	{
		unsigned int count = 0;
		for (unsigned int j = i; j < pusha.size(); j++)
		{
			if (pusha[j] > cpopa[i]) break;
			count++;
		}
		counta[i] = count;
	}

	std::vector<unsigned int> countb(pushb.size());
	for (unsigned int i = 0; i < pushb.size(); i++)
	{
		unsigned int count = 0;
		for (unsigned int j = i; j < pushb.size(); j++)
		{
			if (pushb[j] > cpopb[i]) break;
			count++;
		}
		countb[i] = count;
	}


	std::unordered_set<unsigned int> scounta;
	scounta.insert(counta.begin(), counta.end());

	std::unordered_set<unsigned int> scountb;
	scountb.insert(countb.begin(), countb.end());
	std::vector<unsigned int> u;
	us_isect(u, scounta, scountb);
	
	int ia = 0;
	int ib = 0;

	if (u.size() > 0)
	{
		int s = u[rnd.Next(u.size() - 1)];

		std::vector<unsigned int> x;
		for (unsigned int i = 0; i < counta.size(); i++)
			if (counta[i] == s)
				x.push_back(i);

		std::vector<unsigned int> y;
		for (unsigned int i = 0; i < countb.size(); i++)
			if (countb[i] == s)
				y.push_back(i);

		ia = x[rnd.Next(x.size() - 1)];
		ib = y[rnd.Next(y.size() - 1)];
	}
	
	std::vector<Instruction> na = a.instructions;
	std::vector<Instruction> nb = b.instructions;

	std::vector<Instruction> nna;
	std::vector<float> nnra;

	for (unsigned int i = 0; i < pusha[ia]; i++)
	{
		nna.push_back(na[i]); // i can be higher than number of elements in na
		nnra.push_back(a.ram[i]);
	}
	for (unsigned int i = pushb[ib]; i < cpopb[ib] + 1; i++)
	{
		nna.push_back(nb[i]);
		nnra.push_back(b.ram[i]);
	}
	for (unsigned int i = cpopa[ia] + 1; i < na.size(); i++)
	{
		nna.push_back(na[i]);
		nnra.push_back(a.ram[i]);
	}

	std::vector<Instruction> nnb;
	std::vector<float> nnrb;

	for (unsigned int i = 0; i < pushb[ib]; i++)
	{
		nnb.push_back(nb[i]);
		nnrb.push_back(b.ram[i]);
	}
	for (unsigned int i = pusha[ia]; i < cpopa[ia] + 1; i++)
	{
		nnb.push_back(na[i]);
		nnrb.push_back(a.ram[i]);
	}
	for (unsigned int i = cpopb[ib] + 1; i < nb.size(); i++)
	{
		nnb.push_back(nb[i]);
		nnrb.push_back(b.ram[i]);
	}

	std::array<Individual, 2> children = { Individual(nna, nnra), Individual(nnb, nnrb) };
	return children;
}

Individual Individual::mutate(RandomGenerator& rnd, const Individual& a, float max_mutations)
{
	Individual mutated = a;

	unsigned int mutations = rnd.Next((int)(max_mutations * (mutated.instructions.size()-1)));
	for (unsigned int m = 0; m < mutations; m++)
	{
		mutated.instructions[rnd.Next(mutated.instructions.size()-1)].instruction = rnd.getOpcode();
		mutated.instructions[rnd.Next(mutated.instructions.size()-1)].operand = rnd.getOperand();
		mutated.ram[rnd.Next(mutated.ram.size()-1)] = rnd.getRam();
		unsigned int index = rnd.Next(mutated.instructions.size()-1);
		for (unsigned int i = 0; i < 4; i++)
		{
			float n = rnd.getRam();
			/*
			float f = mutated.instructions[index].constants[i];
			float f_min = f - 0.1f;
			float f_max = f + 0.1f;
			float n = rnd.getRam() * (f_max - f_min) + f_min;
			if (n < 0.0f) n = 0 - n;
			if (n > 1.0f) n = n - (n - 1);
			*/
			mutated.instructions[index].constants[i] = n;
		}
		std::sort(std::begin(mutated.instructions[index].constants), std::end(mutated.instructions[index].constants));
	}
	validize(rnd, mutated.instructions);
	return mutated;
}


template<class T>
unsigned int binary_find_index(const std::vector<T>& v, T val)
{
	typename std::vector<T>::const_iterator i = std::lower_bound(v.begin(), v.end(), val);
	if (i != v.end() && !(*i < val))
		return i - v.begin();
	else
		return v.size() - 1;
}

std::vector<Individual> Individual::nextGeneration(const DataSource& datasource, const std::vector<Individual>& gen, std::vector<std::tuple<unsigned int, float>> results, int generation, MPI_Comm& mpi_comm, float fraction_crossover, float fraction_mutate, unsigned int elite, int dependent, float keep, unsigned int tournament_k, float tournament_p)
{
	unsigned int toKeep = static_cast<unsigned int>((results.size() * keep));

	std::vector<float> cfs;
	cfs.reserve(toKeep);

	cfs.push_back(std::get<1>(results[0]));
	for (unsigned int i = 1; i < toKeep; i++)
		cfs.push_back(cfs[i - 1] + std::get<1>(results[i]));

	// calculate tournament wheel
	std::vector<float> fs;
	fs.push_back(tournament_p);
	for (unsigned int i = 1; i < tournament_k; i++)
		fs.push_back(fs[i - 1] * (1 - tournament_p));
	for (unsigned int i = 1; i < fs.size(); i++)
		fs[i] += fs[i - 1];

	std::vector<unsigned int> picked;
	picked.reserve(gen.size());


	RandomGenerator rndt(datasource.nindependents);
	for (unsigned int j = 0; j < gen.size(); j++)
	{
		// take k elements with weighting of cfs
		std::vector<unsigned int> t;
		for (unsigned int jj = 0; jj < tournament_k; jj++)
		{
			float fi = rndt.getRam() * cfs[cfs.size() - 1];
			int i = binary_find_index<float>(cfs, fi);
			t.push_back(i);
		}
		std::sort(t.begin(), t.end());

		//for (unsigned int jj = 0; jj < tournament_k; jj++)
		//	t[jj] = std::get<0>(results[t[jj]]);

		// pick the one according to roulete wheel
		float rv = rndt.getRam() * fs[fs.size()-1];
		int i = binary_find_index<float>(fs, rv);
		picked.push_back(std::get<0>(results[t[i]]));
	}

	// now picked contains a list of individual indices selected for whatever
	unsigned int eliteMax = elite;
	float sum_fractions = fraction_crossover + fraction_mutate;
	unsigned int crossMax = (unsigned int)(gen.size() * fraction_crossover / (2.0f * sum_fractions));
	unsigned int mutateMax = gen.size() - eliteMax - 2 * crossMax;

	unsigned int index = 0;
	std::vector<Individual> ngen;
	ngen.reserve(gen.size());
	
	#pragma omp parallel
	{
		std::vector<Individual> ngen_private;
		RandomGenerator rnd(datasource.nindependents);

		#pragma omp for nowait
		for (unsigned int i = 0; i < crossMax; i++)
		{
			std::array<Individual, 2> c = crossover(rnd, gen[picked[index]], gen[picked[index+1]]);
			index += 2;
			ngen_private.push_back(c[0]);
			ngen_private.push_back(c[1]);
		}

		#pragma omp for nowait
		for (unsigned int i = 0; i < mutateMax; i++)
			ngen_private.push_back(mutate(rnd, gen[picked[index++]]));

		#pragma omp critical
		ngen.insert(ngen.end(), ngen_private.begin(), ngen_private.end());
	}

	for (unsigned int i = 0; i < eliteMax; i++)
		ngen.push_back(gen[std::get<0>(results[i])]);

	
	const int migrationMax = 3;
	for (unsigned int i=0; i< migrationMax; i++)
	{
		ngen[i] = migration(gen[std::get<0>(results[i])], mpi_comm);
	}
	
	return ngen;
}

Individual Individual::migration(const Individual& individual, MPI_Comm& mpi_comm)
{

	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);

	std::stringstream ss;
	{
		cereal::JSONOutputArchive oarchive(ss);
		oarchive(cereal::make_nvp("Individual", individual));
	}
	std::string dss = ss.str(); 
	std::vector<char> sendbuffer(dss.begin(), dss.end());
	// as a easy hack allocate twice the space to receive the data. Considering that all individuals are of the same length this should not cause problems
	std::vector<char> receivebuffer;
	receivebuffer.resize(sendbuffer.size() * 2);

	Individual received;
	MPI_Status status;
	MPI_Sendrecv(&sendbuffer[0],sendbuffer.size(),MPI_CHAR, (rank + 1) % size, 0, &receivebuffer[0], receivebuffer.size(),MPI_CHAR,(rank-1) % size,0, mpi_comm,&status);

	{
		std::stringstream st(std::string(receivebuffer.begin(), receivebuffer.end()));
		cereal::JSONInputArchive iarchive(st);
		iarchive(received);
	}
	return received;
}

Individual Individual::simplify(const Individual& individual)
{
	std::vector<Instruction> program;
	std::vector<float> ram;
	std::vector<unsigned int> stack;
	std::set<unsigned int> used;
	stack.push_back(individual.instructions.size() - 1);
	int count = 0;
	while (!stack.empty())
	{
		int index = stack.back();
		stack.pop_back();
		if (index < 0) break;
		used.insert(index);
		int instr = individual.instructions[index].instruction;
		switch (instr)
		{
		case 0x05: // NOT
		case 0x00: // PUSH
			stack.push_back(index - 1);
			break;
		case 0x03: // LDA
		case 0x04: // TRAPMF
			break;
		case 0x01: // AND
		case 0x02: // OR
		case 0x06: // IF
			count = 1;
			stack.push_back(index - 1);
			for (int i = index - 1; i > -1; i--)
			{
				int search = individual.instructions[i].instruction;
				if (std::find(MPU::opcodes_pop.begin(), MPU::opcodes_pop.end(), search) != MPU::opcodes_pop.end())
					count++;
				else if (std::find(MPU::opcodes_push.begin(), MPU::opcodes_push.end(), search) != MPU::opcodes_push.end())
				{
					count--;
					if (count == 0)
					{
						stack.push_back(i);
						break;
					}
				}
			}
		}
	}
	for (const auto& elem : used)
	{
		program.push_back(individual.instructions[elem]);
		ram.push_back(individual.ram[elem]);
	}


	// remove not not and not constant
	bool previousNot = false;
	for (int i = program.size() - 1; i > -1; i--)
	{
		int currentInstruction = program[i].instruction;
		if (currentInstruction == 0x05) //NOT
		{
			if (previousNot)
			{
				previousNot = false;
				program.erase(program.begin() + i, program.begin() + i + 2);
			}
			else
				previousNot = true;
		}
		else
			previousNot = false;
	}


	return Individual(program, ram);
}


std::ostream & operator<<(std::ostream & str, Individual const & v) {
	str << Individual::ToPrint(v);
	return str;
}


std::string Individual::ToPrint(const Individual& individual, int position)
{
	int count = 1;
	if (position == -2)
		position = individual.instructions.size() - 1;
	if (position == -1)
		return("C(0)");
	switch (individual.instructions[position].instruction)
	{
	case 0x00: //PUSH
		return ToPrint(individual, position - 1);
		break;
	case 0x01: // AND
		for (int i = position - 1; i > -1; i--)
		{
			if (individual.instructions[i].instruction == 0x01 || individual.instructions[i].instruction == 0x02)
				count++;
			if (individual.instructions[i].instruction == 00)
			{
				count--;
				if (count == 0)
					return("AND(" + ToPrint(individual, position - 1) + "," + ToPrint(individual, i - 1) + ")");
			}
		}
		break;
	case 0x02: // OR
		for (int i = position - 1; i > -1; i--)
		{
			if (individual.instructions[i].instruction == 0x01 || individual.instructions[i].instruction == 0x02)
				count++;
			if (individual.instructions[i].instruction == 00)
			{
				count--;
				if (count == 0)
					return("OR(" + ToPrint(individual, position - 1) + "," + ToPrint(individual, i - 1) + ")");
			}
		}
		break;
	case 0x03: // LDA
		return ("C(" + std::to_string(individual.ram[position]) + ")");
		break;
	case 0x04: // TRAPMF
		return ("TRAPMF(" + std::to_string(individual.instructions[position].operand) + "," +
			std::to_string(individual.instructions[position].constants[0]) + "," +
			std::to_string(individual.instructions[position].constants[1]) + "," +
			std::to_string(individual.instructions[position].constants[2]) + "," +
			std::to_string(individual.instructions[position].constants[3]) + ")");
		break;
	case 0x05: // NOT
		return ("!(" + ToPrint(individual, position - 1) + ")");
		break;
	case 0x06: // IF
		for (int i = position - 1; i > -1; i--)
		{
			if (individual.instructions[i].instruction == 0x01 || individual.instructions[i].instruction == 0x02)
				count++;
			if (individual.instructions[i].instruction == 00)
			{
				count--;
				if (count == 0)
					return("IF(" + ToPrint(individual, position - 1) + ">" + std::to_string(individual.ram[position]) + "," + ToPrint(individual, i - 1) + ")");
			}
		}
		break;

	default:
		std::cout << "ERROR" << std::endl;

	}
	return "";
}