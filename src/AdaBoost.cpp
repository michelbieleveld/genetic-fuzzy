#include "AdaBoost.h"
#include "MPU.h"
#include <map>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cereal/archives/json.hpp>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <mpi.h>
#ifdef OPENMP_FOUND
#include <omp.h>
#endif

void BroadCastIndividual(Individual &individual, int rank, int root)
{
	std::string dss;
	if (rank == root)
	{
		std::stringstream ss;
		{
			cereal::JSONOutputArchive oarchive(ss);
			oarchive(cereal::make_nvp("Individual", individual));
		}
		dss = ss.str(); 
	}

	std::vector<char> buffer(dss.begin(), dss.end());
	int length = buffer.size();
	MPI_Bcast(&length,1,MPI_INT,root,MPI_COMM_WORLD);
	if (length > 0)
		buffer.resize(length+1);
	MPI_Bcast(&buffer[0],length,MPI_CHAR, root, MPI_COMM_WORLD);
				
	{
		std::stringstream st(std::string(buffer.begin(), buffer.end()));
		cereal::JSONInputArchive iarchive(st);
		iarchive(individual);
	}
}

AdaBoost::AdaBoost()
{
}

AdaBoost::AdaBoost(const std::vector<float>& _alpha, const std::vector<Individual>& _classifiers) :
	alpha(_alpha),
	classifiers(_classifiers)
{

}

AdaBoost::~AdaBoost()
{
}

void AdaBoost::save(std::string filepath)
{
	std::ofstream myfile(filepath, std::ios::out | std::ios::trunc | std::ios::binary);
	if (myfile.is_open())
	{
		cereal::JSONOutputArchive oarchive(myfile);
		oarchive(cereal::make_nvp("AdaBoost", *this)); 
	}
}

AdaBoost AdaBoost::load(std::string filepath)
{

	std::ifstream myfile(filepath, std::ios::in | std::ios::binary);
	AdaBoost bs;
	if (myfile.is_open())
	{
		cereal::JSONInputArchive iarchive(myfile);
		iarchive(bs);
	}
	return bs;
}

AdaBoost AdaBoost::run(const DataSource& datasource, const DataSource& datasourceValidate, unsigned int boosts, unsigned int generations, unsigned int populationSize, unsigned int instructions, unsigned int dependent, unsigned int shards)
{
	const int root = 0;
	
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	// divide the nodes to belong to just one deme
	int color = world_rank / (world_size / shards);
	MPI_Comm deme_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &deme_comm);
	int deme_rank, deme_size;
	MPI_Comm_rank(deme_comm, &deme_rank);
	MPI_Comm_size(deme_comm, &deme_size);


	const int individualsPerNode = populationSize / deme_size;

	//std::cout << "Populations per deme: " << individualsPerNode;
	//std::cout << "World rank: " << world_rank << " of: " << world_size << " deme_rank: " << deme_rank << " of: " << deme_size << " color: " << color << std::endl;


	DataSource ds = datasource;

	unsigned int maxrows = ds.rows.size();
	for (unsigned int r = 0; r < maxrows; r++)
	{
		ds.rows[r].times = 1;
		ds.rows[r].weight = 1.0f / (float)maxrows;
	}

	std::vector<Individual> classifiers;
	std::vector<float> alpha;

	for (unsigned int b = 0; b < boosts; b++)
	{
		//std::vector<std::tuple<Individual, float>> bestIndividual;
		Individual bestIndividual;
		
		std::vector<Individual> shard = Individual::initializeGeneration_datasource(individualsPerNode, instructions, datasource);
		
		for (unsigned int g = 0; g < generations; g++)
		{
			std::vector<std::tuple<unsigned int, float>> results;
			#pragma omp parallel num_threads(4)
			{
				std::vector<std::tuple<unsigned int, float>> results_private;
				#pragma omp for nowait
				for (int n=0;n<shard.size();n++)
				{
					std::vector<float> r = MPU::run(shard[n], ds);
					float e = 0;
					for (unsigned int ri = 0; ri < ds.rows.size(); ri++)
						if ((ds.rows[ri].data[dependent] > 0.5) != (r[ri] > 0.5))
							e += ds.rows[ri].weight;
					results_private.push_back(std::tuple<unsigned int, float>(n, e));
				};
				#pragma omp critical
    			results.insert(results.end(), results_private.begin(), results_private.end());
			}	
			
			std::sort(results.begin(), results.end(), [](const std::tuple<unsigned int, float> &left, const std::tuple<unsigned int, float> &right) {
				return std::get<1>(left) < std::get<1>(right);
			});
		
			if (g != generations - 1)
			{
				shard = Individual::nextGeneration(datasource, shard, results, g, deme_comm);
			}
			else
			{
				
				int id = std::get<0>(results[0]);
				float f = std::get<1>(results[0]);
				Individual bestOfShard = shard[id];
				
				if (datasourceValidate.rows.size() > 0)
				{
					// use validation data to check which model is the best to select
					std::vector<float> r = MPU::run(bestOfShard, datasourceValidate);
					f = 0.0f;
					for (unsigned int ri = 0; ri < datasourceValidate.rows.size(); ri++)
						if ((datasourceValidate.rows[ri].data[dependent] > 0.5) != (r[ri] > 0.5))
							f += datasourceValidate.rows[ri].weight;
				}

				Fitness localFitness, globalFitness;
				localFitness.rank = world_rank;
				localFitness.fitness = f;

				MPI_Allreduce(&localFitness, &globalFitness, 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
				//std::cout << "rank: " << world_rank << " local: " << localFitness.fitness << " global: " << globalFitness.fitness << std::endl; 
				bestIndividual = bestOfShard;
				BroadCastIndividual(bestIndividual, world_rank, globalFitness.rank);
			}
		}

		// update weights, everything is already for a specific dependent
		std::vector<float> results = MPU::run(bestIndividual, ds);
		float et = 0;
		for (unsigned int r = 0; r < ds.rows.size(); r++)
		{
			if ((ds.rows[r].data[dependent] > 0.5) != (results[r] > 0.5))
				et += ds.rows[r].weight;
		}

		if (et < 0.5)
		{
			float a = 0.5f * std::log((1 - et) / et);
			alpha.push_back(a);
			classifiers.push_back(bestIndividual);
			float adjust_correct = std::exp(-a);
			float adjust_incorrect = std::exp(a);
			for (unsigned int r = 0; r < ds.rows.size(); r++)
			{
				if ((ds.rows[r].data[dependent] > 0.5) != (results[r] > 0.5))
					ds.rows[r].weight *= adjust_incorrect;
				else
					ds.rows[r].weight *= adjust_correct;
			}
			ds.normalize();
		}
	}
	return AdaBoost(alpha,classifiers);
}

std::vector<float> AdaBoost::classify(const DataSource& datasource)
{
	std::vector<std::vector<float>> result;
	for (int n=0;n<classifiers.size();n++) {	
		std::vector<float> r = MPU::run(classifiers[n], datasource);
		for (unsigned int i = 0; i < r.size(); i++)
			r[i] = ((r[i] > 0.5f) ? 1.0f : -1.0f) * alpha[n];
		result.push_back(r);
	};

	std::vector<float> x;
	x.reserve(datasource.rows.size());

	for (unsigned int r = 0; r < datasource.rows.size(); r++)
	{
		float t = 0;
		for (unsigned int i = 0; i < classifiers.size(); i++)
			t += result[i][r];
		x.push_back((t>0.0f) ? 1.0f : 0.0f);
	}
	return x;

}

