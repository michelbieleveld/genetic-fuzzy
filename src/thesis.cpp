#include <string>
#include <regex>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include "DataSource.h"
#include "RandomGenerator.h"
#include "Individual.h"
#include "MPU.h"
#include "AdaBoost.h"
#include <set>
#include "Utilities.h"
#include "Ensemble.h"
#include <cereal/cereal.hpp>
#include <fstream>
#include <cereal/archives/json.hpp>
#include <experimental/filesystem>
#include <mpi.h>
#include <thesis.h> 

#ifdef GDAL_FOUND
#include "Project.h"
#endif

namespace fs = std::experimental::filesystem;

//INITIALIZE_EASYLOGGINGPP

void evaluate(const DataSource& datasource, Classifier& classifier, const std::string description ="", int dependent = 0)
{
	std::vector<float> result = classifier.classify(datasource);
	std::array<unsigned int, 4> confusion = Utilities::ConfusionMatrix(datasource, result, dependent);
	float tss = Utilities::TSS(confusion);
	float auc = Utilities::AUC(result, datasource);
	//LOG(INFO) << description;
	//LOG(INFO) << "tp: " << confusion[0] << " fp: " << confusion[1] << " fn: " << confusion[2] << " tn: " << confusion[3];
	//LOG(INFO) << "TSS: " << tss;
	//LOG(INFO) << "AUC: " << auc;
}


void predictToCsv(const DataSource& datasource, Classifier& classifier, std::string filename, std::string description="", int dependent=0)
{
	std::vector<float> result = classifier.classify(datasource);
	
	std::ofstream csv;
	csv.open(filename);
	if (csv.is_open())
	{
		const int n = datasource.fields.size();
		for (int i = 0; i < n; i++)
			csv << '"' << datasource.fields[i] << '"' << ((i != n - 1) ? "," : "\r\n");
		
		const int s = result.size();
		for (int i = 0; i < s; i++)
		{
			std::tuple<float,float> loc = datasource.rows[i].loc;
			csv << std::get<0>(loc) << ',' << std::get<1>(loc) << ',' << result[i] << "\r\n";
		}
	}
	csv.close();
}

void projectToCsv(const DataSource& datasource, Classifier& classifier, std::string source, std::string destination, int dependent = 0)
{
	std::vector<float> result = classifier.classify(datasource);

	std::ofstream csv;
	csv.open(destination);
	if (csv.is_open())
	{
		const size_t n = result.size();
		csv << "x,y,predict" << std::endl;
		for (size_t i = 0; i < n; i++)
		{	
			std::tuple<float, float> loc = datasource.rows[i].loc;
			csv << std::get<0>(loc) << ',' << std::get<1>(loc) << ',' << result[i] << std::endl;
		}
	}
	csv.close();
}

void BroadCastDataSource(DataSource &datasource,  int rank, int root)
{
	std::string dss;
	if (rank == root)
	{
		std::stringstream ss;
		{
			cereal::JSONOutputArchive oarchive(ss);
			oarchive(cereal::make_nvp("DataSource", datasource));
		}
		dss = ss.str(); 
	}

	std::vector<char> buffer(dss.begin(), dss.end());
	int length = buffer.size();
	MPI_Bcast(&length,1,MPI_INT,root,MPI_COMM_WORLD);
	if (length > 0)
		buffer.resize(length+1);
	MPI_Bcast(&buffer[0],length,MPI_CHAR, root, MPI_COMM_WORLD);
				
	//DataSource dtest;
	{
		std::stringstream st(std::string(buffer.begin(), buffer.end()));
		cereal::JSONInputArchive iarchive(st);
	//	iarchive(dtest);
		iarchive(datasource);
	}
}




int main(int argc, char** argv)
{

	int  numtasks, rank, len, rc; 
	rc = MPI_Init(&argc,&argv);
   	if (rc != MPI_SUCCESS) {
    	printf ("Error starting MPI program. Terminating.\n");
    	MPI_Abort(MPI_COMM_WORLD, rc);
    }

    const int root = 0;
   	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  	std::cout << "I am process " << rank << " of " << numtasks << "." << std::endl;

  	auto start = std::chrono::steady_clock::now();

	try {
		
		TCLAP::CmdLine cmd("Fuzzy Model Calculator", ' ', "0.9");
		TCLAP::ValueArg<std::string> samplesArg("s", "samples", "Samples file", false, "", "string", cmd);
		TCLAP::ValueArg<std::string> backgroundsamplesArg("", "background-samples", "Background Samples file", false, "", "string", cmd);
		TCLAP::MultiArg<std::string> validationArg("v", "validation", "Validation file(s)", false, "string", cmd);
		TCLAP::ValueArg<std::string> outFileArg("o", "output", "Filename to write model to", false, "", "string", cmd);
		TCLAP::ValueArg<std::string> inFileArg("l", "load", "Filename of model to load", false, "", "string", cmd);
		TCLAP::MultiArg<std::string> projectInCsvArg("", "project-in-csv", "File name(s) of CSV in", false, "string", cmd);
		TCLAP::ValueArg<std::string> projectOutCsvArg("", "project-out-csv", "Filename of CSV out", false, "", "string", cmd);
		
		#ifdef GDAL_FOUND
		TCLAP::ValueArg<std::string> projectInArg("", "project-in", "File name of geotiff in", false, "", "string", cmd);
		TCLAP::ValueArg<std::string> projectOutArg("", "project-out", "File name of geotiff out", false, "", "string", cmd);
		std::string projectIn = projectInArg.getValue();
		std::string projectOut = projectOutArg.getValue();
		#endif

		TCLAP::ValueArg<std::string> outputDirectoryArg("", "output-directory", "?", false, "", "string", cmd);
		TCLAP::ValueArg<unsigned int> runArg("r", "runs", "How many runs to do", false, 1, "int", cmd);
		TCLAP::ValueArg<unsigned int> boostArg("b", "boost", "How many boosts to do", false, 50, "int", cmd);
		TCLAP::ValueArg<unsigned int> shardArg("", "shard", "How many shards to use", false, 5, "int", cmd);
		TCLAP::ValueArg<unsigned int> populationArg("p", "population", "Population size", false, 1000, "int", cmd);
		TCLAP::ValueArg<unsigned int> instructionArg("i", "instruction", "Instructions per individual", false, 64, "int", cmd);
		TCLAP::ValueArg<unsigned int> generationArg("g", "generation", "Generations", false, 100, "int", cmd);
		TCLAP::ValueArg<unsigned int> dependentArg("d", "dependent", "The dependent to optimize for", false, 0, "int", cmd);
		TCLAP::SwitchArg printArg("", "print", "Print the rules of the model", cmd, false);
	

		cmd.parse(argc, argv);

		std::string samplesFile = samplesArg.getValue();
		std::string backgroundSamplesFile = backgroundsamplesArg.getValue();
		std::string outputDirectory = outputDirectoryArg.getValue();
		std::string outFile = outFileArg.getValue();
		std::string inFile = inFileArg.getValue();
		std::vector<std::string> projectInCsvs = projectInCsvArg.getValue();
		std::string projectOutCsv = projectOutCsvArg.getValue();
		std::vector<std::string> validationFiles = validationArg.getValue();
		unsigned int runs = runArg.getValue();
		unsigned int shard = shardArg.getValue();
		unsigned int boost = boostArg.getValue();
		unsigned int population = populationArg.getValue();
		unsigned int generation = generationArg.getValue();
		unsigned int dependent = dependentArg.getValue();
		unsigned int instruction = instructionArg.getValue();
		bool print = printArg.getValue();


		
	
		std::unique_ptr<Classifier> pmodel;
		if (!inFile.empty())
		{
			if (rank == root)
			{
			 	int count = 0;
				try{

					fs::path targetFile (inFile);
					fs::path targetDir( targetFile.parent_path() ); 
					for (fs::directory_entry p : fs::directory_iterator(targetDir))
					{
						if (p.path().extension() == ".model")
							count++;
					}
				    std::cout << count << std::endl;
				}
				catch( fs::filesystem_error const & e){}
	           	if (count < 2)
					pmodel = std::make_unique<AdaBoost>(AdaBoost::load(inFile));
				else
					pmodel = std::make_unique<Ensemble>(inFile);
			}
		}
		else if (!samplesFile.empty())
		{
			std::string saveFile;
			for (unsigned int r = 0; r < runs; r++)
			{
				DataSource datasource, datasourceValidate;
				if (rank == root) {
					if (!fs::exists(status(fs::path(samplesFile))))
						throw TCLAP::ArgException("samples file does not exist");
					datasource = DataSource::fromFileName(samplesFile);
					if (!backgroundSamplesFile.empty())
					{
						if (!fs::exists(status(fs::path(backgroundSamplesFile))))
							throw TCLAP::ArgException("background samples file does not exist");
						DataSource background = DataSource::fromFileName(backgroundSamplesFile);
						datasource.add(background);
					}
				}
				BroadCastDataSource(datasource, rank, root);

				if (validationFiles.size() > 0)
				{
					if (rank == root)
					{
						if (!fs::exists(status(fs::path(validationFiles[0]))))
							throw TCLAP::ArgException("validation file does not exist");
						datasourceValidate = DataSource::fromFileName(validationFiles[0]);
					}
					BroadCastDataSource(datasourceValidate, rank, root);
				}

				AdaBoost a = AdaBoost::run(datasource, datasourceValidate, boost,
					generation, population, instruction,
					dependent, shard);

				if (rank == 0) {
					if (print)
						for (unsigned int i = 0; i < a.classifiers.size(); i++)
							std::cout << a.alpha[i] << " * " << a.classifiers[i] << std::endl << std::endl;
					if (!outFile.empty())
					{
						saveFile = Utilities::GetTimeStampedFilename(outFile);
						a.save(saveFile);
					}
					
					if (!outputDirectory.empty())
					{

						fs::path targetDir(outputDirectory);
						std::string prepend = targetDir.filename().string();
						std::regex f("(_[^_]*?_[^_]*?)$");
						prepend = std::regex_replace(prepend, f, "", std::regex_constants::match_default);
						fs::path filename(prepend + ((r==0) ? "" : std::to_string(r)) + ".model");
						targetDir /= filename;
						saveFile = targetDir.string();
						a.save(saveFile);
					}
					pmodel = std::make_unique<AdaBoost>(a);
				}
			}
			if (runs > 1)
			{
				if (rank == root)
					pmodel = std::make_unique<Ensemble>(saveFile);
			}
		}

		if (pmodel && rank == root)
		{
			for (std::string projectInCsv : projectInCsvs)
			{
				DataSource proj = DataSource::fromFileName(projectInCsv);
				std::string dest = "";
				if (!outputDirectory.empty())
				{
					try{
						fs::path targetFile(projectInCsv); 
						fs::path targetDir(outputDirectory);
						std::string prepend = targetDir.filename().string();
						std::regex f("(_[^_]*?_[^_]*?)$");
						prepend = std::regex_replace(prepend, f, "", std::regex_constants::match_default);
						fs::path filename(prepend + "_" + targetFile.filename().string());
						targetDir /= filename;
						dest = targetDir.string();
						//dest = targetDir.append(filename).string();
						projectToCsv(proj, *pmodel,projectInCsv, dest);
					}
					catch( fs::filesystem_error const & e){}
				}
				else if (!projectOutCsv.empty())
				{
					dest = projectOutCsv;
					projectToCsv(proj, *pmodel, projectInCsv, dest);
				}

			}

			if (!samplesFile.empty())
			{
				DataSource test = DataSource::fromFileName(samplesFile);
				if (!projectOutCsv.empty())
					predictToCsv(test, *pmodel, projectOutCsv);

				evaluate(test, *pmodel, "TEST:");

			}
			for (unsigned int i = 0; i < validationFiles.size(); i++)
			{
				DataSource validation = DataSource::fromFileName(validationFiles[i]);
				evaluate(validation, *pmodel, "VALIDATION: " + validationFiles[i]);
			}
		}

		#ifdef GDAL_FOUND
		if (!projectIn.empty() && pmodel && rank == 0)
		{
			Project::MakeProjection(projectIn, projectOut, *pmodel);
		}
		#endif
	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		MPI_Abort(MPI_COMM_WORLD,0);
	}

	if (rank==root){
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
		std::cout << "Time passed (ms): " << duration.count() << std::endl;
	}

	MPI_Finalize();

	
}

