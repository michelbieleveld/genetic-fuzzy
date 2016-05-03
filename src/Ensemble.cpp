#include "Ensemble.h"
#include <regex>
#include <iostream>
#include "AdaBoost.h"
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

Ensemble::Ensemble()
{
}


Ensemble::~Ensemble()
{
}

Ensemble::Ensemble(const std::string filepath)
{
	AddClassifiers(filepath);
}

std::vector<float> Ensemble::classify(const DataSource& datasource)
{
	std::vector<float> result;
	std::vector<std::vector<float>> res;
	for (unsigned int i = 0; i < classifiers.size(); i++)
		res.push_back(classifiers[i]->classify(datasource));
	for (unsigned int r = 0; r < datasource.rows.size(); r++)
	{
		float t = 0;
		for (unsigned int i = 0; i < res.size(); i++)
			t += res[i][r];
		result.push_back(t / res.size());
	}
	return result;
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
	size_t start_pos = str.find(from);
	if (start_pos == std::string::npos)
		return false;
	str.replace(start_pos, from.length(), to);
	return true;
}


void Ensemble::AddClassifiers(const std::string filepath)
{

	fs::path targetFile (filepath);
	fs::path targetDir( targetFile.parent_path() ); 

	for (fs::directory_entry p : fs::directory_iterator(targetDir))
	{
		if (p.path().extension() == ".model")
		{
    		std::cout << "Selected " << p.path().string() << std::endl;
			std::unique_ptr<AdaBoost> a = std::make_unique<AdaBoost>(AdaBoost::load(p.path().string()));
			classifiers.push_back(std::move(a));
   		}
	}
           
	/*
	
	size_t found = filepath.find_last_of("/\\");
	std::string pathdir;
	if (found != std::string::npos)
		pathdir = filepath.substr(0, found);
	else
		pathdir = ".";

	std::vector<std::string> selectedFiles;

	
	std::regex re(".*?[.]model$");


	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(pathdir.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			std::string s(ent->d_name);
			std::smatch match;
			if (std::regex_search(s, match, re) && match.size() > 0) {
				selectedFiles.push_back(pathdir + '\\' + s);
			}
		}
		closedir(dir);
	}

	for (unsigned int i = 0; i < selectedFiles.size(); i++)
	{
		std::cout << "Selected " << selectedFiles[i] << std::endl;
		std::unique_ptr<AdaBoost> p = std::make_unique<AdaBoost>(AdaBoost::load(selectedFiles[i]));
		classifiers.push_back(std::move(p));
	}
	*/
}