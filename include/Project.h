#pragma once
#ifdef GDAL_FOUND 
#include <string>
#include "Classifier.h"

class Project
{
public:
	Project();
	~Project();

	static void MakeProjection(std::string fileNameIn, std::string fileNameOut, Classifier& classifier);

};
#endif
