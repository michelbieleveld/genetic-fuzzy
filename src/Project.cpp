#ifdef(GDAL_FOUND)
#include "Project.h"
#include <iostream>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <numeric>
#include <cmath>
#include <limits>

Project::Project()
{
}


Project::~Project()
{
}

void Project::MakeProjection(std::string fileNameIn, std::string fileNameOut, Classifier& classifier)
{
	
	GDALDataset  *poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset *)GDALOpen(fileNameIn.c_str(), GA_ReadOnly);
	if (poDataset == NULL)
	{
		std::cout << "ERROR: Could not open " << fileNameIn << std::endl;
		return;
	}

	std::cout << poDataset->GetDriver()->GetDescription() << poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME) << std::endl;
	std::cout << "Size is " << poDataset->GetRasterXSize() << "x" << poDataset->GetRasterYSize() << "x" << poDataset->GetRasterCount() << std::endl;
	std::cout << "Projection is " << poDataset->GetProjectionRef() << std::endl;

	for (int i = 1; i < poDataset->GetRasterCount() + 1; i++)
	{
		GDALRasterBand  *poBand;
		int             nBlockXSize, nBlockYSize;
		int             bGotMin, bGotMax;
		double          adfMinMax[2];

		poBand = poDataset->GetRasterBand(i);
		poBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
		std::cout << "Block=" << nBlockXSize << "x" << nBlockYSize << " Type=" << GDALGetDataTypeName(poBand->GetRasterDataType()) <<
			"ColorInterp=" << GDALGetColorInterpretationName(poBand->GetColorInterpretation()) << std::endl;

		adfMinMax[0] = poBand->GetMinimum(&bGotMin);
		adfMinMax[1] = poBand->GetMaximum(&bGotMax);
		if (!(bGotMin && bGotMax))
			GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);

		std::cout << "Min=" << adfMinMax[0] << ", Max=" << adfMinMax[1] << std::endl;

		if (poBand->GetOverviewCount() > 0)
			std::cout << "Band has " << poBand->GetOverviewCount() << "overviews." << std::endl;

		if (poBand->GetColorTable() != NULL)
			std::cout << "Band has a color table with " << poBand->GetColorTable()->GetColorEntryCount() << " entries." << std::endl;
	}


	double *pafScanline;
	
	int nLayers = poDataset->GetRasterCount();
	pafScanline = (double *)CPLMalloc(sizeof(double)*nLayers);

	int maxX = poDataset->GetRasterXSize();
	int maxY = poDataset->GetRasterYSize();

	double *pafWriteLine;
	pafWriteLine = (double *)CPLMalloc(sizeof(double)*maxX);

	GDALDriverH hMemDriver = GDALGetDriverByName("MEM");
	if (hMemDriver == NULL) {
		std::cout << "ERROR: Failed to find MEM driver." << std::endl;
		return;
	}
	GDALDatasetH hMemDS = GDALCreate(hMemDriver, "msSaveImageGDAL_temp", maxX, maxY, 1, GDT_Float64, NULL);
	if (hMemDS == NULL) {
		std::cout << "ERROR: Failed to create MEM dataset." << std::endl;
		return;
	}
	GDALRasterBandH hBand = GDALGetRasterBand(hMemDS, 1);


	std::cout << "MAX Y: " << maxY << std::endl;
	for (int y = 0; y < maxY; y++)
	{
		std::vector<DataRow> rows;
		std::cout << y << " ";
		for (int x = 0; x < maxX; x++)
		{
			poDataset->RasterIO(GF_Read, x, y, 1, 1, pafScanline, 1, 1, GDT_Float64, nLayers, NULL, 0, 0, 0);
			std::vector<float> layers;
			std::tuple<float, float> loc;
			bool skip = false;
			for (int i = 0; i < nLayers; i++)
			{
				float t = static_cast<float>(pafScanline[i]);
				if (std::isfinite(t))
				{
					layers.push_back(t);
				}
				else
				{
					skip = true;
					break;
				}
			}
			if (!skip)
			{
				DataRow r = { x, 1, 1.0f, loc, layers };
				rows.push_back(r);
			}
		}
		DataSource d(rows, std::vector<std::string>(), 0, nLayers);
		std::vector<float> predictions = classifier.classify(d);

		for (int i = 0; i < maxX; i++)
			pafWriteLine[i] = -std::numeric_limits<double>::infinity();
		for (unsigned int i = 0; i < d.rows.size(); i++)
			pafWriteLine[d.rows[i].id] = static_cast<double>(predictions[i]);
		GDALRasterIO(hBand, GF_Write, 0, y, maxX, 1, pafWriteLine, maxX, 1, GDT_Float64, 0, 0);
	}

	double *bufferTransform;
	bufferTransform = (double *)CPLMalloc(sizeof(double)*6);
	poDataset->GetGeoTransform(bufferTransform);
	GDALSetGeoTransform(hMemDS, bufferTransform);
	CPLFree(bufferTransform);

	const char *pszWKT = poDataset->GetProjectionRef();
	GDALSetProjection(hMemDS, pszWKT);
	GDALSetRasterNoDataValue(hBand, -std::numeric_limits<double>::infinity());


	const char *pszFormat = "GTiff";
	GDALDriver *poDriver;
	poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);

	GDALDatasetH hOutputDS = GDALCreateCopy(poDriver, fileNameOut.c_str(), hMemDS, FALSE, NULL, NULL, NULL);

	GDALClose(hMemDS);
	GDALClose(hOutputDS);

	CPLFree(pafScanline);
	CPLFree(pafWriteLine);
	
}
#endif