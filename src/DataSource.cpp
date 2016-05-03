#include "DataSource.h"
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <array>
#include <tuple>
#include <numeric>


DataSource::DataSource()
{

}

DataSource::DataSource(std::vector<DataRow> _rows, std::vector<std::string> _fields, int _ndependents, int _nindependents)
{
	rows = _rows;
	ndependents = _ndependents;
	nindependents = _nindependents;
	fields = _fields;
	setStats();
}

DataSource::DataSource(const DataSource& _datasource)
	:
	rows(_datasource.rows)
{
	nindependents = _datasource.nindependents;
	ndependents = _datasource.ndependents;
	setStats();
}

DataSource::~DataSource()
{
}

void DataSource::add(const DataSource& _datasource)
{
	int n = rows.size();
	for (size_t i = 0; i < _datasource.rows.size(); i++)
	{
		DataRow r = _datasource.rows[i];
		r.id += n;
		rows.push_back(r);

	}
}

void DataSource::normalize(bool reset_times)
{
	float totalWeight = 0;
	if (reset_times)
		for (unsigned int i = 0; i < rows.size(); i++)
			rows[i].times = 1;
	for (unsigned int i = 0; i < rows.size(); i++)
		totalWeight += rows[i].times * rows[i].weight;
	for (unsigned int i = 0; i < rows.size(); i++)
	{
		rows[i].weight = rows[i].times * rows[i].weight / totalWeight;
	}
}

void DataSource::setStats(int dependent, float threshold)
{
	std::vector<std::vector<float>> d;
	if (rows.size() > 0)
	{
		for (unsigned int c = 0; c < rows[0].data.size(); c++)
			d.push_back(std::vector<float>());

		for (unsigned int r = 0; r < rows.size(); r++)
			if (rows[r].data[dependent] > threshold)
				for (unsigned int c = 0; c < rows[r].data.size(); c++)
					d[c].push_back(rows[r].data[c]);
	}
	stats.clear();
	for (unsigned int c = 0; c < d.size(); c++)
		stats.push_back(stdev(d[c]));
}

std::tuple<float, float> DataSource::stdev(std::vector<float> v)
{
	float sum = (float) std::accumulate(std::begin(v), std::end(v), 0.0);
	float m = sum / v.size();

	float accum = 0.0;
	std::for_each(std::begin(v), std::end(v), [&](const float d) {
		accum += (d - m) * (d - m);
	});

	float stdev = sqrt(accum / (v.size() - 1));
	return std::tuple<float, float>(m, stdev);
}



DataSource DataSource::fromFileName(std::string filename)
{
	std::unordered_set <std::string> ignoreColumns;
	std::unordered_set <std::string> dependentColumns;
	std::vector<DataRow> rows;
	std::vector<std::string> fields;
	std::vector<std::string> ufields;
	std::ifstream file(filename);
	std::string line;
	
	ignoreColumns.insert("PROJ");
	ignoreColumns.insert("BACKGROUND");
	ignoreColumns.insert("PRESENCE");
	ignoreColumns.insert("SPECIES");
	ignoreColumns.insert("PREDICT");
	dependentColumns.insert("presence");

	std::getline(file, line);
	std::stringstream lineStream(line);
	std::string cell;
	while (std::getline(lineStream, cell, ','))
	{
		cell.erase(std::remove(cell.begin(), cell.end(), '\"'), cell.end());
		fields.push_back(cell);
		std::transform(cell.begin(), cell.end(), cell.begin(), ::toupper);
		ufields.push_back(cell);
	}
	

	int rowId = 0;
	int ndependents = 0;
	int nindependents = 0;

	bool isSpeciesFile = (fields[0] == "species");
	std::string species;

	while (std::getline(file, line))
	{
		lineStream.str(line);
		lineStream.clear();
		std::vector<float> dependents;
		std::vector<float> independents;
		std::tuple<float, float> loc;
		int colId = 0;
		while (std::getline(lineStream, cell, ','))
		{
			if (isSpeciesFile) species = cell;
			std::string ufield = ufields[colId];
			std::string field = fields[colId++];
			if (ignoreColumns.count(ufield) > 0) continue;
			float value = std::stof(cell);

			if (ufield[0] == 'X' || ufield == "LON")  
				std::get<0>(loc) = value;
			else if (ufield[0] == 'Y' || ufield == "LAT")
				std::get<1>(loc) = value;
			else if (dependentColumns.count(field) > 0)
				dependents.push_back(value);
			else
				independents.push_back(value);
		}
		
		if (ufields[0] == "PRESENCE" || ufields[0] == "SPECIES")
			dependents.push_back(1.0f);
		else if (ufields[0] == "BACKGROUND")
			dependents.push_back(0.0f);

		ndependents = dependents.size();
		nindependents = independents.size();

		dependents.insert(dependents.end(), independents.begin(), independents.end());
		DataRow r{ rowId++, 1, 1.0f, loc, dependents };
		rows.push_back(r);
	}

	DataSource ds = DataSource(rows, fields, ndependents, nindependents);
	ds.normalize(true);
	ds.setStats();
	ds.species = species;
	return ds;
}