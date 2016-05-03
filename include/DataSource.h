#pragma once
#include <vector>
#include <string>
#include <tuple>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/string.hpp>



struct DataRow
{
	int id;
	int times;
	float weight;
	std::tuple<float, float> loc;
	std::vector<float> data;

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(id), CEREAL_NVP(times), CEREAL_NVP(weight), CEREAL_NVP(loc), CEREAL_NVP(data)); 
	}
};


class DataSource
{
public:
	DataSource();
	DataSource(std::vector<DataRow> rows, std::vector<std::string> fields, int ndependents, int nindependents);
	DataSource(const DataSource& _datasource);

	~DataSource();

	std::vector<DataRow> rows;
	std::vector<std::tuple<float, float>> stats;
	std::vector<std::string> fields;
	std::string species;

	unsigned int ndependents;
	unsigned int nindependents;

	static DataSource fromFileName(std::string filename);
	void normalize(bool reset_times=false);
	void add(const DataSource& _datasource);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(rows), CEREAL_NVP(stats), CEREAL_NVP(fields), CEREAL_NVP(species), CEREAL_NVP(ndependents), CEREAL_NVP(nindependents));
	}

private:
	std::tuple<float, float> stdev(std::vector<float> v);
	void setStats(int dependent = 0, float threshold = 0.5f);

};

