#pragma once
#include <random>

class RandomGenerator
{
public:
	RandomGenerator(int operand);
	~RandomGenerator();

	int getOpcodePush();
	int getOpcodePop();
	int getOpcodeOther();
	int getOpcode();
	int getOperand();
	float getRam();
	unsigned int Next(unsigned int max);

private:
	
	std::mt19937 mt{ std::random_device{}() };
	std::uniform_int_distribution<int> dist_pop;
	std::uniform_int_distribution<int> dist_push;
	std::uniform_int_distribution<int> dist_other;
	std::uniform_int_distribution<int> dist_opcode;
	std::uniform_int_distribution<int> dist_operand;
	std::uniform_real_distribution<float> dist_ram;




};

