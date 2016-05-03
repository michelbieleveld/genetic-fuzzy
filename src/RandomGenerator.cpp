#include "RandomGenerator.h"
#include "MPU.h"


RandomGenerator::RandomGenerator(int operand) :
dist_opcode(0, MPU::opcodes.size()-1),
dist_other(0, MPU::opcodes_other.size()-1),
dist_pop(0, MPU::opcodes_pop.size()-1),
dist_push(0, MPU::opcodes_push.size()-1),
dist_operand(0, operand-1),
dist_ram(0, 1.0f)
{
	
}

RandomGenerator::~RandomGenerator()
{
}

float RandomGenerator::getRam()
{
	return dist_ram(mt);
}

int RandomGenerator::getOpcodePush()
{
	return MPU::opcodes_push[dist_push(mt)];
}

int RandomGenerator::getOpcodePop()
{
	return MPU::opcodes_pop[dist_pop(mt)];
}

int RandomGenerator::getOpcodeOther()
{
	return MPU::opcodes_other[dist_other(mt)];
}

int RandomGenerator::getOpcode()
{
	return MPU::opcodes[dist_opcode(mt)];
}

int RandomGenerator::getOperand()
{
	return dist_operand(mt);
}

unsigned int RandomGenerator::Next(unsigned int max)
{
	std::uniform_int_distribution<unsigned int> dis(0, max);
	return dis(mt);
}
