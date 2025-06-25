#ifndef I2C_DEVICE_HPP
#define I2C_DEVICE_HPP

#include <cstdint>
#include <string>

namespace jetracer::hardware
{
	class I2CDevice
	{
	public:
		I2CDevice(const std::string &device, int address);
		~I2CDevice();

		void write_byte(uint8_t reg, uint8_t value);
		uint8_t read_byte(uint8_t reg);
	private:
		int fd_;
	};
} // namespace jetracer::hardware

#endif // I2C_DEVICE_HPP
