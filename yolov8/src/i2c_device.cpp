#include "jetracer/i2c_device.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <stdexcept>

namespace jetracer::hardware {

I2CDevice::I2CDevice(const std::string& device, int address) {
    fd_ = open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open I2C device: " + device);
    }
    if (ioctl(fd_, I2C_SLAVE, address) < 0) {
        close(fd_);
        throw std::runtime_error("Failed to set I2C address");
    }
}

I2CDevice::~I2CDevice() {
    if (fd_ >= 0) {
        close(fd_);
    }
}

void I2CDevice::write_byte(uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    if (write(fd_, buffer, 2) != 2) {
        close(fd_);
        throw std::runtime_error("Failed to write to I2C device");
    }
}

uint8_t I2CDevice::read_byte(uint8_t reg) {
    if (write(fd_, &reg, 1) != 1) {
        close(fd_);
        throw std::runtime_error("Failed to write register to I2C device");
    }
    uint8_t value;
    if (read(fd_, &value, 1) != 1) {
        close(fd_);
        throw std::runtime_error("Failed to read from I2C device");
    }
    return value;
}

} // namespace jetracer::hardware
