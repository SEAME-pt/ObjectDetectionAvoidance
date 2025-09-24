# Compiler and flags
CC = g++
CFLAGS = -Wall -Wextra -Werror -std=c++17 -O3 -g -Iincludes `pkg-config --cflags opencv4` -MMD
LDFLAGS = -lSDL2 -li2c -lpthread `pkg-config --libs opencv4` -lstdc++fs -lrt -lmosquitto

# Directories
SRCDIR = sources
APPDIR = apps
OBJDIR = build
BINDIR = bin

# Sources by module
SRC_TEST_PID     = $(APPDIR)/main.cpp $(SRCDIR)/pid_controller.cpp $(SRCDIR)/jetracer.cpp $(SRCDIR)/i2c_device.cpp $(SRCDIR)/computer_vision.cpp

# Generated objects
OBJ_TEST_PID     = $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(SRC_TEST_PID)))

# Executables
EXEC_TEST_PID    = $(BINDIR)/jetracer_pid_controler

# Main target
all: $(EXEC_TEST_PID)

# Rules for executables
$(EXEC_TEST_PID): $(OBJ_TEST_PID) | $(BINDIR)
	#$(CC) $(CFLAGS) -o $@ $(addprefix $(OBJDIR)/,$(notdir $^)) -lSDL2 -lpthread `pkg-config --libs opencv4` -lstdc++fs
	$(CC) $(CFLAGS) -o $@ $(addprefix $(OBJDIR)/,$(notdir $^)) $(LDFLAGS)

# Rule for object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(APPDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build/ directory if necessary
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Create bin/ directory if necessary
$(BINDIR):
	mkdir -p $(BINDIR)

# Auxiliary commands
clean:
	rm -rf $(OBJDIR) $(BINDIR)

re: clean all

# Targets to run specific ones
testpid: $(EXEC_TEST_PID)

.PHONY: all clean re control video frames lane frameslane oneframe testpid run

# Include automatically generated dependencies
-include $(OBJDIR)/*.d
