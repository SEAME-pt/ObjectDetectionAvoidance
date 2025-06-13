# Compilador e flags
CC = g++
CFLAGS = -Wall -Wextra -Werror -std=c++17 -O3 -g -Iincludes `pkg-config --cflags opencv4` -MMD
LDFLAGS = -lSDL2 -li2c -lpthread `pkg-config --libs opencv4` -lstdc++fs -lrt

# Diretórios
SRCDIR = sources
APPDIR = apps
OBJDIR = build
BINDIR = bin

# Fontes por módulo
SRC_TEST_PID     = $(APPDIR)/main.cpp $(SRCDIR)/pid_controller.cpp $(SRCDIR)/jetracer.cpp $(SRCDIR)/i2c_device.cpp $(SRCDIR)/computer_vision.cpp

# Objetos gerados
OBJ_TEST_PID     = $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(SRC_TEST_PID)))

# Executáveis
EXEC_TEST_PID    = $(BINDIR)/jetracer_pid_controler

# Alvo principal
all: $(EXEC_TEST_PID)

# Regras para executáveis
$(EXEC_TEST_PID): $(OBJ_TEST_PID) | $(BINDIR)
	#$(CC) $(CFLAGS) -o $@ $(addprefix $(OBJDIR)/,$(notdir $^)) -lSDL2 -lpthread `pkg-config --libs opencv4` -lstdc++fs
	$(CC) $(CFLAGS) -o $@ $(addprefix $(OBJDIR)/,$(notdir $^)) $(LDFLAGS)

# Regra para arquivos objeto
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(APPDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Cria diretório build/ se necessário
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Cria bin/ se necessário
$(BINDIR):
	mkdir -p $(BINDIR)

# Comandos auxiliares
clean:
	rm -rf $(OBJDIR) $(BINDIR)

re: clean all

# Alvos para rodar específicos
testpid: $(EXEC_TEST_PID)

.PHONY: all clean re control video frames lane frameslane oneframe testpid run

# Inclui dependências geradas automaticamente
-include $(OBJDIR)/*.d
