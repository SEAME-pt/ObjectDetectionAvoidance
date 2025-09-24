// File: sources/main.cpp
#include "jetracer/pid_controller.hpp"
#include "jetracer/jetracer.hpp"
#include "jetracer/computer_vision.hpp"
#include "jetracer/pid_controller.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <atomic>
#include <thread>
#include <mosquitto.h>
#include <chrono>

// ====== MQTT CONFIGURATIONS ======
// HiveMQ Cloud (ACTIVE)
#define MQTT_BROKER "972e24210b544ba49bfb9c1d3164d02b.s1.eu.hivemq.cloud"
#define MQTT_PORT 8883
#define MQTT_USERNAME "jetracer"
#define MQTT_PASSWORD "Ft_seame5"

#define MQTT_TOPIC_LANE_TOUCH "jetracer/lane_touch"
#define MQTT_TOPIC_PASSADEIRA "jetracer/passadeira"
#define MQTT_TOPIC_STOP_SIGN "jetracer/stop_sign"
#define MQTT_TOPIC_SPEED_50 "jetracer/speed_50"
#define MQTT_TOPIC_SPEED_80 "jetracer/speed_80"
#define MQTT_TOPIC_JETRACER "jetracer/jetracer"
#define MQTT_TOPIC_GATE "jetracer/gate"

// ====== STREAMING CONFIGURATIONS ======
#define STREAM_IP "100.124.102.80" // Target PC IP
#define STREAM_PORT 5000           // Port for mask streaming
#define STREAM_WIDTH 640           // Stream width
#define STREAM_HEIGHT 480          // Stream height
#define STREAM_FPS 30              // Stream FPS

// ====== EMERGENCY STOP CONFIGURATIONS ======
#define EMERGENCY_THRESHOLD 0.15f        // 15% danger zone occupancy
#define EMERGENCY_RECOVERY_DELAY_MS 2000 // 2000ms (2 seconds) pause after stop

// ====== JETRACER VARIABLES ======
jetracer::control::JetRacer *jetracer_ptr = nullptr;
std::atomic<bool> program_running{true};

// ====== MQTT VARIABLES ======
struct mosquitto *mosq = nullptr;
bool mqtt_connected = false;
std::atomic<bool> mqtt_running{false};

// ====== STREAMING VARIABLES ======
cv::VideoWriter stream_writer;
bool streaming_initialized = false;

// ====== EMERGENCY STOP VARIABLES ======
bool emergency_stop_active = false;
std::chrono::steady_clock::time_point emergency_stop_time;
float speed_before_emergency = 0.0f;
bool cruise_control_before_emergency = false;

// ====== MQTT FUNCTIONS FOR SPECIFIC CLASSES ======
void publishPassadeira(bool detected)
{
    if (!mqtt_connected || !mosq)
    {
        std::cerr << "[MQTT] Not connected, skipping passadeira publication" << std::endl;
        return;
    }

    std::string message = detected ? "1" : "0";
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC_PASSADEIRA, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        std::cerr << "[MQTT] Failed to publish passadeira: " << mosquitto_strerror(ret) << std::endl;
        mqtt_connected = false; // Mark as disconnected to try reconnecting
    }
    else
    {
        // std::cout << "[MQTT] Passadeira: " << message << std::endl;
    }
}

void publishStopSign(bool detected)
{
    if (!mqtt_connected || !mosq)
    {
        std::cerr << "[MQTT] Não conectado, pulando publicação de stop sign" << std::endl;
        return;
    }

    std::string message = detected ? "1" : "0";
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC_STOP_SIGN, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        std::cerr << "[MQTT] Falha ao publicar stop sign: " << mosquitto_strerror(ret) << std::endl;
        mqtt_connected = false; // Marcar como desconectado para tentar reconectar
    }
    else
    {
        // std::cout << "[MQTT] Stop Sign: " << message << std::endl;
    }
}

void publishSpeed50(bool detected)
{
    if (!mqtt_connected || !mosq)
    {
        std::cerr << "[MQTT] Não conectado, pulando publicação de speed 50" << std::endl;
        return;
    }

    std::string message = detected ? "1" : "0";
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC_SPEED_50, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        std::cerr << "[MQTT] Falha ao publicar speed 50: " << mosquitto_strerror(ret) << std::endl;
        mqtt_connected = false; // Marcar como desconectado para tentar reconectar
    }
    else
    {
        // std::cout << "[MQTT] Speed 50: " << message << std::endl;
    }
}

void publishSpeed80(bool detected)
{
    if (!mqtt_connected || !mosq)
    {
        std::cerr << "[MQTT] Não conectado, pulando publicação de speed 80" << std::endl;
        return;
    }

    std::string message = detected ? "1" : "0";
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC_SPEED_80, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        std::cerr << "[MQTT] Falha ao publicar speed 80: " << mosquitto_strerror(ret) << std::endl;
        mqtt_connected = false; // Marcar como desconectado para tentar reconectar
    }
    else
    {
        // std::cout << "[MQTT] Speed 80: " << message << std::endl;
    }
}

void publishJetRacer(bool detected)
{
    if (!mqtt_connected || !mosq)
    {
        std::cerr << "[MQTT] Não conectado, pulando publicação de jetracer" << std::endl;
        return;
    }

    std::string message = detected ? "1" : "0";
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC_JETRACER, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        std::cerr << "[MQTT] Falha ao publicar jetracer: " << mosquitto_strerror(ret) << std::endl;
        mqtt_connected = false; // Marcar como desconectado para tentar reconectar
    }
    else
    {
        // std::cout << "[MQTT] JetRacer: " << message << std::endl;
    }
}

void publishGate(bool detected)
{
    if (!mqtt_connected || !mosq)
    {
        std::cerr << "[MQTT] Não conectado, pulando publicação de gate" << std::endl;
        return;
    }

    std::string message = detected ? "1" : "0";
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC_GATE, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        std::cerr << "[MQTT] Falha ao publicar gate: " << mosquitto_strerror(ret) << std::endl;
        mqtt_connected = false; // Marcar como desconectado para tentar reconectar
    }
    else
    {
        // std::cout << "[MQTT] Gate: " << message << std::endl;
    }
}

// ====== CALLBACKS MQTT ======
void on_connect(struct mosquitto *mosq, void *obj, int rc)
{
    (void)mosq; // Avoid unused parameter warning
    (void)obj;  // Avoid unused parameter warning
    if (rc == 0)
    {
        // std::cout << "[MQTT] Conectado com sucesso ao broker!" << std::endl;
        // std::cout << "[MQTT] Conexão TLS estabelecida e autenticada" << std::endl;
        mqtt_connected = true;
    }
    else
    {
        std::cerr << "[MQTT] Falha na conexão, código: " << rc << std::endl;

        // Interpretar códigos de falha de conexão
        switch (rc)
        {
        case 1:
            std::cerr << "[MQTT] Protocolo incorreto" << std::endl;
            break;
        case 2:
            std::cerr << "[MQTT] Identificador de cliente inválido" << std::endl;
            break;
        case 3:
            std::cerr << "[MQTT] Servidor indisponível" << std::endl;
            break;
        case 4:
            std::cerr << "[MQTT] Credenciais inválidas" << std::endl;
            break;
        case 5:
            std::cerr << "[MQTT] Não autorizado" << std::endl;
            break;
        case 6:
            std::cerr << "[MQTT] Erro de rede" << std::endl;
            break;
        default:
            std::cerr << "[MQTT] Código de erro desconhecido" << std::endl;
            break;
        }

        mqtt_connected = false;
    }
}

// ====== CALLBACKS MQTT ======
void on_disconnect(struct mosquitto *mosq, void *obj, int rc)
{
    (void)mosq; // Avoid unused parameter warning
    (void)obj;  // Avoid unused parameter warning
    // std::cout << "[MQTT] Desconectado do broker, código: " << rc << std::endl;

    // Interpretar códigos de desconexão
    switch (rc)
    {
    case 0:
        // std::cout << "[MQTT] Desconexão solicitada pelo cliente" << std::endl;
        break;
    case 1:
        // std::cout << "[MQTT] Erro de protocolo incorreto" << std::endl;
        break;
    case 2:
        // std::cout << "[MQTT] Identificador de cliente inválido" << std::endl;
        break;
    case 3:
        // std::cout << "[MQTT] Servidor indisponível" << std::endl;
        break;
    case 4:
        // std::cout << "[MQTT] Credenciais inválidas" << std::endl;
        break;
    case 5:
        // std::cout << "[MQTT] Não autorizado" << std::endl;
        break;
    case 6:
        // std::cout << "[MQTT] Erro de rede" << std::endl;
        break;
    case 7:
        // std::cout << "[MQTT] Conexão perdida (timeout/erro TLS)" << std::endl;
        break;
    default:
        // std::cout << "[MQTT] Código de erro desconhecido" << std::endl;
        break;
    }

    mqtt_connected = false;
}

// ====== MQTT INITIALIZATION AND CLEANUP FUNCTIONS ======
void initMQTT()
{
    mosquitto_lib_init();

    // Unique ClientID based on PID
    std::string cid = "yolov8_detector_" + std::to_string(getpid());
    mosq = mosquitto_new(cid.c_str(), true, nullptr);
    if (!mosq)
    {
        throw std::runtime_error("Error creating MQTT client");
    }

    // std::cout << "[MQTT] Cliente criado com ID: " << cid << std::endl;

    // Configure callbacks
    mosquitto_connect_callback_set(mosq, on_connect);
    mosquitto_disconnect_callback_set(mosq, on_disconnect);

    // Configure log callback for debug
    mosquitto_log_callback_set(mosq, [](struct mosquitto *mosq, void *userdata, int level, const char *str)
                               {
        (void)mosq;     // Evitar warning de parâmetro não utilizado
        (void)userdata; // Avoid unused parameter warning
        if (level <= MOSQ_LOG_WARNING) { // Only important logs
            std::cout << "[MQTT LOG] " << str << std::endl;
        } });

    // Additional settings for stability
    mosquitto_max_inflight_messages_set(mosq, 20);
    mosquitto_message_retry_set(mosq, 3);

// Configure authentication if necessary (for HiveMQ Cloud)
#ifdef MQTT_USERNAME
    int auth_ret = mosquitto_username_pw_set(mosq, MQTT_USERNAME, MQTT_PASSWORD);
    if (auth_ret != MOSQ_ERR_SUCCESS)
    {
        throw std::runtime_error("Error configuring MQTT authentication: " + std::string(mosquitto_strerror(auth_ret)));
    }
#endif

    // Configure TLS for HiveMQ Cloud (port 8883)
    if (MQTT_PORT == 8883)
    {
        // Use system CA bundle (safer than tls_insecure_set)
        int tls_ret = mosquitto_tls_set(mosq, "/etc/ssl/certs/ca-certificates.crt", nullptr, nullptr, nullptr, nullptr);
        if (tls_ret != MOSQ_ERR_SUCCESS)
        {
            std::cerr << "[WARNING] Falha ao configurar TLS com CA bundle: " << mosquitto_strerror(tls_ret) << std::endl;
            std::cerr << "[WARNING] Tentando configuração TLS alternativa..." << std::endl;

            // Fallback: basic TLS configuration
            tls_ret = mosquitto_tls_set(mosq, nullptr, nullptr, nullptr, nullptr, nullptr);
            if (tls_ret != MOSQ_ERR_SUCCESS)
            {
                std::cerr << "[ERROR] Falha na configuração TLS alternativa: " << mosquitto_strerror(tls_ret) << std::endl;
            }
        }
        else
        {
            // std::cout << "[MQTT] TLS configurado com CA bundle do sistema" << std::endl;
        }

        // Note: TLS 1.2 is default in OpenSSL 1.1.1f
        // std::cout << "[MQTT] TLS 1.2 será usado por padrão (OpenSSL 1.1.1f)" << std::endl;

        // Configure additional TLS options
        tls_ret = mosquitto_tls_opts_set(mosq, 1, nullptr, nullptr);
        if (tls_ret != MOSQ_ERR_SUCCESS)
        {
            std::cerr << "[WARNING] Falha ao configurar opções TLS: " << mosquitto_strerror(tls_ret) << std::endl;
        }

        // std::cout << "[MQTT] Configuração TLS robusta aplicada para HiveMQ Cloud" << std::endl;
    }

    // Configure optimized keepalive
    int keepalive = 60; // 1 minute (optimized for stability)

    // Connect asynchronously for better performance
    int ret = mosquitto_connect_async(mosq, MQTT_BROKER, MQTT_PORT, keepalive);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        throw std::runtime_error("Error starting MQTT async connection: " + std::string(mosquitto_strerror(ret)));
    }

    // std::cout << "[MQTT] Iniciando conexão assíncrona ao broker MQTT em " << MQTT_BROKER << ":" << MQTT_PORT << " (keepalive: " << keepalive << "s)" << std::endl;

    // Use library loop (more efficient than custom thread)
    ret = mosquitto_loop_start(mosq);
    if (ret != MOSQ_ERR_SUCCESS)
    {
        throw std::runtime_error("Error starting MQTT loop: " + std::string(mosquitto_strerror(ret)));
    }

    // std::cout << "[MQTT] Loop MQTT iniciado com sucesso" << std::endl;
}

void cleanupMQTT()
{
    if (mosq)
    {
        // Parar o loop da biblioteca
        mosquitto_loop_stop(mosq, true);

        // Desconectar e limpar
        mosquitto_disconnect(mosq);
        mosquitto_destroy(mosq);
        mosq = nullptr;
    }

    mosquitto_lib_cleanup();
    mqtt_connected = false;
    mqtt_running = false;
    // std::cout << "[MQTT] Conexão MQTT encerrada" << std::endl;
}

// ====== MQTT RECONNECTION FUNCTION ======
void reconnectMQTT()
{
    if (!mqtt_connected && mosq)
    {
        // std::cout << "[MQTT] Tentando reconectar..." << std::endl;

        // Wait a bit before trying to reconnect
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Check if client still exists and is valid
        if (!mqtt_connected)
        {
            // std::cout << "[MQTT] Cliente não está conectado, tentando reconectar..." << std::endl;

            // Try reconnecting using async system
            int ret = mosquitto_reconnect_async(mosq);
            if (ret != MOSQ_ERR_SUCCESS)
            {
                std::cerr << "[MQTT] Falha na reconexão assíncrona: " << mosquitto_strerror(ret) << std::endl;

                // If it fails, try recreating the connection
                // std::cout << "[MQTT] Tentando recriar conexão..." << std::endl;

                // Stop current loop
                mosquitto_loop_stop(mosq, true);
                mosquitto_disconnect(mosq);
                mosquitto_destroy(mosq);

                // Recreate MQTT client with unique ID
                std::string cid = "yolov8_detector_" + std::to_string(getpid()) + "_reconnect";
                mosq = mosquitto_new(cid.c_str(), true, nullptr);
                if (mosq)
                {
                    // Reconfigure callbacks
                    mosquitto_connect_callback_set(mosq, on_connect);
                    mosquitto_disconnect_callback_set(mosq, on_disconnect);

                    // Reconfigure log callback
                    mosquitto_log_callback_set(mosq, [](struct mosquitto *mosq, void *userdata, int level, const char *str)
                                               {
                        (void)mosq;     // Evitar warning de parâmetro não utilizado
                        (void)userdata; // Avoid unused parameter warning
                        if (level <= MOSQ_LOG_WARNING) {
                            std::cout << "[MQTT LOG] " << str << std::endl;
                        } });

// Reconfigure authentication
#ifdef MQTT_USERNAME
                    mosquitto_username_pw_set(mosq, MQTT_USERNAME, MQTT_PASSWORD);
#endif

                    // Reconfigure TLS
                    if (MQTT_PORT == 8883)
                    {
                        // Use system CA bundle
                        int tls_ret = mosquitto_tls_set(mosq, "/etc/ssl/certs/ca-certificates.crt", nullptr, nullptr, nullptr, nullptr);
                        if (tls_ret != MOSQ_ERR_SUCCESS)
                        {
                            std::cerr << "[WARNING] Falha ao reconfigurar TLS com CA bundle: " << mosquitto_strerror(tls_ret) << std::endl;
                            // Fallback
                            tls_ret = mosquitto_tls_set(mosq, nullptr, nullptr, nullptr, nullptr, nullptr);
                        }

                        // TLS 1.2 is default in OpenSSL 1.1.1f

                        // TLS options
                        mosquitto_tls_opts_set(mosq, 1, nullptr, nullptr);
                    }

                    // Try connecting again asynchronously
                    ret = mosquitto_connect_async(mosq, MQTT_BROKER, MQTT_PORT, 60);
                    if (ret == MOSQ_ERR_SUCCESS)
                    {
                        // Start loop
                        ret = mosquitto_loop_start(mosq);
                        if (ret == MOSQ_ERR_SUCCESS)
                        {
                            // std::cout << "[MQTT] Reconexão bem-sucedida!" << std::endl;
                        }
                        else
                        {
                            std::cerr << "[MQTT] Falha ao iniciar loop após reconexão: " << mosquitto_strerror(ret) << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "[MQTT] Falha na reconexão após recriação: " << mosquitto_strerror(ret) << std::endl;
                    }
                }
            }
            else
            {
                // std::cout << "[MQTT] Reconexão assíncrona iniciada..." << std::endl;
            }
        }
        else
        {
            // std::cout << "[MQTT] Cliente ainda está conectado, verificando status..." << std::endl;
            //  Verificar se realmente está conectado
            if (mqtt_connected)
            {
                // std::cout << "[MQTT] Cliente reconectado com sucesso!" << std::endl;
            }
        }
    }
}

// ====== STREAMING FUNCTIONS ======
bool initStreaming()
{
    try
    {
        // UDP streaming pipeline (streaming only, no camera capture)
        std::string stream_pipeline =
            "appsrc ! videoconvert ! "
            "x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
            "rtph264pay ! udpsink host=" +
            std::string(STREAM_IP) +
            " port=" + std::to_string(STREAM_PORT) + " sync=false";

        stream_writer.open(stream_pipeline, cv::CAP_GSTREAMER, 0, STREAM_FPS,
                           cv::Size(STREAM_WIDTH, STREAM_HEIGHT), true);
        if (!stream_writer.isOpened())
        {
            std::cerr << "[STREAMING] Error opening UDP stream" << std::endl;
            return false;
        }

        streaming_initialized = true;
        std::cout << "[STREAMING] Initialized successfully!" << std::endl;
        std::cout << "[STREAMING] Streaming masks to " << STREAM_IP << ":" << STREAM_PORT << std::endl;
        std::cout << "[STREAMING] Resolution: " << STREAM_WIDTH << "x" << STREAM_HEIGHT << " @ " << STREAM_FPS << "fps" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[STREAMING] Initialization error: " << e.what() << std::endl;
        return false;
    }
}

void cleanupStreaming()
{
    if (streaming_initialized)
    {
        if (stream_writer.isOpened())
        {
            stream_writer.release();
        }
        streaming_initialized = false;
        std::cout << "[STREAMING] Resources released" << std::endl;
    }
}

void signal_handler(int)
{
    std::cout << std::endl
              << "[!] Ctrl+C detected. Stopping the JetRacer..." << std::endl;
    program_running = false;
    if (jetracer_ptr)
        jetracer_ptr->stop();
    cleanupStreaming();
    cleanupMQTT();
    std::_Exit(0);
}

int main()
{
    std::cout << "=== PID with joystick (manual speed) + shared memory ===" << std::endl;

    try
    {
        jetracer::control::JetRacer jetracer(0x40, 0x60);
        jetracer_ptr = &jetracer;

        signal(SIGINT, signal_handler);

        // ===== MODO DE JOYSTICK ATIVADO =====
        jetracer.set_constant_speed_mode(false); // ← ATIVAR MODO DE JOYSTICK

        std::cout << "\n=== MODO DE JOYSTICK ATIVADO ===" << std::endl;
        std::cout << "Controle via joystick habilitado" << std::endl;
        std::cout << "Velocidade máxima limitada a 27% (configuração ideal)" << std::endl;
        std::cout << "Use o joystick esquerdo para controlar a velocidade" << std::endl;

        // Iniciar o sistema principal
        jetracer.start();

        int shm_fd = shm_open("mask_shared", O_RDWR, 0666);
        if (shm_fd == -1)
        {
            std::cerr << "Error oppening shared memory." << std::endl;
            return 1;
        }

        uint8_t *shm_ptr = (uint8_t *)mmap(nullptr, 2 * jetracer::vision::SIZE + 1, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED)
        {
            std::cerr << "Error mapping memory." << std::endl;
            return 1;
        }

        // Estrutura da shared memory: flag + lane_mask + drivable_mask
        uint8_t *flag_ptr = shm_ptr;
        uint8_t *lane_mask_ptr = shm_ptr + 1;
        uint8_t *drivable_mask_ptr = shm_ptr + 1 + jetracer::vision::SIZE;

        cv::Mat lane_mask(jetracer::vision::HEIGHT, jetracer::vision::WIDTH, CV_8UC1, lane_mask_ptr);
        cv::Mat drivable_mask(jetracer::vision::HEIGHT, jetracer::vision::WIDTH, CV_8UC1, drivable_mask_ptr);

        // Note: drivable_mask contains the drivable area detected by YOLO
        // Can be used for additional validation, navigation or terrain analysis

        std::cout << "\n=== SISTEMA PRINCIPAL EM EXECUÇÃO ===" << std::endl;
        std::cout << "Controle via joystick ativo" << std::endl;
        std::cout << "Velocidade máxima limitada a 27% (configuração ideal)" << std::endl;
        std::cout << "Use o joystick esquerdo para controlar a velocidade" << std::endl;
        std::cout << "Recebendo máscaras: Lane + Drivable" << std::endl;
        std::cout << "\n=== CRUISE CONTROL (MODO AUTÔNOMO) ===" << std::endl;
        std::cout << "Botão R2: Ativar/Desativar cruise control" << std::endl;
        std::cout << "• Pressione R2 para manter a velocidade atual" << std::endl;
        std::cout << "• Pressione R2 novamente para voltar ao controle manual" << std::endl;
        std::cout << "• Cruise control é desativado automaticamente em emergência" << std::endl;
        std::cout << "\nUse Ctrl+C para parar o sistema a qualquer momento" << std::endl;

        // ====== MQTT INITIALIZATION ======
        try
        {
            initMQTT();
            std::cout << "[INFO] MQTT inicializado com sucesso!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] Falha ao inicializar MQTT: " << e.what() << std::endl;
            std::cerr << "[WARNING] Continuando sem MQTT..." << std::endl;
        }

        // ====== STREAMING INITIALIZATION ======
        if (!initStreaming())
        {
            std::cerr << "[ERROR] Falha ao inicializar streaming!" << std::endl;
            std::cerr << "[WARNING] Continuando sem streaming..." << std::endl;
        }

        // Variables for streaming
        auto start_time = std::chrono::steady_clock::now();
        int stream_fps_counter = 0;
        float stream_fps = 0.0f;

        while (program_running)
        {

            if (flag_ptr[0] != 1)
            {
                usleep(3000);
                continue;
            }

            // ====== CHECK AND RECONNECT MQTT IF NECESSARY ======
            static int frame_counter = 0;
            if (!mqtt_connected && frame_counter % 30 == 0)
            { // Try reconnecting every 30 frames
                // std::cout << "[MQTT] Status: Desconectado - tentando reconectar..." << std::endl;

                // Check if MQTT client is still valid
                if (mosq && mqtt_connected)
                {
                    // std::cout << "[MQTT] Cliente ainda está conectado, atualizando status..." << std::endl;
                }
                else
                {
                    reconnectMQTT();
                }
            }

            // ====== DETECT AND PUBLISH SPECIFIC CLASSES VIA MQTT ======
            // detectAndPublishClasses(res, labels_map); // Comentado temporariamente

            // ====== EMERGENCY STOP SYSTEM ======
            // Detect lane curves
            float y_ref;
            std::vector<cv::Point> left_curve, right_curve;
            bool lanes_detected = jetracer::vision::extractLanePoints(lane_mask, lane_mask.cols / 2.0f, y_ref, left_curve, right_curve);

            // ====== INTEGRATED EMERGENCY STOP SYSTEM ======
            float lane_occupancy = 0.0f;
            float drivable_occupancy = 0.0f;
            cv::Mat danger_zone_mask, drivable_danger_zone_mask;
            bool emergency_triggered = false;
            std::string emergency_source = "";

            // ====== ZONA DE PERIGO DAS LANES ======
            if (lanes_detected)
            {
                // Criar máscara da zona de perigo para cálculo de ocupação
                const float scale = 40.0f / (lane_mask.cols / 2.0f);
                jetracer::vision::createDangerZoneMask(lane_mask, left_curve, right_curve, 17.0f, scale, danger_zone_mask);

                // Calcular ocupação da zona de perigo usando a máscara da zona de perigo
                lane_occupancy = jetracer::vision::calculateDangerZoneOccupancyFromMask(lane_mask, danger_zone_mask);
            }

            // ====== ZONA DE PERIGO DRIVABLE ======
            // Sempre processar zona de perigo drivable (mesmo se máscara estiver vazia)
            const float scale = 40.0f / (drivable_mask.cols / 2.0f);
            jetracer::vision::createDrivableDangerZoneMask(drivable_mask, 17.0f, scale, drivable_danger_zone_mask);
            drivable_occupancy = jetracer::vision::calculateDrivableDangerZoneOccupancy(drivable_mask, drivable_danger_zone_mask);

            // Log detalhado da ocupação drivable
            if (drivable_occupancy > 0.0f)
            {
                std::cout << "[DRIVABLE] Ocupação na zona de perigo: " << (drivable_occupancy * 100) << "%" << std::endl;
            }

            // Log de debug para verificar se a zona de perigo está sendo criada
            if (cv::countNonZero(drivable_danger_zone_mask) > 0)
            {
                std::cout << "[DEBUG] Zona de perigo drivable criada com " << cv::countNonZero(drivable_danger_zone_mask) << " pixels" << std::endl;
            }

            // Log de debug para verificar o estado das máscaras
            int drivable_pixels = cv::countNonZero(drivable_mask);
            std::cout << "[DEBUG] Máscara drivable: " << drivable_pixels << " pixels, Zona de perigo: " << cv::countNonZero(drivable_danger_zone_mask) << " pixels" << std::endl;

            // ====== VERIFICAÇÃO INTEGRADA DE EMERGÊNCIA ======
            // Verificar se qualquer zona de perigo excede o limiar (25%)
            // Prioridade: LANES primeiro, depois DRIVABLE
            if (lane_occupancy > EMERGENCY_THRESHOLD)
            {
                emergency_triggered = true;
                emergency_source = "LANES";
                std::cout << "[EMERGENCY CHECK] Zona de perigo LANES excedeu limiar: " << (lane_occupancy * 100) << "% > " << (EMERGENCY_THRESHOLD * 100) << "%" << std::endl;
            }
            else if (drivable_occupancy > EMERGENCY_THRESHOLD)
            {
                emergency_triggered = true;
                emergency_source = "DRIVABLE";
                std::cout << "[EMERGENCY CHECK] Zona de perigo DRIVABLE excedeu limiar: " << (drivable_occupancy * 100) << "% > " << (EMERGENCY_THRESHOLD * 100) << "%" << std::endl;
            }

            // Ativar parada de emergência se necessário
            if (emergency_triggered && !emergency_stop_active)
            {
                // Salvar estado antes da emergência
                cruise_control_before_emergency = jetracer.is_cruise_control_active();
                if (cruise_control_before_emergency)
                {
                    // Se estava em cruise control, salvar a velocidade do cruise control
                    speed_before_emergency = jetracer.get_cruise_control_speed();
                }
                else
                {
                    // Se estava em modo manual, precisamos obter a velocidade atual do joystick
                    // Como não temos acesso direto ao joystick aqui, vamos usar uma abordagem diferente
                    // Vamos salvar a velocidade atual do sistema (que pode ser obtida de outras formas)
                    speed_before_emergency = 0.0f; // Será definida pelo usuário após a recuperação
                }

                emergency_stop_active = true;
                emergency_stop_time = std::chrono::steady_clock::now();
                jetracer.emergency_stop();
                std::cout << "[EMERGENCY] Zona de perigo detectada em " << emergency_source << "! ";
                if (emergency_source == "LANES")
                {
                    std::cout << "Ocupação lanes: " << (lane_occupancy * 100) << "%";
                }
                else
                {
                    std::cout << "Ocupação drivable: " << (drivable_occupancy * 100) << "%";
                }
                if (cruise_control_before_emergency)
                {
                    std::cout << " - Cruise control ativo, velocidade salva: " << speed_before_emergency << "%";
                }
                else
                {
                    std::cout << " - Modo manual, velocidade será restaurada pelo joystick";
                }
                std::cout << std::endl;
            }
            // Verificar se pode retomar o controle (ambas as zonas devem estar livres)
            else if (!emergency_triggered && emergency_stop_active)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - emergency_stop_time).count();

                if (elapsed >= EMERGENCY_RECOVERY_DELAY_MS)
                {
                    emergency_stop_active = false;

                    // LIBERAR TRAVAMENTO DO MOTOR
                    jetracer.release_motor_lock();

                    // Restaurar velocidade e estado do cruise control
                    if (cruise_control_before_emergency && std::abs(speed_before_emergency) > 5.0f)
                    {
                        // Restaurar cruise control com velocidade salva
                        jetracer.set_cruise_control_speed(speed_before_emergency);
                        jetracer.set_cruise_control_mode(true);
                        std::cout << "[EMERGENCY] Zonas livres! Restaurando cruise control com velocidade: " << speed_before_emergency << "%" << std::endl;
                    }
                    else if (!cruise_control_before_emergency)
                    {
                        // Modo manual - o joystick retomará o controle automaticamente
                        std::cout << "[EMERGENCY] Zonas livres! Retomando controle manual via joystick." << std::endl;
                    }
                    else
                    {
                        std::cout << "[EMERGENCY] Zonas livres! Retomando controle normal (velocidade baixa não restaurada)." << std::endl;
                    }

                    // Resetar variáveis de estado
                    speed_before_emergency = 0.0f;
                    cruise_control_before_emergency = false;
                }
            }

            // Executar PID apenas se não estiver em parada de emergência
            float pid_angle = 0.0f;
            if (!emergency_stop_active)
            {
                pid_angle = jetracer::pid::PIDexecute(lane_mask.clone());
                jetracer.smooth_steering(static_cast<int>(pid_angle), 5);
            }

            frame_counter++;

            // ====== STREAMING DAS MÁSCARAS (ORIGINAL + ZONA DE PERIGO) ======
            if (streaming_initialized && stream_writer.isOpened())
            {
                cv::Mat mask_for_stream = lane_mask.clone();

                // Construir curvas dinâmicas na metade inferior
                const float image_center = mask_for_stream.cols / 2.0f;
                const float scale = 40.0f / (mask_for_stream.cols / 2.0f); // mantém a tua conversão cm->px
                std::vector<cv::Point> left_curve, right_curve;

                // Opcional: um pouco de fecho morfológico ajuda em falhas pequenas
                // cv::morphologyEx(mask_for_stream, mask_for_stream, cv::MORPH_CLOSE,
                //                  cv::getStructuringElement(cv::MORPH_RECT, {3,3}));

                jetracer::vision::sampleLaneEdgesByRow(
                    mask_for_stream,
                    mask_for_stream.rows * 2 / 5, // y_start (3/5 inferiores)
                    mask_for_stream.rows,         // y_end
                    2,                            // step em píxeis (aumenta para +fps)
                    image_center,
                    left_curve, right_curve, 3 // min_run
                );

                // Zona de perigo dinâmica na máscara original
                jetracer::vision::drawDangerZoneCurved(mask_for_stream, left_curve, right_curve, 17.0f, scale);

                // Criar máscara separada apenas da zona de perigo
                cv::Mat danger_zone_mask;
                jetracer::vision::createDangerZoneMask(lane_mask, left_curve, right_curve, 17.0f, scale, danger_zone_mask);

                // Converter máscaras para formato visual
                cv::Mat lane_visual, drivable_visual, danger_zone_visual;
                cv::cvtColor(mask_for_stream, lane_visual, cv::COLOR_GRAY2BGR);
                cv::cvtColor(drivable_mask, drivable_visual, cv::COLOR_GRAY2BGR);
                cv::cvtColor(danger_zone_mask, danger_zone_visual, cv::COLOR_GRAY2BGR);

                // Redimensionar máscaras para terço da largura do stream
                int third_width = STREAM_WIDTH / 3;
                cv::Mat lane_resized, drivable_resized, danger_zone_resized;
                cv::resize(lane_visual, lane_resized, cv::Size(third_width, STREAM_HEIGHT));
                cv::resize(drivable_visual, drivable_resized, cv::Size(third_width, STREAM_HEIGHT));
                cv::resize(danger_zone_visual, danger_zone_resized, cv::Size(third_width, STREAM_HEIGHT));

                // Criar imagem combinada (3 colunas)
                cv::Mat combined_frame = cv::Mat::zeros(STREAM_HEIGHT, STREAM_WIDTH, CV_8UC3);

                // Colocar máscara lane à esquerda
                cv::Rect left_roi(0, 0, third_width, STREAM_HEIGHT);
                lane_resized.copyTo(combined_frame(left_roi));

                // Colocar máscara drivable no centro
                cv::Rect center_roi(third_width, 0, third_width, STREAM_HEIGHT);
                drivable_resized.copyTo(combined_frame(center_roi));

                // Colocar máscara da zona de perigo à direita
                cv::Rect right_roi(2 * third_width, 0, third_width, STREAM_HEIGHT);
                danger_zone_resized.copyTo(combined_frame(right_roi));

                // HUD
                stream_fps_counter++;
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                if (elapsed >= 1)
                {
                    stream_fps = stream_fps_counter / (float)elapsed;
                    start_time = now;
                    stream_fps_counter = 0;
                }

                // Adicionar textos informativos
                cv::putText(combined_frame, "FPS: " + std::to_string((int)stream_fps), {10, 30},
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
                cv::putText(combined_frame, "TRIPLE MASK STREAMING", {10, 60},
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
                cv::putText(combined_frame, "PID: " + std::to_string(pid_angle), {10, 90},
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);

                // Indicador do Cruise Control
                if (jetracer.is_cruise_control_active())
                {
                    cv::putText(combined_frame, "CRUISE CONTROL: ATIVO", {10, 120},
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
                    cv::putText(combined_frame, "VELOCIDADE: " + std::to_string((int)jetracer.get_cruise_control_speed()) + "%", {10, 150},
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
                }
                else
                {
                    cv::putText(combined_frame, "CRUISE CONTROL: INATIVO", {10, 120},
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, {128, 128, 128}, 2);
                }

                // Labels para as máscaras
                cv::putText(combined_frame, "LANE MASK", {10, STREAM_HEIGHT - 30},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);
                cv::putText(combined_frame, "DRIVABLE MASK", {third_width + 10, STREAM_HEIGHT - 30},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2);
                cv::putText(combined_frame, "DANGER ZONE", {2 * third_width + 10, STREAM_HEIGHT - 30},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 255}, 2);

                // ====== INDICADORES INTEGRADOS DE ZONA DE PERIGO ======
                // Indicador das lanes
                if (lanes_detected)
                {
                    float lane_occupancy_percent = lane_occupancy * 100.0f;

                    // Determinar cor baseada na ocupação das lanes
                    cv::Scalar lane_color;
                    std::string lane_status_text;

                    if (lane_occupancy_percent < 15.0f)
                    {
                        lane_color = cv::Scalar(0, 255, 0); // Verde - seguro
                        lane_status_text = "SEGURO";
                    }
                    else if (lane_occupancy_percent < 25.0f)
                    {
                        lane_color = cv::Scalar(0, 255, 255); // Amarelo - atenção
                        lane_status_text = "ATENCAO";
                    }
                    else
                    {
                        lane_color = cv::Scalar(0, 0, 255); // Vermelho - emergência
                        lane_status_text = "EMERGENCIA";
                    }

                    // Exibir status da zona de perigo das lanes
                    cv::putText(combined_frame, "LANES: " + lane_status_text,
                                cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, lane_color, 2);
                    cv::putText(combined_frame, "Ocupacao: " + std::to_string((int)lane_occupancy_percent) + "%",
                                cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.8, lane_color, 2);
                }
                else
                {
                    // Se não detectar faixas, mostrar aviso
                    cv::putText(combined_frame, "LANES: NAO DETECTADAS",
                                cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                cv::Scalar(0, 0, 255), 2);
                }

                // ====== INDICADORES DE ZONA DE PERIGO DRIVABLE ======
                if (drivable_occupancy > 0.0f)
                {
                    float drivable_occupancy_percent = drivable_occupancy * 100.0f;

                    // Determinar cor baseada na ocupação drivable
                    cv::Scalar drivable_color;
                    std::string drivable_status_text;

                    if (drivable_occupancy_percent < 15.0f)
                    {
                        drivable_color = cv::Scalar(0, 255, 0); // Verde - seguro
                        drivable_status_text = "SEGURO";
                    }
                    else if (drivable_occupancy_percent < 25.0f)
                    {
                        drivable_color = cv::Scalar(0, 255, 255); // Amarelo - atenção
                        drivable_status_text = "ATENCAO";
                    }
                    else
                    {
                        drivable_color = cv::Scalar(0, 0, 255); // Vermelho - perigo
                        drivable_status_text = "EMERGENCIA";
                    }

                    // Exibir status da zona de perigo drivable
                    cv::putText(combined_frame, "DRIVABLE: " + drivable_status_text,
                                cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.8, drivable_color, 2);
                    cv::putText(combined_frame, "Ocupacao: " + std::to_string((int)drivable_occupancy_percent) + "%",
                                cv::Point(10, 210), cv::FONT_HERSHEY_SIMPLEX, 0.8, drivable_color, 2);
                }
                else
                {
                    // Se não há área drivable detectada
                    cv::putText(combined_frame, "DRIVABLE: NAO DETECTADO",
                                cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                cv::Scalar(128, 128, 128), 2);
                }

                // ====== MENSAGEM DE EMERGÊNCIA INTEGRADA ======
                if (emergency_stop_active)
                {
                    cv::putText(combined_frame, "EMERGENCY STOP!",
                                cv::Point(10, 240), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                cv::Scalar(0, 0, 255), 3);

                    // Mostrar qual zona causou a emergência
                    if (emergency_source == "LANES")
                    {
                        cv::putText(combined_frame, "CAUSA: ZONA DE PERIGO LANES",
                                    cv::Point(10, 270), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    cv::Scalar(0, 0, 255), 2);
                    }
                    else if (emergency_source == "DRIVABLE")
                    {
                        cv::putText(combined_frame, "CAUSA: ZONA DE PERIGO DRIVABLE",
                                    cv::Point(10, 270), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    cv::Scalar(0, 0, 255), 2);
                    }

                    // Mostrar informações sobre recuperação
                    if (cruise_control_before_emergency && std::abs(speed_before_emergency) > 5.0f)
                    {
                        cv::putText(combined_frame, "VELOCIDADE SALVA: " + std::to_string((int)speed_before_emergency) + "%",
                                    cv::Point(10, 300), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    cv::Scalar(0, 255, 255), 2);
                        cv::putText(combined_frame, "CRUISE CONTROL SERA RESTAURADO",
                                    cv::Point(10, 330), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    cv::Scalar(0, 255, 255), 2);
                    }
                    else if (!cruise_control_before_emergency)
                    {
                        cv::putText(combined_frame, "CONTROLE MANUAL SERA RESTAURADO",
                                    cv::Point(10, 300), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    cv::Scalar(0, 255, 255), 2);
                    }
                }

                stream_writer.write(combined_frame);
            }

            flag_ptr[0] = 0;

            if (cv::waitKey(1) == 27)
                break;
        }

        // Sistema em modo de joystick - não há teste para parar

        jetracer.stop();
        cleanupStreaming();
        cleanupMQTT();
        std::cout << "Finishing." << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        if (jetracer_ptr)
            jetracer_ptr->stop();
        cleanupStreaming();
        cleanupMQTT();
        return 1;
    }
}
