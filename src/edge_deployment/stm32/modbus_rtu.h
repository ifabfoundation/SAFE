/**
 ******************************************************************************
 * @file    modbus_rtu.h
 * @brief   Modbus RTU protocol parser for STM32U545RE
 * @note    Configurato per sniffing su bus RS-485 (passive listening)
 ******************************************************************************
 */

#ifndef MODBUS_RTU_H
#define MODBUS_RTU_H

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================== */
/* CONFIGURAZIONE MODBUS RTU                                                 */
/* ========================================================================== */

#define MODBUS_SLAVE_ADDR        0x01    // Indirizzo sensore da ascoltare
#define MODBUS_FUNCTION_CODE     0x03    // Read Holding Registers (tipico)
#define MODBUS_MAX_DATA_LEN      128     // Max payload (16 float32 = 64 bytes)
#define MODBUS_FRAME_MAX_SIZE    256     // Max frame completo

// Timing Modbus RTU (@ 9600 bps: 1 char ≈ 1.15 ms)
#define MODBUS_T15_MS            2       // 1.5 caratteri di silenzio
#define MODBUS_T35_MS            4       // 3.5 caratteri di silenzio (fine frame)

/* ========================================================================== */
/* STRUTTURE DATI                                                            */
/* ========================================================================== */

/**
 * @brief Stato della macchina a stati del parser
 */
typedef enum {
    MODBUS_STATE_IDLE = 0,      // Attesa inizio frame
    MODBUS_STATE_RECEIVING,     // Ricezione dati in corso
    MODBUS_STATE_COMPLETE,      // Frame completo ricevuto
    MODBUS_STATE_ERROR          // Errore CRC o timeout
} modbus_state_t;

/**
 * @brief Frame Modbus RTU (struttura completa)
 */
typedef struct {
    uint8_t  address;                        // Slave address
    uint8_t  function;                       // Function code
    uint8_t  data_len;                       // Lunghezza dati
    uint8_t  data[MODBUS_MAX_DATA_LEN];      // Payload
    uint16_t crc16;                          // CRC16 calcolato
    uint16_t crc16_rx;                       // CRC16 ricevuto
    uint8_t  raw[MODBUS_FRAME_MAX_SIZE];     // Buffer raw completo
    uint16_t raw_len;                        // Lunghezza totale frame
    bool     valid;                          // Frame valido (CRC OK)
} modbus_frame_t;

/**
 * @brief Buffer di ricezione Modbus
 */
typedef struct {
    uint8_t         buffer[MODBUS_FRAME_MAX_SIZE];
    uint16_t        index;                   // Indice scrittura buffer
    uint32_t        last_rx_time;            // Timestamp ultimo byte (per T3.5)
    modbus_state_t  state;                   // Stato parser
    modbus_frame_t  frame;                   // Frame corrente
} modbus_rx_buffer_t;

/* ========================================================================== */
/* FUNZIONI PUBBLICHE                                                        */
/* ========================================================================== */

/**
 * @brief Inizializza il parser Modbus RTU
 * @param buf Puntatore al buffer di ricezione
 */
void modbus_init(modbus_rx_buffer_t *buf);

/**
 * @brief Processa un byte ricevuto
 * @param buf Puntatore al buffer di ricezione
 * @param byte Byte ricevuto via UART
 * @param timestamp_ms Timestamp corrente in millisecondi
 * @return true se frame completo disponibile, false altrimenti
 */
bool modbus_process_byte(modbus_rx_buffer_t *buf, uint8_t byte, uint32_t timestamp_ms);

/**
 * @brief Verifica se frame è completo e valido
 * @param buf Puntatore al buffer di ricezione
 * @return true se frame pronto per essere letto
 */
bool modbus_frame_ready(modbus_rx_buffer_t *buf);

/**
 * @brief Ottiene frame completo e lo resetta
 * @param buf Puntatore al buffer di ricezione
 * @param frame Puntatore a struttura destinazione
 * @return true se frame copiato con successo
 */
bool modbus_get_frame(modbus_rx_buffer_t *buf, modbus_frame_t *frame);

/**
 * @brief Calcola CRC16 Modbus (polinomio 0xA001)
 * @param data Puntatore ai dati
 * @param len Lunghezza dati
 * @return Valore CRC16
 */
uint16_t modbus_crc16(const uint8_t *data, uint16_t len);

/**
 * @brief Estrae array di float32 big-endian dal frame
 * @param frame Puntatore al frame Modbus
 * @param output Array di output (minimo 16 elementi)
 * @param max_count Numero massimo di float da estrarre
 * @return Numero di float estratti
 */
int modbus_extract_float32(const modbus_frame_t *frame, float *output, int max_count);

/**
 * @brief Resetta il buffer di ricezione
 * @param buf Puntatore al buffer di ricezione
 */
void modbus_reset(modbus_rx_buffer_t *buf);

#endif /* MODBUS_RTU_H */
