/**
 ******************************************************************************
 * @file    modbus_rtu.c
 * @brief   Modbus RTU protocol parser implementation
 ******************************************************************************
 */

#include "modbus_rtu.h"
#include <string.h>

/* ========================================================================== */
/* FUNZIONI PRIVATE                                                          */
/* ========================================================================== */

/**
 * @brief Converte 4 bytes big-endian in float32 (IEEE 754)
 */
static float bytes_to_float32_be(const uint8_t *bytes) {
    union {
        float    f;
        uint32_t u;
    } converter;
    
    // Big-endian: MSB first
    converter.u = ((uint32_t)bytes[0] << 24) |
                  ((uint32_t)bytes[1] << 16) |
                  ((uint32_t)bytes[2] << 8)  |
                  ((uint32_t)bytes[3]);
    
    return converter.f;
}

/* ========================================================================== */
/* FUNZIONI PUBBLICHE                                                        */
/* ========================================================================== */

void modbus_init(modbus_rx_buffer_t *buf) {
    if (!buf) return;
    
    memset(buf, 0, sizeof(modbus_rx_buffer_t));
    buf->state = MODBUS_STATE_IDLE;
}

bool modbus_process_byte(modbus_rx_buffer_t *buf, uint8_t byte, uint32_t timestamp_ms) {
    if (!buf) return false;
    
    uint32_t time_since_last = timestamp_ms - buf->last_rx_time;
    
    // Timeout T3.5: nuovo frame
    if (time_since_last > MODBUS_T35_MS && buf->index > 0) {
        // Frame precedente terminato (timeout)
        if (buf->index >= 5) {  // Minimo: ADDR + FUNC + LEN + CRC16
            buf->state = MODBUS_STATE_COMPLETE;
            return true;  // Frame disponibile
        } else {
            // Frame incompleto, scarta
            modbus_reset(buf);
        }
    }
    
    // Aggiungi byte al buffer
    if (buf->index < MODBUS_FRAME_MAX_SIZE) {
        buf->buffer[buf->index++] = byte;
        buf->last_rx_time = timestamp_ms;
        buf->state = MODBUS_STATE_RECEIVING;
    } else {
        // Buffer overflow, resetta
        modbus_reset(buf);
        buf->state = MODBUS_STATE_ERROR;
    }
    
    return false;
}

bool modbus_frame_ready(modbus_rx_buffer_t *buf) {
    return (buf && buf->state == MODBUS_STATE_COMPLETE);
}

bool modbus_get_frame(modbus_rx_buffer_t *buf, modbus_frame_t *frame) {
    if (!buf || !frame || buf->state != MODBUS_STATE_COMPLETE) {
        return false;
    }
    
    // Parsing frame
    if (buf->index < 5) {
        modbus_reset(buf);
        return false;
    }
    
    frame->address   = buf->buffer[0];
    frame->function  = buf->buffer[1];
    
    // Per Function Code 0x03 (Read Holding Registers):
    // Byte 2 = data length
    if (frame->function == 0x03) {
        frame->data_len = buf->buffer[2];
        
        if (frame->data_len > MODBUS_MAX_DATA_LEN || 
            buf->index < (3 + frame->data_len + 2)) {
            // Frame malformato
            modbus_reset(buf);
            return false;
        }
        
        // Copia dati
        memcpy(frame->data, &buf->buffer[3], frame->data_len);
        
        // CRC16 (ultimi 2 bytes, little-endian)
        uint16_t crc_offset = 3 + frame->data_len;
        frame->crc16_rx = (uint16_t)buf->buffer[crc_offset] | 
                         ((uint16_t)buf->buffer[crc_offset + 1] << 8);
        
        // Calcola CRC16 sui dati (esclusi CRC bytes)
        frame->crc16 = modbus_crc16(buf->buffer, crc_offset);
        
        // Verifica CRC
        frame->valid = (frame->crc16 == frame->crc16_rx);
        
        // Copia raw frame
        frame->raw_len = buf->index;
        memcpy(frame->raw, buf->buffer, buf->index);
        
    } else {
        // Function code non supportato (per ora)
        frame->valid = false;
    }
    
    // Resetta buffer per prossimo frame
    modbus_reset(buf);
    
    return frame->valid;
}

uint16_t modbus_crc16(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    
    for (uint16_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc >>= 1;
                crc ^= 0xA001;  // Polinomio Modbus
            } else {
                crc >>= 1;
            }
        }
    }
    
    return crc;
}

int modbus_extract_float32(const modbus_frame_t *frame, float *output, int max_count) {
    if (!frame || !output || !frame->valid) {
        return 0;
    }
    
    // Verifica che data_len sia multiplo di 4 (float32 = 4 bytes)
    if (frame->data_len % 4 != 0) {
        return 0;
    }
    
    int float_count = frame->data_len / 4;
    if (float_count > max_count) {
        float_count = max_count;
    }
    
    for (int i = 0; i < float_count; i++) {
        output[i] = bytes_to_float32_be(&frame->data[i * 4]);
    }
    
    return float_count;
}

void modbus_reset(modbus_rx_buffer_t *buf) {
    if (!buf) return;
    
    buf->index = 0;
    buf->state = MODBUS_STATE_IDLE;
    // Non azzeriamo last_rx_time per mantenere timing
}
