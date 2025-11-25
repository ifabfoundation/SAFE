
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "network.h"
#include "network_data.h"

/* USER CODE BEGIN includes */
extern void log_uart(const char* s);

/* ==== MODEL IO SHAPES (da report/modello) ==== */
#define FEAT 16          // <— il tuo modello usa 16 feature, NON 19
#define WIN  30          // finestre 30x16



/* ==== QUANT PARAMS dal modello TFLite ==== */
/**************************************************
 * * * * * DA CAMBIARE SE CAMBI MODELLO * * * * *
 **************************************************/
/* BEGIN qparams 2000/0.5 */
static const float IN_SCALE  = 0.170490965f;
static const int   IN_ZP     = -20;
static const float OUT_SCALE = 0.052055813f;
static const int   OUT_ZP    = 62;
/* END qparams 2000/0.5 */

/* BEGIN qparams 500/0.6 */
//static const float IN_SCALE  = 0.170490965f;
//static const int   IN_ZP     = -20;
//static const float OUT_SCALE = 0.051727000f;
//static const int   OUT_ZP    = 63;
/* END qparams 500/0.6 */
/**************************************************/



/* ==== STANDARDIZZAZIONE PARAMETERS (dal training) ==== */
static const float MEAN[FEAT] = {
  3.54232024f,   // Feature 0: bonfi/gb1_p3_acc_cfa
  -0.17791010f,  // Feature 1: bonfi/gb1_p3_acc_kurt
  0.84796573f,   // Feature 2: bonfi/gb1_p3_acc_max
  -0.82197717f,  // Feature 3: bonfi/gb1_p3_acc_min
  0.23954714f,   // Feature 4: bonfi/gb1_p3_acc_rms
  0.05416340f,   // Feature 5: bonfi/gb1_p3_acc_skew
  0.23894664f,   // Feature 6: bonfi/gb1_p3_acc_std
  48.69563293f,  // Feature 7: bonfi/gb1_p3_temp
  3.34757018f,   // Feature 8: bonfi/gb1_p4_acc_cfa
  -0.00579160f,  // Feature 9: bonfi/gb1_p4_acc_kurt
  1.10108678f,   // Feature 10: bonfi/gb1_p4_acc_max
  -1.23190808f,  // Feature 11: bonfi/gb1_p4_acc_min
  0.32999108f,   // Feature 12: bonfi/gb1_p4_acc_rms
  -0.16345859f,  // Feature 13: bonfi/gb1_p4_acc_skew
  0.32905198f,   // Feature 14: bonfi/gb1_p4_acc_std
  51.14997864f   // Feature 15: bonfi/gb1_p4_temp
};

static const float STD[FEAT] = {
  0.28740004f,   // Feature 0
  0.20240284f,   // Feature 1
  0.09566389f,   // Feature 2
  0.09332504f,   // Feature 3
  0.02045762f,   // Feature 4
  0.06076196f,   // Feature 5
  0.02033368f,   // Feature 6
  2.09906530f,   // Feature 7
  0.29434454f,   // Feature 8
  0.17435325f,   // Feature 9
  0.12140677f,   // Feature 10
  0.14222031f,   // Feature 11
  0.03369605f,   // Feature 12
  0.06418782f,   // Feature 13
  0.03364391f,   // Feature 14
  2.06709480f    // Feature 15
};

static float window[WIN][FEAT];
static int   w_count = 0;      // quante righe caricate (0..30)
static int   window_ready = 0; // flag
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_NETWORK_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_NETWORK_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle network = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  // commento cosi non si blocca
  //do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_create_and_init");
    return -1;
  }

  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

#if defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_network_run(network, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_network_get_error(network),
        "ai_network_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */

/**
 * @brief Push una riga di 16 valori RAW nella finestra
 * @param v Array di FEAT valori RAW (NON standardizzati)
 * @return 0 se accettata, 1 se finestra completa
 *
 * IMPORTANTE: Questa funzione ora STANDARDIZZA i dati in ingresso
 * applicando (x - MEAN) / STD prima di salvarli nella finestra.
 */
int ai_push_line_of_16(const float v[FEAT]) {
  if (w_count < WIN) {
    // Standardizza ogni feature: z = (x - μ) / σ
    for (int i = 0; i < FEAT; i++) {
      window[w_count][i] = (v[i] - MEAN[i]) / STD[i];
    }
    w_count++;

    if (w_count == WIN) {
      window_ready = 1;
      log_vcp("\r\n[AI] *** FINESTRA 30/30 COMPLETA → INFERENZA ***\r\n");
    }
    return 0;
  } else {
    // se arriva extra, shift finestra (sliding) e resta pronta
    memmove(&window[0], &window[1], (WIN-1)*FEAT*sizeof(float));
    // Standardizza anche qui
    for (int i = 0; i < FEAT; i++) {
      window[WIN-1][i] = (v[i] - MEAN[i]) / STD[i];
    }
    window_ready = 1;
    return 0;
  }
}
/* USER CODE END 2 */
static int acquire_and_process_data(ai_i8 **data_in)
{
  (void)data_in;
  if (!window_ready) return -1;

  ai_buffer* in_buf = ai_network_inputs_get(network, NULL);
  if (!in_buf) return -1;

  /* Verifica dimensione attesa dell'input: 1x30x16 int8 = 480 byte */
  const uint32_t want_bytes = WIN * FEAT * sizeof(int8_t);
  if (in_buf[0].size != want_bytes && in_buf[0].format != AI_BUFFER_FORMAT_FLOAT) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "[ERR] Input bytes=%lu attesi=%lu (non-FLOAT)\r\n",
             (unsigned long)in_buf[0].size, (unsigned long)want_bytes);
    log_uart(msg);
    return -1;
  }

  /* Caso A: modello FLOAT32 (raro nel tuo setup) */
  if (in_buf[0].format == AI_BUFFER_FORMAT_FLOAT) {
    float *dst = (float*)in_buf[0].data;
    memcpy(dst, window, sizeof(window));  // [WIN][FEAT]
  }
  /* Caso B: modello INT8 (il tuo) — quantizzazione CORRETTA */
  else {
    int8_t *dst = (int8_t*)in_buf[0].data;
    size_t k = 0;
    for (int r = 0; r < WIN; ++r) {
      for (int c = 0; c < FEAT; ++c) {
        float x = window[r][c];
        // Se i tuoi dati arrivano già standardizzati (tipo z-score),
        // NON clampare a [0,1]. Si quantizza direttamente con IN_SCALE/IN_ZP.
        int q = (int)lrintf(x / IN_SCALE + (float)IN_ZP);
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        dst[k++] = (int8_t)q;
      }
    }
  }

  return 0;  // finestra pronta per ai_run()
}


int post_process(ai_i8* data_outs_local[])
{
  (void)data_outs_local;

  ai_buffer* out_buf = ai_network_outputs_get(network, NULL);
  if (!out_buf) return -1;

  uint32_t out_bytes = out_buf[0].size;

  // Output quantizzato S8 → dequant e stampa
  int8_t  *q8  = (int8_t*)out_buf[0].data;
  uint32_t out_len = out_bytes / sizeof(int8_t);

  char msg[256];
  int o = snprintf(msg, sizeof(msg), "[RX] Inferenza OK (Q→F): ");
  for (uint32_t i = 0; i < out_len; ++i) {
    float y = ( (float)q8[i] - (float)OUT_ZP ) * OUT_SCALE;
    o += snprintf(msg + o, sizeof(msg) - o, "%.3f ", (double)y);
    if (o > (int)sizeof(msg) - 16) {
      log_vcp(msg);
      o = snprintf(msg, sizeof(msg), "    ");
    }
  }
  snprintf(msg + o, sizeof(msg) - o, "\r\n");
  log_vcp(msg);

  // reset finestra per la prossima inferenza
  w_count = 0;
  window_ready = 0;
  log_vcp("[AI] *** RESET w_count=0, pronto per nuova finestra ***\r\n\r\n");
  return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = -1;

  //printf("TEMPLATE - run - main loop\r\n");

  if (network) {
    /* 1 - acquire and pre-process input data */
    res = acquire_and_process_data(data_ins);
    /* 2 - process the data - call inference engine */
    if (res == 0)
      res = ai_run();
    /* 3- post-process the predictions */
    if (res == 0)
      res = post_process(data_outs);
  }

  return;

  // commento cosi non blocca
//  if (res) {
//    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
//    ai_log_err(err, "Process has FAILED");
//  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
