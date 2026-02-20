/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>   // per strtof
#include "modbus_rtu.h"  // Parser Modbus RTU
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RS485_DE_GPIO_Port GPIOA
#define RS485_DE_Pin       GPIO_PIN_1

static inline void RS485_RX(void) { HAL_GPIO_WritePin(RS485_DE_GPIO_Port, RS485_DE_Pin, GPIO_PIN_RESET); }
static inline void RS485_TX(void) { HAL_GPIO_WritePin(RS485_DE_GPIO_Port, RS485_DE_Pin, GPIO_PIN_SET); }

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

UART_HandleTypeDef hlpuart1;
UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
extern int ai_push_line_of_16(const float v[16]);
uint8_t rx_byte;
char    rx_line[512];  // Buffer aumentato per Modbus (69 byte frame)
uint16_t rx_len = 0;

volatile uint8_t line_ready = 0;
char line_copy[512];

static uint32_t rx_bytes = 0;

// Modbus RTU - buffer raw + timestamp
#define MODBUS_RX_BUFFER_SIZE 256
uint8_t modbus_rx_buffer[MODBUS_RX_BUFFER_SIZE];
uint32_t modbus_rx_timestamps[MODBUS_RX_BUFFER_SIZE];  // Timestamp per ogni byte!
volatile uint16_t modbus_rx_head = 0;
volatile uint16_t modbus_rx_tail = 0;
volatile uint32_t modbus_last_rx_time = 0;

// Modbus RTU
modbus_rx_buffer_t modbus_buf;
volatile uint8_t modbus_frame_ready_flag = 0;

// Modalità protocollo: 0=ASCII/CSV, 1=Modbus RTU
#define PROTOCOL_MODE_ASCII   0
#define PROTOCOL_MODE_MODBUS  1
static uint8_t protocol_mode = PROTOCOL_MODE_MODBUS;  // Default: Modbus

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void SystemPower_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_ICACHE_Init(void);
static void MX_LPUART1_UART_Init(void);
/* USER CODE BEGIN PFP */
void log_rs485(const char* s);
void log_vcp(const char* s);
void log_uart(const char* s);
static int parse_and_push_line(char* line);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the System Power */
  SystemPower_Config();

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_ICACHE_Init();
  MX_LPUART1_UART_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
//  RS485_TX();  // DE/RE alto fisso: abilita trasmettitore, disabilita ricevitore
//  const char *msg = "[RS485] PING\r\n";
  RS485_TX();  // DE/RE alto fisso → solo trasmissione
  const char *msg = "[RS485] PING\r\n";
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  RS485_RX();                // parti in ascolto
  HAL_UART_Receive_IT(&hlpuart1, &rx_byte, 1);

  // Inizializza parser Modbus
  modbus_init(&modbus_buf);
  
  // Banner startup
  log_vcp("\r\n");
  log_vcp("╔════════════════════════════════════════════════════════════╗\r\n");
  log_vcp("║  STM32U545RE - SAFE Predictive Maintenance System         ║\r\n");
  log_vcp("║  Firmware v2.1 - MODBUS RTU + ON-BOARD STANDARDIZATION    ║\r\n");
  log_vcp("╠════════════════════════════════════════════════════════════╣\r\n");
  log_vcp("║  Model: CNN+LSTM mt500_b0p6 (INT8, 141.99 KB)             ║\r\n");
  log_vcp("║  Input: 30x16 RAW → Z-Score → Quantized INT8              ║\r\n");
  log_vcp("║  Protocol: Modbus RTU (115200 baud, 8N1)                  ║\r\n");
  log_vcp("╚════════════════════════════════════════════════════════════╝\r\n");
  
  if (protocol_mode == PROTOCOL_MODE_MODBUS) {
    log_vcp("[MODE] Modbus RTU - Listening on RS-485...\r\n\r\n");
    // Non trasmettere sul bus RS-485 in ascolto per evitare collisioni
    log_vcp("[READY] STM32U545RE v2.1 - MODBUS RTU MODE\r\n");
  } else {
    log_vcp("[MODE] ASCII/CSV - Debug mode\r\n");
    log_vcp("[DBG] Invia 16 valori + \\n su /dev/ttyUSB0\r\n\r\n");
    log_rs485("[RX] Board pronta. Invia 16 valori separati da virgola, termina con \\n\r\n");
  }
  
  static uint32_t ok_count = 0;
  static uint32_t modbus_frame_count = 0;

  while (1)
  {
	  // === GESTIONE MODBUS RTU ===
	  if (protocol_mode == PROTOCOL_MODE_MODBUS) {
	    // DEBUG: conta bytes nel buffer
	    static uint32_t last_debug = 0;
	    if (HAL_GetTick() - last_debug > 5000) {  // Ogni 5 secondi
	      uint16_t bytes_in_buffer = (modbus_rx_head >= modbus_rx_tail) ? 
	                                  (modbus_rx_head - modbus_rx_tail) : 
	                                  (MODBUS_RX_BUFFER_SIZE - modbus_rx_tail + modbus_rx_head);
	      char dbg[64];
	      snprintf(dbg, sizeof(dbg), "[DBG] Buffer: %u bytes, head=%u tail=%u\r\n", 
	               bytes_in_buffer, modbus_rx_head, modbus_rx_tail);
	      log_vcp(dbg);
	      last_debug = HAL_GetTick();
	    }
	    
	    // Processa bytes dal buffer circolare
	    while (modbus_rx_tail != modbus_rx_head) {
	      uint8_t byte = modbus_rx_buffer[modbus_rx_tail];
	      uint32_t timestamp = modbus_rx_timestamps[modbus_rx_tail];  // Timestamp REALE!
	      modbus_rx_tail = (modbus_rx_tail + 1) % MODBUS_RX_BUFFER_SIZE;
	      
	      if (modbus_process_byte(&modbus_buf, byte, timestamp)) {
	        // Frame completo!
	        modbus_frame_t frame;
	        if (modbus_get_frame(&modbus_buf, &frame)) {
	          // Filtra per indirizzo e function code
	          if (frame.valid && frame.address == MODBUS_SLAVE_ADDR && frame.function == MODBUS_FUNCTION_CODE) {
	            // Estrai 16 float32 RAW
	            float raw_features[16];
	            int count = modbus_extract_float32(&frame, raw_features, 16);
	            
	            if (count == 16) {
	              // Push al modello (con standardizzazione on-board!)
	              ai_push_line_of_16(raw_features);
                // Log minimale su VCP per visibilità (una riga ogni 2s)
                static uint32_t good_frames = 0;
                good_frames++;
                char line[64];
                snprintf(line, sizeof(line), "[MODBUS] Frame OK #%lu (len=%u)\r\n", good_frames, frame.data_len);
                log_vcp(line);
	            }
	          }
	        }
	      }
	    }
	  }
	  
	  // === GESTIONE ASCII/CSV (legacy) ===
	  if (protocol_mode == PROTOCOL_MODE_ASCII && line_ready) {
	    __disable_irq();
	    strncpy(line_copy, rx_line, sizeof(line_copy));
	    line_copy[sizeof(line_copy)-1] = '\0';
	    line_ready = 0;
	    __enable_irq();

	    int r = parse_and_push_line(line_copy);
	    log_rs485("[RX] OK: riga accettata\r\n");
	      // se vuoi tenere anche il contatore x5, lascialo sotto
	      if ((++ok_count % 5U) == 0U) log_rs485("[RX] OK: riga accettata\r\n");
	  }
	  
	  // === ESEGUI INFERENZA SE FINESTRA PRONTA ===
	  MX_X_CUBE_AI_Process();



    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  // NO DELAY! Deve girare alla massima velocità per processare buffer circolare!

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLMBOOST = RCC_PLLMBOOST_DIV1;
  RCC_OscInitStruct.PLL.PLLM = 2;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLLVCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Power Configuration
  * @retval None
  */
static void SystemPower_Config(void)
{

  /*
   * Switch to SMPS regulator instead of LDO
   */
  if (HAL_PWREx_ConfigSupply(PWR_SMPS_SUPPLY) != HAL_OK)
  {
    Error_Handler();
  }
/* USER CODE BEGIN PWR */
/* USER CODE END PWR */
}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache (default 2-ways set associative cache)
  */
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief LPUART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_LPUART1_UART_Init(void)
{

  /* USER CODE BEGIN LPUART1_Init 0 */

  /* USER CODE END LPUART1_Init 0 */

  /* USER CODE BEGIN LPUART1_Init 1 */

  /* USER CODE END LPUART1_Init 1 */
  hlpuart1.Instance = LPUART1;
  hlpuart1.Init.BaudRate = 115200;  // Company standard for Modbus RTU
  hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
  hlpuart1.Init.StopBits = UART_STOPBITS_1;
  hlpuart1.Init.Parity = UART_PARITY_NONE;
  hlpuart1.Init.Mode = UART_MODE_TX_RX;
  hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  hlpuart1.FifoMode = UART_FIFOMODE_DISABLE;
  if (HAL_UART_Init(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LPUART1_Init 2 */

  /* USER CODE END LPUART1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1, GPIO_PIN_RESET);

  /*Configure GPIO pin : PA1 */
  GPIO_InitStruct.Pin = GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void log_rs485(const char* s) {
  RS485_TX();
  HAL_UART_Transmit(&hlpuart1, (uint8_t*)s, strlen(s), HAL_MAX_DELAY);
  // aspetta fine TX per non tagliare l'ultima byte prima di tornare in RX
  while (__HAL_UART_GET_FLAG(&hlpuart1, UART_FLAG_TC) == RESET) {}
  RS485_RX();
}

void log_vcp(const char* s) {
  extern UART_HandleTypeDef huart1;
  HAL_UART_Transmit(&huart1, (uint8_t*)s, strlen(s), HAL_MAX_DELAY);
}

//wrapper al volo da togliere
void log_uart(const char* s) {
  // compatibilità con codice Cube.AI
  log_rs485(s);   // se vuoi far passare tutto su RS-485
  // oppure log_vcp(s); se vuoi mantenerlo sul VCP
}



static int parse_and_push_line(char* line) {
  // atteso: "v0,v1,...,v18\n"
  float v[16];
  int count = 0;

  for (char* p = line; *p; ) {
    char* end;
    float val = strtof(p, &end);
    if (end == p) break;
    if (count < 16) v[count++] = val;
    if (*end == ',') { p = end + 1; }
    else { p = end; if (*p == '\n' || *p == '\r') p++; }
  }

  if (count != 16) {
    char msg[96];
    snprintf(msg, sizeof(msg), "[RX] ERR parse: attesi 16 valori, trovati %d\r\n", count);
    log_uart(msg);
    return -1;
  }

  /* BEGIN patch main.c — rimuovi clamp z-score */
    // opzionale: clamp 0..1  ← NO
    // for (int i=0;i<16;i++) {
    //   if (v[i] < 0.f) v[i] = 0.f;
    //   if (v[i] > 1.f) v[i] = 1.f;
    // }
  /* END patch main.c */


  return ai_push_line_of_16(v);
}


//void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
//  if (huart->Instance == USART1) {
//    char c = (char)rx_byte;
//    extern UART_HandleTypeDef huart1;
//    //HAL_UART_Transmit(&huart1, (uint8_t*)&c, 1, 0xFFFF);  // echo del carattere ricevuto
//    if (c == '\n' || c == '\r') {
//      if (rx_len > 0) {
//        rx_line[rx_len] = '\0';
//        parse_and_push_line(rx_line);
//
//         log_uart("OK: riga accettata\r\n");
//
//
//        rx_len = 0;
//      }
//    } else {
//      if (rx_len < sizeof(rx_line)-1) rx_line[rx_len++] = c;
//      // se overflow, reset linea
//      else rx_len = 0;
//    }
//    HAL_UART_Receive_IT(huart, &rx_byte, 1); // ri-arma RX
//  }
//}



void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance == LPUART1) {
    
    if (protocol_mode == PROTOCOL_MODE_MODBUS) {
      // === MODALITÀ MODBUS RTU - VELOCISSIMO! ===
      // Salva byte + timestamp nel buffer circolare
      uint32_t now = HAL_GetTick();
      modbus_rx_buffer[modbus_rx_head] = rx_byte;
      modbus_rx_timestamps[modbus_rx_head] = now;
      modbus_rx_head = (modbus_rx_head + 1) % MODBUS_RX_BUFFER_SIZE;
      modbus_last_rx_time = now;
      
      // NO LOG QUI! Troppo lento! Il log blocca per ms!
      
    } else {
      // === MODALITÀ ASCII/CSV (legacy) ===
      char c = (char)rx_byte;
      
      if (c == '\n') {
        if (rx_len > 0 && !line_ready) {
          rx_line[rx_len] = '\0';
          line_ready = 1;
        }
        rx_len = 0;
      } else if (c != '\r') {
        if (rx_len < sizeof(rx_line)-1) rx_line[rx_len++] = c;
        else rx_len = 0; // overflow → reset
      }
    }

    HAL_UART_Receive_IT(&hlpuart1, &rx_byte, 1);
  }
}




/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
