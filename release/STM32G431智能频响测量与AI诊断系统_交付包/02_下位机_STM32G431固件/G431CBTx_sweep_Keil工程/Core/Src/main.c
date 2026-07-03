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
#include "adc.h"
#include "dac.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdbool.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "arm_math.h"
#include "arm_const_structs.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */



/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define adc_buffer_size 256
#define dac_buffer_size 1024
#define MAX_FREQ_STEPS 200
#define TIM1_ADC_PRESCALER 0U
#define TIM1_COUNTER_CLK_HZ 170000000.0f
#define TIM1_MAX_AUTORELOAD 0xFFFFU
#define TIM2_DAC_PRESCALER 0U
#define TIM2_COUNTER_CLK_HZ 170000000.0f
#define ADC_COHERENT_CYCLES 16U
#define ADC_SAMPLES_PER_SIGNAL_PERIOD ((float32_t)adc_buffer_size / (float32_t)ADC_COHERENT_CYCLES)
#define SETTLING_SIGNAL_PERIODS 12.0f
#define SWEEP_CAPTURE_REPEATS 3U
#define ADC_CAPTURE_GUARD_MS 20U
#define UART_CMD_BUFFER_SIZE 96U
#define SWEEP_DEFAULT_F_START 100.0f
#define SWEEP_DEFAULT_F_STOP 20000.0f
#define SWEEP_DEFAULT_F_STEP 100.0f
#define SWEEP_DEFAULT_AMPLITUDE_VPP 1.2f
#define SWEEP_MAX_AMPLITUDE_VPP 3.0f

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
// --- 扫频配置结构体 ---
typedef struct {
    float32_t f_start;          // 起始频率 (Hz)
    float32_t f_stop;           // 终止频率 (Hz)
    float32_t f_step;           // 频率步进值 (Hz)
		float32_t amplitude_V;
    float32_t dac_sample_rate;  // DAC采样率 (Hz)
    uint16_t DAC_MAX_VAL;       // DAC最大值 (例如 4095)
		float32_t adc_sample_rate;
    uint32_t adc_buffersize; // 每个频点需要采集的总点数
	  float32_t f_multiplier;
		uint16_t num_log_steps;
} StepSweepConfig_t;

typedef enum {
	SWEEP_CMD_EMPTY = 0,
	SWEEP_CMD_PING,
	SWEEP_CMD_HELP,
	SWEEP_CMD_DEFAULT,
	SWEEP_CMD_STOP,
	SWEEP_CMD_SWEEP,
	SWEEP_CMD_INVALID_SWEEP,
	SWEEP_CMD_UNKNOWN
} SweepCommandId_t;

typedef enum {
	SWEEP_RUN_DONE = 0,
	SWEEP_RUN_STOPPED
} SweepRunStatus_t;

typedef enum {
	STEP_CAPTURE_TIMEOUT = 0,
	STEP_CAPTURE_DONE,
	STEP_CAPTURE_STOPPED
} StepSweepCaptureStatus_t;

typedef struct {
	SweepCommandId_t id;
	StepSweepConfig_t requested_config;
} SweepCommand_t;

// --- 运行时数据 ---
typedef struct {
    float32_t current_freq;
    uint32_t total_steps;
    uint16_t dac_buffer[dac_buffer_size]; // dac_buffer_size = 1024
    uint16_t adc1_buffer[adc_buffer_size];
    uint16_t adc2_buffer[adc_buffer_size];
    float32_t phase;
    float32_t gain;
		uint16_t new_arr_value;
    float32_t adc_actual_sample_rate;
    uint32_t dac_timer_arr;
    float32_t dac_actual_sample_rate;
    float32_t actual_freq;
    uint32_t dac_actual_length; // <-- 新增: 实际使用的 DAC 缓冲区长度 (N_actual)
    uint32_t dac_periods;       // <-- 新增: 缓冲区内包含的整数周期数 (M)
    float32_t adc1_rms_v;
    float32_t adc2_rms_v;
    float32_t adc1_dc_v;
    float32_t adc2_dc_v;
    float32_t adc1_pp_v;
    float32_t adc2_pp_v;
    uint16_t adc1_min_code;
    uint16_t adc1_max_code;
    uint16_t adc2_min_code;
    uint16_t adc2_max_code;
    uint16_t clip_flags;
    uint32_t adc_timer_prescaler;
} StepSweepData_t;

typedef struct {
    float32_t gain_ratio;     // 增益 (幅度比: Mag2 / Mag1)
    float32_t phase_diff_rad; // 相位差 (Phase2 - Phase1)
} AnalysisOutput_t;

typedef struct {
    uint16_t input_min_code_all;
    uint16_t input_max_code_all;
    uint16_t output_min_code_all;
    uint16_t output_max_code_all;
    uint16_t input_clip_points;
    uint16_t output_clip_points;
} SweepOutputSummary_t;

float32_t fft_input1[adc_buffer_size*2];
float32_t fft_input2[adc_buffer_size*2];
struct PhaseData {
    float32_t phase_unwrapped;  // 解缠绕后的相位差
    float32_t last_phase_raw;   // 上一次原始相位差（未解缠绕）
    float32_t unwrap_offset;    // 累积解缠绕偏移量
    int first_point;            // 标记是否为第一个点
};


/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
volatile bool adc1_flag=0,adc2_flag=0,adc_flag=0;
StepSweepConfig_t config;
StepSweepData_t data;

struct PhaseData data_phase = {
    .phase_unwrapped = 0.0f,
    .last_phase_raw = 0.0f,
    .unwrap_offset = 0.0f,
    .first_point = 1
};

uint16_t GainQ15[MAX_FREQ_STEPS];
int16_t PhaseMilliRad[MAX_FREQ_STEPS];
uint32_t OmegaMilliRadPerSec[MAX_FREQ_STEPS];
uint16_t InputRmsMv[MAX_FREQ_STEPS];
uint16_t OutputRmsMv[MAX_FREQ_STEPS];
uint16_t InputDcMv[MAX_FREQ_STEPS];
uint16_t OutputDcMv[MAX_FREQ_STEPS];
uint16_t InputPpMv[MAX_FREQ_STEPS];
uint16_t OutputPpMv[MAX_FREQ_STEPS];
uint16_t ClipFlags[MAX_FREQ_STEPS];
uint16_t ValidCaptureCount[MAX_FREQ_STEPS];
uint32_t ActualFreqMilliHz[MAX_FREQ_STEPS];
uint32_t AdcSampleRateMilliHz[MAX_FREQ_STEPS];
uint32_t DacSampleRateMilliHz[MAX_FREQ_STEPS];
uint16_t MagnitudeRepeatSpanMilliDb[MAX_FREQ_STEPS];
uint16_t PhaseRepeatSpanMilliRad[MAX_FREQ_STEPS];
uint16_t InputMinCode[MAX_FREQ_STEPS];
uint16_t InputMaxCode[MAX_FREQ_STEPS];
uint16_t OutputMinCode[MAX_FREQ_STEPS];
uint16_t OutputMaxCode[MAX_FREQ_STEPS];
static volatile bool sweep_stop_requested = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
SweepRunStatus_t Sweep_ExecuteStepSweep(StepSweepConfig_t *config, StepSweepData_t *data);
static void SweepProtocol_CheckStopRequest(void);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int fputc(int ch, FILE *f) {
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

static void StepSweep_SetDefaultConfig(StepSweepConfig_t *cfg)
{
	cfg->f_start = SWEEP_DEFAULT_F_START;
	cfg->f_stop = SWEEP_DEFAULT_F_STOP;
	cfg->f_step = SWEEP_DEFAULT_F_STEP;
	cfg->amplitude_V = SWEEP_DEFAULT_AMPLITUDE_VPP;
	cfg->dac_sample_rate = 1000000.0f;
	cfg->DAC_MAX_VAL = 4095;
	cfg->adc_sample_rate = 10000.0f;
	cfg->adc_buffersize = adc_buffer_size;
	cfg->f_multiplier = 0.0f;
	cfg->num_log_steps = 200;
}

static void StepSweep_ResetRuntimeData(StepSweepData_t *state, const StepSweepConfig_t *cfg)
{
	state->current_freq = cfg->f_start;
	state->total_steps = 0;
	state->phase = 0.0f;
	state->gain = 0.0f;
	state->new_arr_value = 999;
	state->adc_actual_sample_rate = cfg->adc_sample_rate;
	state->dac_timer_arr = 169;
	state->dac_actual_sample_rate = cfg->dac_sample_rate;
	state->actual_freq = cfg->f_start;
	state->dac_actual_length = dac_buffer_size;
	state->dac_periods = 1;
	state->adc1_rms_v = 0.0f;
	state->adc2_rms_v = 0.0f;
	state->adc1_dc_v = 0.0f;
	state->adc2_dc_v = 0.0f;
	state->adc1_pp_v = 0.0f;
	state->adc2_pp_v = 0.0f;
	state->adc1_min_code = 0U;
	state->adc1_max_code = 0U;
	state->adc2_min_code = 0U;
	state->adc2_max_code = 0U;
	state->clip_flags = 0U;
	state->adc_timer_prescaler = TIM1_ADC_PRESCALER;
}

static void StepSweep_ResetPhaseTracker(void)
{
	data_phase.phase_unwrapped = 0.0f;
	data_phase.last_phase_raw = 0.0f;
	data_phase.unwrap_offset = 0.0f;
	data_phase.first_point = 1;
}

static void StepSweep_ClearPointResult(uint16_t index)
{
	GainQ15[index] = 0U;
	PhaseMilliRad[index] = 0;
	OmegaMilliRadPerSec[index] = 0UL;
	InputRmsMv[index] = 0U;
	OutputRmsMv[index] = 0U;
	InputDcMv[index] = 0U;
	OutputDcMv[index] = 0U;
	InputPpMv[index] = 0U;
	OutputPpMv[index] = 0U;
	ClipFlags[index] = 0U;
	ValidCaptureCount[index] = 0U;
	ActualFreqMilliHz[index] = 0UL;
	AdcSampleRateMilliHz[index] = 0UL;
	DacSampleRateMilliHz[index] = 0UL;
	MagnitudeRepeatSpanMilliDb[index] = 0U;
	PhaseRepeatSpanMilliRad[index] = 0U;
	InputMinCode[index] = 0U;
	InputMaxCode[index] = 0U;
	OutputMinCode[index] = 0U;
	OutputMaxCode[index] = 0U;
}

static void StepSweep_StopPeripherals(void)
{
	HAL_TIM_Base_Stop(&htim1);
	HAL_TIM_Base_Stop(&htim2);
	HAL_ADC_Stop_DMA(&hadc1);
	HAL_ADC_Stop_DMA(&hadc2);
	HAL_DAC_Stop_DMA(&hdac1, DAC_CHANNEL_1);
	adc1_flag = false;
	adc2_flag = false;
	adc_flag = false;
}

static void SweepOutputSummary_Init(SweepOutputSummary_t *summary)
{
	summary->input_min_code_all = 0xFFFFU;
	summary->input_max_code_all = 0U;
	summary->output_min_code_all = 0xFFFFU;
	summary->output_max_code_all = 0U;
	summary->input_clip_points = 0U;
	summary->output_clip_points = 0U;
}

static void SweepOutputSummary_AddPoint(
	SweepOutputSummary_t *summary,
	uint16_t input_min_code,
	uint16_t input_max_code,
	uint16_t output_min_code,
	uint16_t output_max_code,
	uint16_t clip_flags)
{
	if (input_min_code < summary->input_min_code_all) {
		summary->input_min_code_all = input_min_code;
	}
	if (input_max_code > summary->input_max_code_all) {
		summary->input_max_code_all = input_max_code;
	}
	if (output_min_code < summary->output_min_code_all) {
		summary->output_min_code_all = output_min_code;
	}
	if (output_max_code > summary->output_max_code_all) {
		summary->output_max_code_all = output_max_code;
	}
	if ((clip_flags & 0x01U) != 0U) {
		summary->input_clip_points++;
	}
	if ((clip_flags & 0x02U) != 0U) {
		summary->output_clip_points++;
	}
}

static void StepSweep_ApplyTimerSettings(const StepSweepData_t *state)
{
	HAL_DAC_Stop_DMA(&hdac1, DAC_CHANNEL_1);
	HAL_TIM_Base_Stop(&htim1);
	HAL_TIM_Base_Stop(&htim2);

	__HAL_TIM_SET_PRESCALER(&htim1, state->adc_timer_prescaler);
	__HAL_TIM_SET_AUTORELOAD(&htim1, state->new_arr_value);
	__HAL_TIM_SET_PRESCALER(&htim2, TIM2_DAC_PRESCALER);
	__HAL_TIM_SET_AUTORELOAD(&htim2, state->dac_timer_arr);
	__HAL_TIM_SET_COUNTER(&htim1, 0);
	__HAL_TIM_SET_COUNTER(&htim2, 0);
	HAL_TIM_GenerateEvent(&htim1, TIM_EVENTSOURCE_UPDATE);
	HAL_TIM_GenerateEvent(&htim2, TIM_EVENTSOURCE_UPDATE);
	__HAL_TIM_SET_COUNTER(&htim1, 0);
	__HAL_TIM_SET_COUNTER(&htim2, 0);
}

static uint32_t StepSweep_SettleTimeMs(float32_t frequency_hz)
{
	uint32_t settle_time_ms = (uint32_t)ceilf((SETTLING_SIGNAL_PERIODS / frequency_hz) * 1000.0f);
	if (settle_time_ms < 2U) {
		settle_time_ms = 2U;
	}
	return settle_time_ms;
}

static uint32_t StepSweep_CaptureTimeoutMs(const StepSweepConfig_t *cfg, const StepSweepData_t *state)
{
	uint32_t capture_time_ms = (uint32_t)ceilf(
		((float32_t)cfg->adc_buffersize / state->adc_actual_sample_rate) * 1000.0f
	) + ADC_CAPTURE_GUARD_MS;
	if (capture_time_ms < 5U) {
		capture_time_ms = 5U;
	}
	return capture_time_ms;
}

static void StepSweep_ConfigureAdcTimer(StepSweepData_t *state, float32_t target_sample_rate)
{
	if (target_sample_rate < 1.0f) {
		target_sample_rate = 1.0f;
	}

	float32_t min_divider =
		TIM1_COUNTER_CLK_HZ / (target_sample_rate * ((float32_t)TIM1_MAX_AUTORELOAD + 1.0f));
	uint32_t prescaler = 0U;
	if (min_divider > 1.0f) {
		prescaler = (uint32_t)ceilf(min_divider) - 1U;
	}
	if (prescaler > TIM1_MAX_AUTORELOAD) {
		prescaler = TIM1_MAX_AUTORELOAD;
	}

	float32_t counter_clk = TIM1_COUNTER_CLK_HZ / ((float32_t)prescaler + 1.0f);
	uint32_t arr_candidate = (uint32_t)roundf((counter_clk / target_sample_rate) - 1.0f);
	while (arr_candidate > TIM1_MAX_AUTORELOAD && prescaler < TIM1_MAX_AUTORELOAD) {
		prescaler++;
		counter_clk = TIM1_COUNTER_CLK_HZ / ((float32_t)prescaler + 1.0f);
		arr_candidate = (uint32_t)roundf((counter_clk / target_sample_rate) - 1.0f);
	}

	if (arr_candidate < 1U) {
		arr_candidate = 1U;
	}
	if (arr_candidate > TIM1_MAX_AUTORELOAD) {
		arr_candidate = TIM1_MAX_AUTORELOAD;
	}

	state->adc_timer_prescaler = prescaler;
	state->new_arr_value = (uint16_t)arr_candidate;
	state->adc_actual_sample_rate =
		TIM1_COUNTER_CLK_HZ / (((float32_t)prescaler + 1.0f) * ((float32_t)state->new_arr_value + 1.0f));
}

static StepSweepCaptureStatus_t StepSweep_WaitForAdcCapture(uint32_t timeout_ms)
{
	uint32_t start_tick = HAL_GetTick();
	while (!adc_flag) {
		SweepProtocol_CheckStopRequest();
		if (sweep_stop_requested) {
			StepSweep_StopPeripherals();
			return STEP_CAPTURE_STOPPED;
		}
		if ((HAL_GetTick() - start_tick) >= timeout_ms) {
			return STEP_CAPTURE_TIMEOUT;
		}
	}
	return STEP_CAPTURE_DONE;
}

static bool StepSweep_ValidateConfig(const StepSweepConfig_t *cfg)
{
	if (cfg->f_start <= 0.0f || cfg->f_stop < cfg->f_start || cfg->f_step <= 0.0f) {
		printf("ERR invalid frequency range\r\n");
		return false;
	}

	if (cfg->amplitude_V <= 0.0f || cfg->amplitude_V > SWEEP_MAX_AMPLITUDE_VPP) {
		printf("ERR amplitude must be 0..%.2f Vpp\r\n", SWEEP_MAX_AMPLITUDE_VPP);
		return false;
	}

	uint32_t steps = (uint32_t)((cfg->f_stop - cfg->f_start) / cfg->f_step) + 1U;
	if (steps == 0U) {
		printf("ERR empty sweep\r\n");
		return false;
	}
	if (steps > MAX_FREQ_STEPS) {
		printf("WARN sweep has %lu points, firmware will keep first %u points\r\n",
			(unsigned long)steps, (unsigned int)MAX_FREQ_STEPS);
	}

	return true;
}

static char *SkipSpaces(char *s)
{
	while (*s == ' ' || *s == '\t') {
		s++;
	}
	return s;
}

static bool ParseNextFloat(char **cursor, float32_t *value)
{
	char *start = SkipSpaces(*cursor);
	char *end = start;
	double parsed = strtod(start, &end);
	if (end == start) {
		return false;
	}
	*value = (float32_t)parsed;
	*cursor = end;
	return true;
}

static bool IsCommandBoundary(char ch)
{
	return (ch == '\0' || ch == ' ' || ch == '\t');
}

static bool CommandMatches(const char *cmd, const char *keyword)
{
	size_t keyword_len = strlen(keyword);
	return (strncmp(cmd, keyword, keyword_len) == 0 && IsCommandBoundary(cmd[keyword_len]));
}

static SweepCommand_t SweepCommand_Empty(void)
{
	SweepCommand_t command;
	command.id = SWEEP_CMD_EMPTY;
	command.requested_config = config;
	return command;
}

static bool SweepProtocol_ReadLine(char *line, uint16_t line_size)
{
	uint16_t pos = 0;
	uint8_t ch = 0;

	if (line_size == 0U) {
		return false;
	}

	while (1) {
		if (HAL_UART_Receive(&huart1, &ch, 1, HAL_MAX_DELAY) != HAL_OK) {
			line[0] = '\0';
			return false;
		}

		if (ch == '\r') {
			continue;
		}

		if (ch == '\n') {
			line[pos] = '\0';
			return true;
		}

		if (pos < (uint16_t)(line_size - 1U)) {
			line[pos++] = (char)ch;
		}
	}
}

static bool SweepProtocol_ReadLineNonBlocking(char *line, uint16_t line_size)
{
	static char pending[UART_CMD_BUFFER_SIZE];
	static uint16_t pos = 0;
	uint8_t ch = 0;

	if (line_size == 0U) {
		return false;
	}

	while (HAL_UART_Receive(&huart1, &ch, 1, 0U) == HAL_OK) {
		if (ch == '\r') {
			continue;
		}
		if (ch == '\n') {
			pending[pos] = '\0';
			strncpy(line, pending, line_size - 1U);
			line[line_size - 1U] = '\0';
			pos = 0U;
			return true;
		}
		if (pos < (uint16_t)(sizeof(pending) - 1U)) {
			pending[pos++] = (char)ch;
		}
		if (pos >= (uint16_t)(sizeof(pending) - 1U)) {
			pending[pos] = '\0';
			strncpy(line, pending, line_size - 1U);
			line[line_size - 1U] = '\0';
			pos = 0U;
			return true;
		}
	}

	return false;
}

static void SweepProtocol_CheckStopRequest(void)
{
	char line[UART_CMD_BUFFER_SIZE];
	if (!SweepProtocol_ReadLineNonBlocking(line, sizeof(line))) {
		return;
	}
	if (CommandMatches(SkipSpaces(line), "STOP")) {
		sweep_stop_requested = true;
	}
}

static void UART_PrintHelp(void)
{
	printf("READY STM32G431_SWEEP_V1\r\n");
	printf("Commands:\r\n");
	printf("  SWEEP <start_hz> <stop_hz> <step_hz> <amplitude_vpp>\r\n");
	printf("  STOP\r\n");
	printf("  DEFAULT\r\n");
	printf("  PING\r\n");
	printf("Tip: bias AC-coupled HP/BP nodes to about 1.65V before ADC/LM358.\r\n");
}

static bool SweepProtocol_ParseCommand(char *line, SweepCommand_t *command)
{
	char *cmd = SkipSpaces(line);

	*command = SweepCommand_Empty();

	if (*cmd == '\0') {
		return true;
	}

	if (CommandMatches(cmd, "PING")) {
		command->id = SWEEP_CMD_PING;
		return true;
	}

	if (CommandMatches(cmd, "HELP")) {
		command->id = SWEEP_CMD_HELP;
		return true;
	}

	if (CommandMatches(cmd, "DEFAULT")) {
		command->id = SWEEP_CMD_DEFAULT;
		return true;
	}

	if (CommandMatches(cmd, "STOP")) {
		command->id = SWEEP_CMD_STOP;
		return true;
	}

	if (CommandMatches(cmd, "SWEEP")) {
		StepSweepConfig_t requested = config;
		char *args = cmd + 5;

		if (!ParseNextFloat(&args, &requested.f_start) ||
			!ParseNextFloat(&args, &requested.f_stop) ||
			!ParseNextFloat(&args, &requested.f_step) ||
			!ParseNextFloat(&args, &requested.amplitude_V)) {
			command->id = SWEEP_CMD_INVALID_SWEEP;
			return true;
		}

		command->id = SWEEP_CMD_SWEEP;
		command->requested_config = requested;
		return true;
	}

	command->id = SWEEP_CMD_UNKNOWN;
	return true;
}

static void SweepCommand_Execute(const SweepCommand_t *command)
{
	switch (command->id) {
	case SWEEP_CMD_EMPTY:
		return;

	case SWEEP_CMD_PING:
		printf("PONG\r\n");
		return;

	case SWEEP_CMD_HELP:
		UART_PrintHelp();
		return;

	case SWEEP_CMD_DEFAULT:
		StepSweep_SetDefaultConfig(&config);
		StepSweep_ResetRuntimeData(&data, &config);
		printf("OK DEFAULT %.3f %.3f %.3f %.3f\r\n",
			config.f_start, config.f_stop, config.f_step, config.amplitude_V);
		return;

	case SWEEP_CMD_STOP:
		sweep_stop_requested = true;
		StepSweep_StopPeripherals();
		printf("STOPPED %lu/%lu\r\n", (unsigned long)0UL, (unsigned long)data.total_steps);
		return;

	case SWEEP_CMD_INVALID_SWEEP:
		printf("ERR usage: SWEEP <start_hz> <stop_hz> <step_hz> <amplitude_vpp>\r\n");
		return;

	case SWEEP_CMD_SWEEP:
		if (!StepSweep_ValidateConfig(&command->requested_config)) {
			return;
		}

		config = command->requested_config;
		StepSweep_ResetRuntimeData(&data, &config);
		sweep_stop_requested = false;
		printf("OK SWEEP %.3f %.3f %.3f %.3f\r\n",
			config.f_start, config.f_stop, config.f_step, config.amplitude_V);
		if (Sweep_ExecuteStepSweep(&config, &data) == SWEEP_RUN_STOPPED) {
			return;
		}
		printf("DONE\r\n");
		return;

	case SWEEP_CMD_UNKNOWN:
	default:
		printf("ERR unknown command, send HELP\r\n");
		return;
	}
}

static void UART_ProcessCommand(char *line)
{
	SweepCommand_t command;

	if (!SweepProtocol_ParseCommand(line, &command)) {
		printf("ERR command parse failed\r\n");
		return;
	}

	SweepCommand_Execute(&command);
}


void unwrap_phase(
    float32_t Re1, float32_t Im1,
    float32_t Re2, float32_t Im2,
    struct PhaseData* data)
{
    // 鲁棒性检查：幅度阈值
    float32_t den_sq = Re1 * Re1 + Im1 * Im1;
    if (den_sq < 1.0e-12f) {
        data->phase_unwrapped = 0.0f;  // 信号太弱，不计算相位
        return;
    }

    // 1. 计算两个通道的相位主值 [-PI, PI]
    float32_t phase1 = atan2f(Im1, Re1);
    float32_t phase2 = atan2f(Im2, Re2);

    // 2. 计算当前频率点的相位差主值 [-PI, PI]
    float32_t phase_diff = phase2 - phase1;

    // 限制到 [-PI, PI]
    while (phase_diff > PI)   phase_diff -= 2.0f * PI;
    while (phase_diff < -PI)  phase_diff += 2.0f * PI;

    // 3. 解缠绕：基于上一个点的解缠绕结果
    if (data->first_point) {
        // 第一个点，直接赋值
        data->phase_unwrapped = phase_diff;
        data->first_point = 0;
    } else {
        // 计算与上一个解缠绕值的跳变
        float32_t diff = phase_diff - data->last_phase_raw;

        // 检测是否需要绕圈
        if (diff > PI) {
            data->unwrap_offset -= 2.0f * PI;
        } else if (diff < -PI) {
            data->unwrap_offset += 2.0f * PI;
        }

        // 应用解缠绕偏移
        data->phase_unwrapped = phase_diff + data->unwrap_offset;
    }

    // 更新原始相位差（未解缠绕）用于下一次跳变检测
    data->last_phase_raw = phase_diff;
}



void wave_analys(StepSweepData_t* data){
	float32_t mean1 = 0.0f;
	float32_t mean2 = 0.0f;
	uint16_t min1 = 0xFFFFU;
	uint16_t min2 = 0xFFFFU;
	uint16_t max1 = 0U;
	uint16_t max2 = 0U;

	for (int i = 0; i < adc_buffer_size; i++) {
		uint16_t sample1 = data->adc1_buffer[i];
		uint16_t sample2 = data->adc2_buffer[i];
		mean1 += (float32_t)sample1;
		mean2 += (float32_t)sample2;
		if (sample1 < min1) {
			min1 = sample1;
		}
		if (sample1 > max1) {
			max1 = sample1;
		}
		if (sample2 < min2) {
			min2 = sample2;
		}
		if (sample2 > max2) {
			max2 = sample2;
		}
	}
	mean1 /= (float32_t)adc_buffer_size;
	mean2 /= (float32_t)adc_buffer_size;

	float32_t sumsq1 = 0.0f;
	float32_t sumsq2 = 0.0f;
	for (int i = 0; i < adc_buffer_size; i++) {
		float32_t centered1 = ((float32_t)data->adc1_buffer[i] - mean1) * 3.3f / 4095.0f;
		float32_t centered2 = ((float32_t)data->adc2_buffer[i] - mean2) * 3.3f / 4095.0f;
		sumsq1 += centered1 * centered1;
		sumsq2 += centered2 * centered2;
		fft_input1[2 * i] = centered1;
		fft_input1[2 * i + 1] = 0.0f;
		fft_input2[2 * i] = centered2;
		fft_input2[2 * i + 1] = 0.0f;
	}

	data->adc1_dc_v = mean1 * 3.3f / 4095.0f;
	data->adc2_dc_v = mean2 * 3.3f / 4095.0f;
	data->adc1_rms_v = sqrtf(sumsq1 / (float32_t)adc_buffer_size);
	data->adc2_rms_v = sqrtf(sumsq2 / (float32_t)adc_buffer_size);
	data->adc1_pp_v = ((float32_t)max1 - (float32_t)min1) * 3.3f / 4095.0f;
	data->adc2_pp_v = ((float32_t)max2 - (float32_t)min2) * 3.3f / 4095.0f;
	data->adc1_min_code = min1;
	data->adc1_max_code = max1;
	data->adc2_min_code = min2;
	data->adc2_max_code = max2;
	data->clip_flags = 0U;
	if (min1 <= 4U || max1 >= 4091U) {
		data->clip_flags |= 0x01U;
	}
	if (min2 <= 4U || max2 >= 4091U) {
		data->clip_flags |= 0x02U;
	}

	arm_cfft_f32(&arm_cfft_sR_f32_len256, fft_input1, 0, 1);
	arm_cfft_f32(&arm_cfft_sR_f32_len256, fft_input2, 0, 1);

	float32_t fs = data->adc_actual_sample_rate;
	if (fs <= 1.0f) {
		fs = TIM1_COUNTER_CLK_HZ / ((float32_t)data->new_arr_value + 1.0f);
	}

	int N = adc_buffer_size;
	float32_t f0 = data->current_freq;
	int bin = (int)roundf(f0 * (float32_t)N / fs);
	if (bin < 1) {
		bin = 1;
	}
	if (bin >= N / 2) {
		bin = N / 2 - 1;
	}

	int idx = bin * 2;
	float32_t Re1 = fft_input1[idx];
	float32_t Im1 = fft_input1[idx + 1];
	float32_t Re2 = fft_input2[idx];
	float32_t Im2 = fft_input2[idx + 1];

    float32_t mag1 = sqrtf(Re1 * Re1 + Im1 * Im1);
    float32_t mag2 = sqrtf(Re2 * Re2 + Im2 * Im2);
    /*
     * ADC1/PA0 is the input/reference node and ADC2/PA1 is the output/response
     * node, so the transfer magnitude must be output/input.  The phase remains
     * phase2 - phase1, which makes a low-pass output lag appear negative.
     */
    if (mag1 > 1.0e-6f) {
        data->gain = mag2 / mag1;
    } else {
        data->gain = 0.0f;
    }

	unwrap_phase(Re1, Im1, Re2, Im2, &data_phase);
}

static float32_t MedianFloat32(float32_t *values, uint32_t count)
{
	if (count == 0U) {
		return 0.0f;
	}

	for (uint32_t i = 0U; i + 1U < count; i++) {
		for (uint32_t j = i + 1U; j < count; j++) {
			if (values[j] < values[i]) {
				float32_t tmp = values[i];
				values[i] = values[j];
				values[j] = tmp;
			}
		}
	}

	if ((count & 1U) != 0U) {
		return values[count / 2U];
	}

	return 0.5f * (values[(count / 2U) - 1U] + values[count / 2U]);
}

static uint16_t FloatToScaledU16(float32_t value, float32_t scale)
{
	if (value <= 0.0f) {
		return 0U;
	}
	float32_t scaled = value * scale;
	if (scaled >= 65535.0f) {
		return 65535U;
	}
	return (uint16_t)roundf(scaled);
}

static uint16_t FloatToQ15(float32_t value)
{
	if (value <= 0.0f) {
		return 0U;
	}
	if (value >= 1.9999f) {
		return 65535U;
	}
	return (uint16_t)roundf(value * 32768.0f);
}

static int16_t FloatRadToMilliI16(float32_t value)
{
	float32_t scaled = value * 1000.0f;
	if (scaled > 32767.0f) {
		return 32767;
	}
	if (scaled < -32768.0f) {
		return -32768;
	}
	return (int16_t)roundf(scaled);
}

static uint32_t FloatToScaledU32(float32_t value, float32_t scale)
{
	if (value <= 0.0f) {
		return 0UL;
	}
	return (uint32_t)roundf(value * scale);
}

static void UART_PrintScaledU16Array(const char *name, const uint16_t *values, uint16_t count, float32_t divisor)
{
	printf("%s=[", name);
	for (uint16_t i = 0; i < count; i++) {
		if (i > 0U) {
			printf(",");
		}
		printf("%f", ((float32_t)values[i]) / divisor);
	}
	printf("]\r\n");
}

static void UART_PrintQ15Array(const char *name, const uint16_t *values, uint16_t count)
{
	printf("%s=[", name);
	for (uint16_t i = 0; i < count; i++) {
		if (i > 0U) {
			printf(",");
		}
		printf("%f", ((float32_t)values[i]) / 32768.0f);
	}
	printf("]\r\n");
}

static void UART_PrintMilliRadArray(const char *name, const int16_t *values, uint16_t count)
{
	printf("%s=[", name);
	for (uint16_t i = 0; i < count; i++) {
		if (i > 0U) {
			printf(",");
		}
		printf("%f", ((float32_t)values[i]) / 1000.0f);
	}
	printf("]\r\n");
}

static void UART_PrintU16Array(const char *name, const uint16_t *values, uint16_t count)
{
	printf("%s=[", name);
	for (uint16_t i = 0; i < count; i++) {
		if (i > 0U) {
			printf(",");
		}
		printf("%u", (unsigned int)values[i]);
	}
	printf("]\r\n");
}

static void UART_PrintU32Array(const char *name, const uint32_t *values, uint16_t count)
{
	printf("%s=[", name);
	for (uint16_t i = 0; i < count; i++) {
		if (i > 0U) {
			printf(",");
		}
		printf("%lu", (unsigned long)values[i]);
	}
	printf("]\r\n");
}

static void UART_PrintScaledU32Array(const char *name, const uint32_t *values, uint16_t count, float32_t divisor)
{
	printf("%s=[", name);
	for (uint16_t i = 0; i < count; i++) {
		if (i > 0U) {
			printf(",");
		}
		printf("%f", ((float32_t)values[i]) / divisor);
	}
	printf("]\r\n");
}

/**
 * @brief 使用 CMSIS-DSP 库生成单个固定频率的正弦波数组
 * * @param frequency 要生成的频率 (Hz)
 * @param config 扫频配置参数
 * @param dac_output 存储 DAC 16位整数输出的数组 (大小为 dac_buffer_size)
 * @return arm_status 状态码
 */
arm_status StepSweep_Generate_Data(float32_t frequency, 
	const StepSweepConfig_t *config, StepSweepData_t *stepsweep_data) {
    
    // 从结构体中获取实际的点数和周期数
    const uint32_t N_actual = stepsweep_data->dac_actual_length;
    const uint32_t M = stepsweep_data->dac_periods;
    
    if (frequency <= 0.0f || N_actual == 0 || M == 0) // 使用 N_actual 和 M 检查有效性
			return ARM_MATH_ARGUMENT_ERROR;

    // 总相位必须是 2*PI*M，确保波形在 N_actual 个点后精确闭合 M 个周期
    float32_t phase_total = 2.0f * PI * (float32_t)M; 
    
    // 每个采样点的精确相位增量: 增量 = 总相位 / 点数 N_actual
    float32_t phase_increment = phase_total / (float32_t)N_actual;
    float32_t current_phase = 0.0f; // 从 0 弧度开始
    
    static float32_t temp_float[dac_buffer_size]; // 静态缓冲区大小保持 1024
    
    //生成浮点波形，循环 N_actual 次
    for (int n = 0; n < N_actual; n++) {
        temp_float[n] = sinf(current_phase); 
        current_phase += phase_increment;
    }
    
    float32_t midpoint = (float32_t)config->DAC_MAX_VAL * 0.5f;
    float32_t amplitude_code = (float32_t)config->amplitude_V * (float32_t)config->DAC_MAX_VAL / (2.0f * 3.3f);
    float32_t max_amplitude_code = midpoint - 1.0f;
    if (amplitude_code > max_amplitude_code) {
        amplitude_code = max_amplitude_code;
    }
    if (amplitude_code < 0.0f) {
        amplitude_code = 0.0f;
    }
    
    // a. 乘以缩放系数: 向量操作长度为 N_actual
    arm_scale_f32(temp_float, amplitude_code, temp_float, N_actual); // <-- 使用 N_actual
    
    // b. 加上中值偏移: 向量操作长度为 N_actual
    arm_offset_f32(temp_float, midpoint, temp_float, N_actual); // <-- 使用 N_actual
    
    // c. 转换为 16 位无符号整数
    for (int i = 0; i < N_actual; i++) { // <-- 循环到 N_actual
        // 边界检查和四舍五入
        stepsweep_data->dac_buffer[i] = (uint16_t)fmaxf(0.0f, fminf(config->DAC_MAX_VAL, roundf(temp_float[i])));
    }
    
    return ARM_MATH_SUCCESS;
}

/**
 * @brief 执行步进式扫频的主函数
 * * @param config 扫频配置
 * @param data 运行时数据和缓冲区
 */

SweepRunStatus_t Sweep_ExecuteStepSweep(
    StepSweepConfig_t *config,
    StepSweepData_t *data
){
	if (config->f_step <= 0.0f || config->f_stop < config->f_start) {
		return SWEEP_RUN_DONE;
	}

	StepSweep_ResetPhaseTracker();

	float32_t f_target = config->f_start;
	float32_t f_dac = config->dac_sample_rate;
	float32_t N_max = (float32_t)dac_buffer_size;

	uint32_t computed_steps = (uint32_t)((config->f_stop - config->f_start) / config->f_step) + 1U;
	if (computed_steps > MAX_FREQ_STEPS) {
		computed_steps = MAX_FREQ_STEPS;
	}
	config->num_log_steps = (uint16_t)computed_steps;
	uint16_t steps = config->num_log_steps;
	data->total_steps = steps;
	SweepOutputSummary_t output_summary;
	SweepOutputSummary_Init(&output_summary);

	for (uint16_t i = 0; i < steps; i++) {
		float32_t M_target_f = (f_target * N_max) / f_dac;
		uint32_t M = (uint32_t)floorf(M_target_f);
		if (M == 0U) {
			M = 1U;
		}

		uint32_t N_actual = (uint32_t)roundf(M * (f_dac / f_target));
		if (N_actual > dac_buffer_size) {
			N_actual = dac_buffer_size;
		}
		if (N_actual < 2U) {
			N_actual = 2U;
		}

		data->dac_actual_length = N_actual;
		data->dac_periods = M;

		float32_t dac_target_fs = (f_target * (float32_t)N_actual) / (float32_t)M;
		if (dac_target_fs > f_dac) {
			dac_target_fs = f_dac;
		}
		if (dac_target_fs < 1.0f) {
			dac_target_fs = 1.0f;
		}

		uint32_t dac_arr_candidate = (uint32_t)roundf((TIM2_COUNTER_CLK_HZ / dac_target_fs) - 1.0f);
		if (dac_arr_candidate < 1U) {
			dac_arr_candidate = 1U;
		}

		data->dac_timer_arr = dac_arr_candidate;
		data->dac_actual_sample_rate = TIM2_COUNTER_CLK_HZ / ((float32_t)data->dac_timer_arr + 1.0f);
		data->actual_freq = data->dac_actual_sample_rate * (float32_t)M / (float32_t)N_actual;
		data->current_freq = data->actual_freq;
		OmegaMilliRadPerSec[i] = FloatToScaledU32(data->current_freq * 2.0f * PI, 1000.0f);

		if (StepSweep_Generate_Data(data->current_freq, config, data) != ARM_MATH_SUCCESS) {
			StepSweep_ClearPointResult(i);
			f_target += config->f_step;
			continue;
		}

		float32_t adc_target_fs = data->current_freq * ADC_SAMPLES_PER_SIGNAL_PERIOD;
		StepSweep_ConfigureAdcTimer(data, adc_target_fs);
		StepSweep_ApplyTimerSettings(data);

		HAL_DAC_Start_DMA(&hdac1, DAC_CHANNEL_1, (uint32_t*)data->dac_buffer, data->dac_actual_length, DAC_ALIGN_12B_R);
		HAL_TIM_Base_Start(&htim2);

		HAL_Delay(StepSweep_SettleTimeMs(data->current_freq));

		float32_t gain_samples[SWEEP_CAPTURE_REPEATS];
		float32_t input_rms_samples[SWEEP_CAPTURE_REPEATS];
		float32_t output_rms_samples[SWEEP_CAPTURE_REPEATS];
		float32_t input_dc_samples[SWEEP_CAPTURE_REPEATS];
		float32_t output_dc_samples[SWEEP_CAPTURE_REPEATS];
		float32_t input_pp_samples[SWEEP_CAPTURE_REPEATS];
		float32_t output_pp_samples[SWEEP_CAPTURE_REPEATS];
		uint16_t input_min_code = 0xFFFFU;
		uint16_t output_min_code = 0xFFFFU;
		uint16_t input_max_code = 0U;
		uint16_t output_max_code = 0U;
		uint32_t valid_captures = 0U;
		uint16_t clip_flags_or = 0U;
		uint32_t capture_time_ms = StepSweep_CaptureTimeoutMs(config, data);
		uint32_t repeat = 0U;
		HAL_StatusTypeDef adc1_status;
		HAL_StatusTypeDef adc2_status;
		StepSweepCaptureStatus_t capture_status = STEP_CAPTURE_TIMEOUT;
		bool phase_committed = false;
		float32_t phase_samples[SWEEP_CAPTURE_REPEATS];

		for (repeat = 0U; repeat < SWEEP_CAPTURE_REPEATS; repeat++) {
			adc1_flag = false;
			adc2_flag = false;
			adc_flag = false;

			__HAL_TIM_SET_COUNTER(&htim1, 0);
			adc1_status = HAL_ADC_Start_DMA(&hadc1, (uint32_t*)data->adc1_buffer, config->adc_buffersize);
			adc2_status = HAL_ADC_Start_DMA(&hadc2, (uint32_t*)data->adc2_buffer, config->adc_buffersize);
			capture_status = STEP_CAPTURE_TIMEOUT;

			if (adc1_status == HAL_OK && adc2_status == HAL_OK) {
				HAL_TIM_Base_Start(&htim1);
				capture_status = StepSweep_WaitForAdcCapture(capture_time_ms);
			}

			HAL_TIM_Base_Stop(&htim1);
			HAL_ADC_Stop_DMA(&hadc1);
			HAL_ADC_Stop_DMA(&hadc2);

			if (capture_status == STEP_CAPTURE_STOPPED) {
				StepSweep_StopPeripherals();
				printf("STOPPED %u/%u\r\n", (unsigned int)i, (unsigned int)steps);
				return SWEEP_RUN_STOPPED;
			}

			if (adc1_status == HAL_OK && adc2_status == HAL_OK && capture_status == STEP_CAPTURE_DONE) {
				wave_analys(data);
				gain_samples[valid_captures] = data->gain;
				phase_samples[valid_captures] = data_phase.phase_unwrapped;
				if (!phase_committed) {
					PhaseMilliRad[i] = FloatRadToMilliI16(data_phase.phase_unwrapped);
					phase_committed = true;
				}
				input_rms_samples[valid_captures] = data->adc1_rms_v;
				output_rms_samples[valid_captures] = data->adc2_rms_v;
				input_dc_samples[valid_captures] = data->adc1_dc_v;
				output_dc_samples[valid_captures] = data->adc2_dc_v;
				input_pp_samples[valid_captures] = data->adc1_pp_v;
				output_pp_samples[valid_captures] = data->adc2_pp_v;
				if (data->adc1_min_code < input_min_code) {
					input_min_code = data->adc1_min_code;
				}
				if (data->adc1_max_code > input_max_code) {
					input_max_code = data->adc1_max_code;
				}
				if (data->adc2_min_code < output_min_code) {
					output_min_code = data->adc2_min_code;
				}
				if (data->adc2_max_code > output_max_code) {
					output_max_code = data->adc2_max_code;
				}
				clip_flags_or |= data->clip_flags;
				valid_captures++;
			}

			if (repeat + 1U < SWEEP_CAPTURE_REPEATS) {
				HAL_Delay(1U);
			}
		}

		HAL_TIM_Base_Stop(&htim1);
		HAL_TIM_Base_Stop(&htim2);
		HAL_DAC_Stop_DMA(&hdac1, DAC_CHANNEL_1);

		if (valid_captures == 0U) {
			StepSweep_ClearPointResult(i);
		} else {
			float32_t gain_min = gain_samples[0];
			float32_t gain_max = gain_samples[0];
			float32_t phase_min = phase_samples[0];
			float32_t phase_max = phase_samples[0];
			for (uint32_t span_i = 1U; span_i < valid_captures; span_i++) {
				if (gain_samples[span_i] < gain_min) {
					gain_min = gain_samples[span_i];
				}
				if (gain_samples[span_i] > gain_max) {
					gain_max = gain_samples[span_i];
				}
				if (phase_samples[span_i] < phase_min) {
					phase_min = phase_samples[span_i];
				}
				if (phase_samples[span_i] > phase_max) {
					phase_max = phase_samples[span_i];
				}
			}
			float32_t gain_span_db = 0.0f;
			if (gain_min > 1.0e-9f && gain_max > 1.0e-9f) {
				gain_span_db = 20.0f * log10f(gain_max / gain_min);
			}
			GainQ15[i] = FloatToQ15(MedianFloat32(gain_samples, valid_captures));
			InputRmsMv[i] = FloatToScaledU16(MedianFloat32(input_rms_samples, valid_captures), 1000.0f);
			OutputRmsMv[i] = FloatToScaledU16(MedianFloat32(output_rms_samples, valid_captures), 1000.0f);
			InputDcMv[i] = FloatToScaledU16(MedianFloat32(input_dc_samples, valid_captures), 1000.0f);
			OutputDcMv[i] = FloatToScaledU16(MedianFloat32(output_dc_samples, valid_captures), 1000.0f);
			InputPpMv[i] = FloatToScaledU16(MedianFloat32(input_pp_samples, valid_captures), 1000.0f);
			OutputPpMv[i] = FloatToScaledU16(MedianFloat32(output_pp_samples, valid_captures), 1000.0f);
			ClipFlags[i] = clip_flags_or;
			ValidCaptureCount[i] = (uint16_t)valid_captures;
			ActualFreqMilliHz[i] = FloatToScaledU32(data->actual_freq, 1000.0f);
			AdcSampleRateMilliHz[i] = FloatToScaledU32(data->adc_actual_sample_rate, 1000.0f);
			DacSampleRateMilliHz[i] = FloatToScaledU32(data->dac_actual_sample_rate, 1000.0f);
			MagnitudeRepeatSpanMilliDb[i] = FloatToScaledU16(gain_span_db, 1000.0f);
			PhaseRepeatSpanMilliRad[i] = FloatToScaledU16(fabsf(phase_max - phase_min), 1000.0f);
			InputMinCode[i] = input_min_code;
			InputMaxCode[i] = input_max_code;
			OutputMinCode[i] = output_min_code;
			OutputMaxCode[i] = output_max_code;
			SweepOutputSummary_AddPoint(
				&output_summary,
				input_min_code,
				input_max_code,
				output_min_code,
				output_max_code,
				clip_flags_or);
		}

		printf("PROGRESS %u/%u %.3f\r\n", (unsigned int)(i + 1U), (unsigned int)steps, data->actual_freq);
		f_target += config->f_step;
	}

	UART_PrintScaledU32Array("omega", OmegaMilliRadPerSec, steps, 1000.0f);
	UART_PrintQ15Array("Magnitude_data", GainQ15, steps);
	UART_PrintMilliRadArray("Phase_data_rad", PhaseMilliRad, steps);
	UART_PrintScaledU32Array("Actual_freq_hz", ActualFreqMilliHz, steps, 1000.0f);
	UART_PrintScaledU32Array("Adc_sample_rate_hz", AdcSampleRateMilliHz, steps, 1000.0f);
	UART_PrintScaledU32Array("Dac_sample_rate_hz", DacSampleRateMilliHz, steps, 1000.0f);
	UART_PrintScaledU16Array("Magnitude_repeat_span_db", MagnitudeRepeatSpanMilliDb, steps, 1000.0f);
	UART_PrintScaledU16Array("Phase_repeat_span_rad", PhaseRepeatSpanMilliRad, steps, 1000.0f);
	UART_PrintU16Array("Input_min_code", InputMinCode, steps);
	UART_PrintU16Array("Input_max_code", InputMaxCode, steps);
	UART_PrintU16Array("Output_min_code", OutputMinCode, steps);
	UART_PrintU16Array("Output_max_code", OutputMaxCode, steps);
	UART_PrintScaledU16Array("Input_rms_v", InputRmsMv, steps, 1000.0f);
	UART_PrintScaledU16Array("Output_rms_v", OutputRmsMv, steps, 1000.0f);
	UART_PrintScaledU16Array("Input_dc_v", InputDcMv, steps, 1000.0f);
	UART_PrintScaledU16Array("Output_dc_v", OutputDcMv, steps, 1000.0f);
	UART_PrintU16Array("Clip_flags", ClipFlags, steps);
	UART_PrintU16Array("Valid_capture_count", ValidCaptureCount, steps);
	UART_PrintScaledU16Array("Input_pp_v", InputPpMv, steps, 1000.0f);
	UART_PrintScaledU16Array("Output_pp_v", OutputPpMv, steps, 1000.0f);
	printf("Adc_code_range=[%u,%u,%u,%u]\r\n",
		(unsigned int)output_summary.input_min_code_all,
		(unsigned int)output_summary.input_max_code_all,
		(unsigned int)output_summary.output_min_code_all,
		(unsigned int)output_summary.output_max_code_all);
	printf("Clip_point_count=[%u,%u]\r\n",
		(unsigned int)output_summary.input_clip_points,
		(unsigned int)output_summary.output_clip_points);
	return SWEEP_RUN_DONE;
}


void Generate_Hanning_Window(float32_t *window_coeffs, uint32_t size) {
	if (size == 0) return;
	float32_t N_minus_1 = (float32_t)(size - 1);
	
	for (uint32_t n = 0; n < size; n++) {
			window_coeffs[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float32_t)n / N_minus_1);
	}
}

/* 五系数 flat-top 窗（ISO 18431-2） */
static const float flat_top_coeffs[5] = {
    0.21557895f,
    0.41663158f,
    0.277263158f,
    0.083578947f,
    0.006947368f
};


void Generate_FlatTop_Window(float32_t *window_coeffs, uint32_t size)
{
    if (size == 0) return;
    float32_t N = (float32_t)(size - 1);

    for (uint32_t n = 0; n < size; n++) {
        float32_t w = 0.0f;
        float32_t k  = 2.0f * PI * (float32_t)n / N;
        /* 五项余弦和 */
        w += flat_top_coeffs[0];
        w -= flat_top_coeffs[1] * cosf(k);
        w += flat_top_coeffs[2] * cosf(2.0f * k);
        w -= flat_top_coeffs[3] * cosf(3.0f * k);
        w += flat_top_coeffs[4] * cosf(4.0f * k);
        window_coeffs[n] = w;
    }
}


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

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_TIM1_Init();
  MX_USART1_UART_Init();
  MX_DAC1_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */
	HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);
	HAL_ADCEx_Calibration_Start(&hadc2, ADC_SINGLE_ENDED);

	StepSweep_SetDefaultConfig(&config);
	StepSweep_ResetRuntimeData(&data, &config);
	
	UART_PrintHelp();
	
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
		char cmd_line[UART_CMD_BUFFER_SIZE];
		if (SweepProtocol_ReadLine(cmd_line, sizeof(cmd_line))) {
			UART_ProcessCommand(cmd_line);
		}
		
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
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
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV3;
  RCC_OscInitStruct.PLL.PLLN = 85;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
    if (hadc->Instance == ADC1){
			adc1_flag=1;
			if(adc1_flag&&adc2_flag)
				adc_flag=1;
		}
		if (hadc->Instance == ADC2){
			adc2_flag=1;
			if(adc2_flag&&adc1_flag){
				adc_flag=1;
			}
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

#ifdef  USE_FULL_ASSERT
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
