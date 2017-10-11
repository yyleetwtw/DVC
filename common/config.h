
#ifndef COMMON_INC_CONFIG_H
#define COMMON_INC_CONFIG_H

#define PI                      3.14159265
#define MB_BLOCK_SIZE           16
#define DC_BITDEPTH             10

// Macros for encoder/decoder
#define FPGA                    0
#define WHOLEFLOW               1
#define AC_QSTEP                1
#define RESIDUAL_CODING         0
#define SKIP_MODE               0
#define MODE_DECISION           0
#define INTEGER_DCT             0
#define HARDWARE_FLOW           0
#define HARDWARE_QUANTIZATION   0
#define HARDWARE_LDPC           0
#define HARDWARE_CMS            0
#define HARDWARE_OPT            0
#define BIDIRECT_REFINEMENT     0
#define SI_REFINEMENT           0

// Macros for encoder only
//#ifdef ENCODER
# define TESTPATTERN            0
# define DEBUG                  0
//#endif

// Macros for decoder only
//#ifdef DECODER
# define INVERSE_MATRIX         1
//#endif

#include "types.h"

#endif // COMMON_INC_CONFIG_H

