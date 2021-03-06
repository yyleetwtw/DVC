
#include "codec.h"

# if RESIDUAL_CODING
const int Codec::QuantMatrix[8][BlockSize][BlockSize] = {
 {{4, 3, 0, 0},
  {3, 0, 0, 0},
  {0, 0, 0, 0},
  {0, 0, 0, 0}},
 {{5, 3, 0, 0},
  {3, 0, 0, 0},
  {0, 0, 0, 0},
  {0, 0, 0, 0}},
 {{5, 3, 2, 0},
  {3, 2, 0, 0},
  {2, 0, 0, 0},
  {0, 0, 0, 0}},
 {{5, 4, 3, 2},
  {4, 3, 2, 0},
  {3, 2, 0, 0},
  {2, 0, 0, 0}},
 {{5, 4, 3, 2},
  {4, 3, 2, 2},
  {3, 2, 2, 0},
  {2, 2, 0, 0}},
 {{6, 4, 3, 3},
  {4, 3, 3, 2},
  {3, 3, 2, 0},
  {3, 2, 0, 0}},
 {{6, 5, 4, 3},
  {5, 4, 3, 2},
  {4, 3, 2, 0},
  {3, 2, 0, 0}},
 {{7, 6, 5, 4},
  {6, 5, 4, 3},
  {5, 4, 3, 0},
  {4, 3, 0, 0}}
};

const int Codec::BitPlaneNum[8] = {10, 11, 17, 30, 36, 41, 50, 59};

const int Codec::MaxBitPlane[BlockSize][BlockSize] = {
  {10, 9, 8, 8},
  { 9, 8, 8, 7},
  { 8, 8, 7, 6},
  { 8, 7, 6, 6}
};

const int Codec::MinQStepSize[8][BlockSize][BlockSize] = {
 {{32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32}},
 {{32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32}},
 {{32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32}},
 {{32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32},
  {32, 32, 32, 32}},
 {{16, 16, 16, 16},
  {16, 16, 16, 16},
  {16, 16, 16, 16},
  {16, 16, 16, 16}},
 {{16, 16, 16, 16},
  {16, 16, 16, 16},
  {16, 16, 16, 16},
  {16, 16, 16, 16}},
 {{10, 10, 10, 12},
  {10, 10, 12, 16},
  {10, 12, 16, 16},
  {12, 16, 16, 16}},
 {{ 8,  8,  8, 10},
  { 8,  8, 10, 16},
  { 8, 10, 16, 18},
  {10, 16, 18, 18}}
};

# else // if !RESIDUAL_CODING

const int Codec::QuantMatrix[8][BlockSize][BlockSize] = {
 {{4, 3, 0, 0},
  {3, 0, 0, 0},
  {0, 0, 0, 0},
  {0, 0, 0, 0}},
 {{5, 3, 0, 0},
  {3, 0, 0, 0},
  {0, 0, 0, 0},
  {0, 0, 0, 0}},
 {{5, 3, 2, 0},
  {3, 2, 0, 0},
  {2, 0, 0, 0},
  {0, 0, 0, 0}},
 {{5, 4, 3, 2},
  {4, 3, 2, 0},
  {3, 2, 0, 0},
  {2, 0, 0, 0}},
 {{5, 4, 3, 2},
  {4, 3, 2, 2},
  {3, 2, 2, 0},
  {2, 2, 0, 0}},
 {{6, 4, 3, 3},
  {4, 3, 3, 2},
  {3, 3, 2, 0},
  {3, 2, 0, 0}},
 {{6, 5, 4, 3},
  {5, 4, 3, 2},
  {4, 3, 2, 2},
  {3, 2, 2, 0}},
 {{7, 6, 5, 4},
  {6, 5, 4, 3},
  {5, 4, 3, 2},
  {4, 3, 2, 0}}
};

const int Codec::BitPlaneNum[8] = {10, 11, 17, 30, 36, 41, 50, 63};

const int Codec::MaxBitPlane[BlockSize][BlockSize] = {
  {10, 9, 9, 8},
  { 9, 9, 8, 7},
  { 9, 8, 7, 6},
  { 8, 7, 6, 6}
};

# endif // RESIDUAL_CODING

// Zigzag scan order
const int Codec::ScanOrder[16][2] = {
  {0, 0},
  {1, 0},
  {0, 1},
  {0, 2},
  {1, 1},
  {2, 0},
  {3, 0},
  {2, 1},
  {1, 2},
  {0, 3},
  {1, 3},
  {2, 2},
  {3, 1},
  {3, 2},
  {2, 3},
  {3, 3}
};

const int Codec::HuffmanCodeValue[4][3][16] = {
  // QP = 1 or 2
  {{  3,  1,  4,  0,  3, 23, 21,  4, 40, 11, 83, 82, 21, 41, 40, 22},   // type 0
   {  3,  0,  3,  9,  4, 23, 16, 11, 45, 35, 21, 34, 20, 89, 88, 10},   // type 1
   {  2,  2,  3, 15, 13, 12,  5,  4,  3,  2,  0, 28, 29,  3,  2,  3}},  // type 2
  // QP = 3 or 4
  {{  3,  1,  4, 11,  3,  1, 20,  4,  0, 11,  3,  2, 20, 43, 42, 21},   // type 0
   {  3,  0,  3,  9, 23, 21, 17, 45, 41, 33, 89, 32, 88, 81, 80,  2},   // type 1
   {  2,  3,  5,  2,  0,  3,  6,  2, 19, 15, 14, 17, 16, 37, 36,  3}},  // type 2
  // QP = 5 or 6
  {{  3,  1,  4, 11,  3,  1, 20,  1, 11, 10,  8,  1,  0, 19, 18, 21},   // type 0
   {  0,  7, 13, 10, 25, 23, 17, 49, 45, 33, 32, 97, 96, 89, 88,  9},   // type 1
   {  2, 14, 31, 25, 60, 53, 54, 49,122, 48,123,111,110,105,104,  0}},  // type 2
  // QP = 7 or 8
  {{  0,  7,  4, 12, 27, 23, 20, 52, 44,106, 91,215,214,181,180, 21},   // type 0
   {  0,  6, 15,  9, 29, 21, 16, 57, 41, 35, 34,113,112, 81, 80, 11},   // type 1
   {  0,  4, 11, 30, 21, 20, 62, 56, 58,127,126,119,115,114,118,  6}}   // type 2
};

const int Codec::HuffmanCodeLength[4][3][16] = {
  // QP = 1 or 2
  {{  2,  2,  3,  3,  4,  5,  5,  5,  6,  6,  7,  7,  7,  8,  8,  5},   // type 0
   {  2,  2,  3,  4,  4,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  4},   // type 1
   {  2,  3,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  2}},  // type 2
  // QP = 3 or 4
  {{  2,  2,  3,  4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  8,  8,  5},   // type 0
   {  2,  2,  3,  4,  5,  5,  5,  6,  6,  6,  7,  6,  7,  7,  7,  3},   // type 1
   {  2,  3,  4,  4,  4,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  2}},  // type 2
  // QP = 5 or 6
  {{  2,  2,  3,  4,  4,  4,  5,  5,  6,  6,  6,  6,  6,  7,  7,  5},   // type 0
   {  1,  3,  4,  4,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  4},   // type 1
   {  2,  4,  5,  5,  6,  6,  6,  6,  7,  6,  7,  7,  7,  7,  7,  1}},  // type 2
  // QP = 7 or 8
  {{  1,  3,  3,  4,  5,  5,  5,  6,  6,  7,  7,  8,  8,  8,  8,  5},   // type 0
   {  1,  3,  4,  4,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  4},   // type 1
   {  1,  3,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7,  7,  7,  3}}   // type 2
};

