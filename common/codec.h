
#ifndef COMMON_INC_CODEC_H
#define COMMON_INC_CODEC_H

#include "config.h"

class Bitstream;

class Codec
{
public:
  Codec() {};

  int     getFrameWidth()      { return _frameWidth; };
  int     getFrameHeight()     { return _frameHeight; };
  int     getBitPlaneLength()  { return _bitPlaneLength; };
  int     getQp()              { return _qp; };
  int     getKeyQp()           { return _keyQp; };
  int     getNumChnCodeBands() { return _numChnCodeBands; };

  double* getAverage()         { return _average; };
  double* getAlpha()           { return _alpha; };
  double* getSigma()           { return _sigma; };

  int     getQuantMatrix(int qp, int x, int y) { return QuantMatrix[qp][y][x]; };
  int     getQuantStep(int x, int y) { return _quantStep[y][x]; };

  Bitstream* getBitstream() { return _bs; };

  const static int  ResidualBlockSize = 8;
  const static int  BlockSize         = 4;

protected:
  const static int  QuantMatrix[8][BlockSize][BlockSize];
  const static int  BitPlaneNum[8];
  const static int  MaxBitPlane[BlockSize][BlockSize];
#if RESIDUAL_CODING
  const static int  MinQStepSize[8][BlockSize][BlockSize];
#endif

  const static int  ScanOrder[16][2];
  const static int  HuffmanCodeValue[4][3][16];
  const static int  HuffmanCodeLength[4][3][16];

  int               _quantStep[BlockSize][BlockSize];

  int               _frameWidth;
  int               _frameHeight;
  int               _frameSize;
  int               _numFrames;
  int               _bitPlaneLength;
  int               _qp;
  int               _keyQp;
  int               _gopLevel;
  int               _gop;
  int               _numCodeBands;
  int               _numChnCodeBands;

  bool*             _parity;
  double*           _dParity; // TODO temporary for decoder
  unsigned char*    _crc;
  double*           _average;
  double*           _alpha;
  double*           _sigma;

  Bitstream*        _bs;
};

#endif // COMMON_INC_CODEC_H
