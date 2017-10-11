
#ifndef ENCODER_INC_CAVLCENC_H
#define ENCODER_INC_CAVLCENC_H

#include <vector>

#include <cstdio>

#include "../../common/config.h"
#include "../../common/cavlc.h"

using std::vector;

class File;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class CavlcEnc : public Cavlc
{
public:
  CavlcEnc(Codec *codec, int blockSize);

  int encode(int *frame, int *skipMask);

private:
  void  setupMacroBlock   (int *frame, int mbX, int mbY);
  int   encodeMacroBlock  (int mbX, int mbY);

  int   encodeNumTrail    (int numCoeffs, int numT1s, int vlc);
  int   encodeSignTrail   (vector<int> &sign);
  int   encodeLevelsVlc0  (int level);
  int   encodeLevelsVlcN  (int level, int vlc);
  int   encodeTotalZeros  (int numCoeffs, int numZeros);
  int   encodeRuns        (int runBefore, int zerosLeft);

  void  outputCode        (int length, int code);
};

#endif // ENCODER_INC_CAVLCENC_H

