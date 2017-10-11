
#include <iostream>
#include <sstream>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "cavlcEnc.h"
#include "codec.h"
#include "encoder.h"
#include "bitstream.h"
#include "fileManager.h"

using namespace std;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
CavlcEnc::CavlcEnc(Codec *codec, int blockSize) : Cavlc(codec, blockSize)
{
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encode(int *frame, int *skipMask)
{
  int width    = _codec->getFrameWidth();
  int height   = _codec->getFrameHeight();
  int bitCount = 0;

  for (int y = 0; y < height/_blockSize; y++)
    for (int x = 0; x < width/_blockSize; x++) {
# if SKIP_MODE
      if (skipMask[x+y*(width/_blockSize)] == 0) {
# endif // SKIP_MODE
        setupMacroBlock(frame, x, y);

        bitCount += encodeMacroBlock(x, y);
# if SKIP_MODE
      }
      else
        _mbs[x+y*(width/_blockSize)].nnz = 0;
# endif // SKIP_MODE
    }

  return bitCount;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void CavlcEnc::setupMacroBlock(int *frame, int mbX, int mbY)
{
  int   width  = _codec->getFrameWidth();
  int **buffer = AllocArray2D<int>(4, 4);

  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      int index = (i+mbX*_blockSize) + (j+mbY*_blockSize)*width;
      int qp    = _codec->getQp();

      int mask  = (0x1<<(_codec->getQuantMatrix(qp, i, j)-1))-1;
      int sign  = (frame[index]>>(_codec->getQuantMatrix(qp, i, j)-1)) & 0x1;
      int value = frame[index] & mask;

      buffer[j][i] = (sign == 1) ? -value : value;

# if !RESIDUAL_CODING
      if (i == 0 && j == 0)
        buffer[j][i] = frame[index];
# endif // !RESIDUAL_CODING
    }
  }

  int nnz = 0;

# if MODE_DECISION
  for (int i = 0; i < (16-_codec->getNumChnCodeBands()); i++) {
    int x = ScanOrder[i+_codec->getNumChnCodeBands()][0];
    int y = ScanOrder[i+_codec->getNumChnCodeBands()][1];
# else // if !MODE_DECISION
  for (int i = 0; i < 16; i++) {
    int x = ScanOrder[i][0];
    int y = ScanOrder[i][1];
# endif // MODE_DECISION

    _mbs[mbX + mbY*(width/_blockSize)].coeffs[i] = buffer[y][x];

    if (buffer[y][x] != 0)
      nnz++;
  }

  _mbs[mbX + mbY*(width/_blockSize)].nnz = nnz;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeMacroBlock(int mbX, int mbY)
{
  int         maxCoeffNum;    // Number of bands to be coded

  int         numCoeffs = 0;  // number of total coefficients
  int         numT1s = 0;     // number of trailing ones
  int         numZeros = 0;   // number of total zeros

  int         bitCount = 0;

  vector<int> level;
  vector<int> sign;
  vector<int> runs;
  vector<int> zerosleft;

# if MODE_DECISION
  maxCoeffNum = 16 - _codec->getNumChnCodeBands();
# else // if !MODE_DECISION
  maxCoeffNum = 16;
# endif // MODE_DECISION

  // Traverse coefficients and setup vector data structures accordingly
  // ---------------------------------------------------------------------------
  int   width          = _codec->getFrameWidth();
  int  *coeffs         = _mbs[mbX + mbY*(width/_blockSize)].coeffs;
  bool  doneT1Flag     = 0;
  bool  metNonzeroFlag = 0;

  for (int i = maxCoeffNum-1; i >= 0; i--) {
    if (coeffs[i] != 0) {
      metNonzeroFlag = 1;
      numCoeffs++;

      if (abs(coeffs[i]) == 1 && numT1s < 3 && !doneT1Flag) {
        numT1s++;

        if (coeffs[i] == 1)
          sign.push_back(0);
        else
          sign.push_back(1);
      }
      else {
        doneT1Flag = 1;
        level.push_back(coeffs[i]);
      }

      runs.push_back(0);
    }
    else {
      if (metNonzeroFlag) {
        numZeros++;
        runs.back()++;
      }
    }
  }

  int numZerosLeft = numZeros;

  for (unsigned i = 0; i < runs.size(); i++) {
    zerosleft.push_back(numZerosLeft);
    numZerosLeft -= runs[i];
  }

  // ---------------------------------------------------------------------------
  // STEP 1 - Encode NumTrail
  // ---------------------------------------------------------------------------
  int x = 4 * mbX;
  int y = 4 * mbY;

  int nu = getNumNonzero(x, y-4); // number of non-zero coefficients in up block
  int nl = getNumNonzero(x-4, y); // number of non-zero coefficients in left block
  int n;
  int numVlc;

  // Calculate the number of non-zero coefficients in the neighbor blocks
       if (x == 0 && y == 0) n = 0;
  else if (x == 0 && y != 0) n = nu;
  else if (x != 0 && y == 0) n = nl;
  else                       n = (nu + nl + 1)/2;

  // Select table for coding number of coefficients and trail ones
       if (n < 2) numVlc = 0; // VLC0
  else if (n < 4) numVlc = 1; // VLC1
  else if (n < 8) numVlc = 2; // VLC2
  else            numVlc = 3; // FLC

  bitCount += encodeNumTrail(numCoeffs, numT1s, numVlc);

  // ---------------------------------------------------------------------------
  // STEP 2 - Encode SignTrail
  // ---------------------------------------------------------------------------
  if (numT1s != 0)
    bitCount += encodeSignTrail(sign);

  // ---------------------------------------------------------------------------
  // STEP 3 - Encode Levels
  // ---------------------------------------------------------------------------
  int firstLevel = 0;

  if (numCoeffs > numT1s) {
    firstLevel = level[0];

    // In this case the first level is changed from 2,-2,3,-3,... to 1,-1,2,-2...
    if (numT1s < 3) {
      if (level[0] < 0)
        level[0]++;
      else
        level[0]--;
    }
  }

  int levelVlc = 0;

  if (numCoeffs > 10 && numT1s < 3)
    levelVlc = 1;

  for (int i = 0; i < (numCoeffs-numT1s); i++) {
    if (levelVlc == 0)
      bitCount += encodeLevelsVlc0(level[i]);
    else
      bitCount += encodeLevelsVlcN(level[i], levelVlc);

    if (levelVlc == 0)
      levelVlc = 1;

    if (abs(level[i]) > int(3*pow(2.0, double(levelVlc-1))))
      levelVlc++;

    if ((i == 0) && abs(firstLevel) > 3)
      levelVlc = 2;
  }

  // ---------------------------------------------------------------------------
  // STEP 4 - Encode TotalZeros
  // ---------------------------------------------------------------------------
  if (numCoeffs != 0 && numCoeffs != maxCoeffNum)
    bitCount += encodeTotalZeros(numCoeffs, numZeros);

  // ---------------------------------------------------------------------------
  // STEP 5 - Encode Runs
  // ---------------------------------------------------------------------------
  for (int i = 0; i < numCoeffs-1; i++) {
    if (zerosleft[i] == 0)
      break;

    bitCount += encodeRuns(runs[i], zerosleft[i]);
  }

  return bitCount;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeNumTrail(int numCoeffs, int numT1s, int vlc)
{
  int length;
  int code;

  if (vlc != 3) {
    // Select one of the Num-VLC table
    length = NumVlcTableL[vlc][numT1s][numCoeffs];
    code   = NumVlcTableC[vlc][numT1s][numCoeffs];
  }
  else {
    // Use 6-bit fixed length coding (FLC) xxxxyy
    // xxxx for NumCoeffs-1 and yy for NumT1s
    // Codeword 000011 is used when NumCoeff = 0
    length = 6;

    if (numCoeffs > 0)
      code = (((numCoeffs-1) << 2) | numT1s);
    else
      code = 3;
  }

  outputCode(length, code);

  return length;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeSignTrail(vector<int> &sign)
{
  for (unsigned i = 0; i < sign.size(); i++)
    outputCode(1, sign[i]);

  return sign.size();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeLevelsVlc0(int level)
{
  int sign     = (level < 0) ? 1 : 0;
  int absLevel = abs(level);

  int length;
  int code;

  if (absLevel < 8) {
    length = 2*absLevel - 1 + sign;
    code   = 1;
  }
  else if (absLevel < 16) {
    // Escape code 1: 000000000000001xxxx
    length = 19;
    code   = 16 | ((absLevel << 1) - 16) | sign;
  }
  else {
    // Escape code 2: 0000000000000001xxxxxxxxxxxx
    int escapeBase   = 4096;
    int escapeOffset = absLevel + 2048 - 16;
    int numPrefix    = 0;

    if (escapeOffset >= 4096) {
      numPrefix++;

      while (escapeOffset >= (4096 << numPrefix))
        numPrefix++;
    }

    escapeBase <<= numPrefix;

    length = 28 + (numPrefix << 1);
    code   = escapeBase | ((escapeOffset << 1) - escapeBase) | sign;
  }

  outputCode(length, code);

  return length;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeLevelsVlcN(int level, int vlc)
{
  int sign     = (level < 0 ? 1 : 0);
  int absLevel = abs(level) - 1;

  int shift  = vlc - 1;
  int escape = (15 << shift);

  int length;
  int code;

  if (absLevel < escape) {
    int sufmask = ~((0xffffffff) << shift);
    int suffix  = (absLevel) & sufmask;

    length = ((absLevel) >> shift) + 1 + vlc;
    code   = (2 << shift) | (suffix << 1) | sign;
  }
  else {
    // Escape code: 0000000000000001xxxxxxxxxxxx
    int escapeBase = 4096;
    int escapeOffset = absLevel + 2048 - escape;
    int numPrefix = 0;

    if ((escapeOffset) >= 4096) {
      numPrefix++;

      while ((escapeOffset) >= (4096 << numPrefix))
        numPrefix++;
    }

    escapeBase <<= numPrefix;

    length = 28 + (numPrefix << 1);
    code   = escapeBase | ((escapeOffset << 1) - escapeBase) | sign;
  }

  outputCode(length, code);

  return length;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeTotalZeros(int numCoeffs, int numZeros)
{
  int length = TotalZerosTableL[numCoeffs-1][numZeros];
  int code   = TotalZerosTableC[numCoeffs-1][numZeros];

  outputCode(length, code);

  return length;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int CavlcEnc::encodeRuns(int runBefore, int zerosLeft)
{
  int runsLeft = Min(zerosLeft-1, 6);

  int length   = RunTableL[runsLeft][runBefore];
  int code     = RunTableC[runsLeft][runBefore];

  outputCode(length, code);

  return length;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void CavlcEnc::outputCode(int length, int code)
{
  _codec->getBitstream()->write(code, length);

# if TESTPATTERN
  ((Encoder*)_codec)->getBitstreamCavlc()->write(code, length);
# endif
}

