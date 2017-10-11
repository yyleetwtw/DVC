
#ifndef COMMON_INC_TYPES_H
#define COMMON_INC_TYPES_H

typedef unsigned char byte;
typedef unsigned char imgpel;

// Contain 4 rows by 4 columns of 4x4 blocks
typedef struct _mb {
  int   nnz;          // num of nonzero coef for luma
  int   coeffs  [16];
  int   level   [16];
  int   Ones    [3];
  int   iRun    [16];
} mb;

typedef struct _mvinfo {
  int   iCx;     //current block position
  int   iCy;     //
  int   iMvx;    //x-dir motion vector
  int   iMvy;    //y-dir motion vector
  float fDist;
  bool  bOMBCFlag;
} mvinfo;

template <typename T> inline T Min(T a, T b) { return (a < b) ? a : b; }
template <typename T> inline T Max(T a, T b) { return (a > b) ? a : b; }
template <typename T> inline T Clip(T lowBound, T upBound, T value)
{
  value = Max(value, lowBound);
  value = Min(value, upBound);

  return value;
}

template <typename T> T** AllocArray2D(int width, int height)
{
  T** img = new T*[height];

  for (int i = 0; i < height; i++)
    img[i] = new T[width];

  return img;
}

#endif // COMMON_INC_TYPES_H

