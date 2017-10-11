
#ifndef ENCODER_INC_ENCODER_H
#define ENCODER_INC_ENCODER_H

#include "config.h"
#include "codec.h"
#include <vector>

class FileManager;
class FrameBuffer;
class Transform;
class CavlcEnc;
class LdpcaEnc;
class Codec;

using std::vector;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class Encoder : public Codec
{
public:
  Encoder(char **argv);
  ~Encoder() { /* TODO Remember to free memory space */ };

  void encodeKeyFrame();
  void encodeWzFrame();

# if TESTPATTERN
  Bitstream *getBitstreamCavlc() { return _cavlc_bs; }
# endif

private:
  void initialize();

  void computeResidue(int *residue);
  int  computeSad(imgpel *blk1, int width1, int step1,
                  imgpel *blk2, int width2, int step2, int blockSize);

  void updateMaxValue(int *frame);

  void computeQuantStep();

  void selectCodingMode(int *frame, int CSS_mode);

  void generateSkipMask(int CSS_mode);

  int encodeSkipMask();
  int getHuffmanCode(int type, int symbol, int &code, int &length);

  void encodeByEntropyCode(int *frame);

  void encodeByChannelCode(int *frame);
  void setupLdpcaSource(int *frame, int *src, int offsetX, int offsetY, int bitPos);
  void computeCRC(int *data, const int length, unsigned char *crc);

  void report();

  void loadSVMClassifier();
  int codingStructureSelection(imgpel* prev_raw_fr, imgpel* curr_raw_fr,
                             double* transformed_raw_residue);
  void extractBlkSADFeature (int blk_size, vector<double>& feature_list,
                             imgpel* prev_raw_fr, imgpel* curr_raw_fr);
  void extractDCTFeature(vector<double>& feature_list, double* transformed_raw_residue);
  void generateDownSampleSeq();
  void resample(imgpel* input_img, float resize_ratio, imgpel* resized_img);


private:
  const static int  Scale[3][8];

  FileManager      *_files;

# if TESTPATTERN | FPGA
  Bitstream        *_rlc_bs;
  Bitstream        *_cavlc_bs;
  Bitstream        *_cc_bs;
# endif

  FrameBuffer      *_fb;

  Transform        *_trans;

  CavlcEnc         *_cavlc;
  LdpcaEnc         *_ldpca;

  int               _rcQuantMatrix[BlockSize][BlockSize];
  int               _rcBitPlaneNum;

  int               _maxValue[BlockSize][BlockSize];
  int              *_skipMask;
  int               _prevMode;
  int               _prevType;

  int               _modeCounter[4];

  double           *_trained_mu;
  double           *_trained_std;
  double           *_trained_beta;
  double            _trained_bias;
  int               _feature_dim;
};

#endif // ENCODER_INC_ENCODER_H

