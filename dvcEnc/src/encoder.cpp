
#include <iostream>
#include <sstream>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "encoder.h"
#include "transform.h"
#include "fileManager.h"
#include "frameBuffer.h"
#include "cavlcEnc.h"
#include "ldpcaEnc.h"
#include "bitstream.h"
#include "assert.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#include <fstream> //ddd
#include <vector> //ddd
#include <algorithm> //ddd
#define MULTI_RESOLUTION        0   //ddd
#define RESIDUAL_CODING         0   //ddd

using namespace std;
using namespace cv;

const int Encoder::Scale[3][8] = {
  {8, 6, 6, 4, 4, 3, 2, 1},
  {8, 8, 8, 4, 4, 4, 2, 1},
  {4, 4, 4, 4, 3, 2, 2, 1}
};


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
Encoder::Encoder(char **argv)
{
  _files = FileManager::getManager();

  _qp          = atoi(argv[1]);
  _keyQp       = atoi(argv[2]);
  _frameWidth  = atoi(argv[3]);
  _frameHeight = atoi(argv[4]);
  _numFrames   = atoi(argv[5]);
  _gopLevel    = atoi(argv[6]);

  _files->addFile("src", argv[7])->openFile("rb");
  _files->addFile("wz",  argv[8])->openFile("wb");
  // Don't open reconstructed key frame file now because it may not exist yet
  _files->addFile("key", argv[9]);

  _bs = new Bitstream(1024, _files->getFile("wz")->getFileHandle());

# if TESTPATTERN
  _files->addFile("rlc_bs",   "pattern_rlc.bit")->openFile("wb");
  _files->addFile("cavlc_bs", "pattern_cavlc.bit")->openFile("wb");
  _files->addFile("cc_bs",    "pattern_cc.bit")->openFile("wb");

  _rlc_bs   = new Bitstream(1024, _files->getFile("rlc_bs")->getFileHandle());
  _cavlc_bs = new Bitstream(1024, _files->getFile("cavlc_bs")->getFileHandle());
  _cc_bs    = new Bitstream(1024, _files->getFile("cc_bs")->getFileHandle());
# endif

  initialize();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::initialize()
{
  _frameSize        = _frameWidth * _frameHeight;
  _bitPlaneLength   = _frameSize / (BlockSize*BlockSize);
  _gop              = 1 << _gopLevel;

  // How many bands need to be coded, both entropy and channel coding
  _numCodeBands     = 0;

  for (int band = 0; band < 16; band++) {
    int x = ScanOrder[band][0];
    int y = ScanOrder[band][1];

    if (QuantMatrix[_qp][y][x] > 0)
      _numCodeBands++;
  }

  // By default all bands are channel encoded
  _numChnCodeBands  = 16;

  _average          = new double[16];

  _skipMask         = new int[_bitPlaneLength];

  _prevMode         = 0;
  _prevType         = 0;

  for (int i = 0; i < 4; i++)
    _modeCounter[i] = 0;

  _fb = new FrameBuffer(_frameWidth, _frameHeight);

  _trans = new Transform(this);

  _cavlc = new CavlcEnc(this, 4);

  // Initialize LDPC encoder related variables
  string ladderFile;

  _parity = new bool[_bitPlaneLength * BitPlaneNum[_qp]];

# if HARDWARE_LDPC
  if (_frameWidth == 352 && _frameHeight == 288)
    _crc = new unsigned char[BitPlaneNum[_qp] * 4];
  else
    _crc = new unsigned char[BitPlaneNum[_qp]];

  ladderFile = "ldpca/1584_regDeg3.lad";
# else // !HARDWARE_LDPC
  _crc = new unsigned char[BitPlaneNum[_qp]];

  if (_frameWidth == 352 && _frameHeight == 288)
    ladderFile = "ldpca/6336_regDeg3.lad";
  else
    ladderFile = "ldpca/1584_regDeg3.lad";
# endif // HARDWARE_LDPC

  _ldpca = new LdpcaEnc(ladderFile, this);

# if MULTI_RESOLUTION
  loadSVMClassifier();
  generateDownSampleSeq();
# endif // MULTI_RESOLUTION
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::encodeKeyFrame()
{
  string srcFileName = _files->getFile("src")->getFileName();
  string keyFileName = _files->getFile("key")->getFileName();

  // Fix frame rate to 30 for CIF and 15 for QCIF
  float framerate = 30.0;

  if (_frameWidth == 176 && _frameHeight == 144)
    framerate = 15.0;

  // Round up frame number divided by GOP
  // Plus one because the last WZ frame need the next key frame
# if !MULTI_RESOLUTION
  int numKeyFrames = ((_numFrames + _gop/2) / _gop) + 1;
# else
  int numKeyFrames = _numFrames;
# endif

  // Setup the command to execute JM
  stringstream cmd, cmd1;

# if !FPGA
  // Find the directory where the reconstructed key frame file is located
  unsigned found = keyFileName.find_last_of("/");
  string outPath = (found == string::npos) ? "./" : keyFileName.substr(0, found+1);

  cout << "Running JM to encode key frames in CIF...";
  cout << flush;

  cmd << "./lencod.exe -d encoder.cfg ";
  cmd << "-p InputFile=\""        << srcFileName << "\" ";
  cmd << "-p ReconFile=\""        << keyFileName << "\" ";
  // write to tmp, load off line generated key as keyFilename
  cmd << "-p FrameRate="          << framerate << " ";
  cmd << "-p FramesToBeEncoded="  << numKeyFrames << " ";
  cmd << "-p FrameSkip="          << _gop-1 << " ";
  cmd << "-p QPISlice="           << _keyQp << " ";
  cmd << "-p SourceWidth="        << _frameWidth << " ";
  cmd << "-p SourceHeight="       << _frameHeight << " ";
  cmd << "-p StatsFile="          << outPath << "stats_cif.dat ";
  //cmd << " > " << outPath << "jm.log";
  system(cmd.str().c_str());

# if MULTI_RESOLUTION
  string down_seq_name = keyFileName + ".down_src";
  string tmp_qcif_rec = keyFileName + ".qcif";
  cout << "Running JM to encode key frames in QCIF...";
  cout << flush;
  cmd1 << "./lencod.exe -d encoder.cfg ";
  cmd1 << "-p InputFile=\""        << down_seq_name << "\" ";
  cmd1 << "-p ReconFile=\""        << tmp_qcif_rec << "\" ";
  //cmd << "-p ReconFile=\""        << keyFileName << "\" ";
  // write to tmp, load off line generated key as keyFilename
  cmd1 << "-p FrameRate="          << framerate << " ";
  cmd1 << "-p FramesToBeEncoded="  << numKeyFrames << " ";
  cmd1 << "-p FrameSkip="          << _gop-1 << " ";
  cmd1 << "-p QPISlice="           << _keyQp << " ";
  cmd1 << "-p SourceWidth="        << _frameWidth << " ";
  cmd1 << "-p SourceHeight="       << _frameHeight << " ";
  cmd1 << "-p SourceWidth="        << _frameWidth/2 << " ";
  cmd1 << "-p SourceHeight="       << _frameHeight/2 << " ";
  cmd1 << "-p StatsFile="          << outPath << "stats_qcif.dat ";
  //cmd << " > " << outPath << "jm.log";
  system(cmd1.str().c_str());

  // src-> 264, CIF
  string resampled_key_filename = outPath + "resample.y";
  _files->addFile("key_resample",resampled_key_filename)->openFile("rb");
  //_files->getFile("key_resample")->openFile("rb");
  // src-> decimated-> 264-> upsampled, CIF
# endif

# else // FPGA
  cout << "Running proprietary H264 encoder to encode key frames...";
  cout << flush;

  cmd << "echo \"" << srcFileName     << " #InputSequence\"                > H264Enc.cfg;";
  cmd << "echo \"" << numKeyFrames    << " #NumberOfFramesToBeCoded\"     >> H264Enc.cfg;";
  cmd << "echo \"" << _frameWidth     << " #ImageWidthInPels\"            >> H264Enc.cfg;";
  cmd << "echo \"" << _frameHeight    << " #ImageHeightInPels\"           >> H264Enc.cfg;";
  cmd << "echo \"" << "h264enc.trace" << " #OutputTraceInfo\"             >> H264Enc.cfg;";
  cmd << "echo \"" << keyFileName     << " #OutputSequence\"              >> H264Enc.cfg;";
  cmd << "echo \"" << "h264enc.264"   << " #OutputBitstream\"             >> H264Enc.cfg;";
  cmd << "echo \"" << "h264enc.log"   << " #OutputReport\"                >> H264Enc.cfg;";
  cmd << "echo \"" << _keyQp          << " #QuantIntraFrame\"             >> H264Enc.cfg;";
  cmd << "echo \"" << _keyQp          << " #QuantInterFrames\"            >> H264Enc.cfg;";
  cmd << "echo \"" << (_gop-1)        << " #NumberOfSkippedFrames\"       >> H264Enc.cfg;";
  cmd << "echo \"" << "4"             << " #NumberReferenceFrames\"       >> H264Enc.cfg;";
  cmd << "echo \"" << "16"            << " #HorizontalAbsMaxSearchRange\" >> H264Enc.cfg;";
  cmd << "echo \"" << "16"            << " #VerticalAbsMaxSearchRange\"   >> H264Enc.cfg;";
  cmd << "echo \"" << "1"             << " #PeriodOfIntraFrames\"         >> H264Enc.cfg;";
  cmd << "echo \"" << "0"             << " #SupportHDTV720p\"             >> H264Enc.cfg;";
  cmd << "wine H264Enc_App.exe H264Enc.cfg > /dev/null 2> /dev/null";

  system(cmd.str().c_str());
# endif // FPGA

  cout << "done" << endl << endl;

  // Now open the reconstructed key frame file after it is created by JM
  _files->getFile("key")->openFile("rb");

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::encodeWzFrame()
{
  FILE *srcFh = _files->getFile("src")->getFileHandle();
  FILE *recFh = _files->getFile("key")->getFileHandle();
# if MULTI_RESOLUTION
  FILE *resample_recFH = _files->getFile("key_resample")->getFileHandle();
# endif

  imgpel *currFrame     = _fb->getCurrFrame();
  int    *dctFrame      = _fb->getDctFrame();
  int    *quantDctFrame = _fb->getQuantDctFrame();
  int    *residue       = new int[_frameSize];

  clock_t timeStart;
  clock_t timeEnd;
  double  cpuTime;

# if TESTPATTERN
  // Input pattern for Verilog simulation
  File *patInFile;
  FILE *patInFh;

  // Clean up the file
  patInFile = _files->addFile("pattern_input", "pattern_input.dat");
  patInFile->openFile("w");
  patInFile->closeFile();
# endif

  timeStart = clock();

  // Encode WZ header information
  // ---------------------------------------------------------------------------
  _bs->write(_frameWidth/16,   8); // unit = macroblock
  _bs->write(_frameHeight/16,  8); // unit = macroblock
  _bs->write(_qp,              8);
  _bs->write(_numFrames,       16);
  _bs->write(_gopLevel,        2);

  // Main loop
  // ---------------------------------------------------------------------------
  int keyFrameNo;
  int wzFrameNo;
  int CSS_mode = 0;
  int num_gop_count = 0;
# if !MULTI_RESOLUTION
  for (keyFrameNo = 0; keyFrameNo < _numFrames/_gop; keyFrameNo++) {
# else
  for (keyFrameNo = 0; keyFrameNo < _numFrames/_gop;) {
    // maintain the loop var keyFrameNo according to CSS result
    imgpel *prev_src, *curr_src;
    prev_src = new imgpel [_frameSize];
    curr_src = new imgpel [_frameSize];
    int *residue_src = new int [_frameSize];
    double *transformed_residue_raw = new double [_frameSize];
    if (keyFrameNo>0) // first GOP always using Hybrid
    {
        fseek(srcFh, keyFrameNo*_frameSize*3/2, SEEK_SET);
        fread(curr_src, _frameSize, 1, srcFh);
        fseek(srcFh, (keyFrameNo-1)*_frameSize*3/2, SEEK_SET);
        fread(prev_src, _frameSize, 1, srcFh);
        for (int i=0; i<_frameSize; ++i)
            residue_src[i] = curr_src[i] - prev_src[i];
        _trans->dctTransform(residue_src, transformed_residue_raw);

        CSS_mode = codingStructureSelection(prev_src, curr_src, transformed_residue_raw);
        //CSS_mode = 0;
    }

    _bs->write(CSS_mode, 1);
# endif


    // Read previous key frame from the reconstructed key frame file
    // Set position to 1.5x because the file is in YUV420 format
# if !MULTI_RESOLUTION
    fseek(recFh, keyFrameNo*_frameSize*3/2, SEEK_SET);
    fread(_fb->getPrevFrame(), _frameSize, 1, recFh);
# else
    // prepare residue reference frame into PrevFrame
    // reference frame : resample_recFH for MRDVC with frame# = keyFrameNo
    //                   recFG for HybridDVC with frame# = keyFrameNo
    if (CSS_mode==0){
        fseek(recFh, keyFrameNo*_frameSize*3/2, SEEK_SET);
        fread(_fb->getPrevFrame(), _frameSize, 1, recFh);
    }
    else if (CSS_mode==1){
        fseek(resample_recFH, keyFrameNo*_frameSize, SEEK_SET);
        fread(_fb->getPrevFrame(), _frameSize, 1, resample_recFH);
    }
    else
        cout << "ERROR CSS mode" << endl;
    // use these 2 line when key frame = orig size
# endif

/*
not necessary becuase computeResidue now only need prev_rec & curr_src
# if !MULTI_RESOLUTION
    fseek(recFh, (keyFrameNo+1)*_frameSize*3/2, SEEK_SET);
    fread(_fb->getNextFrame(), _frameSize, 1, recFh);
# else
    if (keyFrameNo != _numFrames/_gop-1){
      fseek(recFh, (keyFrameNo+1)*_frameSize, SEEK_SET); // bicup y only
      fread(_fb->getNextFrame(), _frameSize, 1, recFh);
    } // for orig size key frame
# endif
*/
    // Loop through every GOP level
# if !MULTI_RESOLUTION
    for (int gl = 0; gl < _gopLevel; gl++) {
        int frameStep = _gop / (2*(gl+1));
# else
    for (int gl = 0; gl <= _gopLevel; gl++) {
        // current version MRDVC only compare between GOP 1 & 2
        int frameStep = (CSS_mode==1)? 0 : 1;
# endif
        int idx = frameStep;
      // Start encoding the WZ frame
# if !MULTI_RESOLUTION
      while (idx < _gop) {
# else
      while (idx < frameStep+1) {
      // current version MRDVC only compare between GOP 1 & 2
      // namely, 1 key for 1 WZ no matter MRDVC or HDVC
# endif
        wzFrameNo = keyFrameNo*_gop + idx;

        cout << "Encoding frame " << wzFrameNo << " (Wyner-Ziv frame)" << endl;

        fseek(srcFh, wzFrameNo*_frameSize*3/2, SEEK_SET);
        fread(currFrame, _frameSize, 1, srcFh);

# if TESTPATTERN
        patInFile = _files->getFile("pattern_input");
        patInFile->openFile("a");
        patInFh = patInFile->getFileHandle();

        for (int y = 0; y < _frameHeight; y++)
          for (int x = 0; x < _frameWidth; x++) {
            int pos = x + y*_frameWidth;

            fprintf(patInFh, "%02x", currFrame[pos]);

            if (x % 4 == 3)
              fprintf(patInFh, "\n");
          }

        for (int y = 0; y < _frameHeight; y++)
          for (int x = 0; x < _frameWidth; x++) {
            int pos = x + y*_frameWidth;

            fprintf(patInFh, "%02x", _fb->getPrevFrame()[pos]);

            if (x % 4 == 3)
              fprintf(patInFh, "\n");
          }

        patInFile->closeFile();
# endif // TESTPATTERN

        // ---------------------------------------------------------------------
        // STAGE 1 - Residual coding & DCT
        // ---------------------------------------------------------------------
# if RESIDUAL_CODING
        computeResidue(residue);
        // RCList write to _bs in this function

        _trans->dctTransform(residue, dctFrame);

# else // if !RESIDUAL_CODING
        _trans->dctTransform(currFrame, dctFrame);
# endif // RESIDUAL_CODING

        // Find the largest value of every block entry, which is later used to
        // determine quantization step size
        updateMaxValue(dctFrame);

        // ---------------------------------------------------------------------
        // STAGE 2 - Calculate quantization step size
        // ---------------------------------------------------------------------
        computeQuantStep();

        // ---------------------------------------------------------------------
        // STAGE 3 - Quantization
        // ---------------------------------------------------------------------
        _trans->quantization(dctFrame, quantDctFrame);

        // ---------------------------------------------------------------------
        // STAGE 4 - Mode decision
        // ---------------------------------------------------------------------
# if MODE_DECISION
        selectCodingMode(quantDctFrame, CSS_mode);
# endif // MODE_DECISION

        // ---------------------------------------------------------------------
        // STAGE 5 - Skip mode
        // ---------------------------------------------------------------------
# if SKIP_MODE
        generateSkipMask(CSS_mode);

        encodeSkipMask();
# endif // SKIP_MODE

        // ---------------------------------------------------------------------
        // STAGE 6 - Encode (entropy/channel)
        // ---------------------------------------------------------------------
        encodeByEntropyCode(quantDctFrame);

        encodeByChannelCode(quantDctFrame);
        // Go to next frame within the GOP level
# if !MULTI_RESOLUTION
        idx += 2*frameStep;
# else
        idx += 1;
# endif
      } // End of all frames within the GOP level
    } // End of GOP levels
# if MULTI_RESOLUTION
    //cout << "(css,keyno,wzno) = (" << CSS_mode << "  ,  "
    //     << keyFrameNo << "  ,  " << wzFrameNo << endl;
    if (CSS_mode == 0){
        keyFrameNo += 2;
    } else if (CSS_mode == 1){
        keyFrameNo += 1;
    } else
        cout << "ERROR CSS_mode" << endl;
# endif
    num_gop_count += 1;
  } // End of key frames

  _bs->flush();

# if TESTPATTERN
  _rlc_bs->flush();
  _cavlc_bs->flush();
  _cc_bs->flush();
# endif

  timeEnd = clock();
  cpuTime = (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000.0;

  cout << endl;
  cout << "--------------------------------------------------" << endl;
  cout << "Encode statistics" << endl;
  cout << "--------------------------------------------------" << endl;

  report();

  cout << "Total   encoding time: " << cpuTime << "(ms)" << endl;
  cout << "Average encoding time: " << cpuTime/_numFrames << "(s)" << endl;
  cout << " num gop count: " << num_gop_count << endl;
  cout << "--------------------------------------------------" << endl;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::computeResidue(int *residue)
{
  imgpel *bckRefFrame = _fb->getPrevFrame(); // reconstructed key frame
  imgpel *currFrame   = _fb->getCurrFrame(); // src frame
  //imgpel *fwdRefFrame = _fb->getNextFrame();
  imgpel *refFrame;

  int  blockIdx = 0;
  int  blockCnt = _frameSize / (ResidualBlockSize * ResidualBlockSize);
  int *dirList  = new int[blockCnt];

  memset(dirList, 0, sizeof(int)*blockCnt);

  for (int y = 0; y < _frameHeight; y += ResidualBlockSize){
    for (int x = 0; x < _frameWidth; x += ResidualBlockSize) {
      // First determine the reference frame
# if HARDWARE_OPT
      // Always use backward reference for hardware implementation
      refFrame = bckRefFrame;
# else // if !HARDWARE_OPT

/*
      int bckDist; // backward distortion
      int fwdDist; // forward distortion

      bckDist = computeSad(currFrame   + x + y*_frameWidth, _frameWidth, 1,
                           bckRefFrame + x + y*_frameWidth, _frameWidth, 1,
                           ResidualBlockSize);

      fwdDist = computeSad(currFrame   + x + y*_frameWidth, _frameWidth, 1,
                           fwdRefFrame + x + y*_frameWidth, _frameWidth, 1,
                           ResidualBlockSize);
*/
      // Use frame with lower distortion as reference frame
      // 0: backward reference, 1: forward reference
      dirList[blockIdx] = 0;
      //dirList[blockIdx] = (bckDist <= fwdDist) ? 0 : 1;

      refFrame = bckRefFrame;
      //refFrame = (dirList[blockIdx] == 0) ? bckRefFrame : fwdRefFrame;

# endif // HARDWARE_OPT

      // Then calculate the residue
      for (int j = 0; j < ResidualBlockSize; j++)
        for (int i = 0; i < ResidualBlockSize; i++) {
          int idx = (x+i) + (y+j)*_frameWidth;

          residue[idx] = currFrame[idx] - refFrame[idx];
        }

      blockIdx++;
    }
  }

# if !HARDWARE_FLOW | RESIDUAL_CODING
    // Encode motion vector
    for (int i = 0; i < blockCnt; i++)
      _bs->write(dirList[i], 1);
# endif
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int Encoder::computeSad(imgpel *blk1, int width1, int step1,
                        imgpel *blk2, int width2, int step2, int blockSize)
{
  int sad = 0;

  for (int y = 0; y < blockSize; y++)
    for (int x = 0; x < blockSize; x++) {
      imgpel pel1 = *(blk1 + step1*x + step1*y*width1);
      imgpel pel2 = *(blk2 + step2*x + step2*y*width2);

      sad += abs(pel1 - pel2);
    }

  return sad;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::updateMaxValue(int *frame)
{
  for (int y = 0; y < BlockSize; y++)
    for (int x = 0; x < BlockSize; x++)
      _maxValue[y][x] = 0;

  for (int y = 0; y < _frameHeight; y += BlockSize)
    for (int x = 0; x < _frameWidth; x += BlockSize)
      for (int j = 0; j < BlockSize; j++)
        for (int i = 0; i < BlockSize; i++)
          if (abs(_maxValue[j][i]) < abs(frame[(x+i)+(y+j)*_frameWidth]))
            _maxValue[j][i] = frame[(x+i)+(y+j)*_frameWidth];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::computeQuantStep()
{
  for (int j = 0; j < BlockSize; j++) {
    for (int i = 0; i < BlockSize; i++) {
# if RESIDUAL_CODING // --------------------------------------------------------
#   if !HARDWARE_FLOW
      _bs->write(abs(_maxValue[j][i]), 11);
#   endif

      if (QuantMatrix[_qp][j][i] != 0) {
#   if HARDWARE_QUANTIZATION
        _quantStep[j][i] = 1 << (MaxBitPlane[j][i]+1-QuantMatrix[_qp][j][i]);
#   else // if !HARDWARE_QUANTIZATION
        int iInterval = 1 << QuantMatrix[_qp][j][i];

        _quantStep[j][i] = (int)ceil(double(2*abs(_maxValue[j][i]))/double(iInterval-1));
        _quantStep[j][i] = Max(_quantStep[j][i], MinQStepSize[_qp][j][i]);
        //cout << (int)HARDWARE_QUANTIZATION << ',' << (int)RESIDUAL_CODING << endl;
#   endif // HARDWARE_QUANTIZATION
      }
      else
        _quantStep[j][i] = 1;

# else // if !RESIDUAL_CODING --------------------------------------------------

      if (i != 0 || j != 0) {
        _bs->write(abs(_maxValue[j][i]), 11);
#   if HARDWARE_QUANTIZATION
        if (QuantMatrix[_qp][j][i] != 0)
          _quantStep[j][i] = 1 << (MaxBitPlane[j][i]+1-QuantMatrix[_qp][j][i]);
        else
          _quantStep[j][i] = 0;
#   else // if !HARDWARE_QUANTIZATION
        int iInterval = 1 << QuantMatrix[_qp][j][i];

#     if AC_QSTEP
        if (QuantMatrix[_qp][j][i] != 0) {
          _quantStep[j][i] = (int)ceil(double(2*abs(_maxValue[j][i]))/double(iInterval-1));

          if (_quantStep[j][i] < 0)
            _quantStep[j][i] = 0;
        }
        else
          _quantStep[j][i] = 1;
#     else // if !AC_QSTEP
        if (QuantMatrix[_qp][j][i] != 0)
          _quantStep[j][i] = ceil(double(2*abs(_maxValue[j][i]))/double(iInterval));
        else
          _quantStep[j][i] = 1;
#     endif // AC_QSTEP

#   endif // HARDWARE_QUANTIZATION
      }
      else
        _quantStep[j][i] = 1 << (DC_BITDEPTH-QuantMatrix[_qp][j][i]);
# endif // RESIDUAL_CODING -----------------------------------------------------
    }
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::selectCodingMode(int *frame, int CSS_mode)
{
  // Calculate average value for each band
  memset(_average, 0, 16*sizeof(double));

  for (int j = 0; j < _frameHeight; j++)
    for (int i = 0; i < _frameWidth; i++) {
      //int mask = (0x1 << (QuantMatrix[_qp][j%4][i%4]-1)) - 1;
      int data = abs(frame[i+j*_frameWidth]);   //int data = frame[i + j*_frameWidth] & mask;

      _average[(i%4) + (j%4)*4] += data;
    }

  for (int i = 0; i < 16; i++)
    _average[i] /= _bitPlaneLength;

  double energy1 = 0.0;
  double energy2 = 0.0;
  double energy3 = 0.0;

  for (int i = 0; i < 16; i++) {
    int x = ScanOrder[i][0];
    int y = ScanOrder[i][1];

    if (i < 3)
      energy1 += _average[x + y*4];
    else if (i < 6)
      energy2 += _average[x + y*4];
    else
      energy3 += _average[x + y*4];
  }

  energy1 /= 3;

  if (_numCodeBands > 3) {
    if (_numCodeBands >= 6)
      energy2 /= 3;
    else
      energy2 /= (_numCodeBands-3);
  }
  else
    energy2 = 0;

  if (_numCodeBands > 6)
    energy3 /= (_numCodeBands-6);
  else
    energy3 = 0;

  double th1 = 4; //double th1 = 0.15;
  double th2 = 1.5; //double th2 = 0.05;
  double th3 = 0.3; //double th3 = 0.01;
  int    mode;

    if (CSS_mode == 0){
    // Determine coding mode based on the energy calculated
        if (energy1 > (th1/(double)(Scale[0][_qp]))) {
            if (energy2 > (th2/(double)(Scale[1][_qp]))) {
                if (energy3 > (th3/(double)(Scale[2][_qp])))
                    mode = 0; // channel coding (channel coding for all bands)
                else
                    mode = 2; // hybrid mode 2 (channel coding for lower 6 bands
                              //                entropy coding for other bands)
            }
        else
            mode = 1;   // hybrid mode 1 (channel coding for lower 3 bands
                        //                entropy coding for other bands)
        }
        else
            mode = 3;     // entropy coding (entropy coding for all bands)
    }
    else if (CSS_mode == 1){
       mode = 3;
    }

  _modeCounter[mode]++;

# if HARDWARE_CMS
  int currMode = mode;

  // Use the previous mode for the current frame
  mode = _prevMode;
  // And store current mode for the next frame
  _prevMode = currMode;
# endif // HARDWARE_CMS

  _bs->write(mode, 2);

  if (mode == 0) _numChnCodeBands = 16; else
  if (mode == 1) _numChnCodeBands =  3; else
  if (mode == 2) _numChnCodeBands =  6; else
                 _numChnCodeBands =  0;

  cout << "CMS = " << mode << endl;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::generateSkipMask(int CSS_mode)
{
# if HARDWARE_OPT
  int threshold = 2;
# else // if !HARDWARE_OPT
  int threshold;
  if (CSS_mode==0)
    threshold = 5;
  else
    threshold = 1;
# endif // HARDWARE_OPT

# if RESIDUAL_CODING
  int *frame    = _fb->getQuantDctFrame();
# else // if !RESIDUAL_CODING
  int *frame    = new int[_frameSize];
  int *frameDct = new int[_frameSize];

  for (int i = 0; i < _frameSize; i++)
    frame[i] = _fb->getPrevFrame()[i] - _fb->getCurrFrame()[i];

  _trans->dctTransform(frame, frameDct);
  _trans->quantization(frameDct, frame);

# endif // !RESIDUAL_CODING

  memset(_skipMask, 0, sizeof(int)*_bitPlaneLength);

  for (int j = 0; j < _frameHeight; j += BlockSize)
    for (int i = 0; i < _frameWidth; i += BlockSize) {
      int distortion = 0;
      int blockIndex = i/BlockSize + (j/BlockSize)*(_frameWidth/BlockSize);

      for (int y = 0; y < BlockSize; y++)
        for (int x = 0; x < BlockSize; x++) {
          int mask = (0x1 << (QuantMatrix[_qp][y][x]-1)) - 1;
          int data = frame[(i+x) + (j+y)*_frameWidth] & mask;

# if HARDWARE_OPT
          distortion += data;
# else // if !HARDWARE_OPT
          distortion += data * data;
# endif // HARDWARE_OPT
        }

      _skipMask[blockIndex] = (distortion < threshold) ? 1 : 0;
    }


// experiment code start  ------------------------------
/*
  vector<int> SSEsort;
  vector<int>::iterator itr;
  int cmpSSE = 0;
  int numSkip = 0;
  int targetNumSkip = 6336*0.9;
  int *blkSSE = new int [_frameHeight*_frameWidth/(BlockSize*BlockSize)];

  for (int j = 0; j < _frameHeight; j += BlockSize)
    for (int i = 0; i < _frameWidth; i += BlockSize) {
      int distortion = 0;
      int blockIndex = i/BlockSize + (j/BlockSize)*(_frameWidth/BlockSize);

      for (int y = 0; y < BlockSize; y++)
        for (int x = 0; x < BlockSize; x++) {
          int mask = (0x1 << (QuantMatrix[_qp][y][x]-1)) - 1;
          int data = frame[(i+x) + (j+y)*_frameWidth] & mask;
          distortion += data * data;
        }

      blkSSE[blockIndex] = distortion;
    }

    for (int i=0; i<_frameHeight*_frameWidth/(BlockSize*BlockSize); i++){
        cmpSSE = blkSSE[i];
        itr = find(SSEsort.begin(), SSEsort.end(), cmpSSE);
        if (itr == SSEsort.end() )
            SSEsort.push_back(cmpSSE);
    }

    sort(SSEsort.begin(), SSEsort.end());

    for ( itr=SSEsort.begin(); itr!=SSEsort.end(); itr++ ){
        cmpSSE = *itr;
        for (int j = 0; j < _frameHeight; j += BlockSize){
            for (int i = 0; i < _frameWidth; i += BlockSize) {
                int blockIndex = i/BlockSize + (j/BlockSize)*(_frameWidth/BlockSize);
                if ( blkSSE[blockIndex] == cmpSSE ){
                    _skipMask[blockIndex] = 1;
                    numSkip += 1;
                }
            }
        }
        if (numSkip >= targetNumSkip)
            break;
    }
    cout << "terminate_SSE = " << cmpSSE << endl;
    cout << "numSkip = " << numSkip << endl;
    delete []blkSSE;
*/
// end of test code --------------------------

# if !RESIDUAL_CODING
  delete [] frame;
  delete [] frameDct;
# endif // !RESIDUAL_CODING
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int Encoder::encodeSkipMask()
{
  int n0 = 0; // number of non-skipped blocks
  int bitCount;
  int sign;
  int code;
  int length;
  int run = 0;

# if HARDWARE_FLOW
  static int isFirstFrame = 1;

  // Pad zero to align to 32-bit boundary
  if (isFirstFrame)
    _bs->write(0, 20);
  else
    _bs->write(0, 30);

  if (isFirstFrame)
    isFirstFrame = 0;
# endif // HARDWARE_FLOW

  for (int i = 0; i < _bitPlaneLength; i++)
    if (_skipMask[i] == 0)
      n0++;

  int diff = abs(n0 - _bitPlaneLength/2);

  int type = diff / (_bitPlaneLength/6);

# if HARDWARE_OPT
  int currType = type;

  // Use the previous type for the current frame
  type = _prevType;
  // And store current type for the next frame
  _prevType = currType;
# endif // HARDWARE_OPT

  if (type > 2) type = 2;

  _bs->write(type, 2);
# if TESTPATTERN
  _rlc_bs->write(type, 2);
# endif

  bitCount = 2;

  // Directly output the first bit to bitstream
  sign = _skipMask[0];

  _bs->write(sign, 1);
# if TESTPATTERN
  _rlc_bs->write(sign, 1);
# endif

  run++;
  bitCount++;

  // Huffman code for other bits
  for (int i = 1; i < _bitPlaneLength; i++) {
    if (_skipMask[i] == sign) {
      run++;

      if (run == 16) { // reach maximum run length
        bitCount += getHuffmanCode(type, run-1, code, length);

        _bs->write(code, length);
# if TESTPATTERN
        _rlc_bs->write(code, length);
# endif

        run = 1;
      }
    }
    else {
      bitCount += getHuffmanCode(type, run-1, code, length);

      _bs->write(code, length);
# if TESTPATTERN
      _rlc_bs->write(code, length);
# endif

      sign = _skipMask[i];
      run = 1;
    }
  }

  if (run != 0) {
    bitCount += getHuffmanCode(type, run-1, code, length);

    _bs->write(code, length);
# if TESTPATTERN
    _rlc_bs->write(code, length);
# endif
  }

# if HARDWARE_FLOW
  if (bitCount%32 != 0) { // pad zero to align to 32-bit boundary
    int dummy = 32 - (bitCount%32);
    _bs->write(0, dummy);
  }
# endif // HARDWARE_FLOW

# if TESTPATTERN
  if (bitCount%32 != 0) { // pad zero to align to 32-bit boundary
    int dummy = 32 - (bitCount%32);
    _rlc_bs->write(0, dummy);
  }
# endif // TESTPATTERN

  return bitCount;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int Encoder::getHuffmanCode(int type, int symbol, int &code, int &length)
{
  int table = _qp / 2;

  code   = HuffmanCodeValue [table][type][symbol];
  length = HuffmanCodeLength[table][type][symbol];

  return length;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::encodeByEntropyCode(int *frame)
{
  int bitCount = 0;

  // Entropy encode
  if (_numCodeBands > _numChnCodeBands)
    bitCount = _cavlc->encode(frame, _skipMask);

# if HARDWARE_FLOW
  if (bitCount%32 != 0) { // pad zero to align to 32-bit boundary
    int dummy = 32 - (bitCount%32);
    _bs->write(0, dummy);
  }
# endif // HARDWARE_FLOW

# if TESTPATTERN
  if (bitCount%32 != 0) { // pad zero to align to 32-bit boundary
    int dummy = 32 - (bitCount%32);
    _cavlc_bs->write(0, dummy);
  }
# endif
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::encodeByChannelCode(int* frame)
{
  _rcBitPlaneNum = 0;

  for (int band = 0; band < 16; band++) {
    int x = ScanOrder[band][0];
    int y = ScanOrder[band][1];

    if (band < _numChnCodeBands) {
      // Quantization matrix for residual coding
      // Simply equal to the original one
      _rcQuantMatrix[y][x] = QuantMatrix[_qp][y][x];

      // Add up all numbers to get bit plane number for residual coding
      _rcBitPlaneNum += _rcQuantMatrix[y][x];
    }
  }

  int            *ldpcaSource = new int[_bitPlaneLength + 8]; // add 8 for CRC
  bool           *accumulatedSyndrome = _parity;
  unsigned char  *crc = _crc;

  for (int band = 0; band < _numChnCodeBands; band++) {
    int x = ScanOrder[band][0];
    int y = ScanOrder[band][1];

# if RESIDUAL_CODING
    for (int bitPos = _rcQuantMatrix[y][x]-1; bitPos >= 0; bitPos--)
# else // if !RESIDUAL_CODING
    for (int bitPos = QuantMatrix[_qp][y][x]-1; bitPos >= 0; bitPos--)
# endif // RESIDUAL_CODING
    {
      // Use bits at the bit position as the source for channel encoder
      setupLdpcaSource(frame, ldpcaSource, x, y, bitPos);

# if HARDWARE_LDPC
      if (_bitPlaneLength == 6336) {
        for (int n = 0; n < 4; n++) {
          _ldpca->encode(ldpcaSource + n*1584, accumulatedSyndrome);

          computeCRC(ldpcaSource + n*1584, 1584, crc+n);

          accumulatedSyndrome += _bitPlaneLength/4;
        }

        crc += 4;
        cout << ".";
      }
      else {
        _ldpca->encode(ldpcaSource, accumulatedSyndrome);

        computeCRC(ldpcaSource, _bitPlaneLength, crc);

        accumulatedSyndrome += _bitPlaneLength;

        crc++;
        cout << ".";
      }
# else // if !HARDWARE_LDPC
      _ldpca->encode(ldpcaSource, accumulatedSyndrome);

      computeCRC(ldpcaSource, _bitPlaneLength, crc);

      accumulatedSyndrome += _bitPlaneLength;

      crc++;
      cout << ".";
# endif // HARDWARE_LDPC
    }
  }

  cout << endl;

# if !FPGA
#   if RESIDUAL_CODING
  for (int i = 0; i < _rcBitPlaneNum; i++)
#   else // if !RESIDUAL_CODING
  for (int i = 0; i < BitPlaneNum[_qp]; i++)
#   endif // RESIDUAL_CODING
  {
    for (int j = 0; j < _bitPlaneLength; j++){
      _bs->write(int(_parity[j+i*_bitPlaneLength]), 1);
      }

#   if !HARDWARE_FLOW
#     if HARDWARE_LDPC
    if (_bitPlaneLength == 6336)
      for (int n = 0; n < 4; n++)
        _bs->write(_crc[i*4+n], 8);
    else
      _bs->write(_crc[i], 8);
#     else // if !HARDWARE_LDPC
    _bs->write(_crc[i], 8);

#     endif // HARDWARE_LDPC
#   endif // !HARDWARE_FLOW
  }
# else // FPGA
  for (int j = 0; j < _bitPlaneLength; j++) {
    for (int i = 0; i < 64; i++)
      if (i < _rcBitPlaneNum) {
        _bs->write(int(_parity[j+i*_bitPlaneLength]), 1);
        _cc_bs->write(int(_parity[j+i*_bitPlaneLength]), 1);
      }
      else {
        _bs->write(0, 1);
        _cc_bs->write(0, 1);
      }
  }
# endif // !FPGA

  delete [] ldpcaSource;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::setupLdpcaSource(int *frame, int *src, int offsetX, int offsetY, int bitPos)
{
  for (int y = 0; y < _frameHeight; y += 4)
    for (int x = 0; x < _frameWidth; x += 4) {
      int blockIdx = (x/4) + (y/4)*(_frameWidth/4);
      int frameIdx = (x+offsetX) + (y+offsetY)*_frameWidth;

      // Extract the bit at the bit position of the pixel at the frame index
      // And put it at the block index of the source
# if SKIP_MODE
      if (_skipMask[blockIdx] == 1)
        src[blockIdx] = 0;
      else
        src[blockIdx] = (frame[frameIdx] >> bitPos) & 0x1;
# else // if !SKIP_MODE
      src[blockIdx] = (frame[frameIdx] >> bitPos) & 0x1;
# endif // SKIP_MODE

    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::computeCRC(int *data, const int length, unsigned char *crc)
{
  // CRC8 110011011
  const int code[9] = {1, 1, 0, 0, 1, 1, 0, 1, 1};

  int *buffer = new int[length + 8];

  memcpy(buffer, data, length*sizeof(int));

  for (int i = length; i < length+8; i++)
    buffer[i] = 0;

  for (int i = 0; i < length; i++)
    if (buffer[i] == 1)
      for (int j = 0; j < 9; j++)
        buffer[i+j] = code[j] ^ buffer[i+j];

  *crc = 0;

  for (int i = 0; i < 8; i++)
    *crc |= buffer[length+i] << (7-i);

  delete [] buffer;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::report()
{
# if MODE_DECISION
  cout << "Mode usage: ";

  for (int i = 0; i < 4; i++) {
    float usage = (float)_modeCounter[i]/75.0 * 100.0;
    cout << usage << " ";
  }

  cout << endl;
# endif // MODE_DECISION
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::loadSVMClassifier()
{
    // ------ code for parsing svm classifier --------
    ifstream classifier;
    classifier.open("ldpca/trainedSVM.txt");
    if(!classifier.good())
        cout << "failed loading trainedSVM model" << endl;

    double tmp;
    vector<double> svm_classifier;
    while (!classifier.eof())
        {
            classifier >> tmp;
            svm_classifier.push_back(tmp);
        }
    svm_classifier.pop_back();
    classifier.close();

    int svm_rows = 4;
    // svmfile formate :
    // [mu; std; beta; bias(with padded zero to same #column of feature_dim);];
    _feature_dim = svm_classifier.size()/svm_rows;
    assert(svm_classifier.size()%svm_rows==0);
    _trained_mu = new double [_feature_dim];
    _trained_std = new double [_feature_dim];
    _trained_beta = new double [_feature_dim];
    for (int i=0; i<_feature_dim; ++i){
        _trained_mu[i] = svm_classifier[i];
        _trained_std[i] = svm_classifier[i+_feature_dim];
        _trained_beta[i] = svm_classifier[i+2*_feature_dim];
    }
    _trained_bias = svm_classifier[3*_feature_dim];

}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int Encoder::codingStructureSelection(imgpel* prev_raw_fr, imgpel* curr_raw_fr,
                             double* transformed_raw_residue)
{
    // return 0 for hybrid, 1 for mult-resolution
    // fr abbreaviation of full-resolution
    int css_mode = 0;

    vector<double> feature_list;
    // --- step 1, feature extraction

    extractBlkSADFeature(4,feature_list,prev_raw_fr,curr_raw_fr);
    extractBlkSADFeature(8,feature_list,prev_raw_fr,curr_raw_fr);
    extractBlkSADFeature(16,feature_list,prev_raw_fr,curr_raw_fr);
    extractBlkSADFeature(32,feature_list,prev_raw_fr,curr_raw_fr);

    extractDCTFeature(feature_list, transformed_raw_residue);

    double tmp = 0;
    for (int i=0; i<_feature_dim; ++i){
        tmp = tmp + (feature_list[i] - _trained_mu[i])/_trained_std[i] * _trained_beta[i];
    }
    tmp = tmp + _trained_bias;

    //cout << "score = " << tmp << endl;
    if (tmp >= 0)
        css_mode = 1;

    return css_mode;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::extractBlkSADFeature (int blk_size, vector<double>& feature_list,
                                    imgpel* prev_raw_fr, imgpel* curr_raw_fr)
{
    assert(blk_size%4==0);
    int width = _frameWidth;
    int height = _frameHeight;
    int* temporal_residue = new int [width*height];
    for (int i=0; i<width*height; ++i)
        temporal_residue[i] = abs(curr_raw_fr[i] - prev_raw_fr[i]);

    // prepare container for feature extraction
    int blk_sad_buf = 0;
    vector<double> blk_sad_list;
    for (int y=0; y<height; y=y+blk_size){
        for (int x=0; x<width; x=x+blk_size)
        {
            blk_sad_buf = 0;
            for (int blk_y=0; blk_y<blk_size; ++blk_y){
                for (int blk_x=0; blk_x<blk_size; ++blk_x)
                {
                    blk_sad_buf = blk_sad_buf + temporal_residue[(blk_x+x)+(blk_y+y)*width];
                }
            }
            blk_sad_list.push_back((double)blk_sad_buf);
        }
    }

    // feature extraction (mdeidan,max,interquarterfile)
    sort(blk_sad_list.begin(), blk_sad_list.end());

    double median_blk, max_blk, irq_blk;
    median_blk = (blk_sad_list[ceil((blk_sad_list.size()-1)/2.0)]+
                  blk_sad_list[floor((blk_sad_list.size()-1)/2.0)])/2.0;
    max_blk = blk_sad_list[blk_sad_list.size()-1];
    float quart_1st = (blk_sad_list.size()-1)/4.0;
    float quart_3rd = (blk_sad_list.size()-1)/4.0*3.0;
    irq_blk = (blk_sad_list[ceil(quart_3rd)]+blk_sad_list[floor(quart_3rd)])/2.0 -
              (blk_sad_list[ceil(quart_1st)]+blk_sad_list[floor(quart_1st)])/2.0;

    // feature extraction (mean, std, var)
    double sum = 0;
    for (int i=0; i<blk_sad_list.size(); ++i){
        sum = sum + blk_sad_list[i];
    }

    double mean_blk, var_blk, std_blk;
    mean_blk = sum / (double)blk_sad_list.size();
    double square_sum = 0;
    double tmp;
    for (int i=0; i<blk_sad_list.size(); ++i){
        tmp =  blk_sad_list[i] - mean_blk;
        square_sum = square_sum + tmp * tmp;
    }
    var_blk = square_sum / (double)blk_sad_list.size();
    std_blk = sqrt(var_blk);

    feature_list.push_back(median_blk);
    feature_list.push_back(max_blk);
    feature_list.push_back(mean_blk);
    feature_list.push_back(std_blk);
    feature_list.push_back(var_blk);
    feature_list.push_back(irq_blk);
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::extractDCTFeature(vector<double>& feature_list, double* transformed_raw_residue)
{
    int width = _frameWidth;
    int height = _frameHeight;
    vector<double> dct_band_list;
    double tmp;
    int blk_size = 4;
    double median_band, max_band, irq_band;
    double mean_band, var_band, std_band;
    float quart_1st, quart_3rd;
    double sum, square_sum;

    for (int band_y=0; band_y<blk_size; ++band_y){
        for (int band_x=0; band_x<blk_size; ++band_x){
            dct_band_list.clear();
            for (int y=0; y<height; y=y+blk_size){
                for (int x=0; x<width; x=x+blk_size)
                {
                    tmp = transformed_raw_residue[(band_x+x)+(band_y+y)*width];
                    dct_band_list.push_back(tmp);
                }
            }
            assert(dct_band_list.size()==width*height/blk_size/blk_size);
            sort(dct_band_list.begin(), dct_band_list.end());

            // feature extraction, (median, max, interquarterfile)
            median_band = (dct_band_list[ceil((dct_band_list.size()-1)/2.0)]+
                           dct_band_list[floor((dct_band_list.size()-1)/2.0)])/2.0;
            max_band = dct_band_list[dct_band_list.size()-1];
            quart_1st = (dct_band_list.size()-1)/4.0;
            quart_3rd = (dct_band_list.size()-1)/4.0*3.0;
            irq_band = (dct_band_list[ceil(quart_3rd)]+dct_band_list[floor(quart_3rd)])/2.0 -
                      (dct_band_list[ceil(quart_1st)]+dct_band_list[floor(quart_1st)])/2.0;
            // feature extraction (mean, std, var)
            sum = 0;
            square_sum = 0;
            for (int i=0; i<dct_band_list.size(); ++i){
                sum = sum + dct_band_list[i];
                square_sum = square_sum + dct_band_list[i] * dct_band_list[i];
            }
            mean_band = sum / (double)dct_band_list.size();
            var_band = square_sum / (double)dct_band_list.size() - mean_band* mean_band;
            std_band = sqrt(var_band);

            feature_list.push_back(median_band);
            feature_list.push_back(max_band);
            feature_list.push_back(mean_band);
            feature_list.push_back(std_band);
            feature_list.push_back(var_band);
            feature_list.push_back(irq_band);
        }
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::generateDownSampleSeq()
{
    string srcFileName = _files->getFile("src")->getFileName();
    imgpel *orig_frame, *downsampled_frame, *downsampled_u, *downsampled_v;
    imgpel *orig_u, *orig_v;
    orig_frame = new imgpel [_frameHeight*_frameWidth];
    orig_u = new imgpel [_frameHeight*_frameWidth>>2];
    orig_v = new imgpel [_frameHeight*_frameWidth>>2];

    downsampled_frame = new imgpel [_frameHeight*_frameWidth>>2];
    downsampled_u = new imgpel [_frameHeight*_frameWidth>>4];
    downsampled_v = new imgpel [_frameHeight*_frameWidth>>4];


    FILE *srcFh = _files->getFile("src")->getFileHandle();

    string keyFileName = _files->getFile("key")->getFileName();
    string down_seq_name = keyFileName + ".down_src";
    FILE *downSeq = fopen(down_seq_name.c_str(), "wb");

    for (int f=0; f<_numFrames; ++f){
        fseek(srcFh, f*_frameSize*3/2, SEEK_SET);
        fread(orig_frame, _frameSize, 1, srcFh);
        fread(orig_u, _frameSize>>2, 1, srcFh);
        fread(orig_v, _frameSize>>2, 1, srcFh);

        // bicubic downsample for Y channel
        resample(orig_frame, 0.5, downsampled_frame);

        // NN downsample for U,V channel (dont care quality, just for JM encoding)
        for (int y=0; y<_frameHeight/4; ++y){
            for (int x=0; x<_frameWidth/4; ++x){
                downsampled_u[y*_frameWidth/4+x] = orig_u[2*y*_frameWidth/2+2*x];
                downsampled_v[y*_frameWidth/4+x] = orig_v[2*y*_frameWidth/2+2*x];
            }
        }

        fwrite(downsampled_frame, _frameSize>>2, 1, downSeq);
        fwrite(downsampled_u, _frameSize>>4, 1, downSeq);
        fwrite(downsampled_v, _frameSize>>4, 1, downSeq);
    }
    fclose(downSeq);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Encoder::resample(imgpel* input_img, float resize_ratio, imgpel* resized_img)
{
    assert( (resize_ratio==0.5) | (resize_ratio==2));
    int height = _frameHeight;
    int width = _frameWidth;
    if (resize_ratio == 0.5)
    {
        // anti-aliasing (pre-filtering LPF), reduce high freqency signal
        // TODO

        // downsample process
        Mat src(height, width, CV_32F);
        for (int i = 0; i < height*width; ++i)
            src.at<float>(i) = (float)input_img[i];

        Mat dst;
        resize(src, dst, Size(0, 0), 0.5, 0.5, INTER_CUBIC);
        float tmp;

        assert(dst.rows*dst.cols==width*height>>2);
        for (int i=0; i<dst.rows*dst.cols; ++i){
            tmp = dst.at<float>(i);
            tmp = (tmp<0.0)? 0 : tmp;
            resized_img[i] = (tmp>255.0)? 255 : (unsigned int)(tmp+0.5);
        }
    }
    else // resize_ratio == 2
    {

    }

}

