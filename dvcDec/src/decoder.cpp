
#include <sstream>
#include <fstream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "decoder.h"
#include "fileManager.h"
#include "sideInformation.h"
#include "transform.h"
#include "corrModel.h"
#include "time.h"
#include "cavlcDec.h"
#include "frameBuffer.h"
#include "bitstream.h"
#include "ldpcaDec.h"
#include "regExp.h"
#include "assert.h"

#define MULTI_RESOLUTION       1 //ddd
#define RESIDUAL_CODING        1 //ddd


using namespace std;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
Decoder::Decoder(char **argv)
{
  _files = FileManager::getManager();

  string wzFileName = argv[1];
  string recFileName = wzFileName.substr(0, wzFileName.find(".bin")) + ".y";

  _files->addFile("wz",     argv[1])->openFile("rb");
  _files->addFile("key",    argv[2])->openFile("rb");
  _files->addFile("src",    argv[3])->openFile("rb");
  _files->addFile("rec",    recFileName.c_str())->openFile("wb");

  _bs = new Bitstream(1024, _files->getFile("wz")->getFileHandle());

  decodeWzHeader();

  initialize();

  unsigned found = wzFileName.find_last_of("/");

  string stats_cif_name = (found == string::npos) ? "./" : wzFileName.substr(0, found+1);
  stats_cif_name += "stats_cif.dat";
  string stats_qcif_name = (found == string::npos) ? "./" : wzFileName.substr(0, found+1);
  stats_qcif_name += "stats_qcif.dat";

  // this key rate file should include every keyrate of cif & qcif
  string single_key_rate = (found == string::npos) ? "./" : wzFileName.substr(0, found+1);
  single_key_rate += "keystate.txt";

  string rec_resample_name = (found == string::npos) ? "./" : wzFileName.substr(0, found+1);
  rec_resample_name += "resample.y";
  string siFileName = (found == string::npos) ? "./" : wzFileName.substr(0, found+1);
  siFileName += "si.y";
  _files->addFile("key_resample", rec_resample_name.c_str())->openFile("rb");
  _files->addFile("si", siFileName.c_str())->openFile("rb");

# if MULTI_RESOLUTION
  parseSingleKeyState(single_key_rate.c_str());
# endif // MULTI_RESOLUTION
  parseKeyStat(stats_cif_name.c_str(), 0);
  parseKeyStat(stats_qcif_name.c_str(), 1);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::initialize()
{
  _frameSize        = _frameWidth * _frameHeight;
  _bitPlaneLength   = _frameSize / 16;
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

  _dParity          = new double[_bitPlaneLength * BitPlaneNum[_qp]];

# if HARDWARE_LDPC
  if (_bitPlaneLength == 6336)
    _crc            = new unsigned char[BitPlaneNum[_qp] * 4];
  else
    _crc            = new unsigned char[BitPlaneNum[_qp]];
# else
  _crc              = new unsigned char[BitPlaneNum[_qp]];
# endif

  _average          = new double[16];
  _alpha            = new double[_frameSize];
  _sigma            = new double[16];

# if RESIDUAL_CODING
  int rcBlkSize = 8;
  _rcList           = new int[_frameSize/(rcBlkSize*rcBlkSize)];

  for (int i = 0; i < _frameSize/(rcBlkSize*rcBlkSize); i++)
    _rcList[i] = 0;
# endif

  _skipMask         = new int[_bitPlaneLength];

  _fb = new FrameBuffer(_frameWidth, _frameHeight, _gop);

  _trans = new Transform(this);

  _model = new CorrModel(this, _trans);
  _si    = new SideInformation(this, _model);

  _cavlc = new CavlcDec(this, 4);

  motionSearchInit(64);

  // Initialize LDPC
  string ladderFile;

# if HARDWARE_LDPC
  ladderFile = "ldpca/1584_regDeg3.lad";
# else
  if (_frameWidth == 352 && _frameHeight == 288)
    ladderFile = "ldpca/6336_regDeg3.lad";
  else
    ladderFile = "ldpca/1584_regDeg3.lad";
# endif

  _ldpca = new LdpcaDec(ladderFile, this);

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::decodeWzHeader()
{
  _frameWidth   = _bs->read(8) * 16;
  _frameHeight  = _bs->read(8) * 16;
  _qp           = _bs->read(8);
  _numFrames    = _bs->read(16);
  _gopLevel     = _bs->read(2);

  cout << "--------------------------------------------------" << endl;
  cout << "WZ frame parameters" << endl;
  cout << "--------------------------------------------------" << endl;
  cout << "Width:  " << _frameWidth << endl;
  cout << "Height: " << _frameHeight << endl;
  cout << "Frames: " << _numFrames << endl;
  cout << "QP:     " << _qp << endl;
  cout << "GOP:    " << (1<<_gopLevel) << endl;
  cout << "--------------------------------------------------" << endl << endl;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::decodeWZframe()
{
  double dPSNRAvg=0;
  double dPSNRSIAvg=0;

  clock_t timeStart, timeEnd;
  time_t begintime,endtime;
  double cpuTime;

  imgpel* oriCurrFrame = _fb->getCurrFrame();
  imgpel* imgSI      = _fb->getSideInfoFrame();
  imgpel* imgRefinedSI = new imgpel[_frameSize];

  int* iDCT         = _fb->getDctFrame();
  int* iDCTQ        = _fb->getQuantDctFrame();
  int* iDecoded     = _fb->getDecFrame();
  int* iDecodedInvQ = _fb->getInvQuantDecFrame();

#if RESIDUAL_CODING
  int* iDCTBuffer   = new int [_frameSize];
  int* iDCTResidual = new int [_frameSize];
#endif

  int x,y;
  int CSS_mode;
  int iDecodeWZFrames = 0;
  double totalrate=0;
  double dMVrate=0;

  FILE* fReadPtr    = _files->getFile("src")->getFileHandle();
  FILE* fWritePtr   = _files->getFile("rec")->getFileHandle();
  FILE* fKeyReadPtr = _files->getFile("key")->getFileHandle();
  FILE* resample_recFH = _files->getFile("key_resample")->getFileHandle();
  FILE* fSiPtr      = _files->getFile("si")->getFileHandle();

  timeStart = clock();
  begintime = time(0);
  // Main loop
  // ---------------------------------------------------------------------------
# if !MULTI_RESOLUTION
  for (int keyFrameNo = 0; keyFrameNo < _numFrames/_gop; keyFrameNo++) {

    // Read previous key frame
    fseek(fKeyReadPtr, (3*(keyFrameNo)*_frameSize)>>1, SEEK_SET);
    fread(_fb->getPrevFrame(), _frameSize, 1, fKeyReadPtr);
# else
  for (int keyFrameNo = 0; keyFrameNo < _numFrames/_gop; ) {

    CSS_mode = _bs->read(1);

    //fseek(fKeyReadPtr, keyFrameNo*_frameSize, SEEK_SET);
    //fread(_fb->getPrevFrame(), _frameSize, 1, fKeyReadPtr);

    if (CSS_mode==0){
        fseek(fKeyReadPtr, keyFrameNo*_frameSize*3/2, SEEK_SET);
        fread(_fb->getPrevFrame(), _frameSize, 1, fKeyReadPtr);
    }
    else if (CSS_mode==1){
        fseek(resample_recFH, keyFrameNo*_frameSize, SEEK_SET);
        fread(_fb->getPrevFrame(), _frameSize, 1, resample_recFH);
    }
    else
        cout << "ERROR CSS mode" << endl;
# endif

    // Read next key frame
# if !MULTI_RESOLUTION
    fseek(fKeyReadPtr, (3*(keyFrameNo+1)*_frameSize)>>1, SEEK_SET);
    fread(_fb->getNextFrame(), _frameSize, 1, fKeyReadPtr);
# else

    if ( keyFrameNo+2<_numFrames/_gop ){
      // prepare next key (not resample key) for both mode,
      // even useless in MRDVC
      fseek(fKeyReadPtr, (keyFrameNo+2)*_frameSize*3/2, SEEK_SET);
      fread(_fb->getNextFrame(), _frameSize, 1, fKeyReadPtr);
    }
# endif

    int wzFrameNo;
# if !MULTI_RESOLUTION
    for (int il = 0; il < _gopLevel; il++) {
      int frameStep = _gop / ((il+1)<<1);
# else
    for (int il = 0; il <= _gopLevel; il++) {
        // current version MRDVC only compare between GOP 1 & 2
        int frameStep = (CSS_mode==1)? 0 : 1;
# endif

      int idx = frameStep;

      // Start decoding the WZ frame
# if !MULTI_RESOLUTION
      while (idx < _gop) {
# else
    while (idx < frameStep+1) {
      // current version MRDVC only compare between GOP 1 & 2
      // namely, 1 key for 1 WZ no matter MRDVC or HDVC
# endif
        wzFrameNo = keyFrameNo*_gop + idx;

        cout << "Decoding frame " << wzFrameNo << " (Wyner-Ziv frame)" << endl;

        memset(iDecoded, 0, _frameSize*4);
        memset(iDecodedInvQ, 0, _frameSize*4);

        // Read current frame from the original file
        fseek(fReadPtr, (3*wzFrameNo*_frameSize)>>1, SEEK_SET);
        fread(oriCurrFrame, _frameSize, 1, fReadPtr);

        // Setup frame pointers within the GOP
        int prevIdx = idx - frameStep;
        int nextIdx = idx + frameStep;

# if !MULTI_RESOLUTION
        imgpel* currFrame = _fb->getRecFrames()[idx-1];
        imgpel* prevFrame = (prevIdx == 0)    ? _fb->getPrevFrame() :
                                                _fb->getRecFrames()[prevIdx-1];
        imgpel* nextFrame = (nextIdx == _gop) ? _fb->getNextFrame() :
                                                _fb->getRecFrames()[nextIdx-1];
# else
        imgpel* currFrame = _fb->getRecFrames()[0];
        imgpel* prevFrame = _fb->getPrevFrame();
        imgpel* nextFrame = _fb->getNextFrame();
# endif

        imgpel* prevKeyFrame  = _fb->getPrevFrame();
        imgpel* nextKeyFrame  = _fb->getNextFrame();

        // ---------------------------------------------------------------------
        // STAGE 1 - get maxQstep, syndrome
        // ---------------------------------------------------------------------
        int tmp = getSyndromeData();
        //cout << _numChnCodeBands << endl;

        double dTotalRate = (double)tmp/1024/8;

        // ---------------------------------------------------------------------
        // STAGE 2 -Create side information
        // ---------------------------------------------------------------------

        _si->createSideInfo(prevFrame, nextFrame, imgSI, wzFrameNo,
                            CSS_mode, resample_recFH,fSiPtr);

        _trans->dctTransform(imgSI, iDCT);

# if RESIDUAL_CODING
        _si->getResidualFrame(prevKeyFrame, nextKeyFrame, imgSI, iDCTBuffer, _rcList);
/*
        ofstream residue_dec;
        residue_dec.open("residue_dec.txt",std::ofstream::out | std::ofstream::trunc);
        for (int y=0;y<288;y++)for(int x=0;x<352;x++){
            residue_dec<<iDCTBuffer[y*352+x];
            if(x!=351) residue_dec<<',';
            else residue_dec<<'\n';
        }
        residue_dec.close();
*/
        _trans->dctTransform(iDCTBuffer, iDCTResidual);
        _trans->quantization(iDCTResidual, iDCTQ);

        int iOffset = 0;
        int iDC;

#   if SI_REFINEMENT
        memcpy(iDecodedInvQ, iDCTResidual, 4*_frameSize);
#   endif

        for (int i = 0; i < 16; i++) {
          x = ScanOrder[i][0];
          y = ScanOrder[i][1];

#   if MODE_DECISION
          if (i < _numChnCodeBands)
            dTotalRate += decodeLDPC(iDCTQ, iDCTResidual, iDecoded, x, y, iOffset);
#   else
          dTotalRate += decodeLDPC(iDCTQ, iDCTResidual, iDecoded, x, y, iOffset);
#   endif

#   if SI_REFINEMENT
          //temporal reconstruction
          _trans->invQuantization(iDecoded, iDecodedInvQ, iDCTResidual, x, y);
          _trans->invDctTransform(iDecodedInvQ, iDCTBuffer);

          _si->getRecFrame(prevKeyFrame, nextKeyFrame, iDCTBuffer, currFrame, _rcList);

          iDC = (x == 0 && y == 0) ? 0 : 1;

          _si->getRefinedSideInfo(prevFrame, nextFrame, imgSI, currFrame, imgRefinedSI, iDC);

          memcpy(imgSI, imgRefinedSI, _frameSize);

          _si->getResidualFrame(prevKeyFrame, nextFrame, imgSI, iDCTBuffer, _rcList);

          _trans->dctTransform(iDCTBuffer, iDCTResidual);
          _trans->quantization(iDCTResidual, iDCTQ);
#   endif

          iOffset += QuantMatrix[_qp][y][x];
        }

#   if !SI_REFINEMENT
        _trans->invQuantization(iDecoded, iDecodedInvQ, iDCTResidual);
        _trans->invDctTransform(iDecodedInvQ, iDCTBuffer);

        _si->getRecFrame(prevFrame, nextFrame, iDCTBuffer, currFrame, _rcList);
#   endif

# else // if !RESIDUAL_CODING

        _trans->quantization(iDCT, iDCTQ);

        int iOffset = 0;
        int iDC;

#   if SI_REFINEMENT
        memcpy(iDecodedInvQ, iDCT, 4*_frameSize);
#   endif

        for (int i = 0; i < 16; i++) {
          x = ScanOrder[i][0];
          y = ScanOrder[i][1];

#   if MODE_DECISION
          if (i < _numChnCodeBands)
            dTotalRate += decodeLDPC(iDCTQ, iDCT, iDecoded, x, y, iOffset);
#   else
          dTotalRate += decodeLDPC(iDCTQ, iDCT, iDecoded, x, y, iOffset);
#   endif

#   if SI_REFINEMENT
          _trans->invQuantization(iDecoded, iDecodedInvQ, iDCT, x, y);
          _trans->invDctTransform(iDecodedInvQ, currFrame);

#     if SKIP_MODE
          //reconstruct skipped part of wyner-ziv frame
          getSkippedRecFrame(prevKeyFrame, currFrame, _skipMask);
#     endif
          iDC = (x == 0 && y == 0) ? 0 : 1;

          _si->getRefinedSideInfo(prevFrame, nextFrame, imgSI, currFrame, imgRefinedSI, iDC);

          memcpy(imgSI, imgRefinedSI, _frameSize);

          _trans->dctTransform(imgSI, iDCT);
          _trans->quantization(iDCT, iDCTQ);
#   endif
          iOffset += QuantMatrix[_qp][y][x];
        }

#   if !SI_REFINEMENT

        _trans->invQuantization(iDecoded, iDecodedInvQ, iDCT);

        _trans->invDctTransform(iDecodedInvQ, currFrame);

#     if SKIP_MODE
        _si->getSkippedRecFrame(prevKeyFrame, currFrame, _skipMask);
#     endif

#   endif

# endif // RESIDUAL_CODING

        double key_rate;
        key_rate = getSingleKeyRate(keyFrameNo, wzFrameNo, CSS_mode);
        totalrate += dTotalRate;
        // no matter MR or Hybrid, key rate is always transmitted
        totalrate += key_rate;
        double si_psnr, wz_psnr;
        si_psnr = calcPSNR(oriCurrFrame, imgSI, _frameSize);
        dPSNRSIAvg += si_psnr;
        wz_psnr = calcPSNR(oriCurrFrame, currFrame, _frameSize);
        dPSNRAvg += wz_psnr;

        // added zero when MRDVC mode, cuz GOP=1
        double hybrid_key_psnr;
        hybrid_key_psnr = getSingleKeyPSNR(keyFrameNo, wzFrameNo, CSS_mode);
        dPSNRAvg += hybrid_key_psnr;

        double gop_avg_quality = (CSS_mode==0)? (hybrid_key_psnr+wz_psnr)/2 : wz_psnr;

        cout << endl;  // to escape .s from C.C.
        cout << "WZ rate (Y/frame): " << dTotalRate << " KB" << endl;
        cout << "WZ frame quality: " << wz_psnr << " dB" << endl;
        cout << "GOP key rate: " << key_rate << " KB" << endl;
        cout << "GOP avg quality: " << gop_avg_quality << " dB" << endl;
        cout << "Side Information quality: " << si_psnr << " dB" << endl;
        cout << endl;

# if !MULTI_RESOLUTION
        idx += 2*frameStep;
# else
        idx += 1;
# endif
      }
    }

    //cout << "(css,keyno,wzno) = (" << CSS_mode << "  ,  "
    //<< keyFrameNo << "  ,  " << wzFrameNo << endl;

    if (CSS_mode == 0){
        keyFrameNo += 2;
    } else if (CSS_mode == 1){
        keyFrameNo += 1;
    } else
        cout << "ERROR CSS_mode" << endl;

# if !MULTI_RESOLUTION
    // ---------------------------------------------------------------------
    // Output decoded frames of the whole GOP
    // ---------------------------------------------------------------------
    // First output the key frame
    fwrite(_fb->getPrevFrame(), _frameSize, 1, fWritePtr);

    // Then output the rest WZ frames
    for (int i = 0; i < _gop-1; i++)
      fwrite(_fb->getRecFrames()[i], _frameSize, 1, fWritePtr);
# else
    if (CSS_mode==0){
        fwrite(_fb->getPrevFrame(), _frameSize, 1, fWritePtr);

    // Then output the rest WZ frames
        for (int i = 0; i <= _gop-1; i++)
            fwrite(_fb->getRecFrames()[i], _frameSize, 1, fWritePtr);
    }
    else
    {
        fwrite(_fb->getRecFrames()[0], _frameSize, 1, fWritePtr);
    }

    iDecodeWZFrames += 1;
# endif
  }

  timeEnd = clock();
  endtime = time(0);
  //cpuTime = (timeEnd - timeStart) / CLOCKS_PER_SEC;
  cpuTime = endtime - begintime ;

# if !MULTI_RESOLUTION
  iDecodeWZFrames = ((_numFrames-1)/_gop)*(_gop-1);
  int iNumGOP = (_numFrames-1)/_gop;
  int iTotalFrames = iDecodeWZFrames + iNumGOP;
# else
  //iDecodeWZFrames = _numFrames; <-maintained in loop
  int iNumGOP = iDecodeWZFrames;
  // in MRDVC, no matter hybrid or MR, 1wz correspond to 1 gop
  int iTotalFrames = _numFrames ;
# endif

  float framerate = 30.0;

  if (_frameWidth == 176 && _frameHeight == 144)
    framerate = 15.0;

  cout<<endl;
  cout<<"--------------------------------------------------"<<endl;
  cout<<"Decode statistics"<<endl;
  cout<<"--------------------------------------------------"<<endl;
  cout<<"Total Frames        :   "<<iTotalFrames<<endl;
  dPSNRAvg   /= iTotalFrames;
  dPSNRSIAvg /= iNumGOP;
  cout<<"Total Bytes         :   "<<totalrate<<endl;
  //cout<<"WZ Avg Rate  (kbps) :   "<<totalrate/double(iDecodeWZFrames)*framerate*(iDecodeWZFrames)/(double)iTotalFrames*8.0<<endl;
  //cout<<"Key Avg Rate (kbps) :   "<<_keyBitRate*framerate*(iNumGOP)/(double)iTotalFrames<<endl;
  //cout<<"Avg Rate (Key+WZ)   :   "<<totalrate/double(iDecodeWZFrames)*framerate*(iDecodeWZFrames)/(double)iTotalFrames*8.0+_keyBitRate*framerate*(iNumGOP)/(double)iTotalFrames<<endl;
  cout<<"Avg rate (key+WZ)   :   "<<totalrate/(double)iTotalFrames*framerate<<" kBps"<< endl;
  //cout<<"Key Frame Quality   :   "<<_keyPsnr<<endl;
  //cout<<"SI Avg PSNR         :   "<<dPSNRSIAvg<<endl;
  //cout<<"WZ Avg PSNR         :   "<<dPSNRAvg<<endl;
  //cout<<"Avg    PSNR         :   "<<(dPSNRAvg+_keyPsnr)/2<<endl;
  cout<<"Avg    PSNR         :   "<<dPSNRAvg<<endl;
  cout<<"Total Decoding Time :   "<<cpuTime<<"(s)"<<endl;
  cout<<"Avg Decoding Time   :   "<<cpuTime/(iDecodeWZFrames)<<endl;
  cout<<"--------------------------------------------------"<<endl;

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::parseKeyStat(const char* filename, int mode)
{
  // mode 0 : multiply _keySingleRate with ratio of Y/all rate to CIF key rate
  // mode 1 :                                                     QCIF

  ifstream stats(filename, ios::in);

  if (!stats.is_open())
    return;

  char buf[1024];
  double iSlice_chroma = 0.0;
  double iSliceRate = 0.0;

  RegExp* rgx = RegExp::getInst();

  while (stats.getline(buf, 1024)) {
    string result;

    if (rgx->match(buf, "\\s*SNR Y\\(dB\\)[ |]*([0-9\\.]+)", result)) {
      _keyPsnr = atof(result.c_str());
      continue;
    }

    if (rgx->match(buf, "\\s*Average quant[ |]*([0-9\\.]+)", result)) {
      _keyQp = atoi(result.c_str());
      continue;
    }

    if (rgx->match(buf, "\\s*QP[ |]*([0-9\\.]+)", result)) {
      _keyQp = atoi(result.c_str());
      continue;
    }

    if (rgx->match(buf, "\\s*Coeffs\\. C[ |]*([0-9\\.]+)", result)) {
      iSlice_chroma = atof(result.c_str());
      continue;
    }

    if (rgx->match(buf, "\\s*average bits/frame[ |]*([0-9\\.]+)", result)) {
      iSliceRate = atof(result.c_str());
      continue;
    }
  }

  int count = 0;
  double totalRate = 0;

  if (iSliceRate != 0) {
    totalRate += iSliceRate - iSlice_chroma;
    count++;
  }

  if (count)
    _keyBitRate = totalRate/(1024*count);
  else
    _keyBitRate = 0;

  if (_keySingleRate.size() != 0 ){
    double Y_ratio = totalRate / iSliceRate;
    if (mode==0){
        for (int i=0; i<_keySingleRate.size()/2; ++i){
            _keySingleRate[i] *= Y_ratio;
        }
    }
    else if (mode==1){
        for (int i=_keySingleRate.size()/2; i<_keySingleRate.size(); ++i){
            _keySingleRate[i] *= Y_ratio;
        }
    }
    else
        cout << "ERROR mode at parseKeyStat" << endl;
  }

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::parseSingleKeyState(const char* filename)
{
  // first _numFrame rate is the CIF rate
  // following _numFrame rate is the QCIF rate

  ifstream key_state_file(filename, ios::in);

  if (!key_state_file.is_open())
    return;

  double buf;
# if !MULTI_RESOLUTION
  for (int keynum = 0; keynum<_numFrames; keynum++){
# else
  for (int keynum = 0; keynum< 2*_numFrames; keynum++){
# endif
    key_state_file >> buf;
    buf /= (1024.0*8.0);
    _keySingleRate.push_back(buf);
    key_state_file >> buf;
    _keySinglePSNR.push_back(buf);
  }
  assert(_keySingleRate.size()==2*_numFrames);
  assert(_keySinglePSNR.size()==2*_numFrames);

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int Decoder::getSyndromeData()
{
  int* iDecoded = _fb->getDecFrame();
  int  decodedBits = 0;

# if RESIDUAL_CODING

#   if !HARDWARE_FLOW
  // Decode motion vector
  int rcBlkSize = 8;
  for (int i = 0; i < _frameSize/(rcBlkSize*rcBlkSize); i++)
    _rcList[i] = _bs->read(1);
#   endif

# endif // RESIDUAL_CODING

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
# if RESIDUAL_CODING

#   if !HARDWARE_FLOW
      _maxValue[j][i] = _bs->read(11);
#   endif

      if (QuantMatrix[_qp][j][i] != 0) {
#   if HARDWARE_QUANTIZATION
        _quantStep[j][i] = 1 << (MaxBitPlane[j][i]+1-QuantMatrix[_qp][j][i]);
#   else
        int iInterval = 1 << QuantMatrix[_qp][j][i];

        _quantStep[j][i] = (int)(ceil(double(2*abs(_maxValue[j][i]))/double(iInterval-1)));
        _quantStep[j][i] = Max(_quantStep[j][i], MinQStepSize[_qp][j][i]);
#   endif
      }
      else
        _quantStep[j][i] = 1;

# else // if !RESIDUAL_CODING

      if (i != 0 || j != 0) {
        _maxValue[j][i] = _bs->read(11);

#   if HARDWARE_QUANTIZATION
        if (QuantMatrix[_qp][j][i] != 0)
          _quantStep[j][i] = 1 << (MaxBitPlane[j][i]+1-QuantMatrix[_qp][j][i]);
        else
          _quantStep[j][i] = 0;
#   else
        int iInterval = 1 << QuantMatrix[_qp][j][i];

#     if AC_QSTEP
        if (QuantMatrix[_qp][j][i] != 0) {
          _quantStep[j][i] = (int)(ceil(double(2*abs(_maxValue[j][i]))/double(iInterval-1)));

          if (_quantStep[j][i] < 0)
            _quantStep[j][i] = 0;
        }
        else
          _quantStep[j][i] = 1;
#     else
        if (QuantMatrix[_qp][j][i] != 0)
          _quantStep[j][i] = ceil(double(2*abs(_maxValue[j][i]))/double(iInterval));
        else
          _quantStep[j][i] = 1;
#     endif

#   endif // HARDWARE_QUANTIZATION
      }
      else {
        _maxValue[j][i] = DC_BITDEPTH;

        _quantStep[j][i] = 1 << (_maxValue[j][i]-QuantMatrix[_qp][j][i]);
      }
# endif // RESIDUAL_CODING
    }
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
# if MODE_DECISION
  int codingMode = _bs->read(2);

  if (codingMode == 0) _numChnCodeBands = 16; else
  if (codingMode == 1) _numChnCodeBands =  3; else
  if (codingMode == 2) _numChnCodeBands =  6; else
                       _numChnCodeBands =  0;

  //cout << "CMS = " << codingMode << endl;

  _rcBitPlaneNum = 0;

  for (int band = 0; band < 16; band++) {
    int x = ScanOrder[band][0];
    int y = ScanOrder[band][1];

    if (band < _numChnCodeBands) {
      _rcQuantMatrix[y][x] = QuantMatrix[_qp][y][x];
      _rcBitPlaneNum += _rcQuantMatrix[y][x];
    }
  }

# else // if !MODE_DECISION

  _rcBitPlaneNum = 0;

  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++) {
      _rcQuantMatrix[j][i] = QuantMatrix[_qp][j][i];
      _rcBitPlaneNum += _rcQuantMatrix[j][i];
    }
# endif // MODE_DECISION

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
# if HARDWARE_FLOW
  static int isFirstFrame = 1;

  // Discard padded bits
  if (isFirstFrame)
    _bs->read(20);
  else
    _bs->read(30);

  if (isFirstFrame)
    isFirstFrame = 0;

  int bitCount = _bs->getBitCount();
# endif

# if SKIP_MODE
  decodedBits += decodeSkipMask();
# endif

# if HARDWARE_FLOW
  bitCount = _bs->getBitCount() - bitCount;

  if (bitCount%32 != 0) {
    int dummy = 32 - (bitCount%32);
    _bs->read(dummy);
  }

  bitCount = _bs->getBitCount();
# endif

# if MODE_DECISION
  if (_numCodeBands > _numChnCodeBands) {
    for (int j = 0; j < _frameHeight; j += 4)
      for (int i = 0; i < _frameWidth; i += 4) {
        if (_skipMask[i/4+(j/4)*(_frameWidth/4)] == 0) //not skip
          decodedBits += _cavlc->decode(iDecoded, i, j);
        else
          _cavlc->clearNnz(i/4+(j/4)*(_frameWidth/4));
      }
  }
# endif

# if HARDWARE_FLOW
  bitCount = _bs->getBitCount() - bitCount;

  if (bitCount%32 != 0) {
    int dummy = 32 - (bitCount%32);
    _bs->read(dummy);
  }
# endif

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // Read parity and CRC bits from the bitstream
# if !FPGA
#   if RESIDUAL_CODING
  for (int i = 0; i < _rcBitPlaneNum; i++)
#   else
  for (int i = 0; i < BitPlaneNum[_qp]; i++)
#   endif
  {
    for (int j = 0; j < _bitPlaneLength; j++)
      _dParity[j+i*_bitPlaneLength] = (double)_bs->read(1);

#   if !HARDWARE_FLOW
#     if HARDWARE_LDPC
    if (_bitPlaneLength == 6336)
      for (int n = 0; n < 4; n++)
        _crc[i*4+n] = _bs->read(8);
    else
      _crc[i] = _bs->read(8);
#     else
    _crc[i] = _bs->read(8);
#     endif
#   endif
  }
# else // FPGA
  for (int j = 0; j < _bitPlaneLength; j++)
  {
    for (int i = 0; i < 64; i++)
      if (i < _rcBitPlaneNum)
        _dParity[j+i*_bitPlaneLength] = (double)_bs->read(1);
      else
        _bs->read(1);
  }
# endif // !FPGA

  return decodedBits;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int Decoder::decodeSkipMask()
{
  int type        = 0;
  int sign        = 0;
  int index       = 0;
  int length      = 0;
  int totalLength = 0;

  type = _bs->read(2);
  sign = _bs->read(1);
  totalLength += 3;

  memset(_skipMask, 0, _bitPlaneLength*sizeof(int));

  while (index < _bitPlaneLength) {
    int code = 0;
    int run  = 0;

    for (length = 1; length < 15; length++) {
      code <<= 1;
      code |= _bs->read(1);

      for (int i = 0; i < 16; i++) {
        int table = _qp / 2;

        if (HuffmanCodeValue [table][type][i] == code &&
            HuffmanCodeLength[table][type][i] == length) {
          run = i+1;
          totalLength += length;
          goto DecodeSkipMaskHuffmanCodeDone;
        }
      }
    }

    DecodeSkipMaskHuffmanCodeDone:

    // Reconstruct skip mask
    if (run == 16)
      for (int i = 0; i < 15; i++)
        _skipMask[index++] = sign;
    else {
      for (int i = 0; i < run; i++)
        _skipMask[index++] = sign;

      sign = (sign == 0) ? 1 : 0;
    }
  }
  return totalLength;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
double Decoder::getSingleKeyRate(int keyFrameNo, int wzFrameNo, int CSS_mode)
{
#if MULTI_RESOLUTION
    int total_frame_num = _numFrames;
    assert(_keySingleRate.size() == 2*total_frame_num);

    if (CSS_mode==0){
    // Hybrid structure, key rate = keyFrameNo CIF rate
        return _keySingleRate[keyFrameNo];
    }
    else if (CSS_mode==1){
    // MRDVC structure, key rate = wzFrameNo QCIF rate
        return _keySingleRate[wzFrameNo+total_frame_num];
    }
    else
        cout << "ERROR CSS_mode at Decoder::getSingleKeyRate" << endl;

    return 0;
#else
    return _keyBitRate;
#endif
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
double Decoder::getSingleKeyPSNR(int keyFrameNo, int wzFrameNo, int CSS_mode)
{
#if MULTI_RESOLUTION
    int total_frame_num = _numFrames;
    assert(_keySinglePSNR.size() == 2*total_frame_num);

    if (CSS_mode==0){
    // Hybrid structure, key rate = keyFrameNo CIF rate
        return _keySinglePSNR[keyFrameNo];
    }
    else if (CSS_mode==1){
    // MRDVC structure, key rate = wzFrameNo QCIF rate
        return 0;
    }
    else
        cout << "ERROR CSS_mode at Decoder::getSingleKeyRate" << endl;

    return 0;
#else
    return _keyPsnr;
#endif
}

/*
*Decoding Process of LDPC
*Param
*/
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
double Decoder::decodeLDPC(int* iQuantDCT, int* iDCT, int* iDecoded, int x, int y, int iOffset)
{
  int iCurrPos;
  int* iDecodedTmp;
  double* dLLR;
  double* dAccumulatedSyndrome;
  double* dLDPCDecoded;
  double* dSource;
  double dRate,dTotalRate;
  double dErr;
  double dParityRate;
  unsigned char* ucCRCCode;
  int    iNumCode;

# if HARDWARE_LDPC
  ucCRCCode             = _crc + iOffset*4;
# else
  ucCRCCode             = _crc + iOffset;
# endif
  dAccumulatedSyndrome  = _dParity + _bitPlaneLength*iOffset;

  dLLR         = new double[_bitPlaneLength];
  iDecodedTmp  = new int   [_bitPlaneLength];
  dLDPCDecoded = new double[_bitPlaneLength];
  dSource      = new double[_bitPlaneLength];

  for (int i = 0; i < _bitPlaneLength; i++) {
    dLLR[i]         = 0;
    iDecodedTmp[i]  = 0;
    dLDPCDecoded[i] = 0;
    dSource[i]      = 0;
  }

  dTotalRate   = 0;

  memset(iDecodedTmp, 0, _bitPlaneLength*4);

  dParityRate = 0;

# if RESIDUAL_CODING
  for (iCurrPos = _rcQuantMatrix[y][x]-1; iCurrPos >= 0; iCurrPos--)
# else
  for (iCurrPos = QuantMatrix[_qp][y][x]-1; iCurrPos >= 0; iCurrPos--)
# endif
  {
# if RESIDUAL_CODING
    if (iCurrPos == _rcQuantMatrix[y][x]-1)
      dParityRate = _model->getSoftInput(iQuantDCT, _skipMask, iCurrPos, iDecodedTmp, dLLR, x, y, 1);
    else
      dParityRate = _model->getSoftInput(iDCT, _skipMask, iCurrPos, iDecodedTmp, dLLR, x, y, 2);
# else
    if (x == 0 && y == 0)
      dParityRate = _model->getSoftInput(iDCT, _skipMask, iCurrPos, iDecodedTmp, dLLR, 0, 0, 2);
    else {
      if (iCurrPos == QuantMatrix[_qp][y][x]-1)
        dParityRate = _model->getSoftInput(iQuantDCT, _skipMask, iCurrPos, iDecodedTmp, dLLR, x, y, 1);
      else
        dParityRate = _model->getSoftInput(iDCT, _skipMask, iCurrPos, iDecodedTmp, dLLR, x, y, 2);
    }
# endif
    iNumCode = int(dParityRate*66);

    if (iNumCode <= 2)
      iNumCode = 2;
    if (iNumCode >= 66)
      iNumCode = 66;
    iNumCode = 2;

# if HARDWARE_LDPC
    if (_bitPlaneLength == 6336) {
      double dRateTmp = 0;

      for (int n = 0; n < 4; n++) {
        iNumCode = 2;

        _ldpca->decode(dLLR+n*1584, dAccumulatedSyndrome+n*1584, dSource+n*1584, dLDPCDecoded+n*1584, &dRate, &dErr, *(ucCRCCode+n), iNumCode);
        dRateTmp += (dRate/4.0);
        //cout<<dRate<<endl;
        dRate = 0;
      }
      ucCRCCode += 4;
      cout << ".";

      dTotalRate += dRateTmp;
      dRate = 0;
    }
    else {
      _ldpca->decode(dLLR, dAccumulatedSyndrome, dSource, dLDPCDecoded, &dRate, &dErr, *ucCRCCode, iNumCode);
      cout << ".";

      dTotalRate += dRate;
      dRate = 0;
      ucCRCCode++;
    }
# else
    _ldpca->decode(dLLR, dAccumulatedSyndrome, dSource, dLDPCDecoded, &dRate, &dErr, *ucCRCCode, iNumCode);
    cout << "." <<std::flush;

    dTotalRate += dRate;
    dRate = 0;
    ucCRCCode++;
# endif

    for (int iIndex = 0; iIndex < _bitPlaneLength; iIndex++)
      if (dLDPCDecoded[iIndex] == 1)
        iDecodedTmp[iIndex] |= 0x1<<iCurrPos;

    dAccumulatedSyndrome += _bitPlaneLength;

    memset(dLDPCDecoded, 0, _bitPlaneLength*sizeof(double));
  }

  for (int j = 0; j < _frameHeight; j = j+4)
    for (int i = 0; i < _frameWidth; i = i+4) {
      int tmp = i/4 + j/4*(_frameWidth/4);
      iDecoded[(i+x)+(j+y)*_frameWidth] = iDecodedTmp[tmp];
    }

  delete [] dLLR;
  delete [] iDecodedTmp;
  delete [] dLDPCDecoded;
  delete [] dSource;

  return (dTotalRate*_bitPlaneLength/8/1024);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::getSourceBit(int *dct_q,double *source,int q_i,int q_j,int curr_pos){
  int iWidth,iHeight;
  iWidth  = _frameWidth;
  iHeight = _frameHeight;
  for(int y=0;y<iHeight;y=y+4)
    for(int x=0;x<iWidth;x=x+4)
    {
      source[(x/4)+(y/4)*(iWidth/4)]=(dct_q[(x+q_i)+(y+q_j)*iWidth]>>curr_pos)&(0x1);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void Decoder::motionSearchInit(int maxsearch_range)
{
  _spiralHpelSearchX = new int[(2*maxsearch_range+1)*(2*maxsearch_range+1)];
  _spiralSearchX     = new int[(2*maxsearch_range+1)*(2*maxsearch_range+1)];
  _spiralHpelSearchY = new int[(2*maxsearch_range+1)*(2*maxsearch_range+1)];
  _spiralSearchY     = new int[(2*maxsearch_range+1)*(2*maxsearch_range+1)];

  int k,i,l;

  _spiralSearchX[0] = _spiralSearchY[0] = 0;
  _spiralHpelSearchX[0] = _spiralHpelSearchY[0] = 0;

  for (k=1, l=1; l <= std::max<int>(1,maxsearch_range); l++) {
    for (i=-l+1; i< l; i++) {
      _spiralSearchX[k] =  i;
      _spiralSearchY[k] = -l;
      _spiralHpelSearchX[k] =  i<<1;
      _spiralHpelSearchY[k++] = -l<<1;
      _spiralSearchX[k] =  i;
      _spiralSearchY[k] =  l;
      _spiralHpelSearchX[k] =  i<<1;
      _spiralHpelSearchY[k++] =  l<<1;
    }
    for (i=-l;   i<=l; i++) {
      _spiralSearchX[k] = -l;
      _spiralSearchY[k] =  i;
      _spiralHpelSearchX[k] = -l<<1;
      _spiralHpelSearchY[k++] = i<<1;
      _spiralSearchX[k] =  l;
      _spiralSearchY[k] =  i;
      _spiralHpelSearchX[k] =  l<<1;
      _spiralHpelSearchY[k++] = i<<1;
    }
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
double calcPSNR(unsigned char* img1,unsigned char* img2,int length)
{
  float PSNR;
  float MSE=0;

  for(int i=0;i<length;i++)
    {
      MSE+=pow(float(img1[i]-img2[i]),float(2.0))/length;
    }
  PSNR=10*log10(255*255/MSE);
  //cout<<"PSNR: "<<PSNR<<" dB"<<endl;
  return PSNR;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int getSymbol(int len, int& curr_pos, char* buffer)
{
  int temp = 0;

  for (int count = 0; count < len; count++) {
    int pos = count + curr_pos;

    temp <<= 1;
    temp |= 0x1 & (buffer[pos/8]>>(7-(pos%8)));
  }

  curr_pos += len;

  return temp;
}

