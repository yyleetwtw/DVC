
#include <iostream>

#include "encoder.h"
#include <stdlib.h>

using namespace std;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  if (argc != 10) {
    cerr << endl;
    cerr << "Usage: " << argv[0] << " ";
    cerr << "wz_qp key_qp frame_width frame_height ";
    cerr << "frame_number gop_level ";
    cerr << "src_file output_bitstream_file recon_file" << endl;
    cerr << endl;
  }
  else {
    cout << endl;
    cout << "distributed video coding tool" << endl;
    cout << endl;

    Encoder *encoder = new Encoder(argv);

    encoder->encodeKeyFrame();
    encoder->encodeWzFrame();

    cout << endl;
    cout << "Bye" << endl;
    cout << endl;
  }

  return 0;
}

