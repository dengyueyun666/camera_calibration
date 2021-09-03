#include <iostream>

#include <opencv2/opencv.hpp>

#include "util.h"

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage :" << endl
             << "./main <image_path>" << endl;
        return -1;
    }

    string config_file = argv[1];

    cout << config_file << endl;

    return 0;
}