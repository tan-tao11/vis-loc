#include "vNavigatorInterface.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "probe CONFIG RAW_RGB WIDTH HEIGHT MODES(R=RGB,I=grayIR,T=3chIR)\n"; return 2; }
    int w=std::stoi(argv[3]), h=std::stoi(argv[4]);
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<unsigned char> rgb((std::istreambuf_iterator<char>(f)), {});
    if (w<=0 || h<=0 || rgb.size()!=static_cast<size_t>(w)*h*3) return 2;
    std::vector<unsigned char> gray(static_cast<size_t>(w)*h);
    for (size_t i=0;i<gray.size();++i) gray[i]=(77*rgb[i*3]+150*rgb[i*3+1]+29*rgb[i*3+2])>>8;
    auto n=createVideoNavigator();
    using namespace VNAVIGATORINTERFACE;
    if (!n->initVNavigator(argv[1], 1, 0, false, true)) { std::cerr << "INIT_FAILED\n"; return 3; }
    vTelemParams t;
    // Synthetic fixture coordinate; tests override via environment for other locations.
    t.droneLongtitude=117.001292;
    t.droneLatitude=27.121313;
    t.droneAltitude=164;
    t.hfov=90; t.vfov=90; t.innsPitch=-90; t.isEO=true;
    if (auto v=std::getenv("TEST_LON")) t.droneLongtitude=std::stod(v);
    if (auto v=std::getenv("TEST_LAT")) t.droneLatitude=std::stod(v);
    // Ground truth deliberately invalid: it must not reach inference or its validation.
    t.droneLongtitude_gt=std::numeric_limits<double>::quiet_NaN();
    t.droneLatitude_gt=999; t.droneAltitude_gt=-999;
    bool good=true;
    for (char mode:std::string(argv[5])) {
        int type=mode=='R'?1:0, channels=mode=='I'?1:3;
        vNaviResult result;
        ++t.number;
        int ret=n->stepVNavigator(channels==1?gray.data():rgb.data(),w,h,channels,type,0,false,t,result);
        std::cout << std::setprecision(16) << "RESULT " << mode << " " << ret << " " << result.longtitude
                  << " " << result.latitude << " " << result.altitude << " " << result.selfEval << std::endl;
        good &= ret==1 && std::isfinite(result.altitude);
    }
    vNaviResult out; out.altitude=999;
    good &= n->stepVNavigator(rgb.data(),w,h,3,2,0,false,t,out)==0 && out.altitude==0;
    good &= n->stepVNavigator(gray.data(),w,h,1,1,0,false,t,out)==0;
    good &= n->stepVNavigator(nullptr,w,h,3,1,0,false,t,out)==0;
    n->closeVNavigator(); n->closeVNavigator();
    good &= n->stepVNavigator(rgb.data(),w,h,3,1,0,false,t,out)==0;
    freeVideoNavigator(n);
    return good?0:1;
}
