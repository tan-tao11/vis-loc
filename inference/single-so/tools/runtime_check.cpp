#include <dlfcn.h>
#include <iostream>
int main(int argc,char**argv){if(argc!=2){std::cerr<<"runtime_check /path/libvNavigatorInterface.so\n";return 2;}
void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::cerr<<dlerror()<<std::endl;return 1;}
auto test=(int(*)())dlsym(lib,"uavlocRuntimeCheck");if(!test){std::cerr<<"runtime check export missing\n";return 1;}
return test();}
