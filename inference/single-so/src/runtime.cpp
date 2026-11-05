#include "runtime.hpp"
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <spawn.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>
extern char** environ;
namespace fs=std::filesystem;
namespace embedded {
namespace {
int marker;
struct FD { int v; explicit FD(int fd):v(fd){} ~FD(){if(v>=0)::close(v);} };
#pragma pack(push,1)
struct Footer { char magic[16]; uint64_t offset,size; char sha[64]; };
#pragma pack(pop)
void exact(int fd,void* data,size_t n,off_t off) {
    auto p=static_cast<char*>(data);
    while(n) { auto k=pread(fd,p,n,off); if(k<0&&errno==EINTR)continue;
        if(k<=0)throw std::runtime_error("truncated embedded runtime");
        p+=k;n-=k;off+=k; }
}
int wait(pid_t pid) { int s; while(waitpid(pid,&s,0)<0){if(errno!=EINTR)throw std::runtime_error("waitpid failed");}
    return WIFEXITED(s)?WEXITSTATUS(s):128; }
// OpenSSL is an Ubuntu base-system dependency of this small loader. The runtime
// ships its own user-space crypto libraries for its CPython child processes.
class Hash {
    void* lib=nullptr; void* ctx=nullptr;
    using New=void*(*)(); using Free=void(*)(void*); using Method=const void*(*)();
    using Init=int(*)(void*,const void*,void*); using Update=int(*)(void*,const void*,size_t);
    using Final=int(*)(void*,unsigned char*,unsigned int*);
    Free free_=nullptr;Update update_=nullptr;Final final_=nullptr;
public:
    Hash(){lib=dlopen("libcrypto.so.3",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error("Ubuntu libcrypto.so.3 unavailable");
        auto make=(New)dlsym(lib,"EVP_MD_CTX_new");free_=(Free)dlsym(lib,"EVP_MD_CTX_free");
        auto method=(Method)dlsym(lib,"EVP_sha256");auto init=(Init)dlsym(lib,"EVP_DigestInit_ex");
        update_=(Update)dlsym(lib,"EVP_DigestUpdate");final_=(Final)dlsym(lib,"EVP_DigestFinal_ex");
        if(!make||!free_||!method||!init||!update_||!final_)throw std::runtime_error("OpenSSL SHA256 API missing");
        ctx=make();if(!ctx||init(ctx,method(),nullptr)!=1)throw std::runtime_error("SHA256 init failed"); }
    ~Hash(){if(ctx&&free_)free_(ctx);if(lib)dlclose(lib);}
    void add(const void* p,size_t n){if(update_(ctx,p,n)!=1)throw std::runtime_error("SHA256 update failed");}
    std::string finish(){unsigned char bytes[64];unsigned int n=0;
        if(final_(ctx,bytes,&n)!=1||n!=32)throw std::runtime_error("SHA256 final failed");
        const char* hex="0123456789abcdef";std::string s;for(unsigned int i=0;i<n;++i){s+=hex[bytes[i]>>4];s+=hex[bytes[i]&15];}return s;}
};
void unpack(int fd,const Footer& f,const fs::path& target) {
    int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets))throw std::runtime_error("extract socketpair failed");
    FD input(sockets[0]),output(sockets[1]);
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions,input.v,STDIN_FILENO);
    std::string dst=target.string();
    char* argv[]={const_cast<char*>("/bin/tar"),const_cast<char*>("--no-same-owner"),const_cast<char*>("-xzf"),
                  const_cast<char*>("-"),const_cast<char*>("-C"),dst.data(),nullptr};
    pid_t pid;int rc=posix_spawn(&pid,"/bin/tar",&actions,nullptr,argv,environ);
    posix_spawn_file_actions_destroy(&actions);if(rc)throw std::runtime_error("cannot start /bin/tar");
    ::close(input.v);input.v=-1;
    try {
        Hash hash;std::vector<char> block(1024*1024);uint64_t copied=0;
        while(copied<f.size){size_t n=std::min<uint64_t>(block.size(),f.size-copied);
            exact(fd,block.data(),n,f.offset+copied);hash.add(block.data(),n);
            size_t sent=0;while(sent<n){ssize_t k=send(output.v,block.data()+sent,n-sent,MSG_NOSIGNAL);
                if(k<0&&errno==EINTR)continue;
                if(k<=0)throw std::runtime_error("runtime extraction failed (disk space/permissions?)");
                sent+=k;}
            copied+=n; }
        shutdown(output.v,SHUT_WR);
        int status=wait(pid);pid=-1;
        if(status!=0)throw std::runtime_error("tar failed extracting embedded runtime");
        if(hash.finish()!=std::string(f.sha,64))throw std::runtime_error("embedded payload SHA256 mismatch");
    } catch (...) {shutdown(output.v,SHUT_RDWR);if(pid>0){kill(pid,SIGTERM);wait(pid);}throw;}
}
}
std::vector<std::string> environment(const std::string& dir) {
    std::vector<std::string> env;
    for(char** p=environ;*p;++p){std::string v=*p;
        if(v.rfind("PYTHON",0)==0||v.rfind("LD_LIBRARY_PATH=",0)==0||v.rfind("LD_PRELOAD=",0)==0||
           v.rfind("NAV_EMBEDDED_ROOT=",0)==0||v.rfind("PROJ_LIB=",0)==0||v.rfind("PROJ_DATA=",0)==0||v.rfind("GDAL_DATA=",0)==0)continue;
        env.push_back(v);}
    env.push_back("PYTHONHOME="+dir+"/python");env.push_back("PYTHONNOUSERSITE=1");
    env.push_back("PYTHONDONTWRITEBYTECODE=1");
    env.push_back("LD_LIBRARY_PATH="+dir+"/native:"+dir+"/env/lib/python3.10/site-packages/torch/lib");
    return env;
}
int run(const std::string& dir,const std::vector<std::string>& options) {
    std::vector<std::string> args{dir+"/python/bin/python3.10","-S","-s",dir+"/adapter/bootstrap.py"};
    args.insert(args.end(),options.begin(),options.end());
    auto env=environment(dir);std::vector<char*> av,ev;
    for(auto& s:args)av.push_back(s.data());
    av.push_back(nullptr);
    for(auto& s:env)ev.push_back(s.data());
    ev.push_back(nullptr);
    pid_t pid;int rc=posix_spawn(&pid,args[0].c_str(),nullptr,nullptr,av.data(),ev.data());
    if(rc)throw std::runtime_error("cannot launch bundled interpreter: "+std::string(strerror(rc)));
    return wait(pid);
}
std::string root() {
    static std::mutex mutex;std::lock_guard<std::mutex> guard(mutex);
    static std::string result;if(!result.empty())return result;
    Dl_info info;if(!dladdr(&marker,&info)||!info.dli_fname)throw std::runtime_error("cannot locate loaded .so");
    FD fd(open(info.dli_fname,O_RDONLY|O_CLOEXEC));if(fd.v<0)throw std::runtime_error("cannot open loaded .so");
    struct stat st;if(fstat(fd.v,&st)||st.st_size<(off_t)sizeof(Footer))throw std::runtime_error("invalid .so file");
    Footer f;exact(fd.v,&f,sizeof(f),st.st_size-sizeof(f));
    if(memcmp(f.magic,"UAVLOC_SO_V1",11)||f.offset>uint64_t(st.st_size)||f.size>uint64_t(st.st_size)-f.offset||
       f.offset+f.size+sizeof(f)!=uint64_t(st.st_size))throw std::runtime_error(".so has no valid embedded runtime footer");
    std::string id(f.sha,64);if(id.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("invalid bundle hash");
    fs::path cache;
    if(auto v=getenv("NAV_RUNTIME_CACHE"))cache=fs::absolute(v);
    else if(auto v=getenv("XDG_CACHE_HOME"))cache=fs::path(v)/"uavloc-single";
    else if(auto v=getenv("HOME"))cache=fs::path(v)/".cache/uavloc-single";
    else cache=fs::path("/tmp")/("uavloc-single-"+std::to_string(getuid()));
    fs::create_directories(cache);
    if(lstat(cache.c_str(),&st)||!S_ISDIR(st.st_mode)||st.st_uid!=getuid())throw std::runtime_error("runtime cache must be a directory owned by current user");
    chmod(cache.c_str(),0700);
    FD lock(open((cache/(id+".lock")).c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600));
    if(lock.v<0||flock(lock.v,LOCK_EX))throw std::runtime_error("runtime cache lock failed");
    fs::path dest=cache/id;
    if(!fs::is_regular_file(dest/".ready")) {
        std::string pattern=(cache/("extract-XXXXXX")).string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
        if(!mkdtemp(name.data()))throw std::runtime_error("cannot create runtime extraction directory");
        fs::path temp=name.data();
        try {
            std::cerr<<"[UAVLOC] extracting embedded runtime to "<<dest<<std::endl;
            unpack(fd.v,f,temp);
            if(run(temp.string(),{"--script",(temp/"verify_runtime.py").string()})!=0)throw std::runtime_error("extracted runtime integrity check failed");
            std::ofstream(temp/".ready")<<id<<"\n";
            if(fs::exists(dest))fs::remove_all(dest);
            fs::rename(temp,dest);
        } catch (...) {fs::remove_all(temp);throw;}
    }
    if(!fs::is_regular_file(dest/"python/bin/python3.10")||!fs::is_regular_file(dest/"adapter/worker.py"))
        throw std::runtime_error("runtime cache incomplete; remove this version's cache directory");
    result=dest.string();std::cerr<<"[UAVLOC] runtime="<<result<<std::endl;return result;
}
}
extern "C" int uavlocRuntimeCheck(){try{return embedded::run(embedded::root(),{"--self-test"});}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
