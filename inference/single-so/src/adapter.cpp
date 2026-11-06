#include "vNavigatorInterface.h"
#ifdef NAV_EMBEDDED
#include "runtime.hpp"
#endif
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <spawn.h>
#include <poll.h>
#include <unistd.h>
#include <signal.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

extern char **environ;
namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t MAX_IMAGE = 128 * 1024 * 1024;

// 0 = no limit. Steps have no adapter-side limit by default (NAV_STEP_TIMEOUT_MS sets one for debugging).
int timeout_ms(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v) return fallback;
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (!end || *end || n < 0 || n > 3600000) throw std::runtime_error("invalid timeout environment variable");
    return static_cast<int>(n);
}

Clock::time_point deadline_after(int ms) {
    return ms == 0 ? Clock::time_point::max() : Clock::now() + std::chrono::milliseconds(ms);
}

// One worker per project prevents collisions between their same-named Python packages.
class Worker {
    int fd_ = -1;
    pid_t pid_ = -1;
    void transfer(void* bytes, size_t n, bool writing, Clock::time_point deadline) {
        char* p = static_cast<char*>(bytes);
        const bool unlimited = deadline == Clock::time_point::max();
        while (n) {
            long long ms = -1;   // poll(-1): wait until the worker answers or its socket closes
            if (!unlimited) {
                ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
                if (ms <= 0) throw std::runtime_error("worker timeout");
            }
            pollfd poll_fd{fd_, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
            int ready = poll(&poll_fd, 1, static_cast<int>(ms));
            if (ready < 0 && errno == EINTR) continue;
            if (ready <= 0) throw std::runtime_error(ready == 0 ? "worker timeout" : "poll failed");
            ssize_t k = writing ? send(fd_, p, n, MSG_NOSIGNAL | MSG_DONTWAIT)
                                : recv(fd_, p, n, MSG_DONTWAIT);
            if (k < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            if (k <= 0) throw std::runtime_error("worker connection closed");
            p += k; n -= static_cast<size_t>(k);
        }
    }
public:
    ~Worker() { stop(); }
    bool alive() const { return fd_ >= 0; }
    void stop() noexcept {
        if (fd_ >= 0) { shutdown(fd_, SHUT_RDWR); close(fd_); fd_ = -1; }
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            bool reaped = false;
            for (int i = 0; i < 40; ++i) {
                pid_t r = waitpid(pid_, nullptr, WNOHANG);
                if (r == pid_ || (r < 0 && errno == ECHILD)) { reaped = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            if (!reaped) {
                kill(pid_, SIGKILL);
                while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
            }
            pid_ = -1;
        }
    }
    std::string response(Clock::time_point deadline) {
        uint32_t size = 0;
        transfer(&size, sizeof(size), false, deadline);
        size = ntohl(size);
        if (size == 0 || size > 65536) throw std::runtime_error("invalid worker reply size");
        std::string reply(size, '\0');
        transfer(reply.data(), size, false, deadline);
        return reply;
    }
    bool start(const std::string& config, int modality) {
        stop();
#ifdef NAV_EMBEDDED
        std::string runtime=embedded::root();
#endif
        int fds[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds)) throw std::runtime_error("socketpair failed");
        // Child descriptor is 3; close parent end before dup2 (it may itself be 3).
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, fds[0]);
        posix_spawn_file_actions_adddup2(&actions, fds[1], 3);
        if (fds[1] != 3) posix_spawn_file_actions_addclose(&actions, fds[1]);
        std::string mode = std::to_string(modality);
#ifdef NAV_EMBEDDED
        std::string python_path=runtime+"/python/bin/python3.10";
        const char* python=python_path.c_str();
        std::vector<std::string> args{python,"-S","-s","-u",runtime+"/adapter/bootstrap.py","--fd","3","--modality",std::to_string(modality),"--config",config};
        auto env=embedded::environment(runtime);std::vector<char*> envp;
        for(auto& s:env)envp.push_back(s.data());
        envp.push_back(nullptr);
#else
        const char* python = std::getenv("NAV_PYTHON");
        const char* worker = std::getenv("NAV_WORKER");
        if (!python) python = NAV_PYTHON;
        if (!worker) worker = NAV_WORKER;
        std::vector<std::string> args{python, "-u", worker, "--fd", "3", "--modality", mode, "--config", config};
#endif
        std::vector<char*> argv;
        for (auto& s : args) argv.push_back(s.data());
        argv.push_back(nullptr);
        int error = posix_spawn(&pid_, python, &actions, nullptr, argv.data(),
#ifdef NAV_EMBEDDED
                               envp.data()
#else
                               environ
#endif
                               );
        posix_spawn_file_actions_destroy(&actions);
        close(fds[1]);
        if (error) { close(fds[0]); pid_ = -1; throw std::runtime_error(std::string("spawn failed: ") + strerror(error)); }
        fd_ = fds[0];
        try {
            auto reply = response(deadline_after(timeout_ms("NAV_INIT_TIMEOUT_MS", 600000)));
            if (reply == "DISABLED") { stop(); return false; }
            if (reply != "READY") throw std::runtime_error(reply);
        } catch (...) { stop(); throw; }
        return true;
    }
    std::string step(const std::string& metadata, const unsigned char* data, size_t size) {
        auto deadline = deadline_after(timeout_ms("NAV_STEP_TIMEOUT_MS", 0));
        uint32_t n = htonl(static_cast<uint32_t>(metadata.size()));
        transfer(&n, sizeof(n), true, deadline);
        transfer(const_cast<char*>(metadata.data()), metadata.size(), true, deadline);
        transfer(const_cast<unsigned char*>(data), size, true, deadline);
        return response(deadline);
    }
};

class DualNavigator final : public VNAVIGATORINTERFACE::IVideoNavigator {
    Worker workers_[2];
    std::mutex mutex_;
    bool initialized_ = false;
    bool eo_ = false;
public:
    ~DualNavigator() { closeVNavigator(); }
    int initVNavigator(const std::string& file, int type, int day, bool, bool eo) override {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_ = false;
        for (auto& w : workers_) w.stop();
        try {
            if (type < 0 || type > 1 || day < 0 || day > 1) throw std::runtime_error("invalid image/day type");
            std::string config = std::filesystem::absolute(file).string();
            if (!std::filesystem::is_regular_file(config)) throw std::runtime_error("initialization config missing: " + config);
            // Both models initialize before the evaluator starts its frame timer.
            if (!workers_[type].start(config, type)) throw std::runtime_error("requested modality disabled in config");
            workers_[1-type].start(config, 1-type);
            eo_ = eo;
            initialized_ = true;
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "[DualNavigator:init] " << e.what() << std::endl;
            for (auto& w : workers_) w.stop();
            return 0;
        }
    }
    int closeVNavigator() override {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_ = false;
        for (auto& w : workers_) w.stop();
        return 1;
    }
    int stepVNavigator(const unsigned char* data, int width, int height, int channels,
                       int type, int day, bool slant, const VNAVIGATORINTERFACE::vTelemParams& t,
                       VNAVIGATORINTERFACE::vNaviResult& out, void*) override {
        std::lock_guard<std::mutex> lock(mutex_);
        out = VNAVIGATORINTERFACE::vNaviResult();
        if (!initialized_ || !data || width <= 0 || height <= 0 || width > 32768 || height > 32768 ||
            type < 0 || type > 1 || day < 0 || day > 1 ||
            (type == 1 ? channels != 3 : (channels != 1 && channels != 3))) return 0;
        size_t size = static_cast<size_t>(width) * height * channels;
        if (size > MAX_IMAGE || !workers_[type].alive()) return 0;
        try {
            const double values[] = {t.droneLongtitude, t.droneLatitude, t.droneAltitude,
                t.droneYaw, t.dronePitch, t.droneRoll, t.sensorYaw, t.sensorPitch, t.sensorRoll,
                t.innsYaw, t.innsPitch, t.innsRoll, t.hfov, t.vfov, t.ccdSize};
            for (double v : values) if (!std::isfinite(v)) throw std::runtime_error("nonfinite telemetry");
            std::ostringstream m;
            m.imbue(std::locale::classic()); m << std::setprecision(17);
            m << "{\"width\":" << width << ",\"height\":" << height << ",\"channels\":" << channels
              << ",\"imgType\":" << type << ",\"dayOrNight\":" << day << ",\"isSlant\":" << (slant?1:0)
              << ",\"isEO\":" << (eo_?1:0) << ",\"number\":" << t.number << ",\"telemetry\":[";
            for (size_t i = 0; i < 15; ++i) { if(i) m << ','; m << values[i]; }
            m << "]}";
            // *_gt values never cross the process boundary.
            std::string reply = workers_[type].step(m.str(), data, size);
            if (reply.rfind("OK ", 0) != 0) { std::cerr << "[DualNavigator:step] " << reply << std::endl; return 0; }
            std::istringstream stream(reply.substr(3)); stream.imbue(std::locale::classic());
            double lon, lat, alt, confidence;
            if (!(stream >> lon >> lat >> alt >> confidence) || !std::isfinite(lon) || !std::isfinite(lat) ||
                !std::isfinite(alt) || !std::isfinite(confidence) || std::abs(lon)>180 || std::abs(lat)>90 ||
                (confidence != 0 && confidence != 1)) throw std::runtime_error("invalid localization result");
            out.longtitude=lon; out.latitude=lat; out.altitude=alt; out.selfEval=static_cast<float>(confidence);
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "[DualNavigator:step] " << e.what() << std::endl;
            workers_[type].stop();
            return 0;
        }
    }
};
}

std::shared_ptr<VNAVIGATORINTERFACE::IVideoNavigator> createVideoNavigator() {
    return std::make_shared<DualNavigator>();
}
void freeVideoNavigator(std::shared_ptr<VNAVIGATORINTERFACE::IVideoNavigator> p) {
    if (p) p->closeVNavigator();
}
