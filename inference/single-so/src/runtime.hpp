#pragma once
#include <string>
#include <vector>
namespace embedded {
std::string root();
std::vector<std::string> environment(const std::string& root);
int run(const std::string& root, const std::vector<std::string>& args);
}
