#include <iostream>
#include <vector>
#include <string>

struct PermParameters
{
    float Theta;
    std::vector<float> Values;
    std::vector<std::string> Labels;
    float Selected;
};


class Perm
{
public:

    // Main methods
    Perm();
    void Apply();

    // Get/Set
    PermParameters& Params() { return mParams; }
private:
    PermParameters mParams;

};