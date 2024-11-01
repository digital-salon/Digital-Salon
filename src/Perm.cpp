#include "Perm.h"

Perm::Perm()
{

    // Initialize parameters
    mParams.Theta = 0.f;
    mParams.Values = {0.f, 1.f, 2.f};
    mParams.Labels = {"straight", "wavy", "curly"};
    mParams.Selected = 0.f;
}

void Perm::Apply()
{
    // Main Perm code
}