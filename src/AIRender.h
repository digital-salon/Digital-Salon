#include <iostream>

class AIRender
{

public:

    
    // Main methods
    AIRender(){}
    void Generate();


    std::string input_path="../tmp/controlnet/input.png";
    std::string output_path="../tmp/controlnet/output.png";
    bool imgReceived=false;

    char prompt_gender[1024]="female";
    char prompt_hairstyle[1024]="blonde hair";
    char prompt_headpose[1024]="frontal face";
    char prompt_misc[1024]="wear a sweater";

private:

    std::string run_python_code(const std::string &input_str);

    std::string python_path = "../extern/controlnet";

    std::string python_script = "controlnet_interface";



};