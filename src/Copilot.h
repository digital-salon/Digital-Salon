#include <iostream>


struct CopilotParameters
{
    std::string Prompt;
    bool PromptSent;
    bool hairReceived;
};


class Copilot
{

public:

    
    // Main methods
    Copilot();
    void Generate(std::string prompt);

    // Get/Set
    CopilotParameters& Params() { return mParams; }

    std::string data1_path="../tmp/copilot/tmp1.data";
    std::string data2_path="../tmp/copilot/tmp2.data";
    std::string data3_path="../tmp/copilot/tmp3.data";
    std::string data_path_backup="../data/hair/strands00002.data";
    
    std::string img1_path="../tmp/copilot/tmp1.jpg";
    std::string img2_path="../tmp/copilot/tmp2.jpg";
    std::string img3_path="../tmp/copilot/tmp3.jpg";
    std::string img1_path_batckup="../data/images/strands00014_00061_01100.jpg";
    std::string img2_path_batckup="../data/images/strands00018_00075_11110.jpg";
    std::string img3_path_batckup="../data/images/strands00168_00430_11000.jpg";

private:

    CopilotParameters mParams;

    std::string run_python_code(const std::string &input_str);

    std::string python_path = "../extern/copilot";

    std::string python_script = "copilot_interface";
};