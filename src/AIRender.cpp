#include "AIRender.h"
#include <Python.h>
#include <iostream>
#include <string>




void AIRender::Generate()
{
    // Generation code
    std::string input_str="\""+std::string(prompt_gender)+"\" "+"\""+std::string(prompt_hairstyle)+"\" "+"\""+std::string(prompt_headpose)+"\" "+"\""+std::string(prompt_misc)+"\"";
    //std::string input = "Short Buzzed Cut with Curly Top.";
    std::string result = run_python_code(input_str);
    std::cout << "Result from Python: " << result << std::endl;

    // Once paths 
    imgReceived=true;

}


std::string AIRender::run_python_code(const std::string &input_str) {
    // Initialize the Python interpreter
    Py_Initialize();

    // Modify sys.path to include the directory of the Python script
    PyObject* sysPath = PySys_GetObject("path"); // Get sys.path
    PyObject* scriptPath = PyUnicode_DecodeFSDefault(python_path.data());

    if (PyList_Append(sysPath, scriptPath) < 0) {
        PyErr_Print();
        std::cerr << "Failed to add script directory to sys.path" << std::endl;
        Py_DECREF(scriptPath);
        Py_Finalize();
        return "";
    }

    // Print sys.path for debugging
    PyObject *sysPathList = PySys_GetObject("path");
    Py_ssize_t len = PyList_Size(sysPathList);
    std::cout << "sys.path:" << std::endl;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = PyList_GetItem(sysPathList, i);
        std::cout << PyUnicode_AsUTF8(item) << std::endl;
    }


    PyList_Append(sysPath, scriptPath); // Append the script's directory to sys.path
    Py_DECREF(scriptPath);

    // Load the Python script module
    PyObject* pModule = PyImport_ImportModule(python_script.data());
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load Python script 'copilot_interface.py'" << std::endl;
        Py_Finalize();
        return "";
    }

    // Get the function from the module
    PyObject* pFunc = PyObject_GetAttrString(pModule, "process_string");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        if (PyErr_Occurred())
            PyErr_Print();
        std::cerr << "Cannot find function 'process_string'" << std::endl;
        Py_DECREF(pModule);
        Py_Finalize();
        return "";
    }

    // Prepare the argument for the Python function
    PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(input_str.c_str()));
    
    // Call the Python function
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    // Check the result and convert it to std::string
    if (pValue != nullptr) {
        std::string result = PyUnicode_AsUTF8(pValue);
        Py_DECREF(pValue);
        Py_Finalize();
        return result;
    } else {
        PyErr_Print();
        std::cerr << "Call to Python function failed" << std::endl;
        Py_Finalize();
        return "";
    }
}