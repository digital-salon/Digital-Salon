#ifndef GLOBALS_H
#define GLOBALS_H

#include <fstream>
#include <vector>
#include <map>
#include <list>
#include <limits>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

template<class T> class Singleton
{
//Singleton
//http://www.yolinux.com/TUTORIALS/C++Singleton.html

public:
    static T* inst()
    {
        if (!m_pInstance)
        {
            m_pInstance = new T;
        }
        return m_pInstance;
    }

private:
    Singleton(){};
    Singleton(const Singleton&){};
    Singleton& operator=(const Singleton&){};

    static T* m_pInstance;
};

template<class T> T* Singleton<T>::m_pInstance = NULL;

class HPTimer
{
private:
    struct timespec t0;
    struct timespec t1;
    double et, last_dt;
    bool na;

    double timespec_to_double(const struct timespec& ts)
    {
        return ts.tv_sec + ts.tv_nsec / 1e9;
    }

public:
    HPTimer()
    {
        if (clock_gettime(CLOCK_MONOTONIC, &t0) == -1)
        {
            perror("clock_gettime");
            na = true;
        }
        else
        {
            na = false;
        }
        reset();
    }

    void reset()
    {
        if (clock_gettime(CLOCK_MONOTONIC, &t0) == -1)
        {
            perror("clock_gettime");
            na = true;
        }
        else
        {
            last_dt = 0.0;
            et = 0.0;
        }
    }

    double time()
    {
        double dt;

        if (na) return 0.0;

        if (clock_gettime(CLOCK_MONOTONIC, &t1) == -1)
        {
            perror("clock_gettime");
            return 0.0;
        }

        dt = timespec_to_double(t1) - timespec_to_double(t0);
        if (dt < 0.0)
        {
            et = et + last_dt;
            t0 = t1;
            last_dt = 0.0;
            return et;
        }
        else
        {
            last_dt = dt;
            return et + dt;
        }
    }
};

#endif