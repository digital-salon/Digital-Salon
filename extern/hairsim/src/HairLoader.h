#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

class Loader
{
public:
    void reset();
    bool is_empty() const;
    void clean();
    nlohmann::json query() const;

    void load_data(const std::string& fn);
    void clean_data(); // makes sure there are not single point strands!
    void load_alembic(const std::string& fn);
    void save_data(const std::string& fn) const;
    void save_ply_binary(const std::string& fn) const;
    void save_ply_binary_with_random_strand_colors(const std::string& fn) const;
    void save_obj(const std::string& fn) const;
    void load_obj(const std::string& fn);
    void save_alembic(const std::string& fn) const;
    void trim_randomly(float max_trimming_vertices_proportion);
    void move_roots_randomly(float moving_range);

    void load(const std::string& fn);
    void save(const std::string& fn) const;

    void reverse_strands();

    std::vector<std::vector<Eigen::Vector3f>> strands;
};