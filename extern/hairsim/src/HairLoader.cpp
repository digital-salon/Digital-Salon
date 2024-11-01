#include "HairLoader.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <random>

void Loader::reverse_strands()
{
    for (auto& s : strands) {
        std::reverse(s.begin(), s.end());
    }
}

void Loader::reset()
{
    strands.resize(0);
}

bool Loader::is_empty() const
{
    return strands.empty();
}

void Loader::clean()
{
    for (auto iter = strands.begin(); iter != strands.end();) {
        auto& s = *iter;
        auto pre = s.begin();
        for (auto i = pre + 1; i != s.end();) {
            const auto d = (*i - *pre).norm();
            if (d <= std::numeric_limits<decltype(d)>::epsilon()) {
                i = s.erase(i);
            }
            else {
                pre = i;
                ++i;
            }
        }
        if (static_cast<int>(iter->size()) == 1) {
            iter = strands.erase(iter);
        }
        else {
            ++iter;
        }
    }
}

nlohmann::json Loader::query() const
{
    if (is_empty())
    {
        throw std::runtime_error("failed in hair::query, the hair model is empty");
    }

    int num_strands = static_cast<int>(strands.size());
    int num_vertices = 0;
    int min_vertices_per_strand = std::numeric_limits<int>::max();
    int max_vertices_per_strand = 0;

    int num_duplicated_vertices = 0;
    int num_degenerated_strands = 0;
    int min_vertices_per_non_degenerated_strand = std::numeric_limits<int>::max();
    int max_vertices_per_non_degenerated_strand = 0;

    for (const auto& s : strands) {
        const int n = static_cast<int>(s.size());
        num_vertices += n;
        min_vertices_per_strand = std::min(min_vertices_per_strand, n);
        max_vertices_per_strand = std::max(max_vertices_per_strand, n);

        int _duplicated = 0;
        size_t _i = 0;
        for (int i = 1; i < n; ++i) {
            const auto d = (s[i] - s[_i]).norm();
            if (d <= std::numeric_limits<decltype(d)>::epsilon()) {
                ++_duplicated;
            }
            else {
                _i = i;
            }
        }
        num_duplicated_vertices += _duplicated;
        const int cleaned_n = n - _duplicated;
        if (cleaned_n == 1) {
            ++num_degenerated_strands;
        }
        else {
            min_vertices_per_non_degenerated_strand = std::min(min_vertices_per_non_degenerated_strand, cleaned_n);
            max_vertices_per_non_degenerated_strand = std::max(max_vertices_per_non_degenerated_strand, cleaned_n);
        }
    }

    nlohmann::json j;
    j["num_duplicated_vertices"] = num_duplicated_vertices;
    j["num_degenerated_strands"] = num_degenerated_strands;

    j["without_cleanup"]["num_strands"] = num_strands;
    j["without_cleanup"]["num_vertices"] = num_vertices;
    j["without_cleanup"]["min_vertices_per_strand"] = min_vertices_per_strand;
    j["without_cleanup"]["max_vertices_per_strand"] = max_vertices_per_strand;
    j["without_cleanup"]["ave_vertices_per_strand"] = static_cast<float>(num_vertices) / static_cast<float>(num_strands);

    if (max_vertices_per_non_degenerated_strand > 1)
    {
        j["with_cleanup"]["num_strands"] = num_strands - num_degenerated_strands;
        j["with_cleanup"]["num_vertices"] = num_vertices - num_duplicated_vertices - num_degenerated_strands;
        j["with_cleanup"]["min_vertices_per_strand"] = min_vertices_per_non_degenerated_strand;
        j["with_cleanup"]["max_vertices_per_strand"] = max_vertices_per_non_degenerated_strand;
        j["with_cleanup"]["ave_vertices_per_strand"] = static_cast<float>(num_vertices - num_duplicated_vertices - num_degenerated_strands) / static_cast<float>(num_strands - num_degenerated_strands);
    }

    return j;
}

void Loader::load_data(const std::string& fn)
{
    reset();
    std::ifstream ifs(fn, std::ios_base::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("failed in hair::load_data, cannot open the file");
    }

    int num;
    ifs.read(reinterpret_cast<char*>(&num), sizeof(int));

    if (!num) {
        throw std::runtime_error("failed in hair::load_data, not strands included");
    }
    strands.resize(num);

    for (auto& s : strands) {
        ifs.read(reinterpret_cast<char*>(&num), sizeof(int));
        if (!num) {
            throw std::runtime_error("failed in hair::load_data, a strand is empty");
        }
        s.resize(num);
        ifs.read(reinterpret_cast<char*>(s.data()), sizeof(Eigen::Vector3f) * num);
    }

    if (ifs.eof()) {
        throw std::runtime_error("failed in hair::load_data, more data expected");
    }
    ifs.read(reinterpret_cast<char*>(&num), 1);
    if (!ifs.eof()) {
        throw std::runtime_error("failed in hair::load_data, unknown data not parsed");
    }

    clean_data();
}

void Loader::clean_data()
{
    // Takes only actual strands
    std::vector<std::vector<Eigen::Vector3f>> cleanStrands;
    for (int i = 0; i < strands.size(); i++)
    {
        if (strands[i].size() > 1)
        {
            cleanStrands.push_back(strands[i]);
        }
    }

    // Copy
    strands = cleanStrands;
}

void Loader::save_data(const std::string& fn) const
{
    if (is_empty()) {
        throw std::runtime_error("failed in hair::save_data, the hair model is empty");
    }
    std::ofstream ofs(fn, std::ios_base::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed in hair::save_data, cannot open the file");
    }
    int num = static_cast<int>(strands.size());
    ofs.write(reinterpret_cast<const char*>(&num), sizeof(int));
    for (auto& s : strands) {
        num = static_cast<int>(s.size());
        ofs.write(reinterpret_cast<const char*>(&num), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(s.data()), sizeof(Eigen::Vector3f) * num);
    }
}

void Loader::load(const std::string& fn)
{
    size_t dot = fn.find_last_of(".");
    std::string ext = "";
    if (dot != std::string::npos) {
        ext = fn.substr(dot, fn.size() - dot);
    }
    if (ext == ".data") {
        load_data(fn);
    }
    else if (ext == ".abc") {
        load_alembic(fn);
    }
    else if (ext == ".obj")
    {
        load_obj(fn);
    }
    else {
        throw std::runtime_error("failed in hair::load, invalid file extension");
    }
}

void Loader::save(const std::string& fn) const
{
    size_t dot = fn.find_last_of(".");
    std::string ext = "";
    if (dot != std::string::npos) {
        ext = fn.substr(dot, fn.size() - dot);
    }
    if (ext == ".data") {
        save_data(fn);
    }
    else if (ext == ".abc") {
        save_alembic(fn);
    }
    else if (ext == ".obj") {
        save_obj(fn);
    }
    else if (ext == ".ply") {
        save_ply_binary(fn);
    }
    else {
        throw std::runtime_error("failed in hair::save, invalid file extension");
    }
}

void Loader::load_alembic(const std::string& fn)
{
    //AlembicIO ar(fn, AlembicIO::Read);
    //ar.load_hair(strands);
}

void Loader::save_alembic(const std::string& fn) const
{
    //AlembicIO aw(fn, AlembicIO::Write);
    //aw.create_hair("hair");
    //aw.push_back(strands);
}

void Loader::save_ply_binary(const std::string& fn) const
{
    if (is_empty()) {
        throw std::runtime_error("failed in hair::save_ply_data, the hair model is empty");
    }
    std::ofstream ofs(fn, std::ios_base::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed in hair::save_ply_data, cannot open the file");
    }

    int total_vertices = 0;
    for (const auto& s : strands) {
        total_vertices += static_cast<int>(s.size());
    }
    int total_edges = total_vertices - static_cast<int>(strands.size());

    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << total_vertices << "\n";
    ofs << "property float32 x\n";
    ofs << "property float32 y\n";
    ofs << "property float32 z\n";
    ofs << "element edge " << total_edges << "\n";
    ofs << "property int32 vertex1\n";
    ofs << "property int32 vertex2\n";
    ofs << "end_header\n";
    for (const auto& s : strands) {
        ofs.write(reinterpret_cast<const char*>(s.data()), sizeof(Eigen::Vector3f) * s.size());
    }
    std::vector<int> edge_indices;
    int cur = 0;
    for (const auto& s : strands) {
        int n = static_cast<int>(s.size());
        edge_indices.resize((n - 1) * 2);
        for (int i = 0; i < n - 1; ++i) {
            edge_indices[i * 2 + 0] = cur + i;
            edge_indices[i * 2 + 1] = cur + i + 1;
        }
        ofs.write(
            reinterpret_cast<const char*>(edge_indices.data()),
            sizeof(int) * edge_indices.size());
        cur += n;
    }
}

void Loader::save_ply_binary_with_random_strand_colors(const std::string& fn) const
{
    std::mt19937 mt(777);
    std::uniform_int_distribution<int> dist(0, 255);

    if (is_empty())
    {
        throw std::runtime_error("failed in hair::save_data, the hair model is empty");
    }
    std::ofstream ofs(fn, std::ios_base::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("failed in hair::save_data, cannot open the file");
    }

    int total_vertices = 0;
    for (const auto& s : strands)
    {
        total_vertices += static_cast<int>(s.size());
    }
    int total_edges = total_vertices - static_cast<int>(strands.size());

    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << total_vertices * 2 << "\n";
    ofs << "property float32 x\n";
    ofs << "property float32 y\n";
    ofs << "property float32 z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "element face " << total_edges << "\n";
    ofs << "property list int32 int32 vertex_index\n";
    ofs << "end_header\n";

    //write vertices
    for (const auto& s : strands)
    {
        int n = static_cast<int>(s.size());
        int r = dist(mt);
        int g = dist(mt);
        int b = dist(mt);

        for (int i = 0; i < n; i++)
        {
            ofs.write(reinterpret_cast<const char*>(s[i].data()), sizeof(int) * s[i].size());

            std::vector<unsigned char> color;
            color.resize(3);
            color[0] = r;
            color[1] = g;
            color[2] = b;
            ofs.write(reinterpret_cast<const char*>(color.data()), sizeof(unsigned char) * color.size());
        }
    }

    //write vertices for a second time
    for (const auto& s : strands)
    {
        int n = static_cast<int>(s.size());
        for (int i = 0; i < n; i++)
        {
            ofs.write(reinterpret_cast<const char*>(s[i].data()), sizeof(int) * s[i].size());

            std::vector<unsigned char> color;
            color.resize(3);
            color[0] = 255;
            color[1] = 0;
            color[2] = 0;
            ofs.write(reinterpret_cast<const char*>(color.data()), sizeof(unsigned char) * color.size());
        }
    }

    //write faces
    int cur = 0;
    std::vector<int> faces;
    faces.resize(total_edges * 4);
    int j = 0;
    for (const auto& s : strands)
    {
        int n = static_cast<int>(s.size());
        for (int i = 0; i < n - 1; ++i)
        {
            faces[j] = 3;
            j = j + 1;
            faces[j] = cur + i;
            j = j + 1;
            faces[j] = cur + i + 1;
            j = j + 1;
            faces[j] = cur + i + total_vertices;
            j = j + 1;
        }
        cur += n;
    }
    ofs.write(reinterpret_cast<const char*>(faces.data()), sizeof(int) * faces.size());
}

void Loader::save_obj(const std::string& fn) const
{
    if (is_empty()) {
        throw std::runtime_error("failed in hair::save_obj, the hair model is empty");
    }
    std::ofstream ofs(fn);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed in hair::save_obj, cannot open the file");
    }

    ofs << "o test\n";
    const int num_strands = static_cast<int>(strands.size());

    for (int i = 0; i < num_strands; ++i) {
        const auto& s = strands[i];
        const int n = static_cast<int>(s.size());
        for (const auto& v : s) {
            ofs << "v " << v.transpose() << "\n";
        }
    }

    int vert_base = 0;
    for (int i = 0; i < num_strands; ++i) {
        const auto& s = strands[i];
        const int n = static_cast<int>(s.size());
        for (int j = 0; j < n - 1; ++j)
        {
            ofs << "l " << vert_base + j + 1 << " " << vert_base + j + 2 << "\n";
        }
        vert_base += n;
    }

    //int vert_base = 0;
    //for (int i = 0; i < num_strands; ++i) {
    //    const auto& s = strands[i];
    //    const int n = static_cast<int>(s.size());
    //    ofs << "\n# strand " << i << ": " << n << " vertices\n";
    //    for (const auto& v : s) {
    //        ofs << "v " << v.transpose() << "\n";
    //    }
    //    ofs << "l";
    //    for (int j = 0; j < n; ++j) {
    //        ofs << " " << vert_base + j + 1;
    //    }
    //    ofs << "\n";
    //    vert_base += n;
    //}
    //ofs << "\n";
    //ofs << "# total vertices: " << vert_base << "\n";
}

void Loader::load_obj(const std::string& fn)
{
    printf("TODO load obj hair\n");
}

void Loader::trim_randomly(float max_trimming_vertices_proportion)
{
    std::mt19937 mt(777);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int n = 0; n < strands.size(); n++)
    {
        int num_of_vertices = strands[n].size();

        int max_trimming_num = int(num_of_vertices * max_trimming_vertices_proportion);
        if (max_trimming_num <= 0)
        {
            continue;
        }
        int trimming_num = static_cast<int>(dist(mt) * static_cast<float>(max_trimming_num));

        for (int i = 0; i < trimming_num; i++)
        {
            strands[n].pop_back();
        }
    }
}

void Loader::move_roots_randomly(float moving_range)
{
    std::mt19937 mt(777);
    std::uniform_int_distribution<int> dist(0, 100);

    float x_min = strands[0][0][0];
    float x_max = strands[0][0][0];

    for (int n = 0; n < strands.size(); ++n)
    {
        const float x = strands[n][0][0];
        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
    }
    const float global_scale = (x_max - x_min) / std::sqrt(static_cast<float>(strands.size())) / 0.001f; // 0.001f is an empirical magic number, feel free to change if it does not work anymore.

    for (int n = 0; n < strands.size(); n++)
    {
        Eigen::Vector3f root = strands[n][0];

        if (strands[n].size() < 2)
        {
            continue;
        }

        Eigen::Vector3f normal = strands[n][1] - strands[n][0];
        Eigen::Vector3f move = Eigen::Vector3f(static_cast<float>(dist(mt)), static_cast<float>(dist(mt)), static_cast<float>(dist(mt))) / 100.0f;
        move = move.cross(normal);
        if (move.norm() > 0.0f)
        {
            move = move.normalized();
        }
        float range = static_cast<float>(dist(mt)) / 100.0f * moving_range;
        move = move * range;
        move *= global_scale;

        for (int i = 0; i < strands[n].size(); i++)
        {
            strands[n][i] = strands[n][i] + move;
        }
    }
}