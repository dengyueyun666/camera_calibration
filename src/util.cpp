#include "util.h"

int ReadFilenames(std::vector<std::string>& filenames, const std::string& directory)
{

    DIR* dir;
    class dirent* ent;
    class stat st;

    dir = opendir(directory.c_str());
    if (!dir) {
        return -1;
    }
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        filenames.push_back(file_name);
    }
    closedir(dir);

    std::sort(filenames.begin(), filenames.end());
    return (static_cast<int>(filenames.size()));
}

double RadianToDegree(double radian)
{
    return radian * 180.0 / CV_PI;
}

double DegreeToRadian(double degree)
{
    return degree / 180.0 * CV_PI;
}
