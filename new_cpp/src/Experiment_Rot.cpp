#include "RotFGO.h"
#include "helper.h"
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
vector<string> scene_names = {"S1(workstation)", "S2(office)", "S3(game bar)", "S4(art room)"};
int main(int argc, char** argv)
{
    // load arguments
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << "choose one of four scenes: 1, 2, 3, 4" << std::endl;
        std::cerr << "Usage: " << argv[0] << "choose using gt or predicetd labels: y/n" << std::endl;
        std::cerr << "Usage: " << argv[0] << "q_value" << std::endl;
        return -1;
    }
    int scene_id = std::stoi(argv[1]);
    bool use_gt_labels = (std::string(argv[2]) == "y");
    double q_value = std::stod(argv[3]);
    std::cout << "Chosen scene: " << scene_names[scene_id-1] << std::endl;
    std::cout << "Use gt labels: " << use_gt_labels << std::endl;
    std::cout << "Q value: " << q_value << std::endl;

    // load data
    string data_folder;
    if (use_gt_labels)
        data_folder = "/home/leoj/Github_Repos/SCORE/csv_dataset/S" + std::to_string(scene_id) + "/";
    else
        data_folder = "/home/leoj/Github_Repos/SCORE/csv_dataset/S" + std::to_string(scene_id) + "_pred/";
    auto lines3D_data = helper::readCSV(data_folder + "3Dlines.csv");
    vector<string> query_image_list;
    // read text file and push to query_image_list
    ifstream query_txt(data_folder + "query.txt");
    string line;
    while (getline(query_txt, line))
    {
        // remove .jpg
        line = line.substr(0, line.size() - 4);
        query_image_list.push_back(line);
    }
    query_txt.close();
    std::cout << "Loaded " << query_image_list.size() << " query images" << std::endl;
    
    // params
    double branch_reso = M_PI/512;
    double epsilon_r = 0.015;
    double sample_reso = M_PI/256;
    double prox_thres = branch_reso;
    bool   use_saturated = true;
    RotFGO solver(branch_reso, epsilon_r, sample_reso, prox_thres, use_saturated, q_value);

    return 0;
}