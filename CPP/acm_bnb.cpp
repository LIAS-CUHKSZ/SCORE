#include<iostream>
#include "struct.hpp"
#include <vector>
#include <Eigen/Dense>
#include<fstream>
#include<sstream>
#include <cmath>
#include "BnB.hpp"
int main(){
    std::cout<<"Hello World!"<<std::endl;

    int max_iter=1e6;
    double rotation_epsilon = 0.012;
    double branch_resolution= M_PI/512;
    double sample_resolution= M_PI/16;
    double translation_epsilon=0.05;
    double translation_resolution=0.015;
    string satured_kernel="p-8";
    ACM_BnB* solver = new ACM_BnB(max_iter,rotation_epsilon,branch_resolution,sample_resolution,translation_epsilon,translation_resolution,satured_kernel);

    std::ifstream file("../../test_data/test_data_cpp.csv");
    if(!file.is_open()){
        cout<<"file not found"<<endl;
        return 1;
    }
    // need to implement the line extraction algorithm
    string line;
    vector<Vector3d> line2ds;
    vector<Vector3d> line3ds;
    vector<Vector3d> point3ds;
    vector<int> line_tags;
    char comma;
    double line_tag;
    Vector3d line2d;
    Vector3d line3d;
    // combined_data = [id_cluster, data_2d_N_clustered, data_3d_v_clustered]; % 合并为 N×7 矩阵
    while(getline(file,line)){
        stringstream ss(line);

        ss  >>line_tag>>comma
            >>line2d[0]>>comma>>line2d[1]>>comma>>line2d[2]>>comma
            >>line3d[0]>>comma>>line3d[1]>>comma>>line3d[2];
        line2ds.push_back(line2d);
        line3ds.push_back(line3d);
        line_tags.push_back(line_tag);
        point3ds.push_back(Vector3d::Zero());
    }
    file.close();

    solver->data_load(line2ds, line3ds,point3ds, line_tags);
    solver->rot_bnb_estimate();














    return 0;
}