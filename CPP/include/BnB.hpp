#include <cmath>
#include <iostream>
#include "struct.hpp"
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <utility> // for std::pair
using namespace std;
class ACM_BnB{
    public:
        ACM_BnB(int max_iter,double rotation_epsilon,double branch_resolution,double sample_resolution,double translation_epsilon,double translation_resolution,string satured_kernel):
        max_iter(max_iter),
        rotation_epsilon(rotation_epsilon),
        branch_resolution(branch_resolution),
        sample_resolution(sample_resolution),
        translation_epsilon(translation_epsilon),
        translation_resolution(translation_resolution),
        satured_kernel(satured_kernel){
            if(this->satured_kernel=="p-8"){
                for(int i=0;i<15;i++){
                    this->satured_kernel_cache.push_back(pow((i+1), -8));
                }
            }
        };


        // data input
        void data_load(vector<Vector3d> &line_2ds,vector<Vector3d> &line_3ds,vector<Vector3d> &points_3d,vector<int> &line_tags);


        // data process
        void preprocess(line_pair& lp);
        void rot_bnb_estimate();
        void rot_bnb_epoch(Square &branch);


        pair<VectorXd,VectorXd> h1_bounds(Square &branch);
        pair<VectorXd,VectorXd> h2_bounds(Square &branch);
        vector<double> upper_interval(RowVector3d coef_1,RowVector3d coef_2);
        vector<double> lower_interval(RowVector3d coef);
        pair<double,double> satured_interval_stabbing(vector<double> &interval,vector<int> line_tags);

        // Transition function
        void angle_bisectors(line_pair &lp);
        MatrixXd generate_params(VectorXd &lower,VectorXd &upper);



        ~ACM_BnB();

    private:
        // data
        vector<line_pair> init_line_pairs;
        vector<line_pair> inlier_line_pairs;
        double room[3]={0,0,0};
        // properties
        double lower_bound;
        double upper_bound;
        int max_iter;
        double rotation_epsilon;
        double branch_resolution;
        double sample_resolution;
        double translation_epsilon;
        double translation_resolution;

        string satured_kernel;
        vector<double> satured_kernel_cache;
        //output
        double alpha_hat;
        double phi_hat;
        double theta_hat;
        Vector3d vec_rot=Vector3d::Zero();
        Matrix3d R_hat;
        Vector3d t_hat;
        //debug   
        long long iter_count=0;    

};