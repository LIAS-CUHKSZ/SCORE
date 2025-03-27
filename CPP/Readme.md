# ACM_BNB
## 1. data structure
```cpp
struct help_attribute{
    bool east_or_not;     // east or west sphere
    double norm_of_outer_product;  // norm of outer product
    double inner_product;  // inner product
    Vector3d outer_product;  // outer product
    Vector3d east_outer_product;  // east outer product
    Vector3d west_outer_product;  // west outer product
    Vector2d angle_east_outer_product;  // alpha and phi of east outer product
    Vector2d angle_west_outer_product;  // alpha and phi of west outer product

    Vector3d east_angle_bisector;  // east angle bisector
    Vector3d west_angle_bisector;  // west angle bisector
    Vector3d east_othogonal_angle_bisector;  // the vector orthogonal to the angle bisector in the east sphere
    Vector3d west_othogonal_angle_bisector;  // the vector orthogonal to the angle bisector in the west sphere
    Vector2d angle_east_angle_bisector;  // alpha and phi of east angle bisector
    Vector2d angle_west_angle_bisector;  // alpha and phi of west angle bisector
    Vector2d angle_east_othogonal_angle_bisector;  // alpha and phi of the vector orthogonal to the angle bisector in the east sphere
    Vector2d angle_west_othogonal_angle_bisector;  // alpha and phi of the vector orthogonal to the angle bisector in the west sphere
}
```
The `help_attribute` is used to store the information of the line pair. The alg will preprocess all attributes of the line pairs at the beginning. It will be cached in the `help_attribute` structure to avoid redundant computation.
## 2. class information
```cpp
class FGO_PnL{
    public:
        FGO_PnL(int max_iter,                   // maximum number of iterations
                double rotation_epsilon,       // rotation epsilon
                double branch_resolution,      // branch resolution, it will influence the depth of the branch
                double sample_resolution,      // sample resolution, it will influence the accuracy of the h2 bound
                double translation_epsilon,    // translation epsilon
                double translation_resolution, // translation resolution 
                string satured_kernel):         // satured kernel, it is used to choose the satured kernel
        max_iter(max_iter),
        rotation_epsilon(rotation_epsilon),
        branch_resolution(branch_resolution),
        sample_resolution(sample_resolution),
        translation_epsilon(translation_epsilon),
        translation_resolution(translation_resolution),
        satured_kernel(satured_kernel){
            // if(this->satured_kernel=="p-8"){
            //     for(int i=0;i<15;i++){
            //         this->satured_kernel_cache.push_back(pow((i+1), -8));
            //     }
            // }
        }  
}
```
## 3. important function
`pair<VectorXd,VectorXd> FGO_PnL::h1_bounds(Square &branch)`: calculate the h1 bounds of the branch
`pair<VectorXd,VectorXd> FGO_PnL::h2_bounds(Square &branch)`: calculate the h2 bounds of the branch
`vector<double> FGO_PnL::upper_interval(RowVector3d coef_1,RowVector3d coef_2)`: calculate the upper bound of the interval
`vector<double> FGO_PnL::lower_interval(RowVector3d coef)`: calculate the lower bound of the interval
`pair<double,double> FGO_PnL::satured_interval_stabbing(vector<double> &interval,vector<int> line_tags)`: calculate the satured interval of the interval

## 4. helpful tools
`interval_intersection`: get the intersection of two or four intervals
`xyz_2_polar` : turn the Cartesian coordinates to polar coordinates
`polar_2_xyz` : turn the polar coordinates to Cartesian coordinates
`interval_projection`: find the nearest point in the interval to the given point
`R_error`: calculate the error of two rotation matrices
`t_error`: calculate the error of two translation vectors
`rodrigues`: convert the rotation vector to rotation matrix
`quaternion2angles`: convert the quaternion to $\alpha$, $\phi$, $\theta$
`quaternion2rotvec`: convert the quaternion to rotation vector

# the functions need to parallel
`FGO_PnL::h1_bounds`: 所有的点都没有关系 可以分别计算
`FGO_PnL::h2_bounds`: 同上
`FGO_PnL::rot_bnb_estimate`: 分支的地方分四个支可以并行
`FGO_PnL::rot_bnb_epoch`: epoch中有一些往数组里填数的for 循环可以并行
`FGO_PnL::preprocess`: 这里并不并行无所谓, 因为我们的处理只会进行一次

