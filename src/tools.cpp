#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	//
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() != ground_truth.size()){
		cout << "Estimation and ground truth sizes do not match";
		return rmse;
	}

	// loop over all time instances
	int n = ground_truth.size();

	for(int i=0; i<n; i++){
		VectorXd err = estimations[i] - ground_truth[i];
		err = err.array() * err.array();
		rmse += err;
	}
	rmse = rmse/n;
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
    float rho = px*px + py*py;
	if(fabs(rho)<0.01){
	    std::cout<<"Error: Division by zero during Jacobian calculation";
	    Hj<< 0, 0, 0, 0,
	         0, 0, 0, 0,
	         0, 0, 0, 0;
	}else{
	    float v11 =  px/sqrt(rho);
	    float v12 =  py/sqrt(rho);
	    float v21 = -py/rho;
	    float v22 =  px/rho;
	    float v31 =  py*(vx*py - vy*px)/ (rho*sqrt(rho));
	    float v32 =  px*(vy*px - vx*py)/ (rho*sqrt(rho));

	    Hj << v11, v12,   0,   0,
	          v21, v22,   0,   0,
	          v31, v32, v11, v12;
	}

	return Hj;
}
