#include "fd_interpolate.h"
#include <vector>
#include <cmath>


typedef Eigen::Triplet<double> T;
void fd_interpolate(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const Eigen::RowVector3d & corner,
  const Eigen::MatrixXd & P,
  Eigen::SparseMatrix<double> & W)
{
  Eigen::MatrixXd indices;
  Eigen::MatrixXd residual;
  int max_num_values = 8 * P.rows();
  std::vector<T> weight_vec(max_num_values);

  // <indices> contains the indices of the voxel each point in P lives in
  // <residual> contains fractions b/w 1 and 0
  // the fractions are the normalized unit cube locations
  indices = P.rowwise() - corner;
  indices /= h;
  residual = indices;
  indices = indices.array().floor();
  residual -= indices;

  assert((residual.array() < 1).all());

  for (int idx = 0; idx < indices.rows(); idx++)
  {
    int col_idx;
    double value; 

    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < 2; j++)
      {
	for (int k = 0; k < 2; k++)
	{
	  // (2) calculate the weight of each of the 8 voxels using the equation
	  // here http://paulbourke.net/miscellaneous/interpolation/
	  col_idx = (int) (indices(idx, 0) + i) +
			  (indices(idx, 1) + j) * nx +
			  (indices(idx, 2) + k) * ny * nx;
	  
	  value = ((1-i)-residual(idx, 0)) *
	          ((1-j)-residual(idx, 1)) * 
	          ((1-k)-residual(idx, 2));

	  value = abs(value);

	  weight_vec.push_back(T(idx, col_idx, value));
	}
      }
    }
  }
  W.resize(indices.rows(), nx*ny*nz);
  W.setFromTriplets(weight_vec.begin(), weight_vec.end());
}
